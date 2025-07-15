import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms.functional import to_pil_image
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.classification import Specificity, Precision, Recall  # recall==sensitivity
from torchmetrics.segmentation import DiceScore
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from ResUNet import ResUNet
from PIL import Image
# 反标准化函数

def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(min=0.0, max=1.0)
    return (tensor * 255).type(torch.uint8)

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.unet = None
        self.optimizer = None
        self.device = torch.device('cuda:0')

        # loaders
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # params
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.model_type = config.model_type
        self.t = getattr(config, 't', 2)

        # loss for 2-class segmentation
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics for multiclass (2 classes: background vs vessel)
        self.metric_acc = Accuracy(task="multiclass", num_classes=2).to(self.device)
        self.metric_se  = Recall(task="multiclass", num_classes=2, average='macro').to(self.device)
        self.metric_sp  = Specificity(task="multiclass", num_classes=2, average='macro').to(self.device)
        self.metric_pc  = Precision(task="multiclass", num_classes=2, average='macro').to(self.device)
        self.metric_js  = JaccardIndex(task="multiclass", num_classes=2).to(self.device)
        # use input_format='index' for class indices format
        self.metric_dc  = DiceScore(num_classes=2, average='micro', input_format='index').to(self.device)

        # build
        self.build_model()
        self.scheduler = StepLR(self.optimizer, step_size=self.num_epochs_decay, gamma=0.1)

    def build_model(self):
        # output_ch=2 for background and vessel
        if self.model_type == 'U_Net':
            model = U_Net(1,2)
        elif self.model_type == 'R2U_Net':
            model = R2U_Net(img_ch=1, output_ch=2, t=self.t)
        elif self.model_type == 'AttU_Net':
            model = AttU_Net(img_ch=1, output_ch=2)
        elif self.model_type == 'ResU_Net':
            model =ResUNet(1,2)
        else:
            model = R2AttU_Net(img_ch=1, output_ch=2, t=self.t)
        model.to(self.device)
        self.unet = model
        self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def reset_grad(self):
        self.optimizer.zero_grad()

    def train_epoch(self, loader):
        self.unet.train()
        running_loss = 0.0
        # reset metrics
        for m in (self.metric_acc, self.metric_se, self.metric_sp,
                  self.metric_pc, self.metric_js, self.metric_dc):
            m.reset()

        for images, GT in loader:
            images = images.to(self.device)
            gt_idx = GT.squeeze(1).long().to(self.device)

            logits = self.unet(images)            # [B,2,H,W]
            loss = self.criterion(logits, gt_idx)

            self.reset_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)

            # update metrics
            self.metric_acc.update(preds, gt_idx)
            self.metric_se .update(preds, gt_idx)
            self.metric_sp .update(preds, gt_idx)
            self.metric_pc .update(preds, gt_idx)
            self.metric_js .update(preds, gt_idx)
            self.metric_dc .update(preds, gt_idx)

        n = len(loader.dataset)
        epoch_loss = running_loss / n
        results = {
            'loss': epoch_loss,
            'acc':  self.metric_acc.compute().item(),
            'SE':   self.metric_se.compute().item(),
            'SP':   self.metric_sp.compute().item(),
            'PC':   self.metric_pc.compute().item(),
            'JS':   self.metric_js.compute().item(),
            'DC':   self.metric_dc.compute().item()
        }
        return results

    def validate_epoch(self, loader, epoch):
        self.unet.eval()
        running_loss = 0.0
        for m in (self.metric_acc, self.metric_se, self.metric_sp,
                  self.metric_pc, self.metric_js, self.metric_dc):
            m.reset()

        with torch.no_grad():
            result_dir = os.path.join(self.result_path, f'epoch_{epoch + 1}')
            os.makedirs(result_dir, exist_ok=True)
            for i, (images, GT) in enumerate(loader):
                images = images.to(self.device)
                gt_idx = GT.squeeze(1).long().to(self.device)

                logits = self.unet(images)
                loss = self.criterion(logits, gt_idx)
                running_loss += loss.item() * images.size(0)

                preds = torch.argmax(logits, dim=1)
                self.metric_acc.update(preds, gt_idx)
                self.metric_se .update(preds, gt_idx)
                self.metric_sp .update(preds, gt_idx)
                self.metric_pc .update(preds, gt_idx)
                self.metric_js .update(preds, gt_idx)
                self.metric_dc .update(preds, gt_idx)

                # 保存预测结果
                self._save_predictions(images, logits, GT, result_dir, epoch, i)
        n = len(loader.dataset)
        epoch_loss = running_loss / n
        results = {
            'loss': epoch_loss,
            'acc':  self.metric_acc.compute().item(),
            'SE':   self.metric_se.compute().item(),
            'SP':   self.metric_sp.compute().item(),
            'PC':   self.metric_pc.compute().item(),
            'JS':   self.metric_js.compute().item(),
            'DC':   self.metric_dc.compute().item()
        }
        return results

    def _save_predictions(self, images, logits, GT, result_dir, epoch, batch_idx):
        # restore to multiclass probability
        probs = torch.softmax(logits, dim=1)
        for idx in range(images.size(0)):
            # save original image
            image_tensor = images[idx].cpu().detach()
            image_pil = to_pil_image(image_tensor)
            image_pil.save(os.path.join(result_dir, f'batch_{batch_idx}_img_{idx}_image.png'))

            # save vessel probability map (class=1 channel)
            vessel_prob = probs[idx,1]
            arr = (vessel_prob * 255).type(torch.uint8).cpu().numpy()
            vessel_pil = Image.fromarray(arr, mode='L')
            vessel_pil.save(os.path.join(result_dir, f'batch_{batch_idx}_img_{idx}_vessel.png'))

            # save gt mask
            gt_arr = (GT[idx].squeeze(0) * 255).type(torch.uint8).cpu().numpy()
            gt_pil = Image.fromarray(gt_arr, mode='L')
            gt_pil.save(os.path.join(result_dir, f'batch_{batch_idx}_img_{idx}_gt.png'))

    def train(self):
        best_score = 0.0
        for epoch in range(1, self.num_epochs+1):
            train_res = self.train_epoch(self.train_loader)
            val_res   = self.validate_epoch(self.valid_loader, epoch)

            print(f"epoch:{epoch} [Train] Loss: {train_res['loss']:.4f}  SE: {train_res['SE']:.4f}  SP: {train_res['SP']:.4f}  ACC: {train_res['acc']:.4f}  "
                  f"F1: {train_res['DC']:.4f}  PC: {train_res['PC']:.4f}  IOU: {train_res['JS']:.4f}")
            print(f"[Valid] Loss: {val_res['loss']:.4f}  SE: {val_res['SE']:.4f}  SP: {val_res['SP']:.4f}  ACC: {val_res['acc']:.4f}  "
                f"F1: {val_res['DC']:.4f}  PC: {val_res['PC']:.4f}  IOU: {val_res['JS']:.4f}")

            score = val_res['JS'] + val_res['DC']
            if score > best_score:
                best_score = score
                torch.save(self.unet.state_dict(), os.path.join(self.model_path, f"{self.model_type}_best.pth"))
                print(f"📌 Best model saved at epoch {epoch}, score={best_score:.4f}")

            self.scheduler.step()