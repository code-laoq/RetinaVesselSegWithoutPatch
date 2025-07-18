import os,sys,time
from datetime import datetime
from os.path import join
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.classification import Specificity, Precision, Recall,AveragePrecision, AUROC  # recall==sensitivity
from logs.logger import Print_Logger
from models.UNetFamily import UNet,ResUNet
from PIL import Image
from tools import concat_result

class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        self.args=config
        self.early_stop = config.early_stop
        self.unet = None
        self.optimizer = None
        self.device = config.device
        # AMP
        self.scaler = GradScaler()
        self.autocast = autocast
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
        self.metric_acc = Accuracy(task="binary", threshold=0.5).to(self.device)
        self.metric_se  = Recall(task="binary", threshold=0.5).to(self.device)
        self.metric_sp  = Specificity(task="binary", threshold=0.5).to(self.device)
        self.metric_pc  = Precision(task="binary", threshold=0.5).to(self.device)
        self.metric_js  = JaccardIndex(task="binary", threshold=0.5).to(self.device)
        self.metric_auprc = AveragePrecision(task="binary").to(self.device)
        self.metric_auroc = AUROC(task="binary").to(self.device)
        # build
        self.build_model()
        self.optimizer = optim.Adam(self.unet.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = StepLR(self.optimizer, step_size=self.num_epochs_decay, gamma=0.1)

    def build_model(self):
        # output_ch=2 for background and vessel
        if self.model_type == 'UNet':
            model = UNet(1,2)
        else:
            model =ResUNet(1,2)

        model.to(self.device)
        self.unet = model
        print(f"train {self.model_type}")

    def reset_grad(self):
        self.optimizer.zero_grad()

    def _run_epoch(self, loader, is_train=True, epoch=None):
        # 设置模型模式
        if is_train:
            self.unet.train()
        else:
            self.unet.eval()

        running_loss = 0.0
        # 重置指标
        for m in (self.metric_acc, self.metric_se, self.metric_sp,
        self.metric_pc, self.metric_js,
        self.metric_auprc, self.metric_auroc):
            m.reset()

        # 如果是验证模式，创建结果保存目录
        if not is_train:
            result_dir = join(self.result_path, f'epoch_{epoch + 1}')
            os.makedirs(result_dir, exist_ok=True)

        # 梯度累积步数（可根据显存和需求调整）
        accumulation_steps = 4  # 例如累积4个批次

        for i, (images, GT) in enumerate(loader):
            images = images.to(self.device)
            gt_idx = GT.squeeze(1).long().to(self.device)

            if is_train:
                with self.autocast():
                    logits = self.unet(images)
                    loss = self.criterion(logits, gt_idx)
                running_loss += loss.item() * images.size(0)

                # 更新指标用 fp32 预测
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                self.metric_acc.update(preds, gt_idx)
                self.metric_se.update(preds, gt_idx)
                self.metric_sp.update(preds, gt_idx)
                self.metric_pc.update(preds, gt_idx)
                self.metric_js.update(preds, gt_idx)
                self.metric_auprc.update(probs[:, 1].flatten(), gt_idx.flatten())
                self.metric_auroc.update(probs[:, 1].flatten(), gt_idx.flatten())

                # 缩放损失并反向传播，累积梯度
                self.scaler.scale(loss).backward()

                # 在累积步数达到时更新参数并重置梯度
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.reset_grad()
            else:
                # 验证模式下只需前向传播
                logits = self.unet(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                self.metric_acc.update(preds, gt_idx)
                self.metric_se.update(preds, gt_idx)
                self.metric_sp.update(preds, gt_idx)
                self.metric_pc.update(preds, gt_idx)
                self.metric_js.update(preds, gt_idx)
                self.metric_auprc.update(probs[:, 1].flatten(), gt_idx.flatten())
                self.metric_auroc.update(probs[:, 1].flatten(), gt_idx.flatten())

                loss = self.criterion(logits, gt_idx)
                running_loss += loss.item() * images.size(0)
                self._save_predictions(images, logits, GT, result_dir, i)

        # 计算指标
        n = len(loader.dataset)
        epoch_loss = running_loss / n
        se = self.metric_se.compute()
        pc = self.metric_pc.compute()
        dice_score = 2 * (se * pc) / (se + pc + 1e-6)

        results = {
            'loss': epoch_loss,
            'acc': self.metric_acc.compute().item(),
            'SP': self.metric_sp.compute().item(),
            'JS': self.metric_js.compute().item(),
            'AUPRC': self.metric_auprc.compute().item(),
            'AUROC': self.metric_auroc.compute().item(),
            'SE': se.item(),
            'PC': pc.item(),
            'DC': dice_score.item()
        }
        return results

    def train_epoch(self, loader):
        """训练一个 epoch"""
        return self._run_epoch(loader, is_train=True)

    def validate_epoch(self, loader, epoch):
        """验证一个 epoch"""
        return self._run_epoch(loader, is_train=False, epoch=epoch)

    @staticmethod
    def _save_predictions(images, logits, GT, result_dir, batch_idx):
        # 恢复为多类概率
        probs = torch.softmax(logits, dim=1)
        for idx in range(images.size(0)):
            # 获取原始图像
            ori_img = images[idx].cpu().detach().numpy().transpose(1, 2, 0)
            # 如果 ori_img 是 [0,1] 范围，转换为 [0,255]
            if ori_img.max() <= 1:
                ori_img = ori_img * 255

            # 获取预测结果（血管概率）
            pred_res = probs[idx, 1].cpu().detach().numpy()  # (H, W)

            # 获取真实标签
            gt = GT[idx].squeeze(0).cpu().detach().numpy()  # (H, W)

            # 拼接图像
            total_img = concat_result(ori_img, pred_res, gt)
            if total_img.ndim == 3 and total_img.shape[2] == 1:
                # 单通道图像，移除通道维度并指定模式为 'L'
                total_img = total_img.squeeze(2)  # 形状从 (H, 4*W, 1) 变为 (H, 4*W)
                total_pil = Image.fromarray(total_img, mode='L')
            else:
                # 多通道图像（例如 RGB），直接转换
                total_pil = Image.fromarray(total_img)  # 默认模式为 'RGB'

            # 保存拼接图像
            total_pil.save(join(result_dir, f'batch_{batch_idx}_img_{idx}_concat.png'))

    def train(self,args):
        # 获取当前日期并生成文件名
        today = datetime.now().strftime("%m%d")
        log_path = './logs'  #日志路径
        log_filename = f"{self.model_type}_{today}r.log"
        sys.stdout = Print_Logger(os.path.join(log_path, log_filename))
        print("The computing device used is:" + self.device)

        #如果保存了模型权重，则进行预训练
        if args.pre_trained:
            # Load checkpoint.   checkpoint相当于保存模型的参数，优化器参数，loss，epoch的文件夹
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(join(self.model_path,f"{self.model_type}_latest.pth"),
                map_location=args.device)
            self.unet.load_state_dict(checkpoint['net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1

        best = {'epoch': 0, 'score': 0.5}
        trigger = 0                                             #初始化早停累加器
        for epoch in range(args.start_epoch, self.num_epochs+1):
            train_res = self.train_epoch(self.train_loader)
            val_res   = self.validate_epoch(self.valid_loader, epoch)

            self.scheduler.step()

            #打印当前epoch、学习率和时间
            print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
                  (epoch, self.num_epochs, self.optimizer.state_dict()['param_groups'][0]['lr'],time.asctime()))

            print(f"[Train] Loss: {train_res['loss']:.4f}  SE: {train_res['SE']:.4f}  SP: {train_res['SP']:.4f}  ACC: {train_res['acc']:.4f}  "
                  f"F1: {train_res['DC']:.4f}  PC: {train_res['PC']:.4f}  IOU: {train_res['JS']:.4f}  PR: {train_res['AUPRC']:.4f}  AUROC: {train_res['AUROC']:.4f}")
            print(f"[Valid] Loss: {val_res['loss']:.4f}  SE: {val_res['SE']:.4f}  SP: {val_res['SP']:.4f}  ACC: {val_res['acc']:.4f}  "
                  f"F1: {val_res['DC']:.4f}  PC: {val_res['PC']:.4f}  IOU: {val_res['JS']:.4f}  PR: {val_res['AUPRC']:.4f}  AUROC: {val_res['AUROC']:.4f}")

            #保存当前epoch的模型（权重、优化器状态和epoch，可以在下一次接着训练）
            state = {'net': self.unet.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, join(self.model_path,f"{self.model_type}_latest.pth"))
            trigger += 1
            score =  0.5*val_res['AUPRC']+ 0.5*val_res['DC']        #IOU、SE和F1分别占比为0.2、0.3和0.5

            if score > best['score']:
                best['score'] = score
                best['epoch'] = epoch
                trigger = 0
                torch.save(state, join(self.model_path, f"{self.model_type}_best.pth"))
                print(f"📌 Best model saved at epoch {epoch}, score={best['score']:.4f}")

            if trigger >= self.early_stop:
                print("=> early stopping")
                break

        torch.save(state, join(self.model_path, f"{self.model_type}_best.pth"))

    def test(self):
        """
        测试接口：在 test_loader 上评估已保存的最佳模型，并保存预测图与指标。
        args.device 应与训练时保持一致，例如 'cuda:0' 或 'cpu'。
        """
        # 1. 加载最优模型参数
        best_path = join(self.model_path, f"{self.model_type}_latest.pth")
        assert os.path.exists(best_path), f"找不到最佳模型文件: {best_path}"
        print(f"==> Loading best model from {best_path}")
        checkpoint = torch.load(best_path, map_location=self.device)
        # 如果保存时只保存了 state_dict，直接 load_state_dict(...)
        # 如果像训练时保存了 {'net':state_dict,...}
        if isinstance(checkpoint, dict) and 'net' in checkpoint:
            self.unet.load_state_dict(checkpoint['net'])
        else:
            self.unet.load_state_dict(checkpoint)
        self.unet.to(self.device)
        self.unet.eval()

        # 2. 重置所有指标
        for m in (self.metric_acc, self.metric_se, self.metric_sp,
                  self.metric_pc, self.metric_js,
                  self.metric_auprc, self.metric_auroc):
            m.reset()

        # 3. 创建测试结果保存目录
        test_dir = join(self.result_path, "test")
        os.makedirs(test_dir, exist_ok=True)

        # 4. 遍历 test_loader，不计算梯度，保存每个 batch 的预测图
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, GT) in enumerate(self.test_loader):
                images = images.to(self.device)
                gt_idx = GT.squeeze(1).long().to(self.device)

                logits = self.unet(images)               # [B,2,H,W]
                probs  = torch.softmax(logits, dim=1)    # [B,2,H,W]
                preds  = torch.argmax(logits, dim=1)     # [B,H,W]

                # 更新指标
                self.metric_acc.update(preds, gt_idx)
                self.metric_se .update(preds, gt_idx)
                self.metric_sp .update(preds, gt_idx)
                self.metric_pc .update(preds, gt_idx)
                self.metric_js .update(preds, gt_idx)
                self.metric_auprc.update(probs[:,1].flatten(), gt_idx.flatten())
                self.metric_auroc.update(probs[:,1].flatten(), gt_idx.flatten())

                # 累计损失
                loss = self.criterion(logits, gt_idx)
                total_loss += loss.item() * images.size(0)

                # 保存该 batch 的预测可视化
                self._save_predictions(images, logits, GT, test_dir, i)

        # 5. 计算并打印测试集整体指标
        n = len(self.test_loader.dataset)
        avg_loss = total_loss / n
        se  = self.metric_se.compute()
        pc  = self.metric_pc.compute()
        dice= 2 * (se * pc) / (se + pc + 1e-6)

        results = {
            'loss':  avg_loss,
            'acc':   self.metric_acc.compute().item(),
            'SE':    se.item(),
            'SP':    self.metric_sp.compute().item(),
            'PC':    pc.item(),
            'JS':    self.metric_js.compute().item(),
            'DC':    dice.item(),
            'AUPRC': self.metric_auprc.compute().item(),
            'AUROC': self.metric_auroc.compute().item(),
        }

        print("===== Test Results =====")
        print(f"Loss:  {results['loss']:.4f}")
        print(f"ACC:   {results['acc']:.4f}")
        print(f"SE:    {results['SE']:.4f}")
        print(f"SP:    {results['SP']:.4f}")
        print(f"PC:    {results['PC']:.4f}")
        print(f"JS:    {results['JS']:.4f}")
        print(f"DC(F1):{results['DC']:.4f}")
        print(f"AUPRC: {results['AUPRC']:.4f}")
        print(f"AUROC: {results['AUROC']:.4f}")

        # 返回结果字典，供外部脚本进一步使用
        return results