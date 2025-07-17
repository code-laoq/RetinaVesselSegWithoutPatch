import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
from datetime import datetime

def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['U_Net', 'R2U_Net', 'AttU_Net', 'R2AttU_Net','ResU_Net']:
        print('ERROR!! model_type should be selected in U_Net/R2U_Net/AttU_Net/R2AttU_Net/ResU_Net')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    lr = random.random() * 0.0005 + 0.0000005
    augmentation_prob = random.random() * 0.7
    epoch = random.choice([100, 150, 200, 250])
    decay_ratio = random.random() * 0.8
    decay_epoch = int(epoch * decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = 10
    config.batch_size = 2
    config.lr = lr
    config.num_epochs_decay = decay_epoch
    height = 512
    width = 512
    # print(config)
    train_loader = get_loader(file_path=config.train_file_path,
        height=height,
        width=width,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='train',
        augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(file_path=config.valid_file_path,
        height=height,
        width=width,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='valid',
        augmentation_prob=0.)
    test_loader = get_loader(file_path=config.test_file_path,
        height=height,
        width=width,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        mode='test',
        augmentation_prob=0.)
    solver = Solver(config, train_loader, valid_loader, test_loader)

    # Train and sample the images
    if config.mode == 'train':
        solver.train(config)
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')

    # training hyper parameters
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--early_stop', default=130, type=int,
                        help='early stopping')

    # 预训练
    parser.add_argument('--start_epoch', default=1,
                        help='Start epoch')
    parser.add_argument('--pre_trained', default=False, type=bool,
                        help='(path of trained _model)load trained   '
                             'model to continue train')

    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./checkpoints')
    parser.add_argument('--train_file_path', type=str, default='/home/dell609/dl_pro/VesselSeg/UNet+/data_path_list/ARIA/train.txt')
    parser.add_argument('--valid_file_path', type=str, default='/home/dell609/dl_pro/VesselSeg/UNet+/data_path_list/ARIA/test.txt')
    parser.add_argument('--test_file_path', type=str, default='/home/dell609/dl_pro/VesselSeg/UNet+/data_path_list/ARIA/val.txt')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    config = parser.parse_args()
    main(config)