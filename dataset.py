import argparse
import os
import random
import shutil
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import torchvision.transforms as T
import torchvision.transforms.functional as F

class CLAHE:
    """在 PIL Image（灰度）上执行 OpenCV CLAHE"""
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                      tileGridSize=tileGridSize)

    def __call__(self, img: Image.Image) -> Image.Image:
        # 转成 numpy gray H×W
        arr = np.array(img, dtype=np.uint8)
        eq  = self.clahe.apply(arr)
        return Image.fromarray(eq)

class ImageFolder(data.Dataset):
    def __init__(self, file_path,delimiter = ' ', height=512,width=512, mode='train', augmentation_prob=0.4,transform=None):
        self.img_list = []
        self.gt_list = []
        self.fov_list = []
        """
        读取一个文本文件，里面存储了图像路径，返回三个列表img_list, gt_list, fov_list，分别存original image、GT和FOV的路径
        每一行存储三个路径，用指定分隔符分开。        
        """
        try:
            with open(file_path, 'r') as file_to_read:
                for line in file_to_read:
                    line = line.strip()  #读一行路径
                    if line:
                        parts = line.split(delimiter)  #用delimiter拆分成3个地址
                        if len(parts) != 3:
                            raise ValueError(f"Expected 3 paths per line, got {len(parts)}")
                        self.img_list.append(parts[0])
                        self.gt_list.append(parts[1])
                        self.fov_list.append(parts[2])
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {str(e)}")

        # self.image_size = image_size                                # 固定尺寸（如256）Original
        self.mode = mode
        self.augmentation_prob = augmentation_prob

        if transform is None:
            self.transform = T.Compose([
            T.Resize((width, height)),                              #调整图像尺寸
            T.Grayscale(num_output_channels=1),                     #灰度化，输出 1 通道 PIL Image
            CLAHE(clipLimit=2.0, tileGridSize=(8, 8)),              #CLAHE 均衡（仍然 PIL Image）
            T.Lambda(lambda img: F.adjust_gamma(img, gamma=1.2)),   #Gamma 校正，使用 torchvision 自带的 functional
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image),     #标准化
            T.ToTensor(),                                           #把 H×W×1 的 PIL → [1×H×W] 的 float tensor (0–1)
        ])
        else:
            self.transform = transform

        self.mask_transform=T.Compose([
            T.Resize((width, height)),
            T.ToTensor(),
        ])
        print(f"image count in {mode} path : {len(self.img_list)}")

    def __getitem__(self, index):
        # 加载图像和掩码
        image = Image.open(self.img_list[index])
        mask = Image.open(self.gt_list[index]).convert('L')

        image= self.transform(image)
        mask = self.mask_transform(mask)
        # print(f"image.shape:{image.shape}")
        # print(f"mask.shape:{mask.shape}")

        mask = (mask > 0.5).float()  # 二值化掩码

        return image, mask

    def __len__(self):
        return len(self.img_list)

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)

def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)     #获取路径下的所有文件和文件夹的名称
    data_list = []
    GT_list = []

    for filename in filenames:
        """
        规范化文件名
        os.path.splitext(filename)将filename 拆分成一个元组
        假如filename为‘ISIC_0000079.jpg’，那么os.path.splitext(filename)将返回一个元组，['ISIC_0000079','.jpg']
        filename.split('_')根据_划分字符串，然后放到列表中；[:-len('.jpg')]相当于[:-4]，表示从字符串的开头到倒数第 4 个字符（不包括
        """
        ext = os.path.splitext(filename)[-1]            #获取文件后缀名，只处理jpg文件；索引[-1]是取元组中的最后一个元素
        if ext =='.jpg':
            filename = filename.split('_')[-1][:-len('.jpg')]           #去掉文件前缀和后缀名;，然后取出最后一个，再去除掉.jpg后缀名
            data_list.append('ISIC_'+filename+'.jpg')
            GT_list.append('ISIC_'+filename+'_segmentation.png')

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arrange = list(range(num_total))
    random.shuffle(Arrange)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='../ISIC/dataset/ISIC2018_Task1-2_Training_Input')
    parser.add_argument('--origin_GT_path', type=str, default='../ISIC/dataset/ISIC2018_Task1_Training_GroundTruth')

    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset/valid_GT/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--test_GT_path', type=str, default='./dataset/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)