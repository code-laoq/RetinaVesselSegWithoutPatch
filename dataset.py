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
        # 加载npy文件
        image = np.load(self.img_list[index], mmap_mode="r")
        mask = np.load(self.gt_list[index], mmap_mode="r")
        fov = np.load(self.fov_list[index])

        # 将 NumPy 数组转换为 PIL 图像
        # 假设 image 是 uint8 类型的 (H, W, C) 或 (H, W) 数组
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)  # 如果是 float32，转换为 uint8
        image = Image.fromarray(image)
        # # 假设 mask 是 uint8 类型的 (H, W) 数组
        mask = Image.fromarray(mask)
        # 如果需要处理 FOV
        fov = Image.fromarray(fov)

        image= self.transform(image)
        mask = self.mask_transform(mask)
        # print(f"image.shape:{image.shape}\tmask.shape:{mask.shape}")

        mask = (mask > 0.5).float()  # 二值化掩码

        return image, mask