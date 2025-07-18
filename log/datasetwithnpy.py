import numpy as np
import torch
from torch.utils import data
import cv2
import torchvision.transforms.functional as F

class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8,8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit,
                                      tileGridSize=tileGridSize)
    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        # 接受 H×W 或 H×W×C（RGB）的 uint8 numpy
        # 如果是彩色，先转灰度
        if img_np.ndim == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        return self.clahe.apply(img_np)

class ImageFolder(data.Dataset):
    def __init__(self, file_path,delimiter = ' ', height=512,width=512, mode='train', augmentation_prob=0.4):
        self.height = height
        self.width = width
        self.mode = mode
        self.aug_p = augmentation_prob
        # 用 numpy 层面的 CLAHE，替代 PIL 层
        self.clahe_np = CLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.color_gamma = 1.2

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

        print(f"image count in {mode} path : {len(self.img_list)} from {file_path}")

    def __getitem__(self, idx):
        # 1) mmap 读取 numpy，确保 dtype 是 uint8
        img = np.load(self.img_list[idx], mmap_mode='r')
        mask = np.load(self.gt_list[idx], mmap_mode='r')
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)

        # 2) 可选 resize（用 cv2 性能更好）
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # 3) CLAHE 均衡（输出 H×W uint8）
        img = self.clahe_np(img)

        # 4) 转成 torch.Tensor, [C×H×W], 归一化到 [0,1]
        #    注意：此时 img 是单通道灰度
        img_tensor = torch.tensor(img).unsqueeze(0).float().div(255.0)
        mask_tensor = torch.tensor(mask).unsqueeze(0).float().div(255.0)

        # 5) 在线随机增强（针对 tensor）
        #    比如随机水平翻转
        # if self.mode == 'train' and torch.rand(1).item() < self.aug_p:
        #     img_tensor = F.hflip(img_tensor)
        #     mask_tensor = F.hflip(mask_tensor)

        #    Gamma 校正
        img_tensor = F.adjust_gamma(img_tensor, self.color_gamma)

        #    你可以再加 ColorJitter、RandomCrop、随机旋转……全部用
        #    torchvision.transforms.functional 里的函数，它们都支持 tensor。

        # 6) 二值化 mask
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img_list)