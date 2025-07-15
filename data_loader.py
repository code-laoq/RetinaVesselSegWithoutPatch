import random

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import transforms

from pre_processing import my_PreProc

class ImageFolder(data.Dataset):
    def __init__(self, file_path,delimiter = ' ', image_size=256, mode='train', augmentation_prob=0.4,transform=None):
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

        self.image_size = image_size                                # 固定尺寸（如256）Original
        self.mode = mode
        self.augmentation_prob = augmentation_prob

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((584, 564)),  # 调整为 72x72
                transforms.ToTensor()
            ])
        else:
            self.transform = transform
        print(f"image count in {mode} path : {len(self.img_list)}")

    def __getitem__(self, index):
        # 加载图像和掩码
        image = Image.open(self.img_list[index])
        mask = Image.open(self.gt_list[index])
        img_proc = my_PreProc(image)  # H,W,1，float32
        # transform = T.Resize((self.image_size, self.image_size))
        # image = transform(image)
        # mask = transform(mask)
        # 转换为 Tensor 并标准化

        image= self.transform(img_proc)
        mask = self.transform(mask)
        #image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        mask = (mask > 0.5).float()  # 二值化掩码

        return image, mask

    def __len__(self):
        return len(self.img_list)

def get_loader(file_path, image_size=256, batch_size=16, num_workers=2, mode='train', augmentation_prob=0.4,transform=None):
    dataset = ImageFolder(
        file_path=file_path,
        image_size=image_size,
        mode=mode,
        augmentation_prob=augmentation_prob,
        transform=None
    )
    data_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True
    )
    return data_loader

def test_data_load():
    # 加载数据集
    train_loader = get_loader(file_path='/home/dell609/dl_pro/VesselSeg/UNet+/data_path_list/ARIA/train.txt', image_size=224, batch_size=16, mode='train',
        augmentation_prob=0.4)

    test_loader = get_loader(file_path='/home/dell609/dl_pro/VesselSeg/UNet+/data_path_list/ARIA/test.txt', image_size=224, batch_size=4, mode='test',
        augmentation_prob=0.0)

    # 检查数据形状
    for images, masks in train_loader:
        print(f"训练集图像形状：{images.shape}，掩码形状：{masks.shape}")  # 应输出 (16,3,224,224) 和 (16,1,224,224)
        break

    for images, masks in test_loader:
        print(f"测试集图像形状：{images.shape}，掩码形状：{masks.shape}")
        break



if __name__ == '__main__':
    test_data_load()
