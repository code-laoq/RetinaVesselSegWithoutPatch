from torch.utils import data

from dataset import ImageFolder

def get_loader(file_path, image_size=256, batch_size=16, num_workers=2, mode='train', augmentation_prob=0.4,transform=None):
    """
    加载数据集
    file_path是存放图像路径的文本文件
    image是要resize的尺寸
    """
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
