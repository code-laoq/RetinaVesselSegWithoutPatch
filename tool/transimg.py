import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_to_single_channel(input_path, output_dir):
    """
    将三通道掩码图转换为单通道掩码图并保存。

    参数:
        input_path (str): 输入三通道掩码图的路径。
        output_dir (str): 保存单通道掩码图的目录。
    """
    # 加载图像
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 {input_path}")
        return
    print(f"img.shape:{img.shape}")
    # 转换为灰度图（单通道）
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取文件名并构建输出路径
    file_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, file_name)
    print(f"处理后img.shape:{gray_img.shape}")

    # 保存单通道图像
    cv2.imwrite(output_path, gray_img)
    # print(f"已保存单通道掩码到 {output_path}")


def process_mask_paths(txt_file, output_dir):
    """
    从文本文件中读取掩码路径并将其转换为单通道。

    参数:
        txt_file (str): 包含掩码图路径的文本文件路径。
        output_dir (str): 保存转换后单通道掩码的目录。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 从文本文件中读取路径
    with open(txt_file, 'r') as f:
        paths = f.readlines()

    # 处理每个路径
    for path in paths:
        path = path.strip()  # 移除首尾空白字符
        if path:
            convert_to_single_channel(path, output_dir)

def process_and_save(image_path, save_dir, is_grayscale=False):
    """
    加载图像，将其转换为NumPy数组，并将其保存为.npy文件。
    Args:
        image_path (str): Path to the image file.
        save_dir (str): Directory to save the .npy file.
        is_grayscale (bool): Whether to load the image as grayscale.
    """
    try:
        # Load image
        with Image.open(image_path) as img:
            if is_grayscale:
                img = img.convert('L')  # Convert to grayscale (2D)
            else:
                img = img.convert('RGB')  # Convert to RGB (3D)

            # Convert to NumPy array
            img_array = np.array(img)

            # Extract filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(save_dir, f"{base_name}.npy")

            # Save as .npy file
            np.save(save_path, img_array)
            print(f"Saved {save_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def convert_to_npy(txt_file, save_dirs):
    """
    Read the text file and process all images, saving them to specified directories.

    Args:
        txt_file (str): Path to the text file with image paths.
        save_dirs (dict): Dictionary with save directories for images, labels, and masks.
    """
    # Create save directories if they don’t exist
    for dir_path in save_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # Read the text file
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Process each line
    for line in tqdm(lines, desc="Processing samples"):
        paths = line.strip().split()
        if len(paths) != 3:
            print(f"Invalid line format: {line}")
            continue

        image_path, label_path, mask_path = paths

        # Process original image (RGB)
        process_and_save(image_path, save_dirs['images'], is_grayscale=False)

        # Process mask annotation (grayscale)
        process_and_save(label_path, save_dirs['labels'], is_grayscale=True)

        # Process FOV image (grayscale)
        process_and_save(mask_path, save_dirs['masks'], is_grayscale=True)

def to_single_channel_img():
    txt_file = '/prepare_dataset/data_path_list/FIVES/train1.txt'  # 替换为你的文本文件路径
    output_dir = '/home/dell609/dl_pro/VesselSeg/PANet/Datasets/FIVES/train/label'  # 替换为你的输出目录
    process_mask_paths(txt_file, output_dir)

if __name__ == "__main__":
    # Text file with image paths
    txt_file = "./prepare_dataset/data_path_list/ARIA/train.txt"

    # Define save directories
    save_dirs = {
        'images': "/home/dell609/dl_pro/VesselSeg/PANet/Datasets_npy/ARIA/Train/images",
        'labels': "/home/dell609/dl_pro/VesselSeg/PANet/Datasets_npy/ARIA/Train/labels",
        'masks':  "/home/dell609/dl_pro/VesselSeg/PANet/Datasets_npy/ARIA/Train/masks"
    }

    # Run the conversion
    convert_to_npy(txt_file, save_dirs)

