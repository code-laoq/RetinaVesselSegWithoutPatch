from copy import deepcopy

import numpy as np
import cv2
from PIL import Image

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """输入 H×W×3，输出 H×W"""
    return (rgb[..., 0]*0.299 +
            rgb[..., 1]*0.587 +
            rgb[..., 2]*0.114).astype(np.uint8)

def clahe_equalized(img: np.ndarray) -> np.ndarray:
    """输入 H×W uint8，输出 H×W uint8"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)

def adjust_gamma(img: np.ndarray, gamma: float=1.0) -> np.ndarray:
    """输入 H×W uint8，输出 H×W uint8"""
    invGamma = 1.0 / gamma
    table = np.array([((i/255.0)**invGamma)*255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img, table)

def my_PreProc(data: np.ndarray) -> np.ndarray:
    """
    对单张图像做预处理，返回 H×W×1 的 float32 数组，值域 [0,1]
    支持输入形状 [H,W,3] 或 [H,W]
    """
    data = np.asarray(data)
    if data.ndim == 3 and data.shape[2] == 3:
        gray = rgb2gray(data)
    elif data.ndim == 2:
        gray = data.astype(np.uint8)
    else:
        raise ValueError(f"不支持的输入形状：{data.shape}")

    clahe = clahe_equalized(gray)
    gamma = adjust_gamma(clahe, 1.2)
    output = gamma.astype(np.float32) / 255.0
    output = np.expand_dims(output, axis=-1)  #在数组的最后一个维度上添加一个新的维度，从[h,w]——》[h,w,1]
    return output

if __name__ == '__main__':
    img = np.asarray(
        Image.open('/home/dell609/dl_pro/VesselSeg/PANet/Datasets/FIVES/test/labels/3_A.png'),
        dtype=np.uint8
    )
    # print(f"处理前shape：{img.shape}")
    print(f"处理前shape：{img.shape}")

    proc = my_PreProc(img)

    # #proc = (proc > 0.3)  # 二值化掩码
    #
    # # binary = deepcopy(proc)
    # # binary[binary >= 0.5] = 1
    # # binary[binary < 0.5] = 0
    #
    print(f"处理后shape：{proc.shape}")  # 应为 (H, W) 或 (H, W, 1)
    # 再转回 0–255 uint8
    # out = (proc[...,0] * 255).astype(np.uint8)
    # Image.fromarray(out).save('output.png')
    # Image.fromarray(out).save('output.png')