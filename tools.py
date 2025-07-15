import numpy as np

def concat_result(ori_img, pred_res, gt):
    """将原始图像、预测结果、二值化图像和真实标签拼接成一张结果图"""
    # 确保输入是 (H, W, C) 格式
    if len(ori_img.shape) == 2:
        ori_img = ori_img[:, :, np.newaxis]
    if len(pred_res.shape) == 2:
        pred_res = pred_res[:, :, np.newaxis]
    if len(gt.shape) == 2:
        gt = gt[:, :, np.newaxis]

    # 二值化预测结果，阈值为 0.5
    binary = (pred_res >= 0.5).astype(np.float32)

    # 根据原始图像的通道数调整其他图像
    if ori_img.shape[2] == 3:  # 彩色图像
        pred_res = np.repeat(pred_res, 3, axis=2)
        binary = np.repeat(binary, 3, axis=2)
        gt = np.repeat(gt, 3, axis=2)
    else:  # 单通道图像
        pred_res = pred_res[:, :, 0:1]
        binary = binary[:, :, 0:1]
        gt = gt[:, :, 0:1]

    # 归一化到 [0, 255] 并转换为 uint8
    ori_img = (ori_img * 255).astype(np.uint8) if ori_img.max() <= 1 else ori_img.astype(np.uint8)
    pred_res = (pred_res * 255).astype(np.uint8)
    binary = (binary * 255).astype(np.uint8)
    gt = (gt * 255).astype(np.uint8)

    # 水平拼接
    total_img = np.concatenate((ori_img, pred_res, binary, gt), axis=1)
    return total_img