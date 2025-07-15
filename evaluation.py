import torch

EPS = 1e-6

def _binarize_pred(logits, threshold=0.5):
    """对网络输出做 sigmoid + 二值化"""
    prob = torch.sigmoid(logits)
    return prob > threshold  # 返回 bool Tensor

def _binarize_gt(mask_gt, threshold=0.5):
    """对加载后的 mask 做二值化"""
    return mask_gt > threshold  # mask_gt 应该是 float 或者 bool

def _get_confusion(pred, gt):
    """
    计算 TP, FP, TN, FN，均返回 float Tensor
    pred, gt: bool Tensor
    """
    TP = torch.logical_and(pred,  gt).sum().float()
    FP = torch.logical_and(pred, ~gt).sum().float()
    TN = torch.logical_and(~pred, ~gt).sum().float()
    FN = torch.logical_and(~pred,  gt).sum().float()
    return TP, FP, TN, FN

def get_accuracy(logits, mask_gt, threshold=0.5):
    pred = _binarize_pred(logits, threshold)
    gt   = _binarize_gt(mask_gt, threshold)
    TP, FP, TN, FN = _get_confusion(pred, gt)
    total = TP + TN + FP + FN + EPS
    return ((TP + TN) / total).item()

def get_sensitivity(logits, mask_gt, threshold=0.5):
    pred = _binarize_pred(logits, threshold)
    gt   = _binarize_gt(mask_gt, threshold)
    TP, FP, TN, FN = _get_confusion(pred, gt)
    return (TP / (TP + FN + EPS)).item()

def get_specificity(logits, mask_gt, threshold=0.5):
    pred = _binarize_pred(logits, threshold)
    gt   = _binarize_gt(mask_gt, threshold)
    TP, FP, TN, FN = _get_confusion(pred, gt)
    return (TN / (TN + FP + EPS)).item()

def get_precision(logits, mask_gt, threshold=0.5):
    pred = _binarize_pred(logits, threshold)
    gt   = _binarize_gt(mask_gt, threshold)
    TP, FP, TN, FN = _get_confusion(pred, gt)
    return (TP / (TP + FP + EPS)).item()

def get_F1(logits, mask_gt, threshold=0.5):
    # 可以直接复用 sensitivity 和 precision
    SE = get_sensitivity(logits, mask_gt, threshold)
    PC = get_precision(logits, mask_gt, threshold)
    return 2 * SE * PC / (SE + PC + EPS)

def get_JS(logits, mask_gt, threshold=0.5):
    pred = _binarize_pred(logits, threshold)
    gt   = _binarize_gt(mask_gt, threshold)
    inter = torch.logical_and(pred, gt).sum().float()
    union = torch.logical_or(pred, gt).sum().float()
    return (inter / (union + EPS)).item()

def get_DC(logits, mask_gt, threshold=0.5):
    pred = _binarize_pred(logits, threshold)
    gt   = _binarize_gt(mask_gt, threshold)
    inter = torch.logical_and(pred, gt).sum().float()
    return (2 * inter / (pred.sum().float() + gt.sum().float() + EPS)).item()