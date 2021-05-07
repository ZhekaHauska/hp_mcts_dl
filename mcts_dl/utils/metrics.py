import torch


def calc_acc(outputs, targets, threshold):
    outputs = outputs.squeeze(1) > threshold
    targets = targets.squeeze(1) > threshold
    acc = (outputs == targets).float().sum(1) / targets.shape[1]
    return acc.mean()


def calc_f1(outputs, targets, threshold):
    outputs = outputs.squeeze(1) > threshold
    targets = targets.squeeze(1) > threshold
    y_pred = outputs.float()
    y_true = targets.float()
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    if f1 == 0.0 and True not in targets:
        return 1.0

    return f1


def calc_iou(outputs, targets, threshold=0.5):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1) > threshold
    targets = targets.squeeze(1) > threshold

    intersection = (outputs & targets).float().sum((1, 2))
    union = (outputs | targets).float().sum((1, 2))

    batch_iou = intersection / (union + SMOOTH)

    norm_iou = []
    for i, iou in enumerate(batch_iou):
        if iou == 0. and True not in targets[i]:
            continue
        else:
            norm_iou.append(iou)

    if len(norm_iou) == 0:
        mean_iou = 1.
    else:
        mean_iou = sum(norm_iou) / len(norm_iou)

    return mean_iou
