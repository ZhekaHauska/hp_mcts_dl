def calc_acc(outputs, targets, threshold):
    outputs = outputs.squeeze(1) > threshold
    targets = targets.squeeze(1) > threshold
    acc = (outputs == targets).float().sum(1) / targets.shape[1]
    return acc.mean()


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
