import torch


def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / union
