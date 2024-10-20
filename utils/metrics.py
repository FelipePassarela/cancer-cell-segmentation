import torch
from scipy.ndimage import distance_transform_edt
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self,
                 bce_weight=0.5,
                 dice_weight=0.5,
                 smooth=1.0,
                 eps=1e-7):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.eps = eps

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        if preds.dim() > 2:
            preds = preds.view(preds.size(0), -1)
            targets = targets.view(targets.size(0), -1)

        bce_loss = self.bce(preds, targets)

        preds_sigmoid = torch.sigmoid(preds)
        intersection = (preds_sigmoid * targets).sum(dim=1)
        union = (preds_sigmoid.sum(dim=1) + targets.sum(dim=1))

        soft_dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        soft_dice_loss = soft_dice_loss.mean()
        combined_loss = (self.bce_weight * bce_loss + self.dice_weight * soft_dice_loss)

        return {
            'combined': combined_loss,
            'bce': bce_loss,
            'soft_dice': soft_dice_loss
        }


def hausdorff_distance(pred, target):
    pred = pred.detach().cpu().numpy().astype(bool)
    target = target.detach().cpu().numpy().astype(bool)

    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')

    d1 = distance_transform_edt(~target)[pred].max()
    d2 = distance_transform_edt(~pred)[target].max()
    return max(d1, d2)


def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / union
