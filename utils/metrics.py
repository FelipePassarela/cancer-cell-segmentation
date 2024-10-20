import torch
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

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth + self.eps)
        dice_loss = dice_loss.mean()
        combined_loss = (self.bce_weight * bce_loss + self.dice_weight * dice_loss)

        return {
            'combined': combined_loss,
            'bce': bce_loss,
            'dice': dice_loss
        }