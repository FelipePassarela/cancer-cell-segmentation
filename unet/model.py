from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from dataset.transforms import get_val_transforms


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(DoubleConv, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
        )

    def forward(self, x):
        return self.convs(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, featmaps=None):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmaps = [64, 128, 256, 512] if featmaps is None else featmaps

        self.downs_dropouts = [0.1, 0.15, 0.2, 0.25]
        self.ups_dropouts = [0.2, 0.15, 0.1, 0.05]
        self.bottleneck_dropout = 0.3

        self.downs = nn.ModuleList()
        self.bottleneck = DoubleConv(self.featmaps[-1], self.featmaps[-1] * 2, self.bottleneck_dropout)
        self.ups = nn.ModuleList()
        self.final_conv = nn.Conv2d(self.featmaps[0], out_channels, 1)

        for i, feats in enumerate(self.featmaps):
            self.downs.append(DoubleConv(in_channels, feats, self.downs_dropouts[i]))
            in_channels = feats

        for i, feats in enumerate(reversed(self.featmaps)):
            self.ups.append(nn.ConvTranspose2d(feats * 2, feats, 2, stride=2))
            self.ups.append(DoubleConv(feats * 2, feats, self.ups_dropouts[i]))

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, 2, stride=2)

        x = self.bottleneck(x)
        skip_connections = list(reversed(skip_connections))

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skips = skip_connections[i // 2]
            if x.shape != skips.shape:
                x = TF.resize(x, skips.shape[2:])  # [..., H, W] skip's shape
            x = torch.cat([x, skips], dim=1)
            x = self.ups[i + 1](x)

        return self.final_conv(x)

    def predict(
            self,
            img: np.ndarray | Image.Image | torch.Tensor | list,
            device: str = None,
    ) -> dict[str, Any]:
        """
        Predicts the mask for the input image

        :param img: input image
        :type img: np.ndarray | Image.Image | torch.Tensor | list
        :param device: device to run the model on
        :type device: str
        :return: dictionary containing the predicted mask, input image, and logits
        :rtype: dict[str, Any]
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        img = TF.to_tensor(img) if not isinstance(img, torch.Tensor) else img
        img_transformed = get_val_transforms()(img).to(device)
        if len(img_transformed.shape) == 3:
            img_transformed = img_transformed.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self(img_transformed)
            pred = F.sigmoid(logits)
            pred = (pred > 0.5).float()

        pred_img = TF.to_pil_image(pred.cpu().squeeze())
        img = TF.to_pil_image(img_transformed.cpu().squeeze())
        logits = logits.cpu().squeeze().numpy()

        return {
            "pred": pred_img,
            "img": img,
            "logits": logits,
        }


def test_model():
    x = torch.rand((3, 3, 801, 600))
    out_shape_must_be = torch.Size([3, 1, 801, 600])

    model = UNet(3, 1)
    pred = model(x)

    print(pred)
    print(pred.shape)
    assert pred.shape == out_shape_must_be, f"x shape {x.shape} didn't match shape {out_shape_must_be}"
    print("Model is fine!")


if __name__ == "__main__":
    test_model()
