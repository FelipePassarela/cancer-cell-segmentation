from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from dataset.transforms import get_inference_transforms


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        ]

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone_features=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder / Backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, backbone_features, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(backbone_features),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(backbone_features, backbone_features * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(backbone_features * 2),
            nn.GELU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(backbone_features * 2, backbone_features * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(backbone_features * 4),
            nn.GELU()
        )

        self.aspp = ASPP(backbone_features * 4, 256, [12, 24, 36])

        # Decoder
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(backbone_features, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]

        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # ASPP
        aspp_out = self.aspp(x3)
        aspp_out = F.interpolate(aspp_out, size=x1.shape[-2:], mode='bilinear', align_corners=False)

        # Decoder
        low_level_feat = self.low_level_conv(x1)
        decoder_input = torch.cat([low_level_feat, aspp_out], dim=1)
        out = self.decoder(decoder_input)

        return F.interpolate(out, size=size, mode='bilinear', align_corners=False)

    def predict(
            self,
            img: np.ndarray | Image.Image | torch.Tensor | list,
            threshold=0.5,
            device: str = None,
    ) -> dict[str, Any]:
        """
        Predicts the mask for the input image

        :param img: input image
        :type img: np.ndarray | Image.Image | torch.Tensor | list
        :param threshold: threshold to apply to the predicted mask
        :type threshold: float
        :param device: device to run the model on
        :type device: str
        :return: dictionary containing the predicted mask, input image, and logits
        :rtype: dict[str, Any]
        """

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        img = TF.to_tensor(img) if not isinstance(img, torch.Tensor) else img
        img_transformed = get_inference_transforms()(img).to(device)
        if len(img_transformed.shape) == 3:
            img_transformed = img_transformed.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self(img_transformed)
            pred = F.sigmoid(logits)
            pred = (pred > threshold).float()

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

    model = DeepLabV3Plus(3, 1)
    pred = model(x)

    print(pred.shape)
    assert pred.shape == out_shape_must_be, f"x shape {x.shape} didn't match shape {out_shape_must_be}"
    print("Model is fine!")


if __name__ == "__main__":
    test_model()