import argparse

import torch
from PIL import Image

from models.unet import UNet


def main(args):
    model = UNet(3, 1)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()

    img = Image.open(args.img_path).convert("RGB")
    pred = model.predict(img)

    # pred["img"].show()  # Debugging purposes
    pred["pred"].show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("img_path", type=str, help="Path to the image")
    args = parser.parse_args()
    main(args)
