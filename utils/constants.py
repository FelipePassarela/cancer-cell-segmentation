import os

import numpy as np
import torch

N_EPOCHS = 10
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)
NUM_WORKERS = 0
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed set to {seed}")
