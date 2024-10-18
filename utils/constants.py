import torch


N_EPOCHS = 10
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
IMAGE_SIZE = (256, 256)  # Necessary for avoiding OOM errors
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
