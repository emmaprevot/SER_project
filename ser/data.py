import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

from ser.transform import get_transform


def get_data(batch_size):
   
    ts = get_transform()

    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return training_dataloader, validation_dataloader


