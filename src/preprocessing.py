
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Image settings
IMAGE_SIZE = 64
BATCH_SIZE = 32

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def load_data():

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=transform
    )

    valid_dataset = datasets.ImageFolder(
        DATASET_DIR / "valid",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=transform
    )

    # Verify class mapping
    assert train_dataset.class_to_idx == valid_dataset.class_to_idx
    assert train_dataset.class_to_idx == test_dataset.class_to_idx

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, valid_loader, test_loader, train_dataset.classes
