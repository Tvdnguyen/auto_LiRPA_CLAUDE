"""
GTSRB Dataset Loader
German Traffic Sign Recognition Benchmark
"""
import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Tuple, List, Optional


class GTSRBDataset(Dataset):
    """GTSRB Dataset loader"""

    def __init__(self, root_dir: str, train: bool = True, transform=None):
        """
        Args:
            root_dir: Path to GTSRB dataset
            train: If True, load training data, else test data
            transform: Optional transforms to apply
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        self.data = []
        self.labels = []

        if train:
            self._load_training_data()
        else:
            self._load_test_data()

    def _load_training_data(self):
        """Load training data from Train folder"""
        train_path = os.path.join(self.root_dir, 'Train')

        # Loop through 43 classes (00000 to 00042)
        for class_id in range(43):
            class_folder = os.path.join(train_path, f'{class_id:05d}')
            if not os.path.exists(class_folder):
                print(f"Warning: Class folder {class_folder} not found")
                continue

            # Read GT file for this class
            gt_file = os.path.join(class_folder, f'GT-{class_id:05d}.csv')
            if not os.path.exists(gt_file):
                print(f"Warning: GT file {gt_file} not found")
                continue

            with open(gt_file, 'r') as f:
                reader = csv.DictReader(f, delimiter=';')
                for row in reader:
                    img_path = os.path.join(class_folder, row['Filename'])
                    if os.path.exists(img_path):
                        self.data.append(img_path)
                        self.labels.append(int(row['ClassId']))

        print(f"Loaded {len(self.data)} training images")

    def _load_test_data(self):
        """Load test data from Test folder"""
        test_path = os.path.join(self.root_dir, 'Test')
        gt_file = os.path.join(test_path, 'GT-final_test.csv')

        if not os.path.exists(gt_file):
            print(f"Warning: Test GT file {gt_file} not found")
            return

        with open(gt_file, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Try multiple possible locations for test images
                img_filename = row['Filename']

                # Option 1: Directly in Test/
                img_path = os.path.join(test_path, img_filename)

                # Option 2: In Test/Images/ subfolder
                if not os.path.exists(img_path):
                    img_path = os.path.join(test_path, 'Images', img_filename)

                # Option 3: Just the filename without path in CSV
                if not os.path.exists(img_path):
                    # Extract just filename from path in CSV
                    just_filename = os.path.basename(img_filename)
                    img_path = os.path.join(test_path, just_filename)

                    # Or in Images subfolder
                    if not os.path.exists(img_path):
                        img_path = os.path.join(test_path, 'Images', just_filename)

                if os.path.exists(img_path):
                    self.data.append(img_path)
                    self.labels.append(int(row['ClassId']))
                else:
                    print(f"Warning: Image not found: {img_filename}")

        print(f"Loaded {len(self.data)} test images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label


def get_gtsrb_transforms(train: bool = True, img_size: int = 32) -> transforms.Compose:
    """
    Get standard transforms for GTSRB dataset

    Args:
        train: If True, include data augmentation
        img_size: Target image size (default 32x32)

    Returns:
        Composed transforms
    """
    if train:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                               std=[0.2724, 0.2608, 0.2669])
        ])
    else:
        # Test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.3403, 0.3121, 0.3214],
                               std=[0.2724, 0.2608, 0.2669])
        ])

    return transform


def get_gtsrb_dataloaders(
    root_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Get GTSRB train and test dataloaders

    Args:
        root_dir: Path to GTSRB dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        img_size: Target image size

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = GTSRBDataset(
        root_dir=root_dir,
        train=True,
        transform=get_gtsrb_transforms(train=True, img_size=img_size)
    )

    test_dataset = GTSRBDataset(
        root_dir=root_dir,
        train=False,
        transform=get_gtsrb_transforms(train=False, img_size=img_size)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset loader
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gtsrb_dataset.py <path_to_gtsrb_dataset>")
        sys.exit(1)

    root_dir = sys.argv[1]

    # Test loading
    print("Testing GTSRB dataset loader...")
    train_loader, test_loader = get_gtsrb_dataloaders(root_dir, batch_size=32)

    # Print statistics
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: {labels.min()}-{labels.max()}")
