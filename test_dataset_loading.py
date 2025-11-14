"""Test dataset loading with correct path"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))

from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms

# Test with correct path
data_dir = "gtsrb_project/data/GTSRB_data"

print(f"Testing dataset loading from: {data_dir}")
print(f"Full path: {os.path.abspath(data_dir)}")
print()

# Load test dataset
test_dataset = GTSRBDataset(
    root_dir=data_dir,
    train=False,
    transform=get_gtsrb_transforms(train=False, img_size=32)
)

print(f"Test dataset size: {len(test_dataset)}")

if len(test_dataset) > 0:
    print(f"\nTesting access to index 243:")
    try:
        image, label = test_dataset[243]
        print(f"  Success! Label: {label}, Image shape: {image.shape}")
    except Exception as e:
        print(f"  Error: {e}")
else:
    print("\nDataset is empty! Check the path.")
