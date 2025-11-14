#!/usr/bin/env python3
"""
Debug script to find the root cause of "list index out of range"
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))

from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from collect_correct_samples import load_correct_indices
import random

print("="*80)
print("DEBUG: Index Out of Range Issue")
print("="*80)

# Test 1: Load dataset
print("\n[Test 1] Loading test dataset...")
data_dir = "gtsrb_project/data/GTSRB_data"
test_dataset = GTSRBDataset(
    root_dir=data_dir,
    train=False,
    transform=get_gtsrb_transforms(train=False, img_size=32)
)
print(f"✓ Dataset loaded: {len(test_dataset)} samples")
print(f"  Dataset has 'labels' attribute: {hasattr(test_dataset, 'labels')}")
print(f"  Length of labels: {len(test_dataset.labels)}")

# Test 2: Load correct_indices for class 0
print("\n[Test 2] Loading correct_indices for class 0...")
class_id = 0
correct_indices = load_correct_indices('gtsrb_project/correct_samples', class_id)
print(f"✓ Loaded {len(correct_indices)} correct samples for class {class_id}")
print(f"  First 10 indices: {correct_indices[:10]}")
print(f"  Last 10 indices: {correct_indices[-10:]}")

# Test 3: Check if indices are valid
print("\n[Test 3] Checking index validity...")
max_correct_idx = max(correct_indices)
min_correct_idx = min(correct_indices)
dataset_size = len(test_dataset)

print(f"  Dataset size: {dataset_size}")
print(f"  Min correct index: {min_correct_idx}")
print(f"  Max correct index: {max_correct_idx}")

if max_correct_idx >= dataset_size:
    print(f"  ✗ ERROR: Max correct index ({max_correct_idx}) >= dataset size ({dataset_size})")
    print(f"  → Some indices in correct_samples are out of range!")

    # Find invalid indices
    invalid_indices = [idx for idx in correct_indices if idx >= dataset_size]
    print(f"\n  Found {len(invalid_indices)} invalid indices:")
    print(f"    {invalid_indices[:20]}...")  # Show first 20
else:
    print(f"  ✓ All indices are valid")

# Test 4: Simulate the error
print("\n[Test 4] Simulating the random selection...")
random.seed(42)  # For reproducibility
for attempt in range(5):
    sample_idx = random.randint(0, len(correct_indices) - 1)
    global_idx = correct_indices[sample_idx]

    print(f"\n  Attempt {attempt + 1}:")
    print(f"    sample_idx (within class): {sample_idx}")
    print(f"    global_idx (in dataset): {global_idx}")

    if global_idx >= dataset_size:
        print(f"    ✗ ERROR: global_idx {global_idx} >= dataset size {dataset_size}")
    else:
        try:
            image, label = test_dataset[global_idx]
            print(f"    ✓ Successfully loaded sample, label={label}")
        except Exception as e:
            print(f"    ✗ Error loading sample: {e}")

# Test 5: Check dataset integrity
print("\n[Test 5] Checking dataset integrity...")
print(f"  self.test_dataset.data has {len(test_dataset.data)} paths")
print(f"  self.test_dataset.labels has {len(test_dataset.labels)} labels")

if len(test_dataset.data) != len(test_dataset.labels):
    print(f"  ✗ ERROR: Mismatch between data and labels!")
else:
    print(f"  ✓ Data and labels match")

# Test 6: Check if dataset loaded correctly
print("\n[Test 6] Checking dataset load warnings...")
print("  Re-loading dataset to see warnings...")

test_dataset2 = GTSRBDataset(
    root_dir=data_dir,
    train=False,
    transform=get_gtsrb_transforms(train=False, img_size=32)
)

# Test 7: Check CSV structure
print("\n[Test 7] Checking CSV correctness...")
csv_path = "gtsrb_project/correct_samples/class_00_correct_indices.csv"
print(f"  CSV path: {csv_path}")
print(f"  CSV exists: {os.path.exists(csv_path)}")

if os.path.exists(csv_path):
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    print(f"  Total lines in CSV: {len(lines)}")
    print(f"  First 5 lines:")
    for i, line in enumerate(lines[:5]):
        print(f"    {i}: {line.strip()}")

# Test 8: Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if max_correct_idx >= dataset_size:
    print("\n⚠️  ROOT CAUSE IDENTIFIED:")
    print(f"  The correct_samples CSV contains indices up to {max_correct_idx}")
    print(f"  But the dataset only has {dataset_size} samples")
    print(f"\n  Possible reasons:")
    print(f"  1. Dataset on server is missing some images")
    print(f"  2. Dataset path is wrong (missing some test images)")
    print(f"  3. correct_samples CSV was generated on different dataset version")
    print(f"\n  SOLUTION:")
    print(f"  - Check if all test images exist in gtsrb_project/data/GTSRB_data/Test/")
    print(f"  - Expected: 12,630 test images")
    print(f"  - Actual: {dataset_size} images loaded")
else:
    print("\n✓ No obvious issue detected. Error may be elsewhere.")

print("\n" + "="*80)
