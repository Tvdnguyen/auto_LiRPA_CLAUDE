#!/usr/bin/env python3
"""
Debug script to inspect GTSRB dataset structure
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))

from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
import torch

print("="*80)
print("GTSRB DATASET DEBUG")
print("="*80)

data_dir = "gtsrb_project/data/GTSRB_data"

# Test 1: Load test dataset
print("\n[Test 1] Loading test dataset...")
try:
    test_dataset = GTSRBDataset(
        root_dir=data_dir,
        train=False,
        transform=get_gtsrb_transforms(train=False, img_size=32)
    )
    print(f"✓ Test dataset loaded successfully")
    print(f"  Total samples: {len(test_dataset)}")
except Exception as e:
    print(f"✗ Error loading test dataset: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Check dataset attributes
print("\n[Test 2] Checking dataset attributes...")
print(f"  Dataset type: {type(test_dataset)}")
print(f"  Has 'labels' attribute: {hasattr(test_dataset, 'labels')}")
print(f"  Has 'data' attribute: {hasattr(test_dataset, 'data')}")

if hasattr(test_dataset, 'labels'):
    print(f"  Labels type: {type(test_dataset.labels)}")
    print(f"  Labels length: {len(test_dataset.labels)}")
    print(f"  First 10 labels: {test_dataset.labels[:10]}")
else:
    print(f"  WARNING: No 'labels' attribute found!")

if hasattr(test_dataset, 'data'):
    print(f"  Data type: {type(test_dataset.data)}")
    print(f"  Data length: {len(test_dataset.data)}")
    print(f"  First 3 paths: {test_dataset.data[:3]}")

# Test 3: Try to access first few samples
print("\n[Test 3] Accessing first 10 samples...")
for i in range(min(10, len(test_dataset))):
    try:
        image, label = test_dataset[i]
        print(f"  Sample {i}: label={label}, image shape={image.shape}")
    except Exception as e:
        print(f"  Sample {i}: ERROR - {e}")

# Test 4: Count samples per class
print("\n[Test 4] Counting samples per class...")
class_counts = {}
for i in range(len(test_dataset)):
    try:
        _, label = test_dataset[i]
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    except Exception as e:
        print(f"  Error at index {i}: {e}")
        break

print(f"\nClass distribution (first 10 classes):")
for class_id in sorted(class_counts.keys())[:10]:
    print(f"  Class {class_id}: {class_counts[class_id]} samples")

if len(class_counts) > 10:
    print(f"  ... ({len(class_counts)} total classes)")

# Test 5: Check if labels are integers or something else
print("\n[Test 5] Checking label types...")
sample_labels = []
for i in range(min(5, len(test_dataset))):
    try:
        _, label = test_dataset[i]
        sample_labels.append(label)
        print(f"  Sample {i}: label={label}, type={type(label)}")
    except Exception as e:
        print(f"  Sample {i}: ERROR - {e}")

# Test 6: Check dataset directory structure
print("\n[Test 6] Checking data directory structure...")
import os
print(f"  Data dir: {data_dir}")
print(f"  Exists: {os.path.exists(data_dir)}")

if os.path.exists(data_dir):
    # List subdirectories
    subdirs = []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    print(f"  Subdirectories: {subdirs[:10]}")

    # Check for Test.csv or test.csv
    test_csv = os.path.join(data_dir, "Test.csv")
    test_csv_lower = os.path.join(data_dir, "test.csv")
    print(f"  Test.csv exists: {os.path.exists(test_csv)}")
    print(f"  test.csv exists: {os.path.exists(test_csv_lower)}")

    # Check for Final_Test directory
    final_test = os.path.join(data_dir, "Final_Test")
    print(f"  Final_Test/ exists: {os.path.exists(final_test)}")

# Test 7: Try different approach - check how dataset loads data
print("\n[Test 7] Inspecting dataset loading mechanism...")
if hasattr(test_dataset, '__dict__'):
    print("  Dataset attributes:")
    for key, value in test_dataset.__dict__.items():
        if not key.startswith('_'):
            if isinstance(value, (list, tuple)):
                print(f"    {key}: {type(value).__name__} (length={len(value)})")
            else:
                print(f"    {key}: {type(value).__name__}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
