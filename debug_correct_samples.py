#!/usr/bin/env python3
"""
Debug script to check correct_samples loading
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))

from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from collect_correct_samples import load_correct_indices

print("="*80)
print("DEBUG CORRECT_SAMPLES LOADING")
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

# Test 2: Check dataset attributes
print("\n[Test 2] Checking dataset attributes...")
print(f"  Has 'labels' attribute: {hasattr(test_dataset, 'labels')}")
if hasattr(test_dataset, 'labels'):
    print(f"  Type of labels: {type(test_dataset.labels)}")
    print(f"  Length of labels: {len(test_dataset.labels)}")
    print(f"  First 10 labels: {test_dataset.labels[:10]}")
else:
    print("  WARNING: No 'labels' attribute!")
    print(f"  Available attributes: {[attr for attr in dir(test_dataset) if not attr.startswith('_')]}")

# Test 3: Load correct_samples for a class
print("\n[Test 3] Loading correct_samples...")
test_class_id = 1

try:
    correct_indices = load_correct_indices('gtsrb_project/correct_samples', test_class_id)
    print(f"✓ Loaded correct_samples for class {test_class_id}")
    print(f"  Number of correct samples: {len(correct_indices)}")
    print(f"  First 10 indices: {correct_indices[:10]}")
    print(f"  Last 10 indices: {correct_indices[-10:]}")

    # Test 4: Verify indices are valid
    print(f"\n[Test 4] Verifying indices...")
    max_idx = max(correct_indices)
    min_idx = min(correct_indices)
    print(f"  Min index: {min_idx}")
    print(f"  Max index: {max_idx}")
    print(f"  Dataset size: {len(test_dataset)}")

    if max_idx >= len(test_dataset):
        print(f"  ✗ ERROR: Max index {max_idx} >= dataset size {len(test_dataset)}")
    else:
        print(f"  ✓ All indices are valid")

    # Test 5: Try to load a sample
    print(f"\n[Test 5] Loading a sample...")
    test_idx = correct_indices[663]  # The index from your error
    print(f"  Trying to load global index {test_idx}...")

    try:
        image, label = test_dataset[test_idx]
        print(f"  ✓ Sample loaded successfully")
        print(f"    Image shape: {image.shape}")
        print(f"    Label: {label}")
    except Exception as e:
        print(f"  ✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()

except FileNotFoundError as e:
    print(f"✗ File not found: {e}")
    print(f"\n  Trying fallback method...")

    # Fallback: find samples from dataset
    if hasattr(test_dataset, 'labels'):
        correct_indices = [i for i, label in enumerate(test_dataset.labels)
                         if label == test_class_id]
        print(f"  Found {len(correct_indices)} samples for class {test_class_id}")
    else:
        print(f"  ✗ Cannot use fallback - no 'labels' attribute")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
