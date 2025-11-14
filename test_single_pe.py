#!/usr/bin/env python3
"""
Test single PE sensitivity - Ultra low memory version
Only analyze ONE PE at a time
"""

import sys
import os
import argparse
import torch
import gc
import csv
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'systolic_fault_sim'))

from integrated_pe_sensitivity_analysis_low_memory import LowMemoryIntegratedAnalyzer


def test_single_pe(
    data_dir,
    checkpoint_path,
    pe_row,
    pe_col,
    array_size=8,
    class_id=0,
    test_idx=0,
    duration=1,
    epsilon_max=0.3,
    tolerance=0.05
):
    """
    Test a single PE and save result

    Args:
        data_dir: GTSRB data directory
        checkpoint_path: Model checkpoint
        pe_row: PE row index
        pe_col: PE column index
        array_size: Array size
        class_id: Class to test
        test_idx: Sample index within class
        duration: Fault duration
        epsilon_max: Max epsilon
        tolerance: Binary search tolerance
    """
    print("="*80)
    print(f"  SINGLE PE SENSITIVITY TEST - PE({pe_row},{pe_col})")
    print("="*80)

    # Create analyzer
    print("\nInitializing analyzer...")
    analyzer = LowMemoryIntegratedAnalyzer(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        array_size=array_size,
        device='cpu'
    )

    # Get sample
    print(f"\nLoading sample from class {class_id}...")
    class_samples = []
    for idx in range(len(analyzer.test_dataset)):
        _, label = analyzer.test_dataset[idx]
        if label == class_id:
            class_samples.append(idx)

    if len(class_samples) == 0:
        raise ValueError(f"No samples found for class {class_id}")

    if test_idx >= len(class_samples):
        raise ValueError(f"Index {test_idx} out of range")

    global_idx = class_samples[test_idx]
    image, label = analyzer.test_dataset[global_idx]

    print(f"  Global index: {global_idx}")
    print(f"  True label: {label}")

    # Get prediction
    image_batch = image.unsqueeze(0).to(analyzer.device)
    with torch.no_grad():
        output = analyzer.model(image_batch)
        pred_class = output.argmax(dim=1).item()

    del image_batch, output
    gc.collect()

    print(f"  Predicted class: {pred_class}")

    if pred_class != label:
        print(f"  WARNING: Misclassified!")

    # Test this PE
    print(f"\n{'='*80}")
    print(f"Testing PE({pe_row},{pe_col})")
    print(f"{'='*80}")

    try:
        # Step 1: Get affected region
        print(f"\n[Step 1] Running fault simulation...")
        affected_channels, affected_spatial = analyzer.get_affected_conv1_region(
            pe_row, pe_col, duration
        )

        print(f"  Affected: {len(affected_channels)} channels, {len(affected_spatial)} positions")

        # Step 2: Find max epsilon
        if len(affected_spatial) == 0:
            max_eps = epsilon_max
            print(f"  → No impact, epsilon = {max_eps:.4f}")
        else:
            print(f"\n[Step 2] Finding max epsilon...")
            max_eps = analyzer.find_max_epsilon_for_region(
                image, label, pred_class,
                affected_channels, affected_spatial,
                epsilon_max, tolerance
            )
            print(f"  → Max epsilon: {max_eps:.4f}")

        # Save result
        result = {
            'pe_row': pe_row,
            'pe_col': pe_col,
            'max_epsilon': max_eps,
            'num_affected_channels': len(affected_channels),
            'num_affected_spatial': len(affected_spatial),
            'global_idx': global_idx,
            'class_idx': test_idx,
            'class_id': class_id,
            'true_label': label,
            'pred_class': pred_class,
            'array_size': array_size
        }

        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pe_{pe_row}_{pe_col}_result_{timestamp}.csv'

        with open(filename, 'w', newline='') as f:
            fieldnames = ['pe_row', 'pe_col', 'max_epsilon',
                         'num_affected_channels', 'num_affected_spatial',
                         'global_idx', 'class_idx', 'class_id',
                         'true_label', 'pred_class', 'array_size']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(result)

        print(f"\n✓ Result saved to: {filename}")

        # Print result
        print(f"\n{'='*80}")
        print(f"RESULT SUMMARY")
        print(f"{'='*80}")
        print(f"PE({pe_row},{pe_col}):")
        print(f"  Max epsilon: {max_eps:.4f}")
        print(f"  Affected channels: {len(affected_channels)}")
        print(f"  Affected spatial: {len(affected_spatial)}")
        print(f"  Sensitivity rank: {'HIGH' if max_eps < 0.1 else 'MEDIUM' if max_eps < 0.3 else 'LOW'}")

        return result

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Test single PE sensitivity (ultra low memory)'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB data (e.g., gtsrb_project/data/GTSRB_data)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--pe_row', type=int, required=True,
                       help='PE row index (0-7 for 8x8)')
    parser.add_argument('--pe_col', type=int, required=True,
                       help='PE column index (0-7 for 8x8)')
    parser.add_argument('--array_size', type=int, default=8,
                       help='Array size (default: 8)')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class to test (default: 0)')
    parser.add_argument('--test_idx', type=int, default=0,
                       help='Sample index within class (default: 0)')
    parser.add_argument('--duration', type=int, default=1,
                       help='Fault duration (default: 1)')
    parser.add_argument('--epsilon_max', type=float, default=0.3,
                       help='Max epsilon (default: 0.3)')
    parser.add_argument('--tolerance', type=float, default=0.05,
                       help='Binary search tolerance (default: 0.05)')

    args = parser.parse_args()

    # Validate PE indices
    if not (0 <= args.pe_row < args.array_size):
        print(f"Error: pe_row must be 0-{args.array_size-1}")
        sys.exit(1)

    if not (0 <= args.pe_col < args.array_size):
        print(f"Error: pe_col must be 0-{args.array_size-1}")
        sys.exit(1)

    # Run test
    result = test_single_pe(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        pe_row=args.pe_row,
        pe_col=args.pe_col,
        array_size=args.array_size,
        class_id=args.class_id,
        test_idx=args.test_idx,
        duration=args.duration,
        epsilon_max=args.epsilon_max,
        tolerance=args.tolerance
    )

    if result:
        print("\n✓ Test completed successfully")
    else:
        print("\n✗ Test failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
