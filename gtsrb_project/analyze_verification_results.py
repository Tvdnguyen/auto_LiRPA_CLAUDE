#!/usr/bin/env python3
"""
Phân tích kết quả batch verification

Script này đọc các file CSV từ verification_results/
và tạo report tổng hợp với:
- Statistics tổng quan
- Phân bố verified/not verified
- Visualizations (nếu có matplotlib)
"""

import os
import csv
import argparse
from collections import defaultdict
import glob


def load_verification_results(csv_file):
    """
    Load verification results from CSV

    Args:
        csv_file: Path to CSV file

    Returns:
        List of result dicts
    """
    results = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert types
            row['sample_idx'] = int(row['sample_idx'])
            row['global_idx'] = int(row['global_idx'])
            row['class_id'] = int(row['class_id'])
            row['verified'] = row['verified'].lower() == 'true'

            # Convert floats (may be None)
            for key in ['clean_logit', 'lower_bound', 'upper_bound', 'margin']:
                if row[key] and row[key] != 'None':
                    row[key] = float(row[key])
                else:
                    row[key] = None

            results.append(row)

    return results


def compute_statistics(results):
    """
    Compute statistics from results

    Args:
        results: List of result dicts

    Returns:
        dict with statistics
    """
    total = len(results)
    verified = sum(1 for r in results if r['verified'])
    not_verified = sum(1 for r in results if not r['verified'] and r['reason'] == 'not_verified')
    errors = sum(1 for r in results if 'error' in r['reason'])
    incorrect = sum(1 for r in results if r['reason'] == 'incorrect_prediction')

    # Compute margins
    margins = [r['margin'] for r in results if r['margin'] is not None]

    stats = {
        'total': total,
        'verified': verified,
        'not_verified': not_verified,
        'errors': errors,
        'incorrect': incorrect,
        'verified_rate': 100.0 * verified / total if total > 0 else 0.0,
    }

    if margins:
        stats['avg_margin'] = sum(margins) / len(margins)
        stats['min_margin'] = min(margins)
        stats['max_margin'] = max(margins)
        stats['positive_margins'] = sum(1 for m in margins if m > 0)
        stats['negative_margins'] = sum(1 for m in margins if m <= 0)

    return stats


def print_statistics(stats, class_id=None):
    """Print statistics in readable format"""
    print("\n" + "="*80)
    if class_id is not None:
        print(f"STATISTICS FOR CLASS {class_id}")
    else:
        print("OVERALL STATISTICS")
    print("="*80)

    print(f"\nTotal samples: {stats['total']}")
    print(f"  ✓ Verified robust:   {stats['verified']:5d} ({stats['verified_rate']:5.1f}%)")
    print(f"  ✗ Not verified:      {stats['not_verified']:5d} ({100.0*stats['not_verified']/stats['total']:5.1f}%)")
    print(f"  ⚠ Errors:            {stats['errors']:5d}")
    print(f"  ⚠ Incorrect pred:    {stats['incorrect']:5d}")

    if 'avg_margin' in stats:
        print(f"\nMargin Statistics:")
        print(f"  Average margin:      {stats['avg_margin']:8.4f}")
        print(f"  Min margin:          {stats['min_margin']:8.4f}")
        print(f"  Max margin:          {stats['max_margin']:8.4f}")
        print(f"  Positive margins:    {stats['positive_margins']:5d}")
        print(f"  Negative margins:    {stats['negative_margins']:5d}")

    print("="*80)


def analyze_directory(results_dir):
    """
    Analyze all verification results in directory

    Args:
        results_dir: Directory with CSV files
    """
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_dir, 'class_*_verification_*.csv'))

    if not csv_files:
        print(f"No verification result files found in {results_dir}")
        return

    print(f"Found {len(csv_files)} verification result files")

    # Analyze each file
    all_results = []
    per_class_stats = {}

    for csv_file in sorted(csv_files):
        print(f"\nAnalyzing: {os.path.basename(csv_file)}")

        # Load results
        results = load_verification_results(csv_file)
        all_results.extend(results)

        # Get class ID
        class_id = results[0]['class_id'] if results else None

        # Compute statistics
        stats = compute_statistics(results)
        per_class_stats[class_id] = stats

        # Print statistics
        print_statistics(stats, class_id)

    # Overall statistics
    if len(csv_files) > 1:
        overall_stats = compute_statistics(all_results)
        print_statistics(overall_stats, class_id=None)

    # Summary table
    print("\n" + "="*80)
    print("PER-CLASS SUMMARY")
    print("="*80)
    print(f"{'Class':>6} | {'Total':>6} | {'Verified':>9} | {'Rate':>7} | {'Avg Margin':>11}")
    print("-"*80)

    for class_id in sorted(per_class_stats.keys()):
        stats = per_class_stats[class_id]
        avg_margin = stats.get('avg_margin', 0.0)
        print(f"{class_id:>6} | {stats['total']:>6} | {stats['verified']:>9} | "
              f"{stats['verified_rate']:>6.1f}% | {avg_margin:>11.4f}")

    print("="*80)


def analyze_single_file(csv_file):
    """Analyze single CSV file"""
    print(f"Analyzing: {csv_file}")

    results = load_verification_results(csv_file)
    stats = compute_statistics(results)

    class_id = results[0]['class_id'] if results else None
    print_statistics(stats, class_id)

    # Show some examples
    print("\n" + "="*80)
    print("SAMPLE RESULTS")
    print("="*80)

    # Show first 5 verified
    verified_samples = [r for r in results if r['verified']][:5]
    if verified_samples:
        print("\nFirst 5 Verified Samples:")
        print(f"{'Idx':>5} | {'Global':>7} | {'Clean':>8} | {'LB':>8} | {'UB':>8} | {'Margin':>8}")
        print("-"*80)
        for r in verified_samples:
            print(f"{r['sample_idx']:>5} | {r['global_idx']:>7} | "
                  f"{r['clean_logit']:>8.4f} | {r['lower_bound']:>8.4f} | "
                  f"{r['upper_bound']:>8.4f} | {r['margin']:>8.4f}")

    # Show first 5 not verified
    not_verified_samples = [r for r in results if not r['verified'] and r['reason'] == 'not_verified'][:5]
    if not_verified_samples:
        print("\nFirst 5 Not Verified Samples:")
        print(f"{'Idx':>5} | {'Global':>7} | {'Clean':>8} | {'LB':>8} | {'UB':>8} | {'Margin':>8}")
        print("-"*80)
        for r in not_verified_samples:
            if r['lower_bound'] is not None:
                print(f"{r['sample_idx']:>5} | {r['global_idx']:>7} | "
                      f"{r['clean_logit']:>8.4f} | {r['lower_bound']:>8.4f} | "
                      f"{r['upper_bound']:>8.4f} | {r['margin']:>8.4f}")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze batch verification results'
    )
    parser.add_argument('--results_dir', type=str, default='verification_results',
                       help='Directory with verification results')
    parser.add_argument('--file', type=str, default=None,
                       help='Analyze single CSV file instead of directory')

    args = parser.parse_args()

    if args.file:
        # Analyze single file
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return

        analyze_single_file(args.file)

    else:
        # Analyze directory
        if not os.path.exists(args.results_dir):
            print(f"Directory not found: {args.results_dir}")
            return

        analyze_directory(args.results_dir)


if __name__ == '__main__':
    main()
