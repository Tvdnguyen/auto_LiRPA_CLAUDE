"""
Inference script to collect correctly classified samples per class
Saves indices of correctly classified samples to CSV files
"""
import os
import argparse
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from traffic_sign_net import TrafficSignNet, TrafficSignNetSimple
from gtsrb_dataset import get_gtsrb_dataloaders


def evaluate_and_collect(model, test_loader, device, output_dir='correct_samples'):
    """
    Evaluate model and collect correctly classified sample indices per class

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        output_dir: Directory to save CSV files

    Returns:
        Dictionary mapping class_id -> list of correctly classified indices
        Overall accuracy
    """
    model.eval()

    # Storage for correct predictions per class
    correct_per_class = {i: [] for i in range(43)}

    # Statistics
    total_correct = 0
    total_samples = 0
    predictions_list = []
    targets_list = []

    print("Running inference on test set...")

    with torch.no_grad():
        global_idx = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            # Check correct predictions
            correct_mask = predicted.eq(targets)

            # Process each sample in batch
            for i in range(len(targets)):
                sample_idx = global_idx + i
                target_class = targets[i].item()
                is_correct = correct_mask[i].item()

                predictions_list.append(predicted[i].item())
                targets_list.append(target_class)

                if is_correct:
                    correct_per_class[target_class].append(sample_idx)
                    total_correct += 1

                total_samples += 1

            global_idx += len(targets)

    # Calculate accuracy
    overall_acc = 100.0 * total_correct / total_samples

    print(f"\nOverall Test Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})")

    # Print per-class statistics
    print("\nPer-class statistics:")
    print(f"{'Class':>5} | {'Correct':>7} | {'Samples':>7} | {'Accuracy':>8}")
    print("-" * 40)

    class_total = {i: 0 for i in range(43)}
    for target in targets_list:
        class_total[target] += 1

    for class_id in range(43):
        correct_count = len(correct_per_class[class_id])
        total_count = class_total[class_id]
        class_acc = 100.0 * correct_count / total_count if total_count > 0 else 0.0
        print(f"{class_id:>5} | {correct_count:>7} | {total_count:>7} | {class_acc:>7.2f}%")

    # Save to CSV files
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving correctly classified indices to {output_dir}/...")

    for class_id in range(43):
        csv_file = os.path.join(output_dir, f'class_{class_id:02d}_correct_indices.csv')
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index'])
            for idx in correct_per_class[class_id]:
                writer.writerow([idx])

        print(f"  Class {class_id:02d}: {len(correct_per_class[class_id])} correct samples saved")

    # Save summary
    summary_file = os.path.join(output_dir, 'summary.csv')
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_id', 'correct_count', 'total_count', 'accuracy'])
        for class_id in range(43):
            correct_count = len(correct_per_class[class_id])
            total_count = class_total[class_id]
            class_acc = 100.0 * correct_count / total_count if total_count > 0 else 0.0
            writer.writerow([class_id, correct_count, total_count, f"{class_acc:.2f}"])

    print(f"\nSummary saved to {summary_file}")

    return correct_per_class, overall_acc


def load_correct_indices(output_dir='correct_samples', class_id=0):
    """
    Load correctly classified indices for a specific class

    Args:
        output_dir: Directory containing CSV files
        class_id: Class ID to load

    Returns:
        List of indices
    """
    csv_file = os.path.join(output_dir, f'class_{class_id:02d}_correct_indices.csv')

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    indices = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row['index']))

    return indices


def main():
    parser = argparse.ArgumentParser(description='Collect correctly classified samples')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'simple'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='correct_samples',
                       help='Directory to save CSV files')
    parser.add_argument('--img_size', type=int, default=32,
                       help='Input image size')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading GTSRB dataset from {args.data_dir}...")
    _, test_loader = get_gtsrb_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )

    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'full':
        model = TrafficSignNet(num_classes=43)
    else:
        model = TrafficSignNetSimple(num_classes=43)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Test Accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")

    # Evaluate and collect
    correct_per_class, overall_acc = evaluate_and_collect(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir
    )

    print(f"\nCollection completed!")
    print(f"Files saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
