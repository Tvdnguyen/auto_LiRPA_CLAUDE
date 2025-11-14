"""
Training script for Traffic Sign Recognition on GTSRB dataset
Target: >90% test accuracy
"""
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np

from traffic_sign_net import TrafficSignNet, TrafficSignNetSimple
from gtsrb_dataset import get_gtsrb_dataloaders


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx+1}/{len(train_loader)} | '
                  f'Loss: {running_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.2f}% ({correct}/{total})')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, test_loader, criterion, device):
    """Validate on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=50,
    lr=0.001,
    device='cuda',
    save_path='checkpoints/traffic_sign_net.pth',
    target_acc=90.0,
    min_epochs=10
):
    """
    Train the model

    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        save_path: Path to save best model

    Returns:
        Trained model
    """
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_acc = 0.0
    train_history = []
    test_history = []

    print(f"Training on device: {device}")
    print(f"Total epochs: {num_epochs}")
    print("="*70)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        test_loss, test_acc = validate(model, test_loader, criterion, device)

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - start_time

        # Print epoch summary
        print(f'\nEpoch {epoch}/{num_epochs} Summary:')
        print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        print("="*70)

        # Save history
        train_history.append((train_loss, train_acc))
        test_history.append((test_loss, test_acc))

        # Save best model
        if test_acc > best_acc:
            print(f'>>> New best accuracy: {test_acc:.2f}% (previous: {best_acc:.2f}%)')
            best_acc = test_acc

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }
            torch.save(checkpoint, save_path)
            print(f'>>> Model saved to {save_path}')

        # Early stopping if we achieve target accuracy
        if test_acc >= target_acc:
            print(f'\n*** Target accuracy achieved: {test_acc:.2f}% >= {target_acc}% ***')
            if epoch >= min_epochs:  # Make sure we train for minimum epochs
                print(f'Early stopping after {epoch} epochs...')
                break
            else:
                print(f'Continuing training (minimum {min_epochs} epochs required)...')

        print()

    print(f'\nTraining completed!')
    print(f'Best test accuracy: {best_acc:.2f}%')

    return model, train_history, test_history


def main():
    parser = argparse.ArgumentParser(description='Train Traffic Sign Recognition Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset directory')
    parser.add_argument('--model', type=str, default='full', choices=['full', 'simple'],
                       help='Model architecture (full or simple)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_path', type=str, default='checkpoints/traffic_sign_net.pth',
                       help='Path to save model checkpoint')
    parser.add_argument('--img_size', type=int, default=32,
                       help='Input image size')
    parser.add_argument('--target_acc', type=float, default=90.0,
                       help='Target accuracy for early stopping (default: 90.0)')
    parser.add_argument('--min_epochs', type=int, default=10,
                       help='Minimum epochs before early stopping (default: 10)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"Loading GTSRB dataset from {args.data_dir}...")
    train_loader, test_loader = get_gtsrb_dataloaders(
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

    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Train
    print("\nStarting training...")
    print(f"Early stopping: Enabled (target accuracy: {args.target_acc}%, min epochs: {args.min_epochs})")
    model, train_hist, test_hist = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_path,
        target_acc=args.target_acc,
        min_epochs=args.min_epochs
    )

    print(f"\nModel saved to: {args.save_path}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()
