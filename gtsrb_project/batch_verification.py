#!/usr/bin/env python3
"""
Batch Verification Script for GTSRB Intermediate Perturbations

Chương trình này:
1. Cho phép user chọn một class và cấu hình perturbation
2. Chạy verification trên TẤT CẢ samples correctly classified trong class đó
3. Thu thập thống kê: bao nhiêu verified robust, bao nhiêu not verified
4. Lưu kết quả chi tiết vào CSV file

Khác với main_interactive.py (test từng sample một),
script này batch test toàn bộ class để có thống kê tổng quan.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import csv
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traffic_sign_net import TrafficSignNet, TrafficSignNetSimple, TrafficSignNetNoDropout
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from intermediate_bound_module import IntermediateBoundedModule
from collect_correct_samples import load_correct_indices


class BatchVerifier:
    """Batch verification cho một class cụ thể"""

    def __init__(self, model, checkpoint_path, data_dir, device='cuda'):
        """
        Initialize batch verifier

        Args:
            model: Neural network model
            checkpoint_path: Path to trained checkpoint
            data_dir: Path to GTSRB dataset
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model for display
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model.to(self.device)

        print(f"Model loaded successfully!")
        print(f"  Checkpoint accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")

        # Create verification model (no dropout)
        if hasattr(model, '__class__'):
            if model.__class__.__name__ == 'TrafficSignNet':
                print("\nCreating no-dropout model for verification...")
                verification_model = TrafficSignNetNoDropout(num_classes=43)
                verification_model.load_from_dropout_checkpoint(checkpoint_path)
                verification_model.eval()
                verification_model = verification_model.to(self.device)
                print("No-dropout model created")
            else:
                verification_model = model
        else:
            verification_model = model

        # Create bounded module
        print("\nCreating bounded module...")
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        self.lirpa_model = IntermediateBoundedModule(
            verification_model,
            dummy_input,
            device=self.device
        )

        # Load dataset
        self.data_dir = data_dir
        self.test_dataset = GTSRBDataset(
            root_dir=data_dir,
            train=False,
            transform=get_gtsrb_transforms(train=False, img_size=32)
        )

        print(f"Dataset loaded: {len(self.test_dataset)} test samples")

    def get_layer_info(self):
        """Get layer information for user selection"""
        if hasattr(self.model, 'get_layer_info'):
            return self.model.get_layer_info()
        else:
            # Fallback
            layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])
            return [(name, ltype, None, "unknown") for name, ltype in layers]

    def show_layers(self):
        """Display available layers"""
        print("\n" + "="*80)
        print("Available Layers for Perturbation")
        print("="*80)

        layers_info = self.get_layer_info()
        print(f"{'Index':>5} | {'Layer Name':^15} | {'Type':^10} | {'Output Shape':^15}")
        print("-"*80)
        for i, (name, ltype, layer, shape) in enumerate(layers_info):
            print(f"{i:>5} | {name:^15} | {ltype:^10} | {shape:^15}")

        print("="*80)
        return layers_info

    def get_node_name_from_index(self, layer_idx):
        """Get computation graph node name from layer index"""
        if hasattr(self.model, 'get_layer_info'):
            layers_info = self.model.get_layer_info()
            if 0 <= layer_idx < len(layers_info):
                name, ltype, layer, shape_str = layers_info[layer_idx]

                # Map to graph node name
                graph_layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])
                for node_name, node_type in graph_layers:
                    if name in node_name or node_name.endswith(name):
                        return node_name, ltype, shape_str

                if layer_idx < len(graph_layers):
                    return graph_layers[layer_idx][0], ltype, shape_str

        # Fallback
        graph_layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])
        if 0 <= layer_idx < len(graph_layers):
            node_name, node_type = graph_layers[layer_idx]
            return node_name, node_type, "unknown"

        raise ValueError(f"Invalid layer index: {layer_idx}")

    def load_class_samples(self, class_id, correct_samples_dir='correct_samples'):
        """
        Load all correctly classified samples for a class

        Args:
            class_id: Class ID (0-42)
            correct_samples_dir: Directory with CSV files

        Returns:
            List of (image, label, global_idx)
        """
        # Load correct indices (function takes directory and class_id)
        indices = load_correct_indices(correct_samples_dir, class_id)
        print(f"\nLoading {len(indices)} samples from class {class_id}...")

        samples = []
        for idx in indices:
            image, label = self.test_dataset[idx]
            samples.append((image, label, idx))

        return samples

    def verify_sample(
        self,
        image,
        label,
        node_name,
        epsilon,
        batch_idx=0,
        channel_idx=None,
        height_slice=None,
        width_slice=None,
        method='backward'
    ):
        """
        Verify một sample với perturbation configuration

        Args:
            image: Input image tensor
            label: True label
            node_name: Node to perturb
            epsilon: Perturbation magnitude
            batch_idx, channel_idx, height_slice, width_slice: Mask specs
            method: Bound computation method

        Returns:
            dict with verification results
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Clean forward pass
        with torch.no_grad():
            clean_output = self.model(image)
            pred_class = clean_output.argmax(dim=1).item()

        # Check if prediction correct
        if pred_class != label:
            return {
                'verified': False,
                'reason': 'incorrect_prediction',
                'pred_class': pred_class,
                'true_label': label,
                'clean_logit': None,
                'lower_bound': None,
                'upper_bound': None,
                'margin': None
            }

        clean_logit = clean_output[0, label].item()

        # Create masked perturbation
        perturbation = MaskedPerturbationLpNorm(
            eps=epsilon,
            norm=np.inf,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            height_slice=height_slice,
            width_slice=width_slice
        )

        # Register perturbation
        self.lirpa_model.register_intermediate_perturbation(node_name, perturbation)

        # Compute bounds
        try:
            lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                x=image,
                method=method
            )

            lb = lb.cpu()
            ub = ub.cpu()

            # Check verification
            true_label_lb = lb[0, label].item()
            true_label_ub = ub[0, label].item()

            # Get max bound of other classes
            other_classes_mask = torch.ones(43, dtype=torch.bool)
            other_classes_mask[label] = False
            max_other_ub = ub[0, other_classes_mask].max().item()

            # Verified if lower bound của true class > upper bound của tất cả classes khác
            verified = true_label_lb > max_other_ub
            margin = true_label_lb - max_other_ub

            return {
                'verified': verified,
                'reason': 'verified' if verified else 'not_verified',
                'pred_class': pred_class,
                'true_label': label,
                'clean_logit': clean_logit,
                'lower_bound': true_label_lb,
                'upper_bound': true_label_ub,
                'margin': margin,
                'max_other_ub': max_other_ub
            }

        except Exception as e:
            return {
                'verified': False,
                'reason': f'error: {str(e)}',
                'pred_class': pred_class,
                'true_label': label,
                'clean_logit': clean_logit,
                'lower_bound': None,
                'upper_bound': None,
                'margin': None
            }

    def batch_verify_class(
        self,
        class_id,
        node_name,
        epsilon,
        batch_idx=0,
        channel_idx=None,
        height_slice=None,
        width_slice=None,
        method='backward',
        correct_samples_dir='correct_samples',
        output_dir='verification_results'
    ):
        """
        Batch verify tất cả samples trong một class

        Args:
            class_id: Class ID
            node_name: Node to perturb
            epsilon: Perturbation magnitude
            ... : Mask specifications
            method: Bound computation method
            correct_samples_dir: Directory with correct samples
            output_dir: Directory to save results

        Returns:
            dict with statistics
        """
        # Load samples
        samples = self.load_class_samples(class_id, correct_samples_dir)

        if len(samples) == 0:
            print(f"No samples found for class {class_id}")
            return None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            output_dir,
            f"class_{class_id:02d}_verification_{timestamp}.csv"
        )

        # Statistics
        stats = {
            'total': len(samples),
            'verified': 0,
            'not_verified': 0,
            'errors': 0,
            'incorrect_pred': 0
        }

        results = []

        print(f"\nVerifying {len(samples)} samples from class {class_id}...")
        print(f"Configuration:")
        print(f"  Node: {node_name}")
        print(f"  Epsilon: {epsilon}")
        print(f"  Batch: {batch_idx}")
        print(f"  Channels: {channel_idx}")
        print(f"  Height: {height_slice}")
        print(f"  Width: {width_slice}")
        print(f"  Method: {method}")
        print("")

        # Verify each sample
        for i, (image, label, global_idx) in enumerate(tqdm(samples, desc="Verifying")):
            result = self.verify_sample(
                image=image,
                label=label,
                node_name=node_name,
                epsilon=epsilon,
                batch_idx=batch_idx,
                channel_idx=channel_idx,
                height_slice=height_slice,
                width_slice=width_slice,
                method=method
            )

            # Update statistics
            if result['reason'] == 'incorrect_prediction':
                stats['incorrect_pred'] += 1
            elif result['reason'] == 'verified':
                stats['verified'] += 1
            elif result['reason'] == 'not_verified':
                stats['not_verified'] += 1
            elif 'error' in result['reason']:
                stats['errors'] += 1

            # Store result
            results.append({
                'sample_idx': i,
                'global_idx': global_idx,
                'class_id': class_id,
                'verified': result['verified'],
                'reason': result['reason'],
                'clean_logit': result['clean_logit'],
                'lower_bound': result['lower_bound'],
                'upper_bound': result['upper_bound'],
                'margin': result['margin']
            })

        # Save results to CSV
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['sample_idx', 'global_idx', 'class_id', 'verified',
                         'reason', 'clean_logit', 'lower_bound', 'upper_bound', 'margin']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        # Print statistics
        print("\n" + "="*80)
        print("VERIFICATION RESULTS")
        print("="*80)
        print(f"Class: {class_id}")
        print(f"Total samples: {stats['total']}")
        print(f"  ✓ Verified robust: {stats['verified']} ({100*stats['verified']/stats['total']:.1f}%)")
        print(f"  ✗ Not verified: {stats['not_verified']} ({100*stats['not_verified']/stats['total']:.1f}%)")
        print(f"  ⚠ Errors: {stats['errors']}")
        print(f"  ⚠ Incorrect predictions: {stats['incorrect_pred']}")
        print("="*80)
        print(f"Detailed results saved to: {output_file}")

        # Save summary
        summary_file = os.path.join(output_dir, f"class_{class_id:02d}_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("VERIFICATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Class: {class_id}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write("Configuration:\n")
            f.write(f"  Node: {node_name}\n")
            f.write(f"  Epsilon: {epsilon}\n")
            f.write(f"  Batch: {batch_idx}\n")
            f.write(f"  Channels: {channel_idx}\n")
            f.write(f"  Height slice: {height_slice}\n")
            f.write(f"  Width slice: {width_slice}\n")
            f.write(f"  Method: {method}\n\n")
            f.write("Results:\n")
            f.write(f"  Total samples: {stats['total']}\n")
            f.write(f"  Verified robust: {stats['verified']} ({100*stats['verified']/stats['total']:.1f}%)\n")
            f.write(f"  Not verified: {stats['not_verified']} ({100*stats['not_verified']/stats['total']:.1f}%)\n")
            f.write(f"  Errors: {stats['errors']}\n")
            f.write(f"  Incorrect predictions: {stats['incorrect_pred']}\n")

        return stats, results


def main():
    parser = argparse.ArgumentParser(
        description='Batch verification of intermediate layer perturbations for a class'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default='full',
                       choices=['full', 'simple'],
                       help='Model architecture')
    parser.add_argument('--correct_samples_dir', type=str,
                       default='correct_samples',
                       help='Directory with correct sample indices')
    parser.add_argument('--output_dir', type=str,
                       default='verification_results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Create model
    if args.model == 'full':
        model = TrafficSignNet(num_classes=43)
    else:
        model = TrafficSignNetSimple(num_classes=43)

    # Create verifier
    verifier = BatchVerifier(
        model=model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )

    # Interactive configuration
    print("\n" + "="*80)
    print(" "*25 + "BATCH VERIFICATION")
    print("="*80)

    # Step 1: Select layer
    layers_info = verifier.show_layers()
    while True:
        try:
            layer_idx = int(input("\nSelect layer index to perturb: "))
            node_name, layer_type, shape = verifier.get_node_name_from_index(layer_idx)
            print(f"Selected: {node_name} ({layer_type}, shape: {shape})")
            break
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")

    # Step 2: Select class
    while True:
        try:
            class_id = int(input("\nSelect class ID (0-42): "))
            if 0 <= class_id <= 42:
                break
            else:
                print("Class ID must be between 0 and 42")
        except ValueError:
            print("Invalid input")

    # Step 3: Configure perturbation
    print("\nConfigure perturbation region:")
    print("(Press Enter to skip, will perturb all)")

    # Channels
    channel_input = input("Channel indices (e.g., '0,1,2' or '0-5'): ").strip()
    if channel_input:
        if '-' in channel_input:
            start, end = map(int, channel_input.split('-'))
            channel_idx = list(range(start, end+1))
        else:
            channel_idx = [int(x.strip()) for x in channel_input.split(',')]
        print(f"  Selected channels: {channel_idx}")
    else:
        channel_idx = None
        print("  All channels will be perturbed")

    # Height slice
    height_input = input("Height slice (e.g., '5,10' for [5:10]): ").strip()
    if height_input:
        height_slice = tuple(map(int, height_input.split(',')))
        print(f"  Height slice: {height_slice}")
    else:
        height_slice = None
        print("  All height will be perturbed")

    # Width slice
    width_input = input("Width slice (e.g., '5,10' for [5:10]): ").strip()
    if width_input:
        width_slice = tuple(map(int, width_input.split(',')))
        print(f"  Width slice: {width_slice}")
    else:
        width_slice = None
        print("  All width will be perturbed")

    # Epsilon
    while True:
        try:
            epsilon = float(input("\nEpsilon (perturbation magnitude, e.g., 0.1): "))
            if epsilon > 0:
                break
            else:
                print("Epsilon must be positive")
        except ValueError:
            print("Invalid input")

    # Confirm
    print("\n" + "="*80)
    print("Configuration Summary:")
    print(f"  Class: {class_id}")
    print(f"  Layer: {node_name}")
    print(f"  Channels: {channel_idx if channel_idx else 'All'}")
    print(f"  Height: {height_slice if height_slice else 'All'}")
    print(f"  Width: {width_slice if width_slice else 'All'}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Method: backward (CROWN)")
    print("="*80)

    confirm = input("\nProceed with verification? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Run batch verification
    stats, results = verifier.batch_verify_class(
        class_id=class_id,
        node_name=node_name,
        epsilon=epsilon,
        batch_idx=0,
        channel_idx=channel_idx,
        height_slice=height_slice,
        width_slice=width_slice,
        method='backward',
        correct_samples_dir=args.correct_samples_dir,
        output_dir=args.output_dir
    )

    print("\nBatch verification completed!")


if __name__ == '__main__':
    main()
