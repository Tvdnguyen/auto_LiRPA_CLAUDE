"""
Conv1 Sensitivity Analysis - Find Most Sensitive Regions

Strategy:
1. Divide conv1 output tensor into regions (by channel groups and spatial locations)
2. For each region, gradually increase epsilon until prediction becomes NOT verified robust
3. Region with smallest epsilon = most sensitive
4. Region with largest epsilon = most resilient

Output:
- CSV with epsilon thresholds for all regions
- Heatmap visualization showing resilient (white) vs sensitive (red) regions
"""

import os
import sys
import argparse
import torch
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traffic_sign_net import TrafficSignNet, TrafficSignNetNoDropout
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from intermediate_bound_module import IntermediateBoundedModule
from collect_correct_samples import load_correct_indices


class Conv1SensitivityAnalyzer:
    """Analyze sensitivity of different regions in conv1 output"""

    def __init__(self, model, checkpoint_path, data_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model.to(self.device)

        # Create verification model (no dropout)
        print("Creating no-dropout model for verification...")
        verification_model = TrafficSignNetNoDropout(num_classes=43)
        verification_model.load_from_dropout_checkpoint(checkpoint_path)
        verification_model.eval()
        verification_model = verification_model.to(self.device)

        # Create bounded module
        print("Creating bounded module...")
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        self.lirpa_model = IntermediateBoundedModule(
            verification_model,
            dummy_input,
            device=self.device
        )

        # Load dataset
        self.test_dataset = GTSRBDataset(
            root_dir=data_dir,
            train=False,
            transform=get_gtsrb_transforms(train=False, img_size=32)
        )

        print(f"Dataset loaded: {len(self.test_dataset)} test samples")

        # Get conv1 node name
        self.conv1_node_name = self._get_conv1_node_name()
        print(f"Conv1 node name: {self.conv1_node_name}")

        # Get conv1 output shape
        self.conv1_shape = self._get_conv1_shape()
        print(f"Conv1 output shape: {self.conv1_shape}")

    def _get_conv1_node_name(self):
        """Get the node name for conv1 in the computation graph"""
        layers_info = self.model.get_layer_info()
        conv1_name = layers_info[0][0]  # 'conv1'

        # Find matching node in graph
        layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])
        for node_name, node_type in layers:
            if conv1_name in node_name or node_name.endswith(conv1_name):
                return node_name

        # Fallback: return first conv
        return layers[0][0]

    def _get_conv1_shape(self):
        """Get conv1 output shape by running a forward pass"""
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        with torch.no_grad():
            _ = self.lirpa_model(dummy_input)

        # Get conv1 output from intermediate outputs
        conv1_output = self.lirpa_model.intermediate_outputs.get(self.conv1_node_name)
        if conv1_output is not None:
            # Shape: (batch, channels, height, width)
            return conv1_output.shape[1:]  # Remove batch dimension

        # Default for conv1: (32, 32, 32)
        return (32, 32, 32)

    def define_regions(self, strategy='grid'):
        """
        Define regions to test in conv1 output

        Conv1 output shape: (32 channels, 32 height, 32 width)

        Strategy options:
        - 'grid': Divide into spatial grid (4x4 spatial regions) × channel groups
        - 'channel_groups': Test channel groups independently
        - 'spatial_only': Test spatial regions with all channels
        - 'comprehensive': Combination of all

        Returns:
            List of region dicts with 'name', 'channel_idx', 'height_slice', 'width_slice'
        """
        channels, height, width = self.conv1_shape
        regions = []

        if strategy == 'grid' or strategy == 'comprehensive':
            # Divide spatial dimension into 4x4 grid
            # Divide channels into 4 groups
            h_step = height // 4
            w_step = width // 4
            c_step = channels // 4

            for c_group in range(4):
                c_start = c_group * c_step
                c_end = (c_group + 1) * c_step if c_group < 3 else channels

                for h_idx in range(4):
                    h_start = h_idx * h_step
                    h_end = (h_idx + 1) * h_step if h_idx < 3 else height

                    for w_idx in range(4):
                        w_start = w_idx * w_step
                        w_end = (w_idx + 1) * w_step if w_idx < 3 else width

                        regions.append({
                            'name': f'CG{c_group}_H{h_idx}_W{w_idx}',
                            'channel_idx': list(range(c_start, c_end)),
                            'height_slice': (h_start, h_end),
                            'width_slice': (w_start, w_end),
                            'type': 'grid'
                        })

        if strategy == 'channel_groups' or strategy == 'comprehensive':
            # Test each channel group with full spatial extent
            c_step = channels // 8
            for c_group in range(8):
                c_start = c_group * c_step
                c_end = (c_group + 1) * c_step if c_group < 7 else channels

                regions.append({
                    'name': f'ChannelGroup{c_group}',
                    'channel_idx': list(range(c_start, c_end)),
                    'height_slice': None,
                    'width_slice': None,
                    'type': 'channel_group'
                })

        if strategy == 'spatial_only' or strategy == 'comprehensive':
            # Test spatial regions with all channels
            h_step = height // 8
            w_step = width // 8

            for h_idx in range(8):
                h_start = h_idx * h_step
                h_end = (h_idx + 1) * h_step if h_idx < 7 else height

                for w_idx in range(8):
                    w_start = w_idx * w_step
                    w_end = (w_idx + 1) * w_step if w_idx < 7 else width

                    regions.append({
                        'name': f'Spatial_H{h_idx}_W{w_idx}',
                        'channel_idx': None,  # All channels
                        'height_slice': (h_start, h_end),
                        'width_slice': (w_start, w_end),
                        'type': 'spatial'
                    })

        print(f"\nDefined {len(regions)} regions using '{strategy}' strategy")
        return regions

    def find_threshold_epsilon(
        self,
        image,
        label,
        pred_class,
        region,
        epsilon_min=0.0,
        epsilon_max=1.0,
        tolerance=0.001,
        method='backward'
    ):
        """
        Find the maximum epsilon where prediction is still verified robust using binary search

        Args:
            image: Input image tensor
            label: True label
            pred_class: Predicted class (clean)
            region: Region dict with perturbation mask
            epsilon_min: Minimum epsilon (start of search range)
            epsilon_max: Maximum epsilon (end of search range)
            tolerance: Tolerance for binary search convergence
            method: Bound computation method

        Returns:
            max_robust_epsilon: Maximum epsilon where still robust (None if never robust)
        """

        def is_robust_at_epsilon(eps):
            """Helper function to check if robust at given epsilon"""
            if eps <= 0:
                return True

            # Clear previous perturbations
            self.lirpa_model.clear_intermediate_perturbations()

            # Forward pass
            image_batch = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.lirpa_model(image_batch)

            # Create perturbation
            perturbation = MaskedPerturbationLpNorm(
                eps=eps,
                norm=np.inf,
                batch_idx=0,
                channel_idx=region['channel_idx'],
                height_slice=region['height_slice'],
                width_slice=region['width_slice']
            )

            # Register and compute bounds
            self.lirpa_model.register_intermediate_perturbation(
                self.conv1_node_name, perturbation
            )

            try:
                lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                    x=image_batch,
                    method=method
                )

                # Check robustness
                pred_lb = lb[0, pred_class].item()
                other_ub = ub[0].clone()
                other_ub[pred_class] = -float('inf')
                max_other_ub = other_ub.max().item()

                return pred_lb > max_other_ub

            except Exception as e:
                print(f"      Error at epsilon={eps:.4f}: {e}")
                return False

        # Binary search for maximum epsilon
        left = epsilon_min
        right = epsilon_max
        best_epsilon = None

        # First check if robust at epsilon_min
        if not is_robust_at_epsilon(left):
            print(f"    Not robust even at epsilon={left:.4f}")
            return None

        # Check if robust at epsilon_max
        if is_robust_at_epsilon(right):
            print(f"    Robust up to epsilon_max={right:.4f}")
            return right

        # Binary search
        iteration = 0
        while right - left > tolerance:
            mid = (left + right) / 2.0
            iteration += 1

            print(f"    Iteration {iteration}: Testing epsilon={mid:.4f} (range: [{left:.4f}, {right:.4f}])")

            if is_robust_at_epsilon(mid):
                # Still robust, try higher
                best_epsilon = mid
                left = mid
            else:
                # Not robust, try lower
                right = mid

        return best_epsilon if best_epsilon is not None else left

    def analyze_sample(
        self,
        image,
        label,
        global_idx,
        regions,
        epsilon_min=0.0,
        epsilon_max=1.0,
        tolerance=0.001
    ):
        """
        Analyze all regions for a single sample using binary search

        Returns:
            List of results with max robust epsilon for each region
        """
        print(f"\n{'='*80}")
        print(f"Analyzing sample {global_idx} (label={label})")
        print(f"{'='*80}")

        # Get clean prediction
        image_batch = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_batch)
            pred_class = output.argmax(dim=1).item()

        print(f"Clean prediction: {pred_class}")

        if pred_class != label:
            print(f"Warning: Clean prediction {pred_class} != true label {label}")

        results = []

        for i, region in enumerate(regions):
            print(f"\nRegion {i+1}/{len(regions)}: {region['name']}")
            print(f"  Channels: {region['channel_idx']}")
            print(f"  Height: {region['height_slice']}")
            print(f"  Width: {region['width_slice']}")

            max_robust_eps = self.find_threshold_epsilon(
                image=image,
                label=label,
                pred_class=pred_class,
                region=region,
                epsilon_min=epsilon_min,
                epsilon_max=epsilon_max,
                tolerance=tolerance
            )

            if max_robust_eps is None:
                print(f"  → Not robust even at epsilon_min={epsilon_min:.4f}")
                max_robust_eps = 0.0
            elif max_robust_eps >= epsilon_max:
                print(f"  → Robust up to epsilon_max={epsilon_max:.4f}")
            else:
                print(f"  → Maximum robust epsilon: {max_robust_eps:.4f}")

            results.append({
                'region_name': region['name'],
                'region_type': region['type'],
                'channel_idx': region['channel_idx'],
                'height_slice': region['height_slice'],
                'width_slice': region['width_slice'],
                'max_robust_epsilon': max_robust_eps,
                'sample_idx': global_idx,
                'true_label': label,
                'pred_class': pred_class
            })

        return results

    def visualize_sensitivity_map(
        self,
        results,
        output_path='conv1_sensitivity_map.png',
        strategy='grid'
    ):
        """
        Visualize sensitivity map as heatmap

        Args:
            results: List of analysis results
            output_path: Path to save figure
            strategy: Region strategy used
        """
        print(f"\nCreating sensitivity visualization...")

        channels, height, width = self.conv1_shape

        if strategy == 'grid':
            # Create 4x4 spatial grid × 4 channel groups
            # We'll create 4 subplots (one per channel group)
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle('Conv1 Sensitivity Map (by Channel Group)', fontweight='bold', fontsize=14)

            # Organize results by channel group
            for c_group in range(4):
                ax = axes[c_group // 2, c_group % 2]

                # Create 4x4 spatial heatmap
                heatmap = np.zeros((4, 4))

                for result in results:
                    if result['region_type'] == 'grid':
                        name = result['region_name']
                        if name.startswith(f'CG{c_group}_'):
                            # Parse H and W indices
                            parts = name.split('_')
                            h_idx = int(parts[1][1:])
                            w_idx = int(parts[2][1:])

                            heatmap[h_idx, w_idx] = result['max_robust_epsilon']

                # Plot heatmap (red = sensitive, white = resilient)
                im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto',
                              vmin=0, vmax=np.max([r['max_robust_epsilon'] for r in results]))

                ax.set_title(f'Channel Group {c_group}', fontweight='bold')
                ax.set_xlabel('Width Index')
                ax.set_ylabel('Height Index')
                ax.set_xticks(range(4))
                ax.set_yticks(range(4))

                # Add values as text
                for h in range(4):
                    for w in range(4):
                        text = ax.text(w, h, f'{heatmap[h, w]:.3f}',
                                     ha="center", va="center", color="black", fontsize=8)

            # Add colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), label='Threshold Epsilon',
                        orientation='horizontal', pad=0.05, fraction=0.05)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved sensitivity map to: {output_path}")
            plt.close()

        elif strategy == 'spatial_only':
            # Create single 8x8 spatial heatmap (all channels)
            fig, ax = plt.subplots(figsize=(10, 10))

            heatmap = np.zeros((8, 8))

            for result in results:
                if result['region_type'] == 'spatial':
                    name = result['region_name']
                    parts = name.split('_')
                    h_idx = int(parts[1][1:])
                    w_idx = int(parts[2][1:])

                    heatmap[h_idx, w_idx] = result['max_robust_epsilon']

            # Plot (red = sensitive/low epsilon, white/green = resilient/high epsilon)
            im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto',
                          vmin=0, vmax=np.max([r['max_robust_epsilon'] for r in results]))

            ax.set_title('Conv1 Spatial Sensitivity Map (All Channels)', fontweight='bold', fontsize=14)
            ax.set_xlabel('Width Region Index', fontweight='bold')
            ax.set_ylabel('Height Region Index', fontweight='bold')
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))

            # Add grid
            for h in range(9):
                ax.axhline(h - 0.5, color='black', linewidth=0.5)
            for w in range(9):
                ax.axvline(w - 0.5, color='black', linewidth=0.5)

            # Add values
            for h in range(8):
                for w in range(8):
                    if heatmap[h, w] > 0:
                        text = ax.text(w, h, f'{heatmap[h, w]:.3f}',
                                     ha="center", va="center", color="black", fontsize=9, fontweight='bold')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Threshold Epsilon (Higher = More Resilient)', rotation=270, labelpad=20, fontweight='bold')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved sensitivity map to: {output_path}")
            plt.close()

        elif strategy == 'channel_groups':
            # Bar chart for channel groups
            fig, ax = plt.subplots(figsize=(10, 6))

            channel_results = [r for r in results if r['region_type'] == 'channel_group']
            names = [r['region_name'] for r in channel_results]
            epsilons = [r['max_robust_epsilon'] for r in channel_results]

            colors = ['red' if eps < np.median(epsilons) else 'lightgreen' for eps in epsilons]

            ax.bar(names, epsilons, color=colors, edgecolor='black')
            ax.set_xlabel('Channel Group', fontweight='bold')
            ax.set_ylabel('Max Robust Epsilon', fontweight='bold')
            ax.set_title('Conv1 Channel Group Sensitivity', fontweight='bold', fontsize=14)
            ax.axhline(np.median(epsilons), color='blue', linestyle='--', label='Median')
            ax.legend()

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved channel sensitivity to: {output_path}")
            plt.close()

    def save_results(self, results, filename='conv1_sensitivity_results.csv'):
        """Save results to CSV"""
        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['region_name', 'region_type', 'max_robust_epsilon',
                         'sample_idx', 'true_label', 'pred_class',
                         'channel_idx', 'height_slice', 'width_slice']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in results:
                writer.writerow(result)

        print(f"Results saved!")

    def print_summary(self, results):
        """Print summary statistics"""
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        epsilons = [r['max_robust_epsilon'] for r in results]

        print(f"\nTotal regions tested: {len(results)}")
        print(f"Max Robust Epsilon statistics:")
        print(f"  Min (most sensitive): {np.min(epsilons):.4f}")
        print(f"  Max (most resilient): {np.max(epsilons):.4f}")
        print(f"  Mean: {np.mean(epsilons):.4f}")
        print(f"  Median: {np.median(epsilons):.4f}")
        print(f"  Std: {np.std(epsilons):.4f}")

        # Most sensitive regions (lowest epsilon)
        sorted_results = sorted(results, key=lambda x: x['max_robust_epsilon'])
        print(f"\nTop 5 Most Sensitive Regions (lowest max epsilon):")
        for i, r in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {r['region_name']:30s}: ε_max = {r['max_robust_epsilon']:.4f}")

        # Most resilient regions (highest epsilon)
        print(f"\nTop 5 Most Resilient Regions (highest max epsilon):")
        for i, r in enumerate(sorted_results[-5:][::-1], 1):
            print(f"  {i}. {r['region_name']:30s}: ε_max = {r['max_robust_epsilon']:.4f}")

        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Conv1 Sensitivity Analysis'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class ID to test (0-42)')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index within class')
    parser.add_argument('--strategy', type=str, default='spatial_only',
                       choices=['grid', 'channel_groups', 'spatial_only', 'comprehensive'],
                       help='Region division strategy')
    parser.add_argument('--epsilon_min', type=float, default=0.0,
                       help='Minimum epsilon (binary search lower bound)')
    parser.add_argument('--epsilon_max', type=float, default=1.0,
                       help='Maximum epsilon (binary search upper bound)')
    parser.add_argument('--tolerance', type=float, default=0.001,
                       help='Binary search convergence tolerance')
    parser.add_argument('--correct_samples_dir', type=str,
                       default='correct_samples',
                       help='Directory with correct sample indices')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    print("="*80)
    print(" "*20 + "CONV1 SENSITIVITY ANALYSIS")
    print(" "*15 + "Finding Most Sensitive Regions")
    print("="*80)

    # Create model
    model = TrafficSignNet(num_classes=43)

    # Create analyzer
    analyzer = Conv1SensitivityAnalyzer(
        model=model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )

    # Define regions
    regions = analyzer.define_regions(strategy=args.strategy)

    # Load test sample
    try:
        correct_indices = load_correct_indices(args.correct_samples_dir, args.class_id)
        if args.sample_idx >= len(correct_indices):
            print(f"Warning: Sample index {args.sample_idx} out of range. Using index 0.")
            args.sample_idx = 0
        global_idx = correct_indices[args.sample_idx]
    except FileNotFoundError:
        print(f"Warning: Correct samples not found. Using any sample from class {args.class_id}")
        indices = [i for i, label in enumerate(analyzer.test_dataset.labels)
                  if label == args.class_id]
        global_idx = indices[args.sample_idx] if args.sample_idx < len(indices) else indices[0]

    image, label = analyzer.test_dataset[global_idx]

    # Analyze sample
    results = analyzer.analyze_sample(
        image=image,
        label=label,
        global_idx=global_idx,
        regions=regions,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        tolerance=args.tolerance
    )

    # Summary
    analyzer.print_summary(results)

    # Visualize
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'conv1_sensitivity_map_{args.strategy}_{timestamp}.png'
    analyzer.visualize_sensitivity_map(results, output_path, strategy=args.strategy)

    # Save results
    csv_path = f'conv1_sensitivity_results_{timestamp}.csv'
    analyzer.save_results(results, csv_path)

    print("\n" + "="*80)
    print("Analysis complete!")
    print(f"  → {output_path}: Sensitivity visualization")
    print(f"  → {csv_path}: Detailed results")
    print("="*80)


if __name__ == '__main__':
    main()
