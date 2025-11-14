"""
Integrated PE-to-Conv1 Sensitivity Analysis

Workflow:
1. Run pe_position_analysis.py to find affected regions for each PE
2. For each PE's affected region, run conv1_sensitivity_analysis.py to find max epsilon
3. Rank PEs by epsilon (lower epsilon = more sensitive)

Goal: Find which PE position causes most sensitive perturbation to DNN output
"""

import sys
import os
import numpy as np
import csv
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'systolic_fault_sim'))

from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet, TrafficSignNetNoDropout
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from intermediate_bound_module import IntermediateBoundedModule
from collect_correct_samples import load_correct_indices
from operand_matrix import OperandMatrix


class IntegratedPESensitivityAnalyzer:
    """Analyze sensitivity of each PE by testing affected conv1 regions"""

    def __init__(self, checkpoint_path, data_dir, array_size=8, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model
            data_dir: Path to GTSRB dataset
            array_size: Systolic array size (default: 8)
            device: Device for verification
        """
        self.array_size = array_size
        self.dataflow = 'IS'
        self.component = 'accumulator_register'
        self.layer_idx = 0  # conv1
        self.device = torch.device(device)

        print(f"Initializing Integrated PE Sensitivity Analyzer")
        print(f"  Array size: {array_size}×{array_size}")
        print(f"  Device: {self.device}")

        # Load DNN model for verification
        self._load_verification_model(checkpoint_path, data_dir)

        # Get conv1 node name - will be determined after first forward pass
        self.conv1_node_name = None
        self._determine_conv1_node_name()

        self.results = []

    def _load_verification_model(self, checkpoint_path, data_dir):
        """Load model and create bounded module"""
        print(f"\nLoading verification model...")

        # Load no-dropout model
        self.model = TrafficSignNetNoDropout(num_classes=43)
        self.model.load_from_dropout_checkpoint(checkpoint_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        # Create bounded module
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        self.lirpa_model = IntermediateBoundedModule(
            self.model,
            dummy_input,
            device=self.device
        )

        # Load dataset
        self.test_dataset = GTSRBDataset(
            root_dir=data_dir,
            train=False,
            transform=get_gtsrb_transforms(train=False, img_size=32)
        )

        print(f"Model loaded successfully")

    def _determine_conv1_node_name(self):
        """Determine conv1 node name by doing a forward pass and checking intermediate outputs"""
        print(f"\nDetermining conv1 node name...")

        # Get all conv layers from LiRPA model
        layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])

        print(f"  Available layers in LiRPA model:")
        for i, (node_name, node_type) in enumerate(layers[:5]):
            print(f"    {i}: {node_name} ({node_type})")

        if len(layers) == 0:
            raise ValueError("No Conv layers found in model!")

        # Candidate conv1 node from layer names
        candidate_conv1 = layers[0][0]
        print(f"  Candidate from layer names: {candidate_conv1}")

        # Do a forward pass to populate intermediate_outputs
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        with torch.no_grad():
            _ = self.lirpa_model(dummy_input)

        # Check what keys are actually in intermediate_outputs
        available_keys = list(self.lirpa_model.intermediate_outputs.keys())
        print(f"  Keys in intermediate_outputs: {available_keys[:5]}")

        # Try to match candidate with actual keys
        if candidate_conv1 in available_keys:
            self.conv1_node_name = candidate_conv1
            print(f"  ✓ Using: {self.conv1_node_name}")
        else:
            # Try first key that looks like conv output
            for key in available_keys:
                output = self.lirpa_model.intermediate_outputs[key]
                # Conv1 output should be shape (1, 32, 32, 32) for GTSRB
                if len(output.shape) == 4 and output.shape[1] == 32:
                    self.conv1_node_name = key
                    print(f"  ✓ Using (by shape matching): {self.conv1_node_name}")
                    break

            if self.conv1_node_name is None:
                # Fallback: use first available key
                self.conv1_node_name = available_keys[0]
                print(f"  ⚠ Warning: Using first available key: {self.conv1_node_name}")

    def get_affected_region_for_pe(self, pe_row, pe_col, duration=10):
        """
        Run fault simulation to find affected region in conv1 output

        Returns:
            affected_channels: List of affected channel indices
            affected_spatial: List of (h, w) affected spatial positions
        """
        print(f"  Finding affected region for PE({pe_row},{pe_col})...")

        # Create simulator
        simulator = SystolicFaultSimulator(
            self.array_size,
            self.array_size,
            self.dataflow
        )

        # Get layer config
        model = TrafficSignNet(num_classes=43)
        layer_config = simulator.get_layer_config(model, self.layer_idx)

        # Get critical timing
        op_gen = OperandMatrix(layer_config)
        operand_mats = op_gen.generate_matrices()
        dims = operand_mats['dimensions']

        T = dims['conv_window_size']
        H = self.array_size
        comp_start = H
        comp_end = H + T - 1
        critical_cycle = (comp_start + comp_end) // 2

        # Create fault
        fault = FaultModel(
            fault_type=FaultModel.BIT_FLIP,
            fault_location={
                'pe_row': pe_row,
                'pe_col': pe_col,
                'component': self.component
            },
            fault_timing={
                'start_cycle': critical_cycle,
                'duration': duration
            }
        )

        # Run simulation
        try:
            results = simulator.simulate_layer(layer_config, [fault])
            fault_mask = results['fault_mask']  # Shape: (spatial_pixels, channels)

            # Find affected positions
            affected_positions = np.where(fault_mask)

            if len(affected_positions[0]) == 0:
                return None, None

            # Extract spatial and channel info
            spatial_indices = affected_positions[0]
            channel_indices = affected_positions[1]

            # Convert spatial indices to (h, w)
            out_shape = layer_config['output_shape']
            H_out, W_out = out_shape[1], out_shape[2]

            affected_spatial = []
            for spatial_idx in spatial_indices:
                h = spatial_idx // W_out
                w = spatial_idx % W_out
                affected_spatial.append((h, w))

            # Get unique channels and spatial positions
            unique_channels = np.unique(channel_indices).tolist()
            unique_spatial = list(set(affected_spatial))

            print(f"    Affected: {len(unique_channels)} channels, {len(unique_spatial)} spatial positions")

            return unique_channels, unique_spatial

        except Exception as e:
            print(f"    Error: {str(e)[:80]}")
            return None, None

    def find_max_epsilon_for_region(self, image, label, pred_class,
                                   affected_channels, affected_spatial,
                                   epsilon_min=0.0, epsilon_max=1.0, tolerance=0.001):
        """
        Find maximum epsilon for affected region using binary search

        Args:
            image: Input image
            label: True label
            pred_class: Predicted class
            affected_channels: List of channel indices
            affected_spatial: List of (h, w) tuples
            epsilon_min, epsilon_max, tolerance: Binary search params

        Returns:
            max_robust_epsilon
        """
        def is_robust_at_epsilon(eps):
            """Check if robust at given epsilon"""
            if eps <= 0:
                return True

            # Clear previous perturbations
            self.lirpa_model.clear_intermediate_perturbations()

            # Forward pass
            image_batch = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.lirpa_model(image_batch)

            # Create perturbation for affected region
            # Note: We perturb ALL affected positions together
            perturbation = MaskedPerturbationLpNorm(
                eps=eps,
                norm=np.inf,
                batch_idx=0,
                channel_idx=affected_channels,
                height_slice=None,  # Will handle via custom mask
                width_slice=None
            )

            # Custom mask for exact spatial positions
            # Get conv1 output shape
            if self.conv1_node_name not in self.lirpa_model.intermediate_outputs:
                raise KeyError(
                    f"conv1_node_name '{self.conv1_node_name}' not found in intermediate_outputs. "
                    f"Available: {list(self.lirpa_model.intermediate_outputs.keys())[:5]}"
                )

            conv1_output = self.lirpa_model.intermediate_outputs[self.conv1_node_name]
            mask_shape = conv1_output.shape  # (1, C, H, W)

            # Override perturbation's create_mask to use exact positions
            original_create_mask = perturbation.create_mask

            def custom_create_mask(shape):
                mask = torch.zeros(shape, dtype=torch.bool)
                # Mark affected positions
                for h, w in affected_spatial:
                    if h < shape[2] and w < shape[3]:
                        if affected_channels is None:
                            mask[0, :, h, w] = True
                        else:
                            for c in affected_channels:
                                if c < shape[1]:
                                    mask[0, c, h, w] = True
                return mask

            perturbation.create_mask = custom_create_mask

            # Register and compute bounds
            self.lirpa_model.register_intermediate_perturbation(
                self.conv1_node_name, perturbation
            )

            try:
                lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                    x=image_batch,
                    method='backward'
                )

                # Check robustness
                pred_lb = lb[0, pred_class].item()
                other_ub = ub[0].clone()
                other_ub[pred_class] = -float('inf')
                max_other_ub = other_ub.max().item()

                return pred_lb > max_other_ub

            except Exception as e:
                print(f"      Error at eps={eps:.4f}: {str(e)[:50]}")
                return False

        # Binary search
        left = epsilon_min
        right = epsilon_max
        best_epsilon = None

        # Check boundaries
        if not is_robust_at_epsilon(left):
            return None

        if is_robust_at_epsilon(right):
            return right

        # Binary search
        iteration = 0
        while right - left > tolerance:
            mid = (left + right) / 2.0
            iteration += 1

            print(f"      Iteration {iteration}: eps={mid:.4f}", end="")

            if is_robust_at_epsilon(mid):
                best_epsilon = mid
                left = mid
                print(f" → robust")
            else:
                right = mid
                print(f" → not robust")

        return best_epsilon if best_epsilon is not None else left

    def analyze_all_pe_positions(self, class_id=0, sample_idx=243, duration=10,
                                epsilon_max=1.0, tolerance=0.001):
        """
        Main analysis: test all PE positions

        Args:
            class_id: Test image class (deprecated, kept for compatibility)
            sample_idx: Global index in test dataset (default: 243)
            duration: Fault duration
            epsilon_max: Max epsilon to search
            tolerance: Binary search tolerance
        """
        print(f"\n{'='*80}")
        print(f"INTEGRATED PE SENSITIVITY ANALYSIS")
        print(f"{'='*80}")
        print(f"\nTest configuration:")
        print(f"  Array size: {self.array_size}×{self.array_size}")
        print(f"  Requested sample index: {sample_idx}")
        print(f"  Fault duration: {duration} cycles")
        print(f"  Epsilon range: [0, {epsilon_max}]")
        print(f"  Tolerance: {tolerance}")

        # Use sample_idx as global index directly
        global_idx = sample_idx

        # Check dataset size first
        dataset_size = len(self.test_dataset)
        print(f"\nDataset info:")
        print(f"  Test dataset size: {dataset_size}")
        print(f"  Using global index: {global_idx}")

        if global_idx >= dataset_size:
            raise ValueError(
                f"Index {global_idx} out of range for test dataset (size: {dataset_size}). "
                f"Please use an index between 0 and {dataset_size-1}."
            )

        try:
            image, label = self.test_dataset[global_idx]
            print(f"  Successfully loaded sample at index {global_idx}")
        except Exception as e:
            raise ValueError(f"Failed to load test sample at index {global_idx}: {e}")

        # Get clean prediction
        image_batch = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_batch)
            pred_class = output.argmax(dim=1).item()

        print(f"\nTest sample:")
        print(f"  Global index: {global_idx}")
        print(f"  True label: {label}")
        print(f"  Predicted: {pred_class}")

        # Test each PE
        total_pes = self.array_size * self.array_size
        print(f"\n{'='*80}")
        print(f"Testing {total_pes} PE positions")
        print(f"{'='*80}")

        test_num = 0
        for pe_row in range(self.array_size):
            for pe_col in range(self.array_size):
                test_num += 1
                print(f"\n[{test_num}/{total_pes}] PE({pe_row},{pe_col}):")

                # Step 1: Get affected region
                affected_channels, affected_spatial = self.get_affected_region_for_pe(
                    pe_row, pe_col, duration
                )

                if affected_channels is None or len(affected_channels) == 0:
                    print(f"  → No affected region, skipping")
                    self.results.append({
                        'pe_row': pe_row,
                        'pe_col': pe_col,
                        'max_epsilon': None,
                        'affected_channels': 0,
                        'affected_spatial': 0,
                        'status': 'no_effect'
                    })
                    continue

                # Step 2: Find max epsilon for this region
                print(f"  Finding max epsilon for affected region...")
                max_eps = self.find_max_epsilon_for_region(
                    image=image,
                    label=label,
                    pred_class=pred_class,
                    affected_channels=affected_channels,
                    affected_spatial=affected_spatial,
                    epsilon_max=epsilon_max,
                    tolerance=tolerance
                )

                if max_eps is None:
                    print(f"  → Not robust even at epsilon=0")
                    status = 'extremely_sensitive'
                    max_eps = 0.0
                elif max_eps >= epsilon_max:
                    print(f"  → Robust up to epsilon_max={epsilon_max}")
                    status = 'resilient'
                else:
                    print(f"  → Max epsilon: {max_eps:.4f}")
                    status = 'normal'

                self.results.append({
                    'pe_row': pe_row,
                    'pe_col': pe_col,
                    'max_epsilon': max_eps,
                    'affected_channels': len(affected_channels),
                    'affected_spatial': len(affected_spatial),
                    'status': status
                })

        print(f"\n{'='*80}")
        print(f"Completed {test_num} tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Analyze and display results"""
        print(f"\n{'='*80}")
        print("ANALYSIS: PE Sensitivity Ranking")
        print(f"{'='*80}")

        valid_results = [r for r in self.results if r['max_epsilon'] is not None]
        if not valid_results:
            print("No valid results!")
            return

        epsilons = [r['max_epsilon'] for r in valid_results]

        print(f"\nStatistics:")
        print(f"  Valid PEs: {len(valid_results)}/{len(self.results)}")
        print(f"  Epsilon range: {np.min(epsilons):.4f} - {np.max(epsilons):.4f}")
        print(f"  Mean: {np.mean(epsilons):.4f}")
        print(f"  Median: {np.median(epsilons):.4f}")
        print(f"  Std: {np.std(epsilons):.4f}")

        # Sort by epsilon (ascending = most sensitive first)
        sorted_results = sorted(valid_results, key=lambda x: x['max_epsilon'])

        # Top 10 most sensitive
        print(f"\nTop 10 Most Sensitive PEs (Lowest Epsilon):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Max Epsilon':>12} | {'Affected Ch':>12} | {'Affected Sp':>12}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        for i, r in enumerate(sorted_results[:10], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            print(f"{i:>4} | {pe_str:>8} | {r['max_epsilon']:>12.4f} | "
                  f"{r['affected_channels']:>12} | {r['affected_spatial']:>12}")

        # Top 10 most resilient
        print(f"\nTop 10 Most Resilient PEs (Highest Epsilon):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Max Epsilon':>12} | {'Affected Ch':>12} | {'Affected Sp':>12}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        for i, r in enumerate(sorted_results[-10:][::-1], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            print(f"{i:>4} | {pe_str:>8} | {r['max_epsilon']:>12.4f} | "
                  f"{r['affected_channels']:>12} | {r['affected_spatial']:>12}")

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'integrated_pe_sensitivity_{timestamp}.csv'

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['pe_row', 'pe_col', 'max_epsilon', 'affected_channels',
                         'affected_spatial', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow(result)

        print(f"Results saved!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Integrated PE-to-Conv1 Sensitivity Analysis'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--array_size', type=int, default=8,
                       help='Systolic array size (default: 8)')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Test class ID (deprecated, kept for compatibility)')
    parser.add_argument('--sample_idx', type=int, default=243,
                       help='Global index in test dataset (default: 243)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Fault duration (cycles)')
    parser.add_argument('--epsilon_max', type=float, default=1.0,
                       help='Maximum epsilon to search')
    parser.add_argument('--tolerance', type=float, default=0.001,
                       help='Binary search tolerance')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*80)
    print(" "*15 + "INTEGRATED PE SENSITIVITY ANALYSIS")
    print(" "*10 + "Systolic Array Faults → DNN Robustness")
    print("="*80)

    # Create analyzer
    analyzer = IntegratedPESensitivityAnalyzer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        array_size=args.array_size,
        device=args.device
    )

    # Run analysis
    analyzer.analyze_all_pe_positions(
        class_id=args.class_id,
        sample_idx=args.sample_idx,
        duration=args.duration,
        epsilon_max=args.epsilon_max,
        tolerance=args.tolerance
    )

    # Analyze results
    analyzer.analyze_results()

    # Save
    analyzer.save_results()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
