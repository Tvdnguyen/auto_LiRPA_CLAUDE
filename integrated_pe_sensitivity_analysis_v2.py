"""
Integrated PE-to-Conv1 Sensitivity Analysis (v2)
Uses proven code from main_interactive.py

Workflow:
1. For each PE position in 8x8 array:
   - Step 1: Run fault simulation (fault_simulator.py) with fixed config:
     * IS dataflow
     * Accumulator register only
     * Transient bit-flip at critical timing
     * Conv1 layer only
   - Step 2: Get affected conv1 region from fault mask
   - Step 3: Use main_interactive.py logic to find max epsilon via binary search
2. Rank PEs by epsilon (lower = more sensitive)
"""

import sys
import os
import numpy as np
import csv
import torch
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'systolic_fault_sim'))

# Import from fault simulator
from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet, TrafficSignNetNoDropout
from operand_matrix import OperandMatrix

# Import from GTSRB verification (proven code from main_interactive.py)
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from intermediate_bound_module import IntermediateBoundedModule


class IntegratedAnalyzer:
    """Integrated PE sensitivity analyzer using proven main_interactive.py code"""

    def __init__(self, checkpoint_path, data_dir, array_size=8, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model
            data_dir: Path to GTSRB dataset
            array_size: Systolic array size (8 or 16)
            device: Device for verification
        """
        self.array_size = array_size
        self.dataflow = 'IS'  # Fixed
        self.component = 'accumulator_register'  # Fixed
        self.layer_idx = 0  # conv1 only
        self.device = torch.device(device)

        print(f"="*80)
        print(f"  INTEGRATED PE SENSITIVITY ANALYZER")
        print(f"="*80)
        print(f"Array size: {array_size}×{array_size}")
        print(f"Dataflow: {self.dataflow}")
        print(f"Component: {self.component}")
        print(f"Device: {self.device}")

        # Load verification model (using proven code from main_interactive.py)
        self._load_verification_model(checkpoint_path, data_dir)

        self.results = []

    def _load_verification_model(self, checkpoint_path, data_dir):
        """Load verification model - COPIED FROM main_interactive.py"""
        print(f"\nLoading verification model from {checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create no-dropout model for verification (from main_interactive.py pattern)
        self.model = TrafficSignNetNoDropout(num_classes=43)
        self.model.load_from_dropout_checkpoint(checkpoint_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        print(f"Model loaded successfully!")
        print(f"  Checkpoint accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")

        # Create bounded module (from main_interactive.py)
        print("\nCreating bounded module...")
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

        print(f"Dataset loaded: {len(self.test_dataset)} test samples")

        # Get conv1 node name (from main_interactive.py pattern)
        self._get_conv1_node_name()

    def _get_conv1_node_name(self):
        """Get conv1 node name - adapted from main_interactive.py"""
        # Get layer info from model
        model_temp = TrafficSignNet(num_classes=43)
        layers_info = model_temp.get_layer_info()
        conv1_layer_name = layers_info[0][0]  # First layer is conv1

        # Get node names from bounded module
        layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])

        print(f"\nDetermining conv1 node name:")
        print(f"  Conv1 layer name from model: {conv1_layer_name}")
        print(f"  Available nodes: {[name for name, _ in layers[:3]]}")

        # Find matching node (from main_interactive.py logic)
        for node_name, node_type in layers:
            if conv1_layer_name in node_name or node_name.endswith(conv1_layer_name):
                self.conv1_node_name = node_name
                print(f"  → Using node: {self.conv1_node_name}")
                return

        # Fallback: use first conv node
        self.conv1_node_name = layers[0][0]
        print(f"  → Using first conv node: {self.conv1_node_name}")

    def get_affected_conv1_region(self, pe_row, pe_col, duration=2):
        """
        Run fault simulation to get affected conv1 region

        Uses fault_simulator.py with fixed configuration:
        - IS dataflow
        - Accumulator register
        - Transient bit-flip at critical timing
        - Conv1 layer

        Returns:
            (affected_channels, affected_spatial_positions)
        """
        print(f"\n  [Step 1] Running fault simulation for PE({pe_row},{pe_col})...")

        # Create simulator
        simulator = SystolicFaultSimulator(
            self.array_size,
            self.array_size,
            self.dataflow
        )

        # Get layer config
        model = TrafficSignNet(num_classes=43)
        layer_config = simulator.get_layer_config(model, self.layer_idx)

        # Compute critical timing for IS dataflow
        op_gen = OperandMatrix(layer_config)
        operand_mats = op_gen.generate_matrices()
        dims = operand_mats['dimensions']

        T = dims['conv_window_size']  # Convolution window size
        H = self.array_size

        # IS dataflow critical timing (from fault_simulator.py logic)
        # Input load: 0 to H-1
        # Weight stream (COMPUTATION): H to H+T-1  <- CRITICAL
        # Output drain: H+T to H+T+H-2
        comp_start = H
        comp_end = H + T - 1
        critical_cycle = (comp_start + comp_end) // 2  # Middle of computation

        print(f"    IS dataflow timing:")
        print(f"      Computation phase: cycles {comp_start}-{comp_end}")
        print(f"      Critical cycle: {critical_cycle}")

        # Create fault
        fault = FaultModel(
            fault_type=FaultModel.BIT_FLIP,  # Transient bit-flip
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
        results = simulator.simulate_layer(layer_config, [fault])
        fault_mask = results['fault_mask']
        stats = results['statistics']

        print(f"    Fault coverage: {stats['fault_coverage']*100:.2f}%")
        print(f"    Affected outputs: {stats['affected_outputs']}/{stats['total_outputs']}")

        # Extract affected region
        # fault_mask shape: (spatial_pixels, num_channels)
        # where spatial_pixels = H_out * W_out
        affected_positions = np.where(fault_mask)

        if len(affected_positions[0]) == 0:
            print(f"    WARNING: No outputs affected!")
            return [], []

        # Get unique channels
        unique_channels = np.unique(affected_positions[1]).tolist()

        # Get unique spatial positions
        num_outputs, num_channels = fault_mask.shape
        W_out = int(np.sqrt(num_outputs))
        H_out = W_out

        unique_spatial = []
        for spatial_idx in np.unique(affected_positions[0]):
            h = spatial_idx // W_out
            w = spatial_idx % W_out
            unique_spatial.append((h, w))

        print(f"    Affected: {len(unique_channels)} channels, {len(unique_spatial)} spatial positions")

        return unique_channels, unique_spatial

    def find_max_epsilon_for_region(
        self,
        image,
        label,
        pred_class,
        affected_channels,
        affected_spatial,
        epsilon_max=1.0,
        tolerance=0.001
    ):
        """
        Find max epsilon for affected region using binary search

        LOGIC COPIED FROM main_interactive.py compute_perturbed_bounds()

        Args:
            image: Input image tensor
            label: True label
            pred_class: Predicted class
            affected_channels: List of affected channel indices
            affected_spatial: List of (h, w) affected spatial positions
            epsilon_max: Maximum epsilon to search
            tolerance: Binary search tolerance

        Returns:
            max_robust_epsilon
        """
        print(f"  [Step 2] Finding max epsilon for affected region...")
        print(f"    Affected channels: {len(affected_channels)}")
        print(f"    Affected spatial positions: {len(affected_spatial)}")

        def is_robust_at_epsilon(eps):
            """Check if robust at given epsilon - LOGIC FROM main_interactive.py"""
            if eps <= 0:
                return True

            # Clear previous perturbations (from main_interactive.py)
            self.lirpa_model.clear_intermediate_perturbations()

            # Forward pass (from main_interactive.py)
            image_batch = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.lirpa_model(image_batch)

            # Create custom mask for exact affected positions
            def custom_create_mask(shape):
                """Create mask for exact affected positions"""
                mask = torch.zeros(shape, dtype=torch.bool)
                # shape: (1, C, H, W)
                for h, w in affected_spatial:
                    if h < shape[2] and w < shape[3]:
                        if len(affected_channels) == 0:
                            # All channels at this position
                            mask[0, :, h, w] = True
                        else:
                            # Only specified channels
                            for c in affected_channels:
                                if c < shape[1]:
                                    mask[0, c, h, w] = True
                return mask

            # Create perturbation (pattern from main_interactive.py)
            perturbation = MaskedPerturbationLpNorm(
                eps=eps,
                norm=np.inf,
                batch_idx=0,
                channel_idx=affected_channels if len(affected_channels) > 0 else None,
                height_slice=None,
                width_slice=None
            )

            # Override mask creation
            perturbation.create_mask = custom_create_mask

            # Register perturbation (from main_interactive.py)
            self.lirpa_model.register_intermediate_perturbation(
                self.conv1_node_name,
                perturbation
            )

            try:
                # Compute bounds (from main_interactive.py)
                lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                    x=image_batch,
                    method='backward'
                )

                # Check robustness (from main_interactive.py logic)
                pred_lb = lb[0, pred_class].item()
                other_ub = ub[0].clone()
                other_ub[pred_class] = -float('inf')
                max_other_ub = other_ub.max().item()

                is_robust = pred_lb > max_other_ub

                return is_robust

            except Exception as e:
                print(f"      Error at eps={eps:.4f}: {str(e)[:50]}")
                return False

        # Binary search (from main_interactive.py pattern)
        left = 0.0
        right = epsilon_max
        best_epsilon = None

        # Check boundaries
        if not is_robust_at_epsilon(left):
            print(f"    Not robust even at epsilon=0")
            return 0.0

        if is_robust_at_epsilon(right):
            print(f"    Robust up to epsilon_max={right:.4f}")
            return right

        # Binary search
        iteration = 0
        while right - left > tolerance:
            mid = (left + right) / 2.0
            iteration += 1

            print(f"    Iteration {iteration}: eps={mid:.4f} ", end="")

            if is_robust_at_epsilon(mid):
                best_epsilon = mid
                left = mid
                print("→ robust")
            else:
                right = mid
                print("→ not robust")

        final_epsilon = best_epsilon if best_epsilon is not None else left
        print(f"    → Max epsilon: {final_epsilon:.4f}")

        return final_epsilon

    def analyze_all_pe_positions(self, test_idx=243, duration=2, epsilon_max=1.0, tolerance=0.001):
        """
        Main analysis: test all PE positions

        Args:
            test_idx: Global test sample index
            duration: Fault duration
            epsilon_max: Max epsilon to search
            tolerance: Binary search tolerance
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING ALL PE POSITIONS")
        print(f"{'='*80}")
        print(f"\nConfiguration:")
        print(f"  Test sample index: {test_idx}")
        print(f"  Fault duration: {duration} cycles")
        print(f"  Epsilon range: [0, {epsilon_max}]")
        print(f"  Binary search tolerance: {tolerance}")

        # Load test sample
        print(f"\nLoading test sample...")
        if test_idx >= len(self.test_dataset):
            raise ValueError(f"Test index {test_idx} out of range (dataset size: {len(self.test_dataset)})")

        image, label = self.test_dataset[test_idx]

        # Get clean prediction (from main_interactive.py)
        image_batch = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_batch)
            pred_class = output.argmax(dim=1).item()

        print(f"  Global index: {test_idx}")
        print(f"  True label: {label}")
        print(f"  Predicted class: {pred_class}")

        # Test each PE
        total_pes = self.array_size * self.array_size
        print(f"\n{'='*80}")
        print(f"Testing {total_pes} PE positions")
        print(f"{'='*80}")

        test_num = 0
        for pe_row in range(self.array_size):
            for pe_col in range(self.array_size):
                test_num += 1
                print(f"\n[{test_num}/{total_pes}] PE({pe_row},{pe_col})")

                try:
                    # Step 1: Get affected region from fault simulation
                    affected_channels, affected_spatial = self.get_affected_conv1_region(
                        pe_row, pe_col, duration
                    )

                    # Step 2: Find max epsilon
                    if len(affected_spatial) == 0:
                        max_eps = epsilon_max  # No impact = max robustness
                        print(f"  → No impact, epsilon = {max_eps:.4f}")
                    else:
                        max_eps = self.find_max_epsilon_for_region(
                            image, label, pred_class,
                            affected_channels, affected_spatial,
                            epsilon_max, tolerance
                        )

                    # Record result
                    self.results.append({
                        'pe_row': pe_row,
                        'pe_col': pe_col,
                        'max_epsilon': max_eps,
                        'num_affected_channels': len(affected_channels),
                        'num_affected_spatial': len(affected_spatial),
                        'test_idx': test_idx,
                        'true_label': label,
                        'pred_class': pred_class
                    })

                except Exception as e:
                    print(f"  ERROR: {e}")
                    self.results.append({
                        'pe_row': pe_row,
                        'pe_col': pe_col,
                        'max_epsilon': -1,
                        'num_affected_channels': 0,
                        'num_affected_spatial': 0,
                        'test_idx': test_idx,
                        'true_label': label,
                        'pred_class': pred_class,
                        'error': str(e)
                    })

        print(f"\n{'='*80}")
        print(f"Completed {test_num} PE tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Analyze and display results"""
        print(f"\n{'='*80}")
        print(f"RESULTS ANALYSIS")
        print(f"{'='*80}")

        valid_results = [r for r in self.results if r['max_epsilon'] >= 0]

        if len(valid_results) == 0:
            print("No valid results!")
            return

        epsilons = [r['max_epsilon'] for r in valid_results]

        print(f"\nStatistics:")
        print(f"  Valid tests: {len(valid_results)}/{len(self.results)}")
        print(f"  Epsilon range: {np.min(epsilons):.4f} - {np.max(epsilons):.4f}")
        print(f"  Mean epsilon: {np.mean(epsilons):.4f}")
        print(f"  Median epsilon: {np.median(epsilons):.4f}")
        print(f"  Std dev: {np.std(epsilons):.4f}")

        # Sort by epsilon (lower = more sensitive)
        sorted_results = sorted(valid_results, key=lambda x: x['max_epsilon'])

        print(f"\nTop 10 Most Sensitive PEs (lowest epsilon):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Epsilon':>10} | {'Affected Ch':>12} | {'Affected Sp':>12}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
        for i, r in enumerate(sorted_results[:10], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            print(f"{i:>4} | {pe_str:>8} | {r['max_epsilon']:>9.4f} | {r['num_affected_channels']:>12} | {r['num_affected_spatial']:>12}")

        print(f"\nTop 10 Least Sensitive PEs (highest epsilon):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Epsilon':>10} | {'Affected Ch':>12} | {'Affected Sp':>12}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
        for i, r in enumerate(sorted_results[-10:][::-1], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            print(f"{i:>4} | {pe_str:>8} | {r['max_epsilon']:>9.4f} | {r['num_affected_channels']:>12} | {r['num_affected_spatial']:>12}")

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'pe_sensitivity_results_{self.array_size}x{self.array_size}_{timestamp}.csv'

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['pe_row', 'pe_col', 'max_epsilon',
                         'num_affected_channels', 'num_affected_spatial',
                         'test_idx', 'true_label', 'pred_class']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
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
    parser.add_argument('--test_idx', type=int, default=243,
                       help='Global test index (default: 243)')
    parser.add_argument('--duration', type=int, default=2,
                       help='Fault duration (cycles, default: 2 for lower memory)')
    parser.add_argument('--epsilon_max', type=float, default=1.0,
                       help='Maximum epsilon to search')
    parser.add_argument('--tolerance', type=float, default=0.001,
                       help='Binary search tolerance')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("="*80)
    print(" "*10 + "INTEGRATED PE SENSITIVITY ANALYSIS (v2)")
    print(" "*15 + "Using proven main_interactive.py code")
    print("="*80)

    # Create analyzer
    analyzer = IntegratedAnalyzer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        array_size=args.array_size,
        device=args.device
    )

    # Run analysis
    analyzer.analyze_all_pe_positions(
        test_idx=args.test_idx,
        duration=args.duration,
        epsilon_max=args.epsilon_max,
        tolerance=args.tolerance
    )

    # Analyze results
    analyzer.analyze_results()

    # Save results
    analyzer.save_results()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
