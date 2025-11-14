"""
Integrated PE-to-Conv1 Sensitivity Analysis (Low Memory Version)
Optimized for systems with limited RAM (8GB)

Key optimizations:
1. Reduced fault duration (1 cycle instead of 2)
2. Aggressive tensor cleanup and garbage collection
3. Reduced binary search precision (tolerance=0.01 instead of 0.001)
4. Process PEs in batches with memory cleanup between batches
5. Save intermediate results to avoid data loss
"""

import sys
import os
import numpy as np
import csv
import torch
import gc
from datetime import datetime

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gtsrb_project'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'systolic_fault_sim'))

# Import from fault simulator
from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet, TrafficSignNetNoDropout
from operand_matrix import OperandMatrix

# Import from GTSRB verification
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from intermediate_bound_module import IntermediateBoundedModule


class LowMemoryIntegratedAnalyzer:
    """Memory-optimized PE sensitivity analyzer for systems with limited RAM"""

    def __init__(self, checkpoint_path, data_dir, array_size=8, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model
            data_dir: Path to GTSRB dataset
            array_size: Systolic array size (8 or 16)
            device: Device for verification (recommend 'cpu' for low memory)
        """
        self.array_size = array_size
        self.dataflow = 'IS'  # Fixed
        self.component = 'accumulator_register'  # Fixed
        self.layer_idx = 0  # conv1 only
        self.device = torch.device(device)

        print(f"="*80)
        print(f"  LOW MEMORY PE SENSITIVITY ANALYZER")
        print(f"="*80)
        print(f"Array size: {array_size}×{array_size}")
        print(f"Dataflow: {self.dataflow}")
        print(f"Component: {self.component}")
        print(f"Device: {self.device}")
        print(f"\nMemory optimizations enabled:")
        print(f"  - Reduced fault duration (1 cycle)")
        print(f"  - Aggressive garbage collection")
        print(f"  - Reduced search precision (tolerance=0.01)")
        print(f"  - Batch processing with cleanup")

        # Load verification model
        self._load_verification_model(checkpoint_path, data_dir)

        self.results = []

    def _load_verification_model(self, checkpoint_path, data_dir):
        """Load verification model - COPIED FROM main_interactive.py"""
        print(f"\nLoading verification model from {checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create no-dropout model
        self.model = TrafficSignNetNoDropout(num_classes=43)
        self.model.load_from_dropout_checkpoint(checkpoint_path)
        self.model.eval()
        self.model = self.model.to(self.device)

        print(f"Model loaded successfully!")
        print(f"  Checkpoint accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")

        # Create bounded module
        print("\nCreating bounded module...")
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        self.lirpa_model = IntermediateBoundedModule(
            self.model,
            dummy_input,
            device=self.device
        )

        # Cleanup dummy input
        del dummy_input, checkpoint
        gc.collect()

        # Load dataset
        self.test_dataset = GTSRBDataset(
            root_dir=data_dir,
            train=False,
            transform=get_gtsrb_transforms(train=False, img_size=32)
        )

        print(f"Dataset loaded: {len(self.test_dataset)} test samples")

        # Get conv1 node name
        self._get_conv1_node_name()

    def _get_conv1_node_name(self):
        """Get conv1 node name - adapted from main_interactive.py"""
        model_temp = TrafficSignNet(num_classes=43)
        layers_info = model_temp.get_layer_info()
        conv1_layer_name = layers_info[0][0]

        layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])

        print(f"\nDetermining conv1 node name:")
        print(f"  Conv1 layer name from model: {conv1_layer_name}")

        for node_name, node_type in layers:
            if conv1_layer_name in node_name or node_name.endswith(conv1_layer_name):
                self.conv1_node_name = node_name
                print(f"  → Using node: {self.conv1_node_name}")
                return

        self.conv1_node_name = layers[0][0]
        print(f"  → Using first conv node: {self.conv1_node_name}")

    def get_affected_conv1_region(self, pe_row, pe_col, duration=1):
        """
        Run fault simulation - OPTIMIZED with duration=1 for lower memory

        Returns:
            (affected_channels, affected_spatial_positions)
        """
        # Create simulator
        simulator = SystolicFaultSimulator(
            self.array_size,
            self.array_size,
            self.dataflow
        )

        # Get layer config
        model = TrafficSignNet(num_classes=43)
        layer_config = simulator.get_layer_config(model, self.layer_idx)

        # Compute critical timing
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
                'duration': duration  # Only 1 cycle for memory efficiency
            }
        )

        # Run simulation
        results = simulator.simulate_layer(layer_config, [fault])
        fault_mask = results['fault_mask']
        stats = results['statistics']

        # Extract affected region
        affected_positions = np.where(fault_mask)

        if len(affected_positions[0]) == 0:
            return [], []

        unique_channels = np.unique(affected_positions[1]).tolist()

        num_outputs, num_channels = fault_mask.shape
        W_out = int(np.sqrt(num_outputs))
        H_out = W_out

        unique_spatial = []
        for spatial_idx in np.unique(affected_positions[0]):
            h = spatial_idx // W_out
            w = spatial_idx % W_out
            unique_spatial.append((h, w))

        # Cleanup
        del simulator, results, fault_mask
        gc.collect()

        return unique_channels, unique_spatial

    def find_max_epsilon_for_region(
        self,
        image,
        label,
        pred_class,
        affected_channels,
        affected_spatial,
        epsilon_max=1.0,
        tolerance=0.01  # Reduced precision for memory efficiency
    ):
        """
        Find max epsilon - OPTIMIZED with aggressive cleanup
        Following main_interactive.py pattern exactly
        """

        def is_robust_at_epsilon(eps):
            """Check robustness with memory cleanup"""
            if eps <= 0:
                return True

            # Clear previous perturbations
            self.lirpa_model.clear_intermediate_perturbations()

            # Forward pass
            image_batch = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                _ = self.lirpa_model(image_batch)

            # Convert affected_spatial to bounding box slices (EFFICIENT!)
            # This matches main_interactive.py pattern
            height_slice = None
            width_slice = None
            channel_idx = None

            if len(affected_spatial) > 0:
                # Extract all h, w coordinates
                h_coords = [h for h, w in affected_spatial]
                w_coords = [w for h, w in affected_spatial]

                # Compute bounding box
                min_h = min(h_coords)
                max_h = max(h_coords)
                min_w = min(w_coords)
                max_w = max(w_coords)

                # Create slices (inclusive end, so +1)
                height_slice = (min_h, max_h + 1)
                width_slice = (min_w, max_w + 1)

            if len(affected_channels) > 0:
                channel_idx = affected_channels

            # Create perturbation - EXACTLY like main_interactive.py
            # No custom mask override - use built-in efficient slicing
            perturbation = MaskedPerturbationLpNorm(
                eps=eps,
                norm=np.inf,
                batch_idx=0,
                channel_idx=channel_idx,
                height_slice=height_slice,
                width_slice=width_slice
            )

            # Register perturbation
            self.lirpa_model.register_intermediate_perturbation(
                self.conv1_node_name,
                perturbation
            )

            try:
                # Compute bounds
                lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                    x=image_batch,
                    method='backward'
                )

                # Check robustness
                pred_lb = lb[0, pred_class].item()
                other_ub = ub[0].clone()
                other_ub[pred_class] = -float('inf')
                max_other_ub = other_ub.max().item()

                is_robust = pred_lb > max_other_ub

                # AGGRESSIVE CLEANUP
                del lb, ub, other_ub, image_batch, perturbation
                gc.collect()

                return is_robust

            except Exception as e:
                # Cleanup on error
                gc.collect()
                return False

        # Binary search with reduced precision
        left = 0.0
        right = epsilon_max
        best_epsilon = None

        if not is_robust_at_epsilon(left):
            return 0.0

        if is_robust_at_epsilon(right):
            return right

        # Binary search (fewer iterations due to larger tolerance)
        iteration = 0
        while right - left > tolerance:
            mid = (left + right) / 2.0
            iteration += 1

            if is_robust_at_epsilon(mid):
                best_epsilon = mid
                left = mid
            else:
                right = mid

            # Periodic cleanup
            if iteration % 3 == 0:
                gc.collect()

        final_epsilon = best_epsilon if best_epsilon is not None else left

        return final_epsilon

    def analyze_all_pe_positions(
        self,
        test_idx=0,
        class_id=0,
        duration=1,
        epsilon_max=1.0,
        tolerance=0.01,
        batch_size=8,
        save_every=8
    ):
        """
        Analyze all PEs with batch processing and intermediate saves

        Args:
            test_idx: Index within the class (default: 0 = first sample of the class)
            class_id: Class to test (default: 0)
            duration: Fault duration (1 cycle for low memory)
            epsilon_max: Max epsilon
            tolerance: Binary search tolerance (0.01 for low memory)
            batch_size: PEs to process before cleanup
            save_every: Save results every N PEs
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING ALL PE POSITIONS (Low Memory Mode)")
        print(f"{'='*80}")
        print(f"\nConfiguration:")
        print(f"  Class ID: {class_id}")
        print(f"  Sample index within class: {test_idx}")
        print(f"  Fault duration: {duration} cycle")
        print(f"  Epsilon range: [0, {epsilon_max}]")
        print(f"  Binary search tolerance: {tolerance}")
        print(f"  Batch size: {batch_size} PEs")
        print(f"  Save every: {save_every} PEs")

        # Find samples of target class in test dataset
        print(f"\nSearching for samples of class {class_id} in test set...")
        class_samples = []
        for idx in range(len(self.test_dataset)):
            _, label = self.test_dataset[idx]
            if label == class_id:
                class_samples.append(idx)

        print(f"Found {len(class_samples)} samples of class {class_id} in test set")

        if len(class_samples) == 0:
            raise ValueError(f"No samples found for class {class_id}")

        if test_idx >= len(class_samples):
            raise ValueError(f"Index {test_idx} out of range for class {class_id} (found {len(class_samples)} samples)")

        # Get the actual dataset index
        global_idx = class_samples[test_idx]
        print(f"Using sample at dataset index {global_idx} (index {test_idx} within class {class_id})")

        # Load test sample
        image, label = self.test_dataset[global_idx]

        # Get clean prediction
        image_batch = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_batch)
            pred_class = output.argmax(dim=1).item()

        del image_batch, output
        gc.collect()

        print(f"  Global index: {global_idx}")
        print(f"  True label: {label}")
        print(f"  Predicted class: {pred_class}")

        # Verify prediction is correct
        if pred_class != label:
            print(f"\n  WARNING: Model misclassified this sample!")
            print(f"  True label: {label}, Predicted: {pred_class}")
            print(f"  Continuing anyway...")

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
                    # Step 1: Get affected region
                    affected_channels, affected_spatial = self.get_affected_conv1_region(
                        pe_row, pe_col, duration
                    )

                    print(f"  Affected: {len(affected_channels)} channels, {len(affected_spatial)} positions")

                    # Step 2: Find max epsilon
                    if len(affected_spatial) == 0:
                        max_eps = epsilon_max
                        print(f"  → No impact, epsilon = {max_eps:.4f}")
                    else:
                        max_eps = self.find_max_epsilon_for_region(
                            image, label, pred_class,
                            affected_channels, affected_spatial,
                            epsilon_max, tolerance
                        )
                        print(f"  → Max epsilon: {max_eps:.4f}")

                    # Record result
                    self.results.append({
                        'pe_row': pe_row,
                        'pe_col': pe_col,
                        'max_epsilon': max_eps,
                        'num_affected_channels': len(affected_channels),
                        'num_affected_spatial': len(affected_spatial),
                        'global_idx': global_idx,
                        'class_idx': test_idx,
                        'class_id': class_id,
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
                        'global_idx': global_idx,
                        'class_idx': test_idx,
                        'class_id': class_id,
                        'true_label': label,
                        'pred_class': pred_class,
                        'error': str(e)
                    })

                # Batch cleanup
                if test_num % batch_size == 0:
                    print(f"\n  [Cleanup] Processed {test_num} PEs, running garbage collection...")
                    gc.collect()

                # Intermediate save
                if test_num % save_every == 0:
                    self._save_intermediate_results(test_num)

        print(f"\n{'='*80}")
        print(f"Completed {test_num} PE tests")
        print(f"{'='*80}")

    def _save_intermediate_results(self, pe_count):
        """Save intermediate results to avoid data loss"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'pe_sensitivity_intermediate_{pe_count}PEs_{timestamp}.csv'

        with open(filename, 'w', newline='') as f:
            fieldnames = ['pe_row', 'pe_col', 'max_epsilon',
                         'num_affected_channels', 'num_affected_spatial',
                         'global_idx', 'class_idx', 'class_id',
                         'true_label', 'pred_class']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
                    writer.writerow(result)

        print(f"    → Intermediate results saved to: {filename}")

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

        # Sort by epsilon
        sorted_results = sorted(valid_results, key=lambda x: x['max_epsilon'])

        print(f"\nTop 10 Most Sensitive PEs (lowest epsilon):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Epsilon':>10} | {'Affected Ch':>12} | {'Affected Sp':>12}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*12}-+-{'-'*12}")
        for i, r in enumerate(sorted_results[:10], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            print(f"{i:>4} | {pe_str:>8} | {r['max_epsilon']:>9.4f} | {r['num_affected_channels']:>12} | {r['num_affected_spatial']:>12}")

    def save_results(self, filename=None):
        """Save final results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'pe_sensitivity_final_{self.array_size}x{self.array_size}_{timestamp}.csv'

        print(f"\nSaving final results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['pe_row', 'pe_col', 'max_epsilon',
                         'num_affected_channels', 'num_affected_spatial',
                         'global_idx', 'class_idx', 'class_id',
                         'true_label', 'pred_class']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
                    writer.writerow(result)

        print(f"Results saved!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Low Memory PE Sensitivity Analysis'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--array_size', type=int, default=8,
                       help='Systolic array size (default: 8)')
    parser.add_argument('--class_id', type=int, default=0,
                       help='Class to test (default: 0)')
    parser.add_argument('--test_idx', type=int, default=0,
                       help='Index within class (default: 0 = first sample)')
    parser.add_argument('--duration', type=int, default=1,
                       help='Fault duration (default: 1 cycle for low memory)')
    parser.add_argument('--epsilon_max', type=float, default=1.0,
                       help='Maximum epsilon')
    parser.add_argument('--tolerance', type=float, default=0.01,
                       help='Binary search tolerance (default: 0.01 for low memory)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='PEs per batch before cleanup (default: 8)')
    parser.add_argument('--save_every', type=int, default=8,
                       help='Save intermediate results every N PEs (default: 8)')

    args = parser.parse_args()

    print("="*80)
    print(" "*10 + "LOW MEMORY PE SENSITIVITY ANALYSIS")
    print(" "*15 + "Optimized for 8GB RAM systems")
    print("="*80)

    # Create analyzer
    analyzer = LowMemoryIntegratedAnalyzer(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        array_size=args.array_size,
        device='cpu'  # Force CPU for low memory
    )

    # Run analysis
    analyzer.analyze_all_pe_positions(
        test_idx=args.test_idx,
        class_id=args.class_id,
        duration=args.duration,
        epsilon_max=args.epsilon_max,
        tolerance=args.tolerance,
        batch_size=args.batch_size,
        save_every=args.save_every
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
