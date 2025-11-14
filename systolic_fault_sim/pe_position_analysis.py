"""
PE Position Fault Coverage Analysis

Find which PE position has HIGHEST fault coverage under fixed conditions:
- Dataflow: IS (Input Stationary) only
- Array sizes: 8x8 and 16x16
- Fault location: Accumulator Register only
- Fault type: Bit-flip (transient)
- Timing: Critical computation phase (single optimal timing)
- Duration: Fixed duration across all tests
- Layer: conv1 only

Goal: Test each PE position, find which has max fault coverage
"""

import sys
import os
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet
from operand_matrix import OperandMatrix


class PEPositionAnalyzer:
    """Analyze fault coverage for each PE position"""

    def __init__(self, array_size):
        """
        Args:
            array_size: Array size (8 or 16)
        """
        self.array_size = array_size
        self.dataflow = 'IS'  # Fixed
        self.component = 'accumulator_register'  # Fixed
        self.layer_idx = 0  # conv1 only
        self.results = []

        print(f"Initialized PE Position Analyzer")
        print(f"  Array size: {array_size}x{array_size}")
        print(f"  Dataflow: {self.dataflow}")
        print(f"  Component: {self.component}")
        print(f"  Layer: conv1")

    def _compute_critical_timing(self):
        """
        Compute the most critical timing for IS dataflow on conv1

        IS dataflow phases:
        1. Input load: cycles 0 to H-1 (load inputs)
        2. Weight stream: cycles H to H+T-1 (computation happens here)
        3. Output drain: cycles H+T to H+T+H-2

        Most critical: Middle of weight stream phase
        """
        # Get layer config
        model = TrafficSignNet(num_classes=43)
        simulator = SystolicFaultSimulator(self.array_size, self.array_size, self.dataflow)
        layer_config = simulator.get_layer_config(model, self.layer_idx)

        # Get dimensions
        op_gen = OperandMatrix(layer_config)
        operand_mats = op_gen.generate_matrices()
        dims = operand_mats['dimensions']

        T = dims['conv_window_size']  # Convolution window size
        H = self.array_size

        # Critical phase: weight stream (computation)
        comp_start = H
        comp_end = H + T - 1

        # Choose middle of computation phase as most critical
        critical_cycle = (comp_start + comp_end) // 2

        print(f"\nTiming Analysis (IS dataflow):")
        print(f"  Input load: cycles 0-{H-1}")
        print(f"  Weight stream (COMPUTATION): cycles {comp_start}-{comp_end}")
        print(f"  Output drain: cycles {comp_end+1}-{comp_end+H-1}")
        print(f"  → Critical cycle selected: {critical_cycle}")

        return critical_cycle, comp_start, comp_end

    def test_pe_position(self, pe_row, pe_col, start_cycle, duration):
        """
        Test single PE position with fixed fault configuration

        Args:
            pe_row: PE row index
            pe_col: PE column index
            start_cycle: Fault start cycle
            duration: Fault duration

        Returns:
            Dict with test results
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

        # Create fault
        fault = FaultModel(
            fault_type=FaultModel.BIT_FLIP,  # Transient bit-flip
            fault_location={
                'pe_row': pe_row,
                'pe_col': pe_col,
                'component': self.component
            },
            fault_timing={
                'start_cycle': start_cycle,
                'duration': duration
            }
        )

        # Run simulation
        try:
            results = simulator.simulate_layer(layer_config, [fault])
            stats = results['statistics']
            coverage = stats['fault_coverage'] * 100

            return {
                'pe_row': pe_row,
                'pe_col': pe_col,
                'coverage': coverage,
                'total_outputs': stats['total_outputs'],
                'affected_outputs': stats['affected_outputs'],
                'start_cycle': start_cycle,
                'duration': duration
            }

        except Exception as e:
            print(f"  ERROR at PE({pe_row},{pe_col}): {str(e)[:80]}")
            return {
                'pe_row': pe_row,
                'pe_col': pe_col,
                'coverage': -1,
                'total_outputs': 0,
                'affected_outputs': 0,
                'start_cycle': start_cycle,
                'duration': duration,
                'error': str(e)
            }

    def run_all_tests(self, duration=10):
        """
        Test all PE positions in the array

        Args:
            duration: Fixed fault duration for all tests (default: 10 cycles)
        """
        print(f"\n{'='*80}")
        print(f"Testing All PE Positions ({self.array_size}x{self.array_size})")
        print(f"{'='*80}")

        # Compute critical timing
        critical_cycle, comp_start, comp_end = self._compute_critical_timing()

        print(f"\nFault Configuration:")
        print(f"  Component: {self.component}")
        print(f"  Type: Bit-flip (transient)")
        print(f"  Start cycle: {critical_cycle}")
        print(f"  Duration: {duration} cycles")
        print(f"  End cycle: {critical_cycle + duration - 1}")

        total_tests = self.array_size * self.array_size
        print(f"\nTotal PE positions to test: {total_tests}")
        print(f"")

        # Test each PE
        test_num = 0
        for pe_row in range(self.array_size):
            for pe_col in range(self.array_size):
                test_num += 1
                print(f"[{test_num}/{total_tests}] Testing PE({pe_row},{pe_col})... ", end="")

                result = self.test_pe_position(
                    pe_row=pe_row,
                    pe_col=pe_col,
                    start_cycle=critical_cycle,
                    duration=duration
                )

                if result['coverage'] >= 0:
                    print(f"Coverage: {result['coverage']:.2f}%")
                else:
                    print(f"ERROR")

                self.results.append(result)

        print(f"\n{'='*80}")
        print(f"Completed {test_num} tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Analyze and display results"""
        print(f"\n{'='*80}")
        print(f"ANALYSIS: PE Position Impact")
        print(f"{'='*80}")

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            print("No valid results!")
            return

        coverages = [r['coverage'] for r in valid_results]

        print(f"\nStatistics:")
        print(f"  Valid tests: {len(valid_results)}/{len(self.results)}")
        print(f"  Coverage range: {np.min(coverages):.2f}% - {np.max(coverages):.2f}%")
        print(f"  Mean coverage: {np.mean(coverages):.2f}%")
        print(f"  Median coverage: {np.median(coverages):.2f}%")
        print(f"  Std dev: {np.std(coverages):.2f}%")

        # Sort by coverage
        sorted_results = sorted(valid_results, key=lambda x: x['coverage'], reverse=True)

        # Top 10 highest coverage (most critical PEs)
        print(f"\nTop 10 Most Critical PE Positions (Highest Coverage):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Coverage':>10} | {'Affected':>15}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*15}")
        for i, r in enumerate(sorted_results[:10], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            affected_str = f"{r['affected_outputs']}/{r['total_outputs']}"
            print(f"{i:>4} | {pe_str:>8} | {r['coverage']:>9.2f}% | {affected_str:>15}")

        # Bottom 10 (least critical)
        print(f"\nTop 10 Least Critical PE Positions (Lowest Coverage):")
        print(f"{'Rank':>4} | {'PE':>8} | {'Coverage':>10} | {'Affected':>15}")
        print(f"{'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*15}")
        for i, r in enumerate(sorted_results[-10:][::-1], 1):
            pe_str = f"({r['pe_row']},{r['pe_col']})"
            affected_str = f"{r['affected_outputs']}/{r['total_outputs']}"
            print(f"{i:>4} | {pe_str:>8} | {r['coverage']:>9.2f}% | {affected_str:>15}")

        # Position patterns
        print(f"\nPosition Pattern Analysis:")

        # Corner positions
        corners = [
            (0, 0),
            (0, self.array_size-1),
            (self.array_size-1, 0),
            (self.array_size-1, self.array_size-1)
        ]
        corner_coverages = [r['coverage'] for r in valid_results
                           if (r['pe_row'], r['pe_col']) in corners]
        if corner_coverages:
            print(f"  Corners (n={len(corner_coverages)}): {np.mean(corner_coverages):.2f}% avg")

        # Edge positions (excluding corners)
        edge_positions = []
        for r in range(self.array_size):
            for c in range(self.array_size):
                if (r == 0 or r == self.array_size-1 or
                    c == 0 or c == self.array_size-1) and \
                   (r, c) not in corners:
                    edge_positions.append((r, c))

        edge_coverages = [r['coverage'] for r in valid_results
                         if (r['pe_row'], r['pe_col']) in edge_positions]
        if edge_coverages:
            print(f"  Edges (n={len(edge_coverages)}): {np.mean(edge_coverages):.2f}% avg")

        # Center positions
        center_start = self.array_size // 4
        center_end = 3 * self.array_size // 4
        center_positions = []
        for r in range(center_start, center_end):
            for c in range(center_start, center_end):
                center_positions.append((r, c))

        center_coverages = [r['coverage'] for r in valid_results
                           if (r['pe_row'], r['pe_col']) in center_positions]
        if center_coverages:
            print(f"  Center (n={len(center_coverages)}): {np.mean(center_coverages):.2f}% avg")

    def visualize_heatmap(self, save_path=None):
        """Create heatmap of coverage by PE position"""
        print(f"\nGenerating heatmap...")

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            print("No valid results for heatmap!")
            return

        # Create coverage matrix
        coverage_map = np.zeros((self.array_size, self.array_size))
        for r in valid_results:
            coverage_map[r['pe_row'], r['pe_col']] = r['coverage']

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot heatmap
        im = ax.imshow(coverage_map, cmap='YlOrRd', aspect='auto',
                      vmin=0, vmax=np.max(coverage_map))

        # Labels
        ax.set_xlabel('PE Column', fontweight='bold', fontsize=12)
        ax.set_ylabel('PE Row', fontweight='bold', fontsize=12)
        ax.set_title(f'Fault Coverage by PE Position ({self.array_size}×{self.array_size} Array)\n'
                    f'Dataflow: IS | Component: Accumulator Register | Layer: conv1',
                    fontweight='bold', fontsize=14)

        # Ticks
        ax.set_xticks(range(self.array_size))
        ax.set_yticks(range(self.array_size))

        # Add grid
        ax.set_xticks(np.arange(-0.5, self.array_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.array_size, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # Add text annotations (show values)
        for i in range(self.array_size):
            for j in range(self.array_size):
                if coverage_map[i, j] > 0:
                    text = ax.text(j, i, f'{coverage_map[i, j]:.1f}',
                                 ha="center", va="center",
                                 color="black" if coverage_map[i, j] < 50 else "white",
                                 fontsize=8, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Fault Coverage (%)', rotation=270, labelpad=20, fontweight='bold')

        plt.tight_layout()

        if save_path is None:
            save_path = f'pe_position_heatmap_{self.array_size}x{self.array_size}.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        plt.close()

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'pe_position_results_{self.array_size}x{self.array_size}_{timestamp}.csv'

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['pe_row', 'pe_col', 'coverage', 'total_outputs',
                         'affected_outputs', 'start_cycle', 'duration']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
                    writer.writerow(result)

        print(f"Results saved!")


def compare_array_sizes(sizes=[8, 16], duration=10):
    """
    Compare results across different array sizes

    Args:
        sizes: List of array sizes to test
        duration: Fault duration
    """
    print("="*80)
    print(" "*15 + "PE POSITION ANALYSIS - MULTIPLE SIZES")
    print("="*80)

    all_results = {}

    for size in sizes:
        print(f"\n{'='*80}")
        print(f"Testing Array Size: {size}×{size}")
        print(f"{'='*80}")

        analyzer = PEPositionAnalyzer(array_size=size)
        analyzer.run_all_tests(duration=duration)
        analyzer.analyze_results()
        analyzer.visualize_heatmap()
        analyzer.save_results()

        all_results[size] = analyzer.results

    # Comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    for size in sizes:
        valid_results = [r for r in all_results[size] if r['coverage'] >= 0]
        if valid_results:
            coverages = [r['coverage'] for r in valid_results]

            # Find max coverage PE
            max_result = max(valid_results, key=lambda x: x['coverage'])

            print(f"\n{size}×{size} Array:")
            print(f"  Mean coverage: {np.mean(coverages):.2f}%")
            print(f"  Max coverage: {np.max(coverages):.2f}%")
            print(f"  Most critical PE: ({max_result['pe_row']},{max_result['pe_col']}) "
                  f"with {max_result['coverage']:.2f}% coverage")

    print(f"\n{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='PE Position Fault Coverage Analysis'
    )
    parser.add_argument('--size', type=int, default=None,
                       help='Single array size to test (8 or 16)')
    parser.add_argument('--duration', type=int, default=10,
                       help='Fault duration in cycles (default: 10)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare both 8×8 and 16×16 arrays')

    args = parser.parse_args()

    if args.compare:
        # Compare both sizes
        compare_array_sizes(sizes=[8, 16], duration=args.duration)
    elif args.size:
        # Test single size
        if args.size not in [8, 16]:
            print("Error: Size must be 8 or 16")
            return

        print("="*80)
        print(" "*20 + "PE POSITION ANALYSIS")
        print("="*80)

        analyzer = PEPositionAnalyzer(array_size=args.size)
        analyzer.run_all_tests(duration=args.duration)
        analyzer.analyze_results()
        analyzer.visualize_heatmap()
        analyzer.save_results()

        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)
    else:
        # Default: compare both
        compare_array_sizes(sizes=[8, 16], duration=args.duration)


if __name__ == '__main__':
    main()
