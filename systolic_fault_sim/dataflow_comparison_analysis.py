"""
Dataflow Comparison Analysis

Comprehensive comparison of fault coverage across OS, WS, and IS dataflows.
Tests multiple scenarios to determine which dataflow is most resilient (lowest coverage).

Test dimensions:
1. Number of faulty PEs: 1, 2, 4, 8, 16, 32
2. Fault locations: corner, center, edge, diagonal, random
3. Layers: conv1 (large), conv3 (medium), conv5 (small)
4. Fault duration: permanent, long transient, short transient
5. PE components: entire_PE, MAC_unit, accumulator

Total tests: ~300+ configurations to ensure statistical significance
"""

import sys
import os
import numpy as np
import csv
from datetime import datetime
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet


class DataflowComparisonAnalyzer:
    """Compare fault resilience across different dataflows"""

    def __init__(self):
        self.results = []
        self.model = TrafficSignNet(num_classes=43)

    def _get_layer_info(self, layer_idx):
        """Get layer dimensions"""
        layers_info = self.model.get_layer_info()
        name, ltype, _, shape = layers_info[layer_idx]
        return name, ltype, shape

    def _generate_pe_locations(self, pattern, num_pes, array_size=(8, 8)):
        """
        Generate PE locations based on pattern

        Args:
            pattern: 'corner', 'center', 'edge', 'diagonal', 'row', 'column', 'random', 'scattered'
            num_pes: Number of PEs to select
            array_size: (h, w) tuple

        Returns:
            List of (row, col) tuples
        """
        h, w = array_size
        locations = []

        if pattern == 'corner':
            # Top-left corner
            for i in range(num_pes):
                r = i // int(np.sqrt(num_pes))
                c = i % int(np.sqrt(num_pes))
                if r < h and c < w:
                    locations.append((r, c))

        elif pattern == 'center':
            # Center region
            center_r, center_c = h // 2, w // 2
            radius = int(np.ceil(np.sqrt(num_pes) / 2))
            for r in range(max(0, center_r - radius), min(h, center_r + radius + 1)):
                for c in range(max(0, center_c - radius), min(w, center_c + radius + 1)):
                    if len(locations) < num_pes:
                        locations.append((r, c))

        elif pattern == 'edge':
            # Top edge
            for i in range(min(num_pes, w)):
                locations.append((0, i))
            # Fill remaining with left edge
            for i in range(1, min(num_pes - w + 1, h)):
                if len(locations) < num_pes:
                    locations.append((i, 0))

        elif pattern == 'diagonal':
            # Main diagonal
            for i in range(min(num_pes, min(h, w))):
                locations.append((i, i))
            # Anti-diagonal if needed
            for i in range(min(h, w), num_pes):
                r = i - min(h, w)
                c = w - 1 - r
                if 0 <= r < h and 0 <= c < w:
                    locations.append((r, c))

        elif pattern == 'row':
            # First complete row, then second row, etc.
            for i in range(num_pes):
                r = i // w
                c = i % w
                if r < h:
                    locations.append((r, c))

        elif pattern == 'column':
            # First column top to bottom, then second column, etc.
            for i in range(num_pes):
                c = i // h
                r = i % h
                if c < w:
                    locations.append((r, c))

        elif pattern == 'scattered':
            # Evenly distributed across array
            step = max(1, int(np.sqrt((h * w) / num_pes)))
            for r in range(0, h, step):
                for c in range(0, w, step):
                    if len(locations) < num_pes:
                        locations.append((r, c))

        elif pattern == 'random':
            # Random locations
            all_positions = [(r, c) for r in range(h) for c in range(w)]
            random.shuffle(all_positions)
            locations = all_positions[:num_pes]

        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        return locations[:num_pes]

    def test_single_configuration(self, test_id, layer_idx, num_pes,
                                  pattern, timing_type, component):
        """
        Test one configuration across all three dataflows

        Args:
            test_id: Unique test identifier
            layer_idx: Which layer to test (0=conv1, 2=conv3, 4=conv5)
            num_pes: Number of faulty PEs
            pattern: PE location pattern
            timing_type: 'permanent', 'long_transient', 'short_transient'
            component: PE component to fault

        Returns:
            Dict with results for OS, WS, IS
        """
        layer_name, _, _ = self._get_layer_info(layer_idx)

        # Generate PE locations (same for all dataflows for fair comparison)
        pe_locations = self._generate_pe_locations(pattern, num_pes)

        # Determine fault timing based on type
        if timing_type == 'permanent':
            timing = {'start_cycle': 0, 'duration': float('inf')}
        elif timing_type == 'long_transient':
            # Long transient: 50 cycles starting from cycle 10
            timing = {'start_cycle': 10, 'duration': 50}
        elif timing_type == 'short_transient':
            # Short transient: 5 cycles starting from cycle 10
            timing = {'start_cycle': 10, 'duration': 5}
        else:
            timing = {'start_cycle': 0, 'duration': float('inf')}

        print(f"\n{'='*80}")
        print(f"Test #{test_id}")
        print(f"{'='*80}")
        print(f"  Layer: {layer_name} (idx={layer_idx})")
        print(f"  Num PEs: {num_pes}")
        print(f"  Pattern: {pattern}")
        print(f"  Timing: {timing_type}")
        print(f"  Component: {component}")
        print(f"  PE locations: {pe_locations[:5]}..." if len(pe_locations) > 5 else f"  PE locations: {pe_locations}")

        results_per_dataflow = {}

        # Test each dataflow
        for dataflow in ['OS', 'WS', 'IS']:
            print(f"\n  Testing {dataflow}...")

            try:
                # Create simulator
                simulator = SystolicFaultSimulator(8, 8, dataflow)

                # Get layer config
                layer_config = simulator.get_layer_config(self.model, layer_idx)

                # Create faults
                faults = []
                for r, c in pe_locations:
                    fault = FaultModel(
                        fault_type=FaultModel.PERMANENT,
                        fault_location={
                            'pe_row': r,
                            'pe_col': c,
                            'component': component
                        },
                        fault_timing=timing
                    )
                    faults.append(fault)

                # Run simulation (minimal output)
                results = simulator.simulate_layer(layer_config, faults)
                stats = results['statistics']

                coverage = stats['fault_coverage'] * 100

                print(f"    Coverage: {coverage:.2f}%")

                results_per_dataflow[dataflow] = {
                    'coverage': coverage,
                    'total_outputs': stats['total_outputs'],
                    'affected_outputs': stats['affected_outputs']
                }

            except Exception as e:
                print(f"    ERROR: {e}")
                results_per_dataflow[dataflow] = {
                    'coverage': -1,
                    'total_outputs': 0,
                    'affected_outputs': 0,
                    'error': str(e)
                }

        # Determine winner (lowest coverage = most resilient)
        valid_results = {k: v for k, v in results_per_dataflow.items()
                        if v['coverage'] >= 0}

        if valid_results:
            winner = min(valid_results.keys(), key=lambda k: valid_results[k]['coverage'])
            winner_coverage = valid_results[winner]['coverage']

            print(f"\n  → Winner: {winner} ({winner_coverage:.2f}%)")
        else:
            winner = 'NONE'
            winner_coverage = -1

        # Record result
        result = {
            'test_id': test_id,
            'layer': layer_name,
            'layer_idx': layer_idx,
            'num_pes': num_pes,
            'pattern': pattern,
            'timing_type': timing_type,
            'component': component,
            'os_coverage': results_per_dataflow.get('OS', {}).get('coverage', -1),
            'ws_coverage': results_per_dataflow.get('WS', {}).get('coverage', -1),
            'is_coverage': results_per_dataflow.get('IS', {}).get('coverage', -1),
            'winner': winner,
            'winner_coverage': winner_coverage
        }

        self.results.append(result)
        return result

    def run_comprehensive_tests(self):
        """Run comprehensive test suite"""
        print("\n" + "="*80)
        print(" "*20 + "DATAFLOW COMPARISON ANALYSIS")
        print(" "*15 + "Testing OS vs WS vs IS Resilience")
        print("="*80)

        test_id = 1

        # Test parameters
        layers = [
            (0, 'conv1'),  # 32x32 spatial, 32 channels
            (2, 'conv3'),  # 16x16 spatial, 64 channels
            (4, 'conv5'),  # 8x8 spatial, 128 channels
        ]

        num_pes_list = [1, 2, 4, 8, 16, 32]
        patterns = ['corner', 'center', 'edge', 'diagonal', 'row', 'column', 'scattered', 'random']
        timing_types = ['permanent', 'long_transient', 'short_transient']
        components = ['entire_PE', 'MAC_unit', 'accumulator_register']

        # Strategy: Test all combinations systematically
        print(f"\nPlanning tests...")
        print(f"  Layers: {len(layers)}")
        print(f"  PE counts: {len(num_pes_list)}")
        print(f"  Patterns: {len(patterns)}")
        print(f"  Timing types: {len(timing_types)}")
        print(f"  Components: {len(components)}")
        print(f"  Total combinations: {len(layers) * len(num_pes_list) * len(patterns) * len(timing_types) * len(components)}")

        # For efficiency, test key combinations
        print(f"\nRunning strategic subset (high-priority tests)...")

        # Priority 1: Vary PE count (all patterns, permanent, entire_PE)
        print("\n" + "="*80)
        print("Priority 1: PE Count Variation (all patterns, permanent faults)")
        print("="*80)

        for layer_idx, layer_name in layers:
            for num_pes in num_pes_list:
                for pattern in patterns:
                    self.test_single_configuration(
                        test_id=test_id,
                        layer_idx=layer_idx,
                        num_pes=num_pes,
                        pattern=pattern,
                        timing_type='permanent',
                        component='entire_PE'
                    )
                    test_id += 1

        # Priority 2: Timing variation (key patterns, vary timing)
        print("\n" + "="*80)
        print("Priority 2: Timing Variation (selected patterns)")
        print("="*80)

        key_patterns = ['corner', 'center', 'row', 'random']
        key_pe_counts = [4, 8, 16]

        for layer_idx, layer_name in layers:
            for num_pes in key_pe_counts:
                for pattern in key_patterns:
                    for timing_type in ['long_transient', 'short_transient']:
                        self.test_single_configuration(
                            test_id=test_id,
                            layer_idx=layer_idx,
                            num_pes=num_pes,
                            pattern=pattern,
                            timing_type=timing_type,
                            component='entire_PE'
                        )
                        test_id += 1

        # Priority 3: Component variation (key configurations)
        print("\n" + "="*80)
        print("Priority 3: Component Variation")
        print("="*80)

        for layer_idx, layer_name in layers:
            for num_pes in [4, 16]:
                for pattern in ['corner', 'center']:
                    for component in ['MAC_unit', 'accumulator_register']:
                        self.test_single_configuration(
                            test_id=test_id,
                            layer_idx=layer_idx,
                            num_pes=num_pes,
                            pattern=pattern,
                            timing_type='permanent',
                            component=component
                        )
                        test_id += 1

        print(f"\n{'='*80}")
        print(f"Completed {test_id - 1} tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Analyze and summarize results"""
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)

        if not self.results:
            print("No results to analyze!")
            return

        # Overall winner count
        print("\nOverall Winner Count:")
        winner_counts = {'OS': 0, 'WS': 0, 'IS': 0, 'NONE': 0}
        for r in self.results:
            winner_counts[r['winner']] += 1

        total = len(self.results)
        for dataflow in ['OS', 'WS', 'IS']:
            count = winner_counts[dataflow]
            pct = count / total * 100 if total > 0 else 0
            print(f"  {dataflow}: {count}/{total} ({pct:.1f}%) - Most resilient")

        # Average coverage by dataflow
        print("\nAverage Fault Coverage by Dataflow:")
        for dataflow in ['OS', 'WS', 'IS']:
            key = f'{dataflow.lower()}_coverage'
            coverages = [r[key] for r in self.results if r[key] >= 0]
            if coverages:
                avg = np.mean(coverages)
                std = np.std(coverages)
                min_cov = np.min(coverages)
                max_cov = np.max(coverages)
                print(f"  {dataflow}: {avg:.2f}% ± {std:.2f}% (min={min_cov:.2f}%, max={max_cov:.2f}%)")

        # Breakdown by layer
        print("\nWinner Count by Layer:")
        layers = set(r['layer'] for r in self.results)
        for layer in sorted(layers):
            layer_results = [r for r in self.results if r['layer'] == layer]
            layer_winners = {}
            for r in layer_results:
                winner = r['winner']
                layer_winners[winner] = layer_winners.get(winner, 0) + 1

            print(f"  {layer}:")
            for dataflow in ['OS', 'WS', 'IS']:
                count = layer_winners.get(dataflow, 0)
                pct = count / len(layer_results) * 100 if layer_results else 0
                print(f"    {dataflow}: {count}/{len(layer_results)} ({pct:.1f}%)")

        # Breakdown by PE count
        print("\nWinner Count by Number of Faulty PEs:")
        pe_counts = sorted(set(r['num_pes'] for r in self.results))
        for num_pes in pe_counts:
            pe_results = [r for r in self.results if r['num_pes'] == num_pes]
            pe_winners = {}
            for r in pe_results:
                winner = r['winner']
                pe_winners[winner] = pe_winners.get(winner, 0) + 1

            print(f"  {num_pes} PEs:")
            for dataflow in ['OS', 'WS', 'IS']:
                count = pe_winners.get(dataflow, 0)
                pct = count / len(pe_results) * 100 if pe_results else 0
                print(f"    {dataflow}: {count}/{len(pe_results)} ({pct:.1f}%)")

        # Statistical significance
        print("\nStatistical Comparison (paired t-test):")
        from scipy import stats as scipy_stats

        os_coverages = [r['os_coverage'] for r in self.results if r['os_coverage'] >= 0]
        ws_coverages = [r['ws_coverage'] for r in self.results if r['ws_coverage'] >= 0]
        is_coverages = [r['is_coverage'] for r in self.results if r['is_coverage'] >= 0]

        if len(os_coverages) == len(ws_coverages) == len(is_coverages):
            # OS vs WS
            t_stat_os_ws, p_val_os_ws = scipy_stats.ttest_rel(os_coverages, ws_coverages)
            print(f"  OS vs WS: t={t_stat_os_ws:.3f}, p={p_val_os_ws:.6f}")
            if p_val_os_ws < 0.05:
                winner = "OS" if np.mean(os_coverages) < np.mean(ws_coverages) else "WS"
                print(f"    → {winner} significantly more resilient (p < 0.05)")

            # OS vs IS
            t_stat_os_is, p_val_os_is = scipy_stats.ttest_rel(os_coverages, is_coverages)
            print(f"  OS vs IS: t={t_stat_os_is:.3f}, p={p_val_os_is:.6f}")
            if p_val_os_is < 0.05:
                winner = "OS" if np.mean(os_coverages) < np.mean(is_coverages) else "IS"
                print(f"    → {winner} significantly more resilient (p < 0.05)")

            # WS vs IS
            t_stat_ws_is, p_val_ws_is = scipy_stats.ttest_rel(ws_coverages, is_coverages)
            print(f"  WS vs IS: t={t_stat_ws_is:.3f}, p={p_val_ws_is:.6f}")
            if p_val_ws_is < 0.05:
                winner = "WS" if np.mean(ws_coverages) < np.mean(is_coverages) else "IS"
                print(f"    → {winner} significantly more resilient (p < 0.05)")

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataflow_comparison_{timestamp}.csv"

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['test_id', 'layer', 'layer_idx', 'num_pes', 'pattern',
                         'timing_type', 'component', 'os_coverage', 'ws_coverage',
                         'is_coverage', 'winner', 'winner_coverage']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow(result)

        print(f"Results saved!")


def main():
    print("="*80)
    print(" "*15 + "DATAFLOW RESILIENCE COMPARISON STUDY")
    print(" "*20 + "OS vs WS vs IS")
    print("="*80)

    analyzer = DataflowComparisonAnalyzer()

    # Run tests
    analyzer.run_comprehensive_tests()

    # Analyze results
    analyzer.analyze_results()

    # Save results
    analyzer.save_results()

    print("\n" + "="*80)
    print("Study complete!")
    print("="*80)


if __name__ == '__main__':
    main()
