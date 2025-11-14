"""
Fault Coverage Analysis Script

Tests various fault scenarios to verify theoretical predictions about fault coverage:
- 0-10%: Low impact scenarios
- 10-20%: Low-medium impact
- 20-50%: Medium-high impact
- 50%+: High impact

Runs multiple configurations and records results.
"""

import sys
import os
import numpy as np
import csv
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet


class FaultCoverageAnalyzer:
    """Analyze fault coverage across different scenarios"""

    def __init__(self):
        self.results = []

    def test_scenario(self, scenario_name, array_size, dataflow, layer_idx,
                     fault_specs, expected_range):
        """
        Test a single fault scenario

        Args:
            scenario_name: Descriptive name
            array_size: (h, w) tuple
            dataflow: 'OS', 'WS', or 'IS'
            layer_idx: Layer index in TrafficSignNet
            fault_specs: List of (pe_row, pe_col, component, timing) tuples
            expected_range: Expected coverage range string (e.g., "0-10%")
        """
        print(f"\n{'='*80}")
        print(f"Testing: {scenario_name}")
        print(f"{'='*80}")
        print(f"  Array: {array_size[0]}x{array_size[1]}")
        print(f"  Dataflow: {dataflow}")
        print(f"  Faults: {len(fault_specs)} PE(s)")
        print(f"  Expected: {expected_range}")

        # Create simulator
        h, w = array_size
        simulator = SystolicFaultSimulator(h, w, dataflow)

        # Get layer config
        model = TrafficSignNet(num_classes=43)
        layer_config = simulator.get_layer_config(model, layer_idx)

        print(f"  Layer: {layer_config['name']} ({layer_config['type']})")

        # Create faults
        faults = []
        for pe_row, pe_col, component, timing in fault_specs:
            fault = FaultModel(
                fault_type=FaultModel.PERMANENT,
                fault_location={
                    'pe_row': pe_row,
                    'pe_col': pe_col,
                    'component': component
                },
                fault_timing=timing
            )
            faults.append(fault)

        # Run simulation (suppress detailed output)
        try:
            results = simulator.simulate_layer(layer_config, faults)

            stats = results['statistics']
            coverage = stats['fault_coverage'] * 100

            print(f"\n  RESULT: {coverage:.2f}%")

            # Check if within expected range
            if expected_range == "0-10%":
                in_range = 0 <= coverage <= 10
            elif expected_range == "10-20%":
                in_range = 10 < coverage <= 20
            elif expected_range == "20-50%":
                in_range = 20 < coverage <= 50
            elif expected_range == "50%+":
                in_range = coverage > 50
            else:
                in_range = None

            status = "✓ PASS" if in_range else "✗ FAIL"
            print(f"  {status}")

            # Record result
            self.results.append({
                'scenario': scenario_name,
                'array_size': f"{h}x{w}",
                'dataflow': dataflow,
                'layer': layer_config['name'],
                'num_faults': len(fault_specs),
                'expected_range': expected_range,
                'actual_coverage': coverage,
                'total_outputs': stats['total_outputs'],
                'affected_outputs': stats['affected_outputs'],
                'status': 'PASS' if in_range else 'FAIL'
            })

            return coverage

        except Exception as e:
            print(f"  ERROR: {e}")
            self.results.append({
                'scenario': scenario_name,
                'array_size': f"{h}x{w}",
                'dataflow': dataflow,
                'layer': layer_config['name'],
                'num_faults': len(fault_specs),
                'expected_range': expected_range,
                'actual_coverage': -1,
                'total_outputs': 0,
                'affected_outputs': 0,
                'status': 'ERROR'
            })
            return -1

    def run_all_tests(self):
        """Run all test scenarios"""
        print("\n" + "="*80)
        print(" "*20 + "FAULT COVERAGE ANALYSIS")
        print(" "*15 + "Theoretical Predictions vs Actual Results")
        print("="*80)

        # ===================================================================
        # 0-10% Coverage Tests
        # ===================================================================
        print("\n" + "="*80)
        print("Category 1: LOW IMPACT (0-10% coverage)")
        print("="*80)

        # OS: Single PE, small layer
        self.test_scenario(
            scenario_name="OS-1PE-Small-Layer",
            array_size=(8, 8),
            dataflow='OS',
            layer_idx=0,  # conv1: 32x32 spatial, 32 channels
            fault_specs=[
                (2, 3, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
            ],
            expected_range="0-10%"
        )

        # WS: Single PE, transient short
        self.test_scenario(
            scenario_name="WS-1PE-Transient-Short",
            array_size=(8, 8),
            dataflow='WS',
            layer_idx=0,
            fault_specs=[
                (2, 3, 'MAC_unit', {'start_cycle': 10, 'duration': 5})
            ],
            expected_range="0-10%"
        )

        # IS: Single PE, transient
        self.test_scenario(
            scenario_name="IS-1PE-Transient",
            array_size=(8, 8),
            dataflow='IS',
            layer_idx=0,
            fault_specs=[
                (1, 1, 'entire_PE', {'start_cycle': 10, 'duration': 3})
            ],
            expected_range="0-10%"
        )

        # ===================================================================
        # 10-20% Coverage Tests
        # ===================================================================
        print("\n" + "="*80)
        print("Category 2: LOW-MEDIUM IMPACT (10-20% coverage)")
        print("="*80)

        # OS: 2-4 PEs
        self.test_scenario(
            scenario_name="OS-4PE-Permanent",
            array_size=(8, 8),
            dataflow='OS',
            layer_idx=0,
            fault_specs=[
                (2, 2, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
                (2, 3, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
                (3, 2, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
                (3, 3, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
            ],
            expected_range="10-20%"
        )

        # WS: 1 PE permanent (affects 1 full channel)
        self.test_scenario(
            scenario_name="WS-1PE-Permanent-FullChannel",
            array_size=(8, 8),
            dataflow='WS',
            layer_idx=0,
            fault_specs=[
                (0, 3, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
            ],
            expected_range="10-20%"
        )

        # ===================================================================
        # 20-50% Coverage Tests
        # ===================================================================
        print("\n" + "="*80)
        print("Category 3: MEDIUM-HIGH IMPACT (20-50% coverage)")
        print("="*80)

        # OS: 8-16 PEs (1-2 rows)
        self.test_scenario(
            scenario_name="OS-8PE-OneRow",
            array_size=(8, 8),
            dataflow='OS',
            layer_idx=0,
            fault_specs=[
                (2, c, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
                for c in range(8)
            ],
            expected_range="20-50%"
        )

        # WS: 2-3 PEs permanent (multiple channels)
        self.test_scenario(
            scenario_name="WS-3PE-MultiChannel",
            array_size=(8, 8),
            dataflow='WS',
            layer_idx=0,
            fault_specs=[
                (0, 2, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
                (0, 3, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
                (0, 4, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')}),
            ],
            expected_range="20-50%"
        )

        # IS: 12 PEs
        self.test_scenario(
            scenario_name="IS-12PE-Mixed",
            array_size=(8, 8),
            dataflow='IS',
            layer_idx=0,
            fault_specs=[
                (r, c, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
                for r in range(3) for c in range(4)
            ],
            expected_range="20-50%"
        )

        # ===================================================================
        # 50%+ Coverage Tests
        # ===================================================================
        print("\n" + "="*80)
        print("Category 4: HIGH IMPACT (50%+ coverage)")
        print("="*80)

        # OS: 32 PEs (half array)
        self.test_scenario(
            scenario_name="OS-32PE-HalfArray",
            array_size=(8, 8),
            dataflow='OS',
            layer_idx=0,
            fault_specs=[
                (r, c, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
                for r in range(4) for c in range(8)
            ],
            expected_range="50%+"
        )

        # WS: 4+ PEs (half channels)
        self.test_scenario(
            scenario_name="WS-4PE-HalfChannels",
            array_size=(8, 8),
            dataflow='WS',
            layer_idx=0,
            fault_specs=[
                (0, c, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
                for c in range(4)
            ],
            expected_range="50%+"
        )

        # IS: 40 PEs
        self.test_scenario(
            scenario_name="IS-40PE-MajorityArray",
            array_size=(8, 8),
            dataflow='IS',
            layer_idx=0,
            fault_specs=[
                (r, c, 'entire_PE', {'start_cycle': 0, 'duration': float('inf')})
                for r in range(5) for c in range(8)
            ],
            expected_range="50%+"
        )

    def save_results(self, filename='fault_coverage_results.csv'):
        """Save results to CSV"""
        print(f"\n{'='*80}")
        print("Saving results...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['scenario', 'array_size', 'dataflow', 'layer', 'num_faults',
                         'expected_range', 'actual_coverage', 'total_outputs',
                         'affected_outputs', 'status']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                writer.writerow(result)

        print(f"Results saved to: {filename}")

    def print_summary(self):
        """Print summary statistics"""
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")

        total = len(self.results)
        passed = sum(1 for r in self.results if r['status'] == 'PASS')
        failed = sum(1 for r in self.results if r['status'] == 'FAIL')
        errors = sum(1 for r in self.results if r['status'] == 'ERROR')

        print(f"\nTotal tests: {total}")
        print(f"  Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"  Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"  Errors: {errors} ({errors/total*100:.1f}%)")

        # Breakdown by category
        print(f"\nBreakdown by expected coverage:")
        for exp_range in ["0-10%", "10-20%", "20-50%", "50%+"]:
            results_in_range = [r for r in self.results if r['expected_range'] == exp_range]
            if results_in_range:
                passed_in_range = sum(1 for r in results_in_range if r['status'] == 'PASS')
                print(f"  {exp_range:>8}: {passed_in_range}/{len(results_in_range)} passed")

        # Show failures
        failures = [r for r in self.results if r['status'] == 'FAIL']
        if failures:
            print(f"\nFailed tests:")
            for r in failures:
                print(f"  - {r['scenario']}: Expected {r['expected_range']}, "
                      f"got {r['actual_coverage']:.2f}%")

        print(f"\n{'='*80}")


def main():
    analyzer = FaultCoverageAnalyzer()

    # Run all tests
    analyzer.run_all_tests()

    # Print summary
    analyzer.print_summary()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fault_coverage_results_{timestamp}.csv"
    analyzer.save_results(filename)

    print(f"\nAnalysis complete!")


if __name__ == '__main__':
    main()
