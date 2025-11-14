"""
Transient Fault Resilience Analysis

Focus on TRANSIENT faults (realistic scenario) with varying:
- Fault duration (1, 5, 10, 20, 50 cycles)
- Fault timing (early, mid, late in computation phase)
- PE components (acc_reg, weight_reg, input_reg)
- Dataflows (OS, WS, IS)

Key Question: Which dataflow is most resilient to SHORT, REALISTIC transient faults?

Test Matrix:
- 1 PE fault (single point failure)
- 5 durations: 1, 5, 10, 20, 50 cycles
- 5 start times: computed phase start, +25%, +50%, +75%, near end
- 3 components: accumulator, weight, input registers
- 3 dataflows: OS, WS, IS
- 3 layers: conv1, conv3, conv5

Total: 5 × 5 × 3 × 3 × 3 = 675 tests
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


class TransientFaultAnalyzer:
    """Analyze transient fault impact"""

    def __init__(self):
        self.results = []
        self.model = TrafficSignNet(num_classes=43)

    def _compute_computation_phase(self, dataflow, layer_config):
        """
        Compute the computation phase cycles for a layer

        Returns:
            (comp_start, comp_end) tuple
        """
        # Generate operand matrices to get dimensions
        op_gen = OperandMatrix(layer_config)
        operand_mats = op_gen.generate_matrices()
        dims = operand_mats['dimensions']

        Sr = dims['ofmap_pixels']
        T = dims['conv_window_size']
        arr_h, arr_w = 8, 8

        if dataflow == 'OS':
            return (0, T - 1)
        elif dataflow == 'WS':
            comp_start = arr_h
            comp_end = arr_h + Sr - 1
            return (comp_start, comp_end)
        elif dataflow == 'IS':
            comp_start = arr_h
            comp_end = arr_h + T - 1
            return (comp_start, comp_end)
        else:
            return (0, 100)

    def _get_timing_points(self, comp_start, comp_end):
        """
        Generate 5 timing points across computation phase

        Returns:
            List of (start_cycle, label) tuples
        """
        duration = comp_end - comp_start + 1

        points = [
            (comp_start, 'early'),
            (comp_start + int(duration * 0.25), 'early-mid'),
            (comp_start + int(duration * 0.50), 'mid'),
            (comp_start + int(duration * 0.75), 'mid-late'),
            (max(comp_start, comp_end - 10), 'late')
        ]

        return points

    def test_transient_fault(self, test_id, dataflow, layer_idx, component,
                            start_cycle, duration, timing_label):
        """
        Test single transient fault

        Args:
            test_id: Test identifier
            dataflow: OS, WS, IS
            layer_idx: Layer index
            component: accumulator_register, weight_register, input_register
            start_cycle: When fault starts
            duration: How long fault lasts (cycles)
            timing_label: Descriptive label (early, mid, late, etc.)
        """
        # Get layer info
        layers_info = self.model.get_layer_info()
        layer_name, ltype, _, shape = layers_info[layer_idx]

        # Use center PE (4,4) for all tests
        pe_row, pe_col = 4, 4

        print(f"Test #{test_id:4d}: {dataflow:2s} | {layer_name:6s} | {component[:3]:3s}_reg | "
              f"cyc {start_cycle:4d}+{duration:2d} ({timing_label:9s})", end=" ")

        try:
            # Create simulator
            simulator = SystolicFaultSimulator(8, 8, dataflow)

            # Get layer config
            layer_config = simulator.get_layer_config(self.model, layer_idx)

            # Create transient fault
            fault = FaultModel(
                fault_type=FaultModel.BIT_FLIP,  # Realistic fault type
                fault_location={
                    'pe_row': pe_row,
                    'pe_col': pe_col,
                    'component': component
                },
                fault_timing={
                    'start_cycle': start_cycle,
                    'duration': duration
                }
            )

            # Run simulation (suppress output)
            results = simulator.simulate_layer(layer_config, [fault])
            stats = results['statistics']

            coverage = stats['fault_coverage'] * 100

            print(f"→ {coverage:6.2f}%")

            # Record result
            result = {
                'test_id': test_id,
                'dataflow': dataflow,
                'layer': layer_name,
                'layer_idx': layer_idx,
                'component': component,
                'component_short': component.replace('_register', ''),
                'start_cycle': start_cycle,
                'duration': duration,
                'timing_label': timing_label,
                'pe_row': pe_row,
                'pe_col': pe_col,
                'coverage': coverage,
                'total_outputs': stats['total_outputs'],
                'affected_outputs': stats['affected_outputs']
            }

            self.results.append(result)
            return result

        except Exception as e:
            print(f"→ ERROR: {str(e)[:40]}")
            result = {
                'test_id': test_id,
                'dataflow': dataflow,
                'layer': layer_name,
                'layer_idx': layer_idx,
                'component': component,
                'component_short': component.replace('_register', ''),
                'start_cycle': start_cycle,
                'duration': duration,
                'timing_label': timing_label,
                'pe_row': pe_row,
                'pe_col': pe_col,
                'coverage': -1,
                'total_outputs': 0,
                'affected_outputs': 0,
                'error': str(e)
            }
            self.results.append(result)
            return result

    def run_comprehensive_tests(self):
        """Run all transient fault tests"""
        print("\n" + "="*80)
        print(" "*15 + "TRANSIENT FAULT RESILIENCE ANALYSIS")
        print(" "*20 + "(Realistic Fault Scenarios)")
        print("="*80)

        # Test parameters
        durations = [1, 5, 10, 20, 50]  # cycles
        components = [
            'accumulator_register',
            'weight_register',
            'input_register'
        ]
        dataflows = ['OS', 'WS', 'IS']
        layers = [0, 2, 4]  # conv1, conv3, conv5

        print(f"\nTest Configuration:")
        print(f"  Durations: {durations} cycles")
        print(f"  Components: {len(components)}")
        print(f"  Dataflows: {len(dataflows)}")
        print(f"  Layers: {len(layers)}")
        print(f"  Timing points per config: 5 (early to late)")
        print(f"  Total tests: {len(durations) * len(components) * len(dataflows) * len(layers) * 5}")

        test_id = 1

        # Iterate through all combinations
        for layer_idx in layers:
            layer_name = self.model.get_layer_info()[layer_idx][0]
            print(f"\n{'='*80}")
            print(f"Testing Layer: {layer_name}")
            print(f"{'='*80}")

            for dataflow in dataflows:
                # Get computation phase for this layer/dataflow
                simulator_temp = SystolicFaultSimulator(8, 8, dataflow)
                layer_config = simulator_temp.get_layer_config(self.model, layer_idx)
                comp_start, comp_end = self._compute_computation_phase(dataflow, layer_config)

                print(f"\n  {dataflow} Dataflow: Computation phase = cycles {comp_start}-{comp_end}")

                # Get timing points
                timing_points = self._get_timing_points(comp_start, comp_end)

                for component in components:
                    for duration in durations:
                        for start_cycle, timing_label in timing_points:
                            self.test_transient_fault(
                                test_id=test_id,
                                dataflow=dataflow,
                                layer_idx=layer_idx,
                                component=component,
                                start_cycle=start_cycle,
                                duration=duration,
                                timing_label=timing_label
                            )
                            test_id += 1

        print(f"\n{'='*80}")
        print(f"Completed {test_id - 1} tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Comprehensive analysis"""
        print("\n" + "="*80)
        print("ANALYSIS: Transient Fault Resilience")
        print("="*80)

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            print("No valid results!")
            return

        # 1. Overall winner by dataflow
        print("\n1. Overall Resilience Ranking (Lower coverage = More resilient):")
        print("-" * 80)

        dataflow_avg = {}
        for dataflow in ['OS', 'WS', 'IS']:
            subset = [r for r in valid_results if r['dataflow'] == dataflow]
            if subset:
                avg = np.mean([r['coverage'] for r in subset])
                std = np.std([r['coverage'] for r in subset])
                dataflow_avg[dataflow] = avg
                print(f"  {dataflow}: {avg:6.2f}% ± {std:5.2f}%")

        sorted_dataflows = sorted(dataflow_avg.items(), key=lambda x: x[1])
        print(f"\n  → Winner: {sorted_dataflows[0][0]} (most resilient with {sorted_dataflows[0][1]:.2f}% avg coverage)")

        # 2. Impact of fault duration
        print("\n2. Impact of Fault Duration:")
        print("-" * 80)
        print(f"{'Duration':<10} | {'OS':<10} | {'WS':<10} | {'IS':<10} | {'Winner'}")
        print("-" * 80)

        for duration in [1, 5, 10, 20, 50]:
            row = f"{duration:>3d} cycles |"
            duration_coverages = {}

            for dataflow in ['OS', 'WS', 'IS']:
                subset = [r for r in valid_results
                         if r['duration'] == duration and r['dataflow'] == dataflow]
                if subset:
                    avg = np.mean([r['coverage'] for r in subset])
                    duration_coverages[dataflow] = avg
                    row += f" {avg:>6.2f}%   |"
                else:
                    row += f" {'N/A':>6s}    |"

            if duration_coverages:
                winner = min(duration_coverages.keys(), key=lambda k: duration_coverages[k])
                row += f" {winner}"

            print(row)

        # 3. Impact of fault timing
        print("\n3. Impact of Fault Timing (within computation phase):")
        print("-" * 80)
        print(f"{'Timing':<12} | {'OS':<10} | {'WS':<10} | {'IS':<10} | {'Winner'}")
        print("-" * 80)

        for timing in ['early', 'early-mid', 'mid', 'mid-late', 'late']:
            row = f"{timing:<12} |"
            timing_coverages = {}

            for dataflow in ['OS', 'WS', 'IS']:
                subset = [r for r in valid_results
                         if r['timing_label'] == timing and r['dataflow'] == dataflow]
                if subset:
                    avg = np.mean([r['coverage'] for r in subset])
                    timing_coverages[dataflow] = avg
                    row += f" {avg:>6.2f}%   |"
                else:
                    row += f" {'N/A':>6s}    |"

            if timing_coverages:
                winner = min(timing_coverages.keys(), key=lambda k: timing_coverages[k])
                row += f" {winner}"

            print(row)

        # 4. Component sensitivity (transient faults)
        print("\n4. Component Sensitivity for Transient Faults:")
        print("-" * 80)
        print(f"{'Component':<15} | {'OS':<10} | {'WS':<10} | {'IS':<10} | {'Winner'}")
        print("-" * 80)

        for component in ['accumulator', 'weight', 'input']:
            comp_full = f'{component}_register'
            row = f"{component:<15} |"
            comp_coverages = {}

            for dataflow in ['OS', 'WS', 'IS']:
                subset = [r for r in valid_results
                         if r['component'] == comp_full and r['dataflow'] == dataflow]
                if subset:
                    avg = np.mean([r['coverage'] for r in subset])
                    comp_coverages[dataflow] = avg
                    row += f" {avg:>6.2f}%   |"
                else:
                    row += f" {'N/A':>6s}    |"

            if comp_coverages:
                winner = min(comp_coverages.keys(), key=lambda k: comp_coverages[k])
                row += f" {winner}"

            print(row)

        # 5. Short vs long transient comparison
        print("\n5. Short vs Long Transient Faults:")
        print("-" * 80)

        for dataflow in ['OS', 'WS', 'IS']:
            print(f"\n  {dataflow}:")

            # Short: 1-5 cycles
            short = [r for r in valid_results
                    if r['dataflow'] == dataflow and r['duration'] <= 5]
            short_avg = np.mean([r['coverage'] for r in short]) if short else 0

            # Long: 20-50 cycles
            long = [r for r in valid_results
                   if r['dataflow'] == dataflow and r['duration'] >= 20]
            long_avg = np.mean([r['coverage'] for r in long]) if long else 0

            print(f"    Short (1-5 cycles):  {short_avg:6.2f}%")
            print(f"    Long (20-50 cycles): {long_avg:6.2f}%")
            print(f"    Ratio: {long_avg/short_avg:.2f}x" if short_avg > 0 else "    Ratio: N/A")

    def create_plots(self):
        """Create visualization plots"""
        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            return

        # Plot 1: Duration vs Coverage
        fig, ax = plt.subplots(figsize=(10, 6))

        for dataflow in ['OS', 'WS', 'IS']:
            durations = []
            coverages = []

            for duration in [1, 5, 10, 20, 50]:
                subset = [r for r in valid_results
                         if r['dataflow'] == dataflow and r['duration'] == duration]
                if subset:
                    durations.append(duration)
                    coverages.append(np.mean([r['coverage'] for r in subset]))

            ax.plot(durations, coverages, marker='o', linewidth=2, markersize=8, label=dataflow)

        ax.set_xlabel('Fault Duration (cycles)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Fault Coverage (%)', fontweight='bold', fontsize=12)
        ax.set_title('Transient Fault Impact vs Duration\n(Lower is better - more resilient)',
                    fontweight='bold', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')

        plt.tight_layout()
        plt.savefig('transient_duration_plot.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved: transient_duration_plot.png")
        plt.close()

        # Plot 2: Timing vs Coverage
        fig, ax = plt.subplots(figsize=(10, 6))

        timing_order = ['early', 'early-mid', 'mid', 'mid-late', 'late']
        x_positions = range(len(timing_order))

        for dataflow in ['OS', 'WS', 'IS']:
            coverages = []

            for timing in timing_order:
                subset = [r for r in valid_results
                         if r['dataflow'] == dataflow and r['timing_label'] == timing]
                if subset:
                    coverages.append(np.mean([r['coverage'] for r in subset]))
                else:
                    coverages.append(0)

            ax.plot(x_positions, coverages, marker='s', linewidth=2, markersize=8, label=dataflow)

        ax.set_xlabel('Fault Timing in Computation Phase', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Fault Coverage (%)', fontweight='bold', fontsize=12)
        ax.set_title('Transient Fault Impact vs Timing\n(Lower is better - more resilient)',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(timing_order)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('transient_timing_plot.png', dpi=150, bbox_inches='tight')
        print("Plot saved: transient_timing_plot.png")
        plt.close()

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transient_fault_results_{timestamp}.csv"

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['test_id', 'dataflow', 'layer', 'layer_idx', 'component',
                         'component_short', 'start_cycle', 'duration', 'timing_label',
                         'coverage', 'total_outputs', 'affected_outputs']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
                    writer.writerow(result)

        print("Results saved!")


def main():
    print("="*80)
    print(" "*10 + "TRANSIENT FAULT RESILIENCE STUDY")
    print(" "*15 + "Realistic Fault Scenarios")
    print("="*80)

    analyzer = TransientFaultAnalyzer()

    # Run tests
    analyzer.run_comprehensive_tests()

    # Analysis
    analyzer.analyze_results()

    # Plots
    analyzer.create_plots()

    # Save
    analyzer.save_results()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("  → Check transient_duration_plot.png")
    print("  → Check transient_timing_plot.png")
    print("  → Check CSV file for detailed results")
    print("="*80)


if __name__ == '__main__':
    main()
