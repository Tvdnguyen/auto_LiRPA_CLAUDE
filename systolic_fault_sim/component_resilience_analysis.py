"""
Component-Level Resilience Analysis (Single PE Fault)

Analyzes which PE component is most critical for each dataflow when ONLY 1 PE fails.

Test Matrix:
- 1 PE fault (fixed)
- 3 components: accumulator_register, weight_register, input_register
- 3 dataflows: OS, WS, IS
- 4 PE positions: corner, center, edge, off-center
- 3 layers: conv1, conv3, conv5

Total: 108 tests

Outputs:
1. Component criticality ranking per dataflow
2. Heatmap of coverage by (component, dataflow, position)
3. CSV with detailed results
"""

import sys
import os
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from fault_simulator import SystolicFaultSimulator, FaultModel
from traffic_sign_net import TrafficSignNet


class ComponentResilienceAnalyzer:
    """Analyze component-level fault resilience"""

    def __init__(self):
        self.results = []
        self.model = TrafficSignNet(num_classes=43)

    def test_single_pe_component(self, test_id, pe_row, pe_col, component,
                                 dataflow, layer_idx):
        """
        Test single PE with specific component fault

        Args:
            test_id: Unique identifier
            pe_row, pe_col: PE coordinates
            component: 'accumulator_register', 'weight_register', 'input_register'
            dataflow: 'OS', 'WS', 'IS'
            layer_idx: Layer index

        Returns:
            Dict with test results
        """
        # Get layer info
        layers_info = self.model.get_layer_info()
        layer_name, ltype, _, shape = layers_info[layer_idx]

        print(f"Test #{test_id:3d}: PE({pe_row},{pe_col}) | {component:20s} | {dataflow:2s} | {layer_name:6s}", end=" ")

        try:
            # Create simulator
            simulator = SystolicFaultSimulator(8, 8, dataflow)

            # Get layer config
            layer_config = simulator.get_layer_config(self.model, layer_idx)

            # Create single fault
            fault = FaultModel(
                fault_type=FaultModel.PERMANENT,
                fault_location={
                    'pe_row': pe_row,
                    'pe_col': pe_col,
                    'component': component
                },
                fault_timing={'start_cycle': 0, 'duration': float('inf')}
            )

            # Run simulation
            results = simulator.simulate_layer(layer_config, [fault])
            stats = results['statistics']

            coverage = stats['fault_coverage'] * 100

            print(f"→ {coverage:6.2f}%")

            # Record result
            result = {
                'test_id': test_id,
                'pe_row': pe_row,
                'pe_col': pe_col,
                'pe_position': f"({pe_row},{pe_col})",
                'component': component,
                'dataflow': dataflow,
                'layer': layer_name,
                'layer_idx': layer_idx,
                'coverage': coverage,
                'total_outputs': stats['total_outputs'],
                'affected_outputs': stats['affected_outputs']
            }

            self.results.append(result)
            return result

        except Exception as e:
            print(f"→ ERROR: {str(e)[:50]}")
            result = {
                'test_id': test_id,
                'pe_row': pe_row,
                'pe_col': pe_col,
                'pe_position': f"({pe_row},{pe_col})",
                'component': component,
                'dataflow': dataflow,
                'layer': layer_name,
                'layer_idx': layer_idx,
                'coverage': -1,
                'total_outputs': 0,
                'affected_outputs': 0,
                'error': str(e)
            }
            self.results.append(result)
            return result

    def run_comprehensive_tests(self):
        """Run all test combinations"""
        print("\n" + "="*80)
        print(" "*15 + "COMPONENT-LEVEL RESILIENCE ANALYSIS")
        print(" "*20 + "(Single PE Fault Study)")
        print("="*80)

        # Test parameters
        pe_positions = [
            (0, 0, 'corner'),
            (4, 4, 'center'),
            (0, 4, 'edge'),
            (2, 2, 'off-center')
        ]

        components = [
            'accumulator_register',
            'weight_register',
            'input_register'
        ]

        dataflows = ['OS', 'WS', 'IS']

        layers = [
            0,  # conv1: 32x32 spatial, 32 channels
            2,  # conv3: 16x16 spatial, 64 channels
            4,  # conv5: 8x8 spatial, 128 channels
        ]

        print(f"\nTest Configuration:")
        print(f"  PE positions: {len(pe_positions)}")
        print(f"  Components: {len(components)}")
        print(f"  Dataflows: {len(dataflows)}")
        print(f"  Layers: {len(layers)}")
        print(f"  Total tests: {len(pe_positions) * len(components) * len(dataflows) * len(layers)}")

        test_id = 1

        # Iterate through all combinations
        for layer_idx in layers:
            layer_name = self.model.get_layer_info()[layer_idx][0]
            print(f"\n{'='*80}")
            print(f"Testing Layer: {layer_name}")
            print(f"{'='*80}")

            for pe_row, pe_col, pos_label in pe_positions:
                print(f"\n  Position: ({pe_row},{pe_col}) - {pos_label}")

                for component in components:
                    for dataflow in dataflows:
                        self.test_single_pe_component(
                            test_id=test_id,
                            pe_row=pe_row,
                            pe_col=pe_col,
                            component=component,
                            dataflow=dataflow,
                            layer_idx=layer_idx
                        )
                        test_id += 1

        print(f"\n{'='*80}")
        print(f"Completed {test_id - 1} tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Comprehensive analysis"""
        print("\n" + "="*80)
        print("ANALYSIS: Component Criticality by Dataflow")
        print("="*80)

        if not self.results:
            print("No results to analyze!")
            return

        # Filter valid results
        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            print("No valid results!")
            return

        # 1. Average coverage by (dataflow, component)
        print("\n1. Average Coverage by Dataflow and Component:")
        print("-" * 80)
        print(f"{'Dataflow':<10} | {'Component':<22} | {'Avg Coverage':<15} | {'Std Dev':<10}")
        print("-" * 80)

        for dataflow in ['OS', 'WS', 'IS']:
            for component in ['accumulator_register', 'weight_register', 'input_register']:
                subset = [r for r in valid_results
                         if r['dataflow'] == dataflow and r['component'] == component]

                if subset:
                    coverages = [r['coverage'] for r in subset]
                    avg = np.mean(coverages)
                    std = np.std(coverages)
                    print(f"{dataflow:<10} | {component:<22} | {avg:>6.2f}%         | {std:>6.2f}%")

        # 2. Component ranking per dataflow
        print("\n2. Component Criticality Ranking (Higher coverage = More critical):")
        print("-" * 80)

        for dataflow in ['OS', 'WS', 'IS']:
            print(f"\n  {dataflow} Dataflow:")

            component_avg = {}
            for component in ['accumulator_register', 'weight_register', 'input_register']:
                subset = [r for r in valid_results
                         if r['dataflow'] == dataflow and r['component'] == component]
                if subset:
                    component_avg[component] = np.mean([r['coverage'] for r in subset])

            # Sort by coverage (descending)
            sorted_components = sorted(component_avg.items(), key=lambda x: x[1], reverse=True)

            for rank, (comp, avg_cov) in enumerate(sorted_components, 1):
                comp_short = comp.replace('_register', '')
                print(f"    #{rank}. {comp_short:<15s}: {avg_cov:6.2f}% average coverage")

        # 3. Position sensitivity
        print("\n3. Position Sensitivity (Average coverage by position):")
        print("-" * 80)

        position_labels = ['corner', 'center', 'edge', 'off-center']
        position_coords = ['(0,0)', '(4,4)', '(0,4)', '(2,2)']

        for pos_coord, pos_label in zip(position_coords, position_labels):
            subset = [r for r in valid_results if r['pe_position'] == pos_coord]
            if subset:
                avg = np.mean([r['coverage'] for r in subset])
                print(f"  {pos_label:<12s} {pos_coord}: {avg:6.2f}%")

        # 4. Best dataflow per component
        print("\n4. Most Resilient Dataflow per Component:")
        print("-" * 80)

        for component in ['accumulator_register', 'weight_register', 'input_register']:
            print(f"\n  {component}:")

            dataflow_avg = {}
            for dataflow in ['OS', 'WS', 'IS']:
                subset = [r for r in valid_results
                         if r['component'] == component and r['dataflow'] == dataflow]
                if subset:
                    dataflow_avg[dataflow] = np.mean([r['coverage'] for r in subset])

            # Best = lowest coverage
            best_dataflow = min(dataflow_avg.items(), key=lambda x: x[1])
            worst_dataflow = max(dataflow_avg.items(), key=lambda x: x[1])

            print(f"    Most resilient: {best_dataflow[0]} ({best_dataflow[1]:.2f}%)")
            print(f"    Least resilient: {worst_dataflow[0]} ({worst_dataflow[1]:.2f}%)")

        # 5. Layer sensitivity
        print("\n5. Layer Sensitivity:")
        print("-" * 80)

        for layer_idx in [0, 2, 4]:
            layer_name = self.model.get_layer_info()[layer_idx][0]
            subset = [r for r in valid_results if r['layer_idx'] == layer_idx]
            if subset:
                avg = np.mean([r['coverage'] for r in subset])
                print(f"  {layer_name}: {avg:6.2f}% average coverage")

    def create_heatmap(self, save_path='component_heatmap.png'):
        """Create heatmap visualization"""
        print("\nGenerating heatmap...")

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            print("No valid results for heatmap!")
            return

        # Create pivot table: rows=components, cols=dataflows
        components = ['accumulator_register', 'weight_register', 'input_register']
        dataflows = ['OS', 'WS', 'IS']

        data = np.zeros((len(components), len(dataflows)))

        for i, component in enumerate(components):
            for j, dataflow in enumerate(dataflows):
                subset = [r for r in valid_results
                         if r['component'] == component and r['dataflow'] == dataflow]
                if subset:
                    data[i, j] = np.mean([r['coverage'] for r in subset])

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

        # Labels
        ax.set_xticks(np.arange(len(dataflows)))
        ax.set_yticks(np.arange(len(components)))
        ax.set_xticklabels(dataflows)
        ax.set_yticklabels([c.replace('_register', '') for c in components])

        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Add values as text
        for i in range(len(components)):
            for j in range(len(dataflows)):
                text = ax.text(j, i, f'{data[i, j]:.1f}%',
                              ha="center", va="center", color="black", fontweight='bold')

        ax.set_title("Average Fault Coverage by Component and Dataflow\n(Single PE Fault)", fontweight='bold')
        ax.set_xlabel("Dataflow", fontweight='bold')
        ax.set_ylabel("Component", fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Fault Coverage (%)', rotation=270, labelpad=20)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        plt.close()

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"component_resilience_{timestamp}.csv"

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['test_id', 'pe_row', 'pe_col', 'pe_position', 'component',
                         'dataflow', 'layer', 'layer_idx', 'coverage',
                         'total_outputs', 'affected_outputs']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
                    writer.writerow(result)

        print(f"Results saved!")

    def print_summary_table(self):
        """Print concise summary table"""
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)

        valid_results = [r for r in self.results if r['coverage'] >= 0]

        print("\nAverage Fault Coverage (%) by Dataflow and Component:")
        print("-" * 80)
        print(f"{'Component':<22} | {'OS':<10} | {'WS':<10} | {'IS':<10} | {'Winner':<10}")
        print("-" * 80)

        for component in ['accumulator_register', 'weight_register', 'input_register']:
            comp_short = component.replace('_register', '')
            row = f"{comp_short:<22} |"

            coverages = {}
            for dataflow in ['OS', 'WS', 'IS']:
                subset = [r for r in valid_results
                         if r['component'] == component and r['dataflow'] == dataflow]
                if subset:
                    avg = np.mean([r['coverage'] for r in subset])
                    coverages[dataflow] = avg
                    row += f" {avg:>6.2f}%   |"
                else:
                    row += f" {'N/A':>6s}    |"

            # Winner = lowest coverage
            if coverages:
                winner = min(coverages.keys(), key=lambda k: coverages[k])
                row += f" {winner:<10}"

            print(row)

        print("-" * 80)


def main():
    print("="*80)
    print(" "*10 + "COMPONENT-LEVEL FAULT RESILIENCE STUDY")
    print(" "*15 + "Single PE Fault Analysis")
    print("="*80)

    analyzer = ComponentResilienceAnalyzer()

    # Run tests
    analyzer.run_comprehensive_tests()

    # Analysis
    analyzer.analyze_results()

    # Summary table
    analyzer.print_summary_table()

    # Heatmap
    analyzer.create_heatmap()

    # Save results
    analyzer.save_results()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == '__main__':
    main()
