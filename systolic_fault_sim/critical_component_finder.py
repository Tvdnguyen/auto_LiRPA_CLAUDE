"""
Critical Component Finder - Which Register is Most Dangerous?

Goal: Identify which PE component (register) has HIGHEST fault coverage
      (i.e., most critical/dangerous when faulted)

Test Strategy:
- Test ALL 64 PE positions (8x8 array)
- Test 3 register types: accumulator, weight, input
- Test all 3 dataflows: OS, WS, IS
- Use transient faults (realistic)
- Multiple durations: 1, 5, 10, 20 cycles

Total: 64 positions × 3 components × 3 dataflows × 4 durations = 2,304 tests
per layer. We'll test 3 layers = 6,912 tests total.

Output:
1. Ranking: Which register is most critical (highest avg coverage)
2. Heatmap: Fault coverage by PE position for each register
3. Worst-case scenarios: PE position + register combinations with max impact
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


class CriticalComponentFinder:
    """Find which register type is most critical"""

    def __init__(self):
        self.results = []
        self.model = TrafficSignNet(num_classes=43)

    def _compute_mid_computation_cycle(self, dataflow, layer_config):
        """Get middle of computation phase"""
        op_gen = OperandMatrix(layer_config)
        operand_mats = op_gen.generate_matrices()
        dims = operand_mats['dimensions']

        Sr = dims['ofmap_pixels']
        T = dims['conv_window_size']
        arr_h = 8

        if dataflow == 'OS':
            return T // 2
        elif dataflow == 'WS':
            return arr_h + Sr // 2
        elif dataflow == 'IS':
            return arr_h + T // 2
        return 10

    def test_single_configuration(self, test_id, pe_row, pe_col, component,
                                  dataflow, layer_idx, duration):
        """Test single PE position with specific component"""

        layers_info = self.model.get_layer_info()
        layer_name = layers_info[layer_idx][0]

        # Print progress indicator
        if test_id % 100 == 0:
            print(f"Progress: Test #{test_id}...")

        try:
            # Create simulator
            simulator = SystolicFaultSimulator(8, 8, dataflow)
            layer_config = simulator.get_layer_config(self.model, layer_idx)

            # Get mid computation cycle
            start_cycle = self._compute_mid_computation_cycle(dataflow, layer_config)

            # Create transient fault
            fault = FaultModel(
                fault_type=FaultModel.BIT_FLIP,
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

            # Run simulation (minimal output)
            results = simulator.simulate_layer(layer_config, [fault])
            stats = results['statistics']

            coverage = stats['fault_coverage'] * 100

            # Record result
            result = {
                'test_id': test_id,
                'pe_row': pe_row,
                'pe_col': pe_col,
                'component': component,
                'component_short': component.replace('_register', ''),
                'dataflow': dataflow,
                'layer': layer_name,
                'layer_idx': layer_idx,
                'duration': duration,
                'start_cycle': start_cycle,
                'coverage': coverage,
                'total_outputs': stats['total_outputs'],
                'affected_outputs': stats['affected_outputs']
            }

            self.results.append(result)
            return result

        except Exception as e:
            result = {
                'test_id': test_id,
                'pe_row': pe_row,
                'pe_col': pe_col,
                'component': component,
                'component_short': component.replace('_register', ''),
                'dataflow': dataflow,
                'layer': layer_name,
                'layer_idx': layer_idx,
                'duration': duration,
                'coverage': -1,
                'error': str(e)
            }
            self.results.append(result)
            return result

    def run_comprehensive_tests(self):
        """Run all tests"""
        print("\n" + "="*80)
        print(" "*10 + "CRITICAL COMPONENT FINDER")
        print(" "*15 + "Which Register is Most Dangerous?")
        print("="*80)

        components = [
            'accumulator_register',
            'weight_register',
            'input_register'
        ]

        dataflows = ['OS', 'WS', 'IS']

        # Test representative layers
        layers = [
            0,  # conv1: large
            2,  # conv3: medium
            4,  # conv5: small
        ]

        durations = [1, 5, 10, 20]  # cycles

        # All 64 PE positions
        pe_positions = [(r, c) for r in range(8) for c in range(8)]

        total_tests = len(pe_positions) * len(components) * len(dataflows) * len(layers) * len(durations)

        print(f"\nTest Configuration:")
        print(f"  PE positions: {len(pe_positions)} (all positions in 8x8 array)")
        print(f"  Components: {len(components)}")
        print(f"  Dataflows: {len(dataflows)}")
        print(f"  Layers: {len(layers)}")
        print(f"  Durations: {len(durations)}")
        print(f"  Total tests: {total_tests}")
        print(f"\nEstimated time: ~{total_tests * 3 / 60:.1f} minutes")

        test_id = 1

        for layer_idx in layers:
            layer_name = self.model.get_layer_info()[layer_idx][0]
            print(f"\n{'='*80}")
            print(f"Testing Layer: {layer_name}")
            print(f"{'='*80}")

            for dataflow in dataflows:
                print(f"\n  Dataflow: {dataflow}")

                for duration in durations:
                    print(f"    Duration: {duration} cycles")

                    for component in components:
                        for pe_row, pe_col in pe_positions:
                            self.test_single_configuration(
                                test_id=test_id,
                                pe_row=pe_row,
                                pe_col=pe_col,
                                component=component,
                                dataflow=dataflow,
                                layer_idx=layer_idx,
                                duration=duration
                            )
                            test_id += 1

        print(f"\n{'='*80}")
        print(f"Completed {test_id - 1} tests")
        print(f"{'='*80}")

    def analyze_results(self):
        """Comprehensive analysis"""
        print("\n" + "="*80)
        print("ANALYSIS: Critical Component Identification")
        print("="*80)

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            print("No valid results!")
            return

        # 1. Overall ranking: Which register is most critical?
        print("\n1. CRITICAL COMPONENT RANKING (Higher = More Dangerous):")
        print("="*80)

        component_stats = {}
        for component in ['accumulator', 'weight', 'input']:
            comp_full = f'{component}_register'
            subset = [r for r in valid_results if r['component'] == comp_full]

            if subset:
                coverages = [r['coverage'] for r in subset]
                component_stats[component] = {
                    'mean': np.mean(coverages),
                    'std': np.std(coverages),
                    'max': np.max(coverages),
                    'min': np.min(coverages),
                    'median': np.median(coverages)
                }

        # Sort by mean coverage (descending = most critical first)
        sorted_components = sorted(component_stats.items(),
                                   key=lambda x: x[1]['mean'],
                                   reverse=True)

        for rank, (comp, stats) in enumerate(sorted_components, 1):
            print(f"\n  Rank #{rank}: {comp.upper()} Register")
            print(f"    Average coverage: {stats['mean']:6.2f}%")
            print(f"    Std deviation:    {stats['std']:6.2f}%")
            print(f"    Max coverage:     {stats['max']:6.2f}%")
            print(f"    Min coverage:     {stats['min']:6.2f}%")
            print(f"    Median:           {stats['median']:6.2f}%")

        print(f"\n  → MOST CRITICAL: {sorted_components[0][0].upper()} register")
        print(f"    (Highest average fault coverage: {sorted_components[0][1]['mean']:.2f}%)")

        # 2. Per-dataflow ranking
        print("\n2. Critical Component by Dataflow:")
        print("="*80)

        for dataflow in ['OS', 'WS', 'IS']:
            print(f"\n  {dataflow} Dataflow:")

            comp_avg = {}
            for component in ['accumulator', 'weight', 'input']:
                comp_full = f'{component}_register'
                subset = [r for r in valid_results
                         if r['dataflow'] == dataflow and r['component'] == comp_full]
                if subset:
                    avg = np.mean([r['coverage'] for r in subset])
                    comp_avg[component] = avg

            sorted_comps = sorted(comp_avg.items(), key=lambda x: x[1], reverse=True)
            for rank, (comp, avg) in enumerate(sorted_comps, 1):
                print(f"    #{rank}. {comp:15s}: {avg:6.2f}%")

        # 3. Worst-case scenarios
        print("\n3. WORST-CASE SCENARIOS (Top 10 Highest Coverage):")
        print("="*80)

        # Sort by coverage descending
        sorted_results = sorted(valid_results, key=lambda x: x['coverage'], reverse=True)

        print(f"{'Rank':<5} | {'Coverage':<9} | {'Component':<15} | {'Dataflow':<8} | "
              f"{'PE Position':<12} | {'Layer':<6} | {'Duration':<8}")
        print("-"*80)

        for rank, r in enumerate(sorted_results[:10], 1):
            print(f"{rank:<5} | {r['coverage']:>6.2f}%   | {r['component_short']:<15} | "
                  f"{r['dataflow']:<8} | ({r['pe_row']},{r['pe_col']}){' '*7} | "
                  f"{r['layer']:<6} | {r['duration']:2d} cyc")

        # 4. Best-case scenarios (least critical)
        print("\n4. BEST-CASE SCENARIOS (Top 10 Lowest Coverage):")
        print("="*80)

        # Sort by coverage ascending
        sorted_results_asc = sorted(valid_results, key=lambda x: x['coverage'])

        print(f"{'Rank':<5} | {'Coverage':<9} | {'Component':<15} | {'Dataflow':<8} | "
              f"{'PE Position':<12} | {'Layer':<6} | {'Duration':<8}")
        print("-"*80)

        for rank, r in enumerate(sorted_results_asc[:10], 1):
            print(f"{rank:<5} | {r['coverage']:>6.2f}%   | {r['component_short']:<15} | "
                  f"{r['dataflow']:<8} | ({r['pe_row']},{r['pe_col']}){' '*7} | "
                  f"{r['layer']:<6} | {r['duration']:2d} cyc")

        # 5. Duration sensitivity per component
        print("\n5. Duration Sensitivity per Component:")
        print("="*80)

        for component in ['accumulator', 'weight', 'input']:
            comp_full = f'{component}_register'
            print(f"\n  {component.upper()} register:")

            for duration in [1, 5, 10, 20]:
                subset = [r for r in valid_results
                         if r['component'] == comp_full and r['duration'] == duration]
                if subset:
                    avg = np.mean([r['coverage'] for r in subset])
                    print(f"    {duration:2d} cycles: {avg:6.2f}%")

    def create_heatmaps(self):
        """Create heatmaps for each component showing PE position criticality"""
        print("\nGenerating heatmaps...")

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            return

        components = ['accumulator', 'weight', 'input']
        dataflows = ['OS', 'WS', 'IS']

        # Create figure with subplots
        fig, axes = plt.subplots(len(components), len(dataflows),
                                 figsize=(15, 12))

        for i, component in enumerate(components):
            comp_full = f'{component}_register'

            for j, dataflow in enumerate(dataflows):
                ax = axes[i, j]

                # Create 8x8 matrix for average coverage
                heatmap_data = np.zeros((8, 8))

                for r in range(8):
                    for c in range(8):
                        subset = [res for res in valid_results
                                 if res['component'] == comp_full
                                 and res['dataflow'] == dataflow
                                 and res['pe_row'] == r
                                 and res['pe_col'] == c]

                        if subset:
                            heatmap_data[r, c] = np.mean([res['coverage'] for res in subset])

                # Plot heatmap
                im = ax.imshow(heatmap_data, cmap='YlOrRd', vmin=0, vmax=100)

                # Add text annotations
                for r in range(8):
                    for c in range(8):
                        text = ax.text(c, r, f'{heatmap_data[r, c]:.1f}',
                                      ha="center", va="center", color="black",
                                      fontsize=7)

                # Labels
                ax.set_title(f'{component.upper()} - {dataflow}',
                            fontweight='bold', fontsize=10)
                ax.set_xlabel('Column', fontsize=8)
                ax.set_ylabel('Row', fontsize=8)
                ax.set_xticks(range(8))
                ax.set_yticks(range(8))

        # Add colorbar
        fig.colorbar(im, ax=axes.ravel().tolist(), label='Fault Coverage (%)',
                    fraction=0.02, pad=0.04)

        plt.suptitle('Fault Coverage Heatmap by PE Position\n'
                    '(Higher = More Critical)',
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig('critical_component_heatmap.png', dpi=150, bbox_inches='tight')
        print("Heatmap saved: critical_component_heatmap.png")
        plt.close()

    def create_comparison_chart(self):
        """Create bar chart comparing components"""
        print("Generating comparison chart...")

        valid_results = [r for r in self.results if r['coverage'] >= 0]
        if not valid_results:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        components = ['accumulator', 'weight', 'input']
        dataflows = ['OS', 'WS', 'IS']

        x = np.arange(len(components))
        width = 0.25

        for i, dataflow in enumerate(dataflows):
            coverages = []

            for component in components:
                comp_full = f'{component}_register'
                subset = [r for r in valid_results
                         if r['component'] == comp_full and r['dataflow'] == dataflow]
                if subset:
                    coverages.append(np.mean([r['coverage'] for r in subset]))
                else:
                    coverages.append(0)

            ax.bar(x + i * width, coverages, width, label=dataflow)

        ax.set_xlabel('Component', fontweight='bold', fontsize=12)
        ax.set_ylabel('Average Fault Coverage (%)', fontweight='bold', fontsize=12)
        ax.set_title('Component Criticality Comparison\n(Higher = More Dangerous)',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels([c.capitalize() for c in components])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig('component_comparison.png', dpi=150, bbox_inches='tight')
        print("Chart saved: component_comparison.png")
        plt.close()

    def save_results(self, filename=None):
        """Save results to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"critical_component_results_{timestamp}.csv"

        print(f"\nSaving results to {filename}...")

        with open(filename, 'w', newline='') as f:
            fieldnames = ['test_id', 'pe_row', 'pe_col', 'component', 'component_short',
                         'dataflow', 'layer', 'layer_idx', 'duration', 'start_cycle',
                         'coverage', 'total_outputs', 'affected_outputs']
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                if 'error' not in result:
                    writer.writerow(result)

        print("Results saved!")


def main():
    print("="*80)
    print(" "*15 + "CRITICAL COMPONENT FINDER")
    print(" "*10 + "Which Register Has HIGHEST Fault Coverage?")
    print("="*80)

    analyzer = CriticalComponentFinder()

    # Run tests
    print("\nStarting comprehensive tests...")
    print("This will take some time (~30-60 minutes for 6,912 tests)")
    print("")

    analyzer.run_comprehensive_tests()

    # Analysis
    analyzer.analyze_results()

    # Visualizations
    analyzer.create_heatmaps()
    analyzer.create_comparison_chart()

    # Save
    analyzer.save_results()

    print("\n" + "="*80)
    print("Analysis complete!")
    print("  → critical_component_heatmap.png: PE position heatmaps")
    print("  → component_comparison.png: Component comparison chart")
    print("  → CSV file: Detailed results")
    print("="*80)


if __name__ == '__main__':
    main()
