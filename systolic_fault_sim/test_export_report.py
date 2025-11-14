"""
Test script to demonstrate fault report export functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from fault_simulator import SystolicFaultSimulator, FaultModel

def test_report_export():
    """Test exporting fault report to text file"""

    # Create simulator
    simulator = SystolicFaultSimulator(8, 8, 'OS')

    # Define a simple Conv layer
    layer_config = {
        'name': 'test_conv',
        'type': 'Conv',
        'input_channels': 3,
        'output_channels': 32,
        'input_size': (32, 32),
        'kernel_size': (3, 3),
        'stride': 1,
        'padding': 1
    }

    # Create multiple faults
    faults = [
        FaultModel(
            fault_type=FaultModel.STUCK_AT_0,
            fault_location={
                'pe_row': 2,
                'pe_col': 3,
                'component': 'accumulator_register'
            },
            fault_timing={'start_cycle': 0, 'duration': float('inf')}
        ),
        FaultModel(
            fault_type=FaultModel.BIT_FLIP,
            fault_location={
                'pe_row': 4,
                'pe_col': 5,
                'component': 'MAC_unit'
            },
            fault_timing={'start_cycle': 0, 'duration': float('inf')}
        )
    ]

    print("Running simulation with 2 faults...")
    results = simulator.simulate_layer(layer_config, faults)

    print("\nExporting visualization...")
    simulator.visualize_results(results, 'test_fault_impact.png')

    print("\nExporting detailed text report...")
    simulator.export_fault_report(results, 'test_fault_report.txt')

    print("\n" + "="*80)
    print("Test completed!")
    print("Generated files:")
    print("  - test_fault_impact.png (visualization)")
    print("  - test_fault_report.txt (detailed report)")
    print("="*80)

    # Print a snippet of the report
    print("\nReport snippet:")
    print("-"*80)
    with open('test_fault_report.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:30]:  # Print first 30 lines
            print(line.rstrip())
    print("...")


if __name__ == '__main__':
    test_report_export()
