"""
Test WS dataflow fault propagation fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from fault_simulator import SystolicFaultSimulator, FaultModel

def test_ws_transient_fault():
    """Test WS dataflow with transient fault during computation"""

    # Create simulator with WS dataflow
    simulator = SystolicFaultSimulator(8, 8, 'WS')

    # Small layer for testing
    layer_config = {
        'name': 'test_ws',
        'type': 'Conv',
        'input_channels': 3,
        'output_channels': 8,  # Small to debug
        'input_size': (8, 8),  # Small spatial
        'kernel_size': (3, 3),
        'stride': 1,
        'padding': 1
    }

    # Fault at PE (2,3) during computation phase
    # WS: H=8, Sr=64, so computation is cycles 8-71
    # Fault at cycles 10-11 should affect outputs!
    faults = [
        FaultModel(
            fault_type=FaultModel.BIT_FLIP,
            fault_location={
                'pe_row': 2,
                'pe_col': 3,
                'component': 'MAC_unit'
            },
            fault_timing={'start_cycle': 10, 'duration': 2}
        )
    ]

    print("Testing WS dataflow with transient fault...")
    print(f"Layer: {layer_config['output_channels']} channels, {layer_config['input_size']} spatial")
    print(f"Fault: PE(2,3), MAC_unit, bit_flip, cycles 10-12")
    print()

    results = simulator.simulate_layer(layer_config, faults)

    stats = results['statistics']
    print("Results:")
    print(f"  Total outputs: {stats['total_outputs']}")
    print(f"  Affected outputs: {stats['affected_outputs']}")
    print(f"  Fault coverage: {stats['fault_coverage']*100:.2f}%")
    print()

    if stats['affected_outputs'] > 0:
        print("✅ SUCCESS: Fault propagation working correctly!")
        print(f"   {stats['affected_outputs']} outputs affected")
    else:
        print("❌ FAILED: No outputs affected (bug still exists)")

    return stats['affected_outputs'] > 0


if __name__ == '__main__':
    success = test_ws_transient_fault()
    exit(0 if success else 1)
