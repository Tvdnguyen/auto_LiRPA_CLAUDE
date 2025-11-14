"""
Debug WS dataflow PE-to-output mapping
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from operand_matrix import OperandMatrix
from systolic_compute_ws import SystolicComputeWS

def debug_ws_mapping():
    """Debug WS dataflow demand matrices"""

    # Small layer
    layer_config = {
        'name': 'test',
        'type': 'Conv',
        'input_channels': 3,
        'output_channels': 8,
        'input_size': (8, 8),
        'kernel_size': (3, 3),
        'stride': 1,
        'padding': 1
    }

    print("Generating operand matrices...")
    op_gen = OperandMatrix(layer_config)
    operand_mats = op_gen.generate_matrices()

    dims = operand_mats['dimensions']
    print(f"Dimensions: {dims}")
    print()

    print("Generating WS demand matrices...")
    compute_sim = SystolicComputeWS(8, 8)
    demand_mats = compute_sim.generate_demand_matrices(operand_mats)

    print(f"Total cycles: {demand_mats['total_cycles']}")
    print()

    # Check ofmap_demand matrix
    ofmap_demand = demand_mats['ofmap_demand']
    print(f"ofmap_demand shape: {ofmap_demand.shape}")
    print()

    # Count how many output addresses each PE has
    print("PE-to-output mapping:")
    print("-" * 60)

    num_cycles, num_pes = ofmap_demand.shape
    pe_to_outputs = {}

    for pe_idx in range(num_pes):
        output_addrs = ofmap_demand[:, pe_idx]
        valid_outputs = output_addrs[output_addrs >= 0]
        if len(valid_outputs) > 0:
            pe_to_outputs[pe_idx] = set(valid_outputs)

    print(f"Total PEs with outputs: {len(pe_to_outputs)}")
    print()

    # Show first few PEs
    for pe_idx in range(min(16, num_pes)):
        if pe_idx in pe_to_outputs:
            num_outputs = len(pe_to_outputs[pe_idx])
            pe_row = pe_idx // 8
            pe_col = pe_idx % 8
            print(f"PE ({pe_row},{pe_col}): {num_outputs} unique outputs")

    print()
    print("Expected behavior:")
    print(f"  - Total outputs: {dims['ofmap_pixels'] * dims['num_filters']} (64 spatial × 8 channels)")
    print(f"  - Each PE column should produce ALL spatial outputs for its channel")
    print(f"  - PE in column c should have ~{dims['ofmap_pixels']} outputs")


if __name__ == '__main__':
    debug_ws_mapping()
