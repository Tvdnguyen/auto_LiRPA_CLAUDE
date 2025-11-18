"""
SCALE-Sim OS Dataflow Demand Matrix Generation

Based on: SCALE-Sim-main/scalesim/compute/systolic_compute_os.py

Output Stationary (OS):
- Each PE accumulates partial sums for one output pixel
- IFMAP and Filter are streamed through the array
- Outputs remain stationary until complete
"""

import numpy as np
import math
from typing import Dict, Tuple


class SystolicComputeOS:
    """
    OS (Output Stationary) dataflow demand matrix generator.

    Based on SCALE-Sim systolic_compute_os.py
    """

    def __init__(self, arr_h: int, arr_w: int):
        """
        Initialize OS dataflow simulator.

        Args:
            arr_h: Array height (number of rows)
            arr_w: Array width (number of columns)
        """
        self.arr_h = arr_h
        self.arr_w = arr_w

    def generate_demand_matrices(self, operand_mats: Dict) -> Dict:
        """
        Generate demand matrices for OS dataflow.

        Based on systolic_compute_os.py lines 97-236

        Args:
            operand_mats: Dictionary containing:
                - ifmap_matrix: (ofmap_px_per_filt, conv_window_size)
                - filter_matrix: (conv_window_size, num_filters)
                - ofmap_matrix: (ofmap_px_per_filt, num_filters)
                - dimensions: layer dimensions

        Returns:
            Dictionary containing:
                - ifmap_demand: (num_cycles, num_pes)
                - filter_demand: (num_cycles, num_pes)
                - ofmap_demand: (num_cycles, num_pes)
                - total_cycles: int
                - num_tiles: int
        """
        ifmap_mat = operand_mats['ifmap_matrix']
        filter_mat = operand_mats['filter_matrix']
        ofmap_mat = operand_mats['ofmap_matrix']

        # Matrix dimensions (from systolic_compute_os.py lines 84-92)
        Sr = ifmap_mat.shape[0]  # ofmap pixels per filter
        T = ifmap_mat.shape[1]   # conv window size
        Sc = filter_mat.shape[1]  # num filters

        # Tiling factors
        row_fold = math.ceil(Sr / self.arr_h)
        col_fold = math.ceil(Sc / self.arr_w)
        num_tiles = row_fold * col_fold

        print(f"  Tiling: {row_fold} row folds × {col_fold} col folds = {num_tiles} tiles")

        # Initialize demand matrices
        # Each tile has: T cycles (accumulation) + arr_w - 1 (output drain)
        cycles_per_tile = T + self.arr_w - 1
        total_cycles = cycles_per_tile * num_tiles
        num_pes = self.arr_h * self.arr_w

        ifmap_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)
        filter_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)
        ofmap_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)

        cycle_offset = 0

        # Process each tile
        for fr in range(row_fold):
            for fc in range(col_fold):
                tile_start_cycle = cycle_offset

                # Determine tile boundaries
                r_start = fr * self.arr_h
                r_end = min(r_start + self.arr_h, Sr)
                c_start = fc * self.arr_w
                c_end = min(c_start + self.arr_w, Sc)

                r_size = r_end - r_start
                c_size = c_end - c_start

                # Get tile operands
                ifmap_tile = ifmap_mat[r_start:r_end, :]  # (r_size, T)
                filter_tile = filter_mat[:, c_start:c_end]  # (T, c_size)
                ofmap_tile = ofmap_mat[r_start:r_end, c_start:c_end]  # (r_size, c_size)

                # Generate demand for this tile
                self._generate_tile_demands(
                    ifmap_tile, filter_tile, ofmap_tile,
                    r_size, c_size, T,
                    tile_start_cycle,
                    ifmap_demand, filter_demand, ofmap_demand
                )

                cycle_offset += cycles_per_tile

        return {
            'ifmap_demand': ifmap_demand,
            'filter_demand': filter_demand,
            'ofmap_demand': ofmap_demand,
            'total_cycles': total_cycles,
            'num_tiles': num_tiles,
            'cycles_per_tile': cycles_per_tile
        }

    def _generate_tile_demands(self, ifmap_tile, filter_tile, ofmap_tile,
                               r_size, c_size, T, tile_start_cycle,
                               ifmap_demand, filter_demand, ofmap_demand):
        """
        Generate demand matrices for one tile (OS dataflow).

        OS Dataflow Schedule:
        - Cycle 0 to T-1: Accumulation phase
          - Stream ifmap (diagonal) and filter (broadcast columns)
        - Cycle T to T+W-2: Output drain phase
          - Outputs drain diagonally
        """
        # Phase 1: Accumulation (cycles 0 to T-1)
        for t in range(T):
            cycle = tile_start_cycle + t

            # IFMAP: Diagonal streaming (systolic flow)
            # Each row receives ifmap at different times
            for r in range(r_size):
                for c in range(c_size):
                    pe_idx = r * self.arr_w + c

                    # Diagonal delay: ifmap arrives at PE(r,c) at cycle t-r
                    if t >= r and t - r < T:
                        ifmap_idx = t - r
                        ifmap_demand[cycle, pe_idx] = ifmap_tile[r, ifmap_idx]

            # FILTER: Column broadcast
            # All PEs in same column get same filter at same time
            for c in range(c_size):
                if t < T:
                    for r in range(r_size):
                        pe_idx = r * self.arr_w + c
                        filter_demand[cycle, pe_idx] = filter_tile[t, c]

        # Phase 2: Output drain (cycles T to T+W-2)
        for d in range(self.arr_w - 1):
            cycle = tile_start_cycle + T + d

            # Outputs drain diagonally
            for c in range(c_size):
                if c <= d < c + r_size:
                    r = d - c
                    if r < r_size:
                        pe_idx = r * self.arr_w + c
                        ofmap_demand[cycle, pe_idx] = ofmap_tile[r, c]

    def compute_mapping_efficiency(self, operand_mats: Dict) -> float:
        """
        Compute mapping efficiency.

        Efficiency = (active PEs) / (total PEs) averaged across tiles
        """
        Sr = operand_mats['ifmap_matrix'].shape[0]
        Sc = operand_mats['filter_matrix'].shape[1]

        row_fold = math.ceil(Sr / self.arr_h)
        col_fold = math.ceil(Sc / self.arr_w)

        total_util = 0
        for fr in range(row_fold):
            for fc in range(col_fold):
                r_start = fr * self.arr_h
                r_end = min(r_start + self.arr_h, Sr)
                c_start = fc * self.arr_w
                c_end = min(c_start + self.arr_w, Sc)

                r_size = r_end - r_start
                c_size = c_end - c_start

                util = (r_size * c_size) / (self.arr_h * self.arr_w)
                total_util += util

        return total_util / (row_fold * col_fold)


if __name__ == '__main__':
    print("Testing OS Dataflow Generator\n")

    # Import operand matrix generator
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from scalesim_operand_matrix import OperandMatrixGenerator

    # Test with simple case
    print("="*60)
    print("Test: 4x4 conv, 2x2 array, OS dataflow")
    print("="*60)

    gen = OperandMatrixGenerator()
    gen.set_layer_params(
        ifmap_h=4, ifmap_w=4,
        filter_h=3, filter_w=3,
        num_channels=3, num_filters=2,
        stride=1, padding=1
    )

    ifmap_mat, filter_mat, ofmap_mat = gen.create_all_matrices()

    operand_mats = {
        'ifmap_matrix': ifmap_mat,
        'filter_matrix': filter_mat,
        'ofmap_matrix': ofmap_mat,
        'dimensions': gen.get_layer_info()
    }

    os_compute = SystolicComputeOS(arr_h=2, arr_w=2)
    demand_mats = os_compute.generate_demand_matrices(operand_mats)

    print(f"\nResults:")
    print(f"  Total cycles: {demand_mats['total_cycles']}")
    print(f"  Cycles per tile: {demand_mats['cycles_per_tile']}")
    print(f"  Number of tiles: {demand_mats['num_tiles']}")

    print(f"\nDemand matrix shapes:")
    print(f"  IFMAP: {demand_mats['ifmap_demand'].shape}")
    print(f"  Filter: {demand_mats['filter_demand'].shape}")
    print(f"  OFMAP: {demand_mats['ofmap_demand'].shape}")

    # Count valid accesses
    ifmap_valid = np.sum(demand_mats['ifmap_demand'] >= 0)
    filter_valid = np.sum(demand_mats['filter_demand'] >= 0)
    ofmap_valid = np.sum(demand_mats['ofmap_demand'] >= 0)

    print(f"\nValid accesses:")
    print(f"  IFMAP: {ifmap_valid}")
    print(f"  Filter: {filter_valid}")
    print(f"  OFMAP: {ofmap_valid}")

    mapping_eff = os_compute.compute_mapping_efficiency(operand_mats)
    print(f"\nMapping efficiency: {mapping_eff*100:.2f}%")

    print("\n✓ OS dataflow test complete!")
