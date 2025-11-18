"""
SCALE-Sim WS Dataflow Demand Matrix Generation

Based on: SCALE-Sim-main/scalesim/compute/systolic_compute_ws.py

Weight Stationary (WS):
- Each PE holds one weight (filter element)
- Inputs stream through the array
- Outputs accumulate and drain
"""

import numpy as np
import math
from typing import Dict


class SystolicComputeWS:
    """
    WS (Weight Stationary) dataflow demand matrix generator.

    Based on SCALE-Sim systolic_compute_ws.py
    """

    def __init__(self, arr_h: int, arr_w: int):
        """
        Initialize WS dataflow simulator.

        Args:
            arr_h: Array height (number of rows)
            arr_w: Array width (number of columns)
        """
        self.arr_h = arr_h
        self.arr_w = arr_w

    def generate_demand_matrices(self, operand_mats: Dict) -> Dict:
        """
        Generate demand matrices for WS dataflow.

        WS has 3 phases per tile (from systolic_compute_ws.py lines 92-107):
        1. Weight loading: arr_h cycles
        2. Input streaming: Sr cycles (ofmap pixels)
        3. Output drain: arr_w - 1 + Sr cycles

        Args:
            operand_mats: Dictionary with operand matrices

        Returns:
            Dictionary with demand matrices
        """
        ifmap_mat = operand_mats['ifmap_matrix']
        filter_mat = operand_mats['filter_matrix']
        ofmap_mat = operand_mats['ofmap_matrix']

        # Dimensions
        Sr = ifmap_mat.shape[0]  # ofmap pixels per filter
        T = ifmap_mat.shape[1]   # conv window size
        Sc = filter_mat.shape[1]  # num filters

        # WS tiling: tile by filters (columns)
        # Each tile processes all spatial positions for a subset of filters
        col_fold = math.ceil(Sc / self.arr_w)
        row_fold = 1  # WS doesn't fold rows
        num_tiles = col_fold

        print(f"  Tiling: 1 row fold × {col_fold} col folds = {num_tiles} tiles")

        # Cycles per tile (from systolic_compute_ws.py lines 92-107)
        weight_load_cycles = self.arr_h
        input_stream_cycles = Sr
        output_drain_cycles = self.arr_w - 1 + Sr
        cycles_per_tile = weight_load_cycles + input_stream_cycles + output_drain_cycles

        total_cycles = cycles_per_tile * num_tiles
        num_pes = self.arr_h * self.arr_w

        ifmap_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)
        filter_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)
        ofmap_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)

        cycle_offset = 0

        # Process each tile (each tile = subset of filters)
        for fc in range(col_fold):
            tile_start_cycle = cycle_offset

            # Filter tile boundaries
            c_start = fc * self.arr_w
            c_end = min(c_start + self.arr_w, Sc)
            c_size = c_end - c_start

            # Get filter tile: all window positions, subset of filters
            filter_tile = filter_mat[:, c_start:c_end]  # (T, c_size)

            # Get ofmap tile: all spatial positions, subset of filters
            ofmap_tile = ofmap_mat[:, c_start:c_end]  # (Sr, c_size)

            # Generate demand for this tile
            self._generate_tile_demands_ws(
                ifmap_mat, filter_tile, ofmap_tile,
                Sr, T, c_size,
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
            'cycles_per_tile': cycles_per_tile,
            'weight_load_cycles': weight_load_cycles,
            'input_stream_cycles': input_stream_cycles,
            'output_drain_cycles': output_drain_cycles
        }

    def _generate_tile_demands_ws(self, ifmap_mat, filter_tile, ofmap_tile,
                                  Sr, T, c_size, tile_start_cycle,
                                  ifmap_demand, filter_demand, ofmap_demand):
        """
        Generate WS demand for one tile.

        WS Schedule (from systolic_compute_ws.py):
        Phase 1 (cycles 0 to arr_h-1): Weight loading
          - Load weights row by row into PEs
        Phase 2 (cycles arr_h to arr_h+Sr-1): Input streaming
          - Stream inputs through array
          - Accumulate partial sums
        Phase 3 (cycles arr_h+Sr to end): Output drain
          - Drain outputs diagonally
        """
        # Phase 1: Weight loading (cycles 0 to arr_h-1)
        for h in range(self.arr_h):
            cycle = tile_start_cycle + h

            # Load weights for row h
            for c in range(c_size):
                # Each PE in row h gets filter element based on its position
                # WS: PE(r,c) holds filter[r, c]
                if h < T:  # Only load if within window size
                    pe_idx = h * self.arr_w + c
                    filter_demand[cycle, pe_idx] = filter_tile[h, c]

        # Phase 2: Input streaming (cycles arr_h to arr_h+Sr-1)
        for s in range(Sr):
            cycle = tile_start_cycle + self.arr_h + s

            # Stream inputs
            # Each spatial position s streams through columns
            for c in range(c_size):
                # All PEs in column c participate
                for r in range(self.arr_h):
                    pe_idx = r * self.arr_w + c

                    # IFMAP access: PE(r,c) accesses ifmap[s, r]
                    if r < T:
                        ifmap_demand[cycle, pe_idx] = ifmap_mat[s, r]

                    # FILTER access: Weight already loaded, read from local register
                    # (In real hardware, filter stays in register)
                    # For demand matrix, we mark -1 (no SRAM access during compute)
                    # But for fault injection, we need to know PE is using this filter
                    # So we keep the address for fault tracking
                    if r < T:
                        filter_demand[cycle, pe_idx] = filter_tile[r, c]

        # Phase 3: Output drain (cycles arr_h+Sr to arr_h+Sr+arr_w-1+Sr-1)
        for d in range(self.arr_w - 1 + Sr):
            cycle = tile_start_cycle + self.arr_h + Sr + d

            # Diagonal drain
            for c in range(c_size):
                for s in range(Sr):
                    if d == c + s:  # Diagonal condition
                        # Find which PE row produced this output
                        # In WS, all rows contribute to each output
                        # We drain from the last row
                        r = self.arr_h - 1
                        pe_idx = r * self.arr_w + c
                        ofmap_demand[cycle, pe_idx] = ofmap_tile[s, c]
                        break

    def compute_mapping_efficiency(self, operand_mats: Dict) -> float:
        """Compute mapping efficiency for WS."""
        Sc = operand_mats['filter_matrix'].shape[1]
        T = operand_mats['filter_matrix'].shape[0]

        col_fold = math.ceil(Sc / self.arr_w)

        total_util = 0
        for fc in range(col_fold):
            c_start = fc * self.arr_w
            c_end = min(c_start + self.arr_w, Sc)
            c_size = c_end - c_start

            # Row utilization: depends on T vs arr_h
            r_util = min(T, self.arr_h) / self.arr_h
            c_util = c_size / self.arr_w

            util = r_util * c_util
            total_util += util

        return total_util / col_fold


if __name__ == '__main__':
    print("Testing WS Dataflow Generator\n")

    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from scalesim_operand_matrix import OperandMatrixGenerator

    print("="*60)
    print("Test: 4x4 conv, 2x2 array, WS dataflow")
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

    ws_compute = SystolicComputeWS(arr_h=2, arr_w=2)
    demand_mats = ws_compute.generate_demand_matrices(operand_mats)

    print(f"\nResults:")
    print(f"  Total cycles: {demand_mats['total_cycles']}")
    print(f"  Cycles per tile: {demand_mats['cycles_per_tile']}")
    print(f"    - Weight load: {demand_mats['weight_load_cycles']}")
    print(f"    - Input stream: {demand_mats['input_stream_cycles']}")
    print(f"    - Output drain: {demand_mats['output_drain_cycles']}")
    print(f"  Number of tiles: {demand_mats['num_tiles']}")

    print(f"\nDemand matrix shapes:")
    print(f"  IFMAP: {demand_mats['ifmap_demand'].shape}")
    print(f"  Filter: {demand_mats['filter_demand'].shape}")
    print(f"  OFMAP: {demand_mats['ofmap_demand'].shape}")

    ifmap_valid = np.sum(demand_mats['ifmap_demand'] >= 0)
    filter_valid = np.sum(demand_mats['filter_demand'] >= 0)
    ofmap_valid = np.sum(demand_mats['ofmap_demand'] >= 0)

    print(f"\nValid accesses:")
    print(f"  IFMAP: {ifmap_valid}")
    print(f"  Filter: {filter_valid}")
    print(f"  OFMAP: {ofmap_valid}")

    mapping_eff = ws_compute.compute_mapping_efficiency(operand_mats)
    print(f"\nMapping efficiency: {mapping_eff*100:.2f}%")

    print("\n✓ WS dataflow test complete!")
