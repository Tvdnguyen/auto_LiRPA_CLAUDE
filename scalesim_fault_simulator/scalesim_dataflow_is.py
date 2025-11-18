"""
SCALE-Sim IS Dataflow Demand Matrix Generation

Based on: SCALE-Sim-main/scalesim/compute/systolic_compute_is.py

Input Stationary (IS):
- Each PE holds one input (ifmap element)
- Weights stream through the array
- Outputs accumulate and drain
"""

import numpy as np
import math
from typing import Dict


class SystolicComputeIS:
    """IS (Input Stationary) dataflow demand matrix generator."""

    def __init__(self, arr_h: int, arr_w: int):
        self.arr_h = arr_h
        self.arr_w = arr_w

    def generate_demand_matrices(self, operand_mats: Dict) -> Dict:
        """
        Generate demand matrices for IS dataflow.

        IS has 3 phases per tile:
        1. Input loading: arr_h cycles
        2. Weight streaming: T cycles (conv window)
        3. Output drain: arr_h - 1 cycles
        """
        ifmap_mat = operand_mats['ifmap_matrix']
        filter_mat = operand_mats['filter_matrix']
        ofmap_mat = operand_mats['ofmap_matrix']

        Sr = ifmap_mat.shape[0]
        T = ifmap_mat.shape[1]
        Sc = filter_mat.shape[1]

        # IS tiling: tile by spatial rows
        row_fold = math.ceil(Sr / self.arr_h)
        col_fold = math.ceil(Sc / self.arr_w)
        num_tiles = row_fold * col_fold

        print(f"  Tiling: {row_fold} row folds × {col_fold} col folds = {num_tiles} tiles")

        input_load_cycles = self.arr_h
        weight_stream_cycles = T
        output_drain_cycles = self.arr_h - 1
        cycles_per_tile = input_load_cycles + weight_stream_cycles + output_drain_cycles

        total_cycles = cycles_per_tile * num_tiles
        num_pes = self.arr_h * self.arr_w

        ifmap_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)
        filter_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)
        ofmap_demand = np.full((total_cycles, num_pes), -1, dtype=np.int32)

        cycle_offset = 0

        for fr in range(row_fold):
            for fc in range(col_fold):
                tile_start_cycle = cycle_offset

                r_start = fr * self.arr_h
                r_end = min(r_start + self.arr_h, Sr)
                c_start = fc * self.arr_w
                c_end = min(c_start + self.arr_w, Sc)

                r_size = r_end - r_start
                c_size = c_end - c_start

                ifmap_tile = ifmap_mat[r_start:r_end, :]
                filter_tile = filter_mat[:, c_start:c_end]
                ofmap_tile = ofmap_mat[r_start:r_end, c_start:c_end]

                self._generate_tile_demands_is(
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

    def _generate_tile_demands_is(self, ifmap_tile, filter_tile, ofmap_tile,
                                  r_size, c_size, T, tile_start_cycle,
                                  ifmap_demand, filter_demand, ofmap_demand):
        """Generate IS demand for one tile."""
        # Phase 1: Input loading
        for h in range(self.arr_h):
            cycle = tile_start_cycle + h
            for r in range(min(h + 1, r_size)):
                for c in range(c_size):
                    pe_idx = r * self.arr_w + c
                    if h < T:
                        ifmap_demand[cycle, pe_idx] = ifmap_tile[r, h]

        # Phase 2: Weight streaming
        for t in range(T):
            cycle = tile_start_cycle + self.arr_h + t
            for r in range(r_size):
                for c in range(c_size):
                    pe_idx = r * self.arr_w + c
                    filter_demand[cycle, pe_idx] = filter_tile[t, c]
                    # Ifmap already loaded
                    ifmap_demand[cycle, pe_idx] = ifmap_tile[r, t]

        # Phase 3: Output drain
        for d in range(self.arr_h - 1):
            cycle = tile_start_cycle + self.arr_h + T + d
            for r in range(r_size):
                if r <= d:
                    for c in range(c_size):
                        pe_idx = r * self.arr_w + c
                        ofmap_demand[cycle, pe_idx] = ofmap_tile[r, c]

    def compute_mapping_efficiency(self, operand_mats: Dict) -> float:
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
