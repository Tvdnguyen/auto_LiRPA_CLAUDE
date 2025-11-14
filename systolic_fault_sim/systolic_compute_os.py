"""
Output Stationary Dataflow Implementation
Adapted from SCALE-Sim

Each PE accumulates one output element
Inputs and weights flow through the array
"""

import numpy as np


class SystolicComputeOS:
    """Output Stationary dataflow simulator"""

    def __init__(self, array_rows, array_cols):
        """
        Args:
            array_rows: Number of PE rows
            array_cols: Number of PE columns
        """
        self.arr_h = array_rows
        self.arr_w = array_cols

    def generate_demand_matrices(self, operand_matrices):
        """
        Generate demand matrices from operand matrices

        Returns:
            Dict with 'ifmap_demand', 'filter_demand', 'ofmap_demand'
            Each is a 2D array: (num_cycles, num_pes)
            Values are addresses or -1 for no access
        """
        ifmap_mat = operand_matrices['ifmap']
        filter_mat = operand_matrices['filter']
        ofmap_mat = operand_matrices['ofmap']

        dims = operand_matrices['dimensions']
        Sr = dims['ofmap_pixels']  # Spatial dimension
        Sc = dims['num_filters']   # Output channels
        T = dims['conv_window_size']  # Temporal (accumulation)

        # Compute folding
        row_fold = int(np.ceil(Sr / self.arr_h))
        col_fold = int(np.ceil(Sc / self.arr_w))

        all_ifmap_demands = []
        all_filter_demands = []
        all_ofmap_demands = []

        pe_mapping = {}  # Maps (cycle, pe_row, pe_col) -> address

        global_cycle = 0

        for fr in range(row_fold):
            for fc in range(col_fold):
                # Extract tile
                r_start = fr * self.arr_h
                r_end = min(r_start + self.arr_h, Sr)
                c_start = fc * self.arr_w
                c_end = min(c_start + self.arr_w, Sc)

                ifmap_block = ifmap_mat[r_start:r_end, :]  # (tile_h, T)
                filter_block = filter_mat[:, c_start:c_end]  # (T, tile_w)
                ofmap_block = ofmap_mat[r_start:r_end, c_start:c_end]  # (tile_h, tile_w)

                # Pad if needed
                tile_h = r_end - r_start
                tile_w = c_end - c_start

                if tile_h < self.arr_h:
                    pad_h = self.arr_h - tile_h
                    ifmap_block = np.vstack([ifmap_block, -np.ones((pad_h, T), dtype=np.int32)])
                    ofmap_block = np.vstack([ofmap_block, -np.ones((pad_h, tile_w), dtype=np.int32)])

                if tile_w < self.arr_w:
                    pad_w = self.arr_w - tile_w
                    filter_block = np.hstack([filter_block, -np.ones((T, pad_w), dtype=np.int32)])
                    ofmap_block = np.hstack([ofmap_block, -np.ones((self.arr_h, pad_w), dtype=np.int32)])

                # Generate demand sequence for this tile
                # OS dataflow: T cycles of accumulation, then output
                num_cycles = T + self.arr_w - 1  # Systolic pipeline depth

                tile_ifmap_demand = np.zeros((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)
                tile_filter_demand = np.zeros((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)
                tile_ofmap_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)

                # Accumulation phase (T cycles)
                for t in range(T):
                    for r in range(self.arr_h):
                        for c in range(self.arr_w):
                            pe_idx = r * self.arr_w + c

                            # IFMAP: broadcast along columns
                            tile_ifmap_demand[t, pe_idx] = ifmap_block[r, t]

                            # FILTER: broadcast along rows
                            tile_filter_demand[t, pe_idx] = filter_block[t, c]

                            # Record mapping
                            pe_mapping[(global_cycle + t, r, c)] = {
                                'ifmap': ifmap_block[r, t],
                                'filter': filter_block[t, c]
                            }

                # Output phase (arr_w - 1 cycles for pipeline drain)
                for t in range(T, num_cycles):
                    output_cycle = t - T
                    if output_cycle < self.arr_w:
                        for r in range(self.arr_h):
                            for c in range(self.arr_w):
                                if c <= output_cycle:
                                    pe_idx = r * self.arr_w + c
                                    tile_ofmap_demand[t, pe_idx] = ofmap_block[r, c]

                                    pe_mapping[(global_cycle + t, r, c)] = {
                                        'ofmap': ofmap_block[r, c]
                                    }

                # Accumulate for all tiles
                all_ifmap_demands.append(tile_ifmap_demand)
                all_filter_demands.append(tile_filter_demand)
                all_ofmap_demands.append(tile_ofmap_demand)

                global_cycle += num_cycles

        # Concatenate all tiles
        ifmap_demand = np.vstack(all_ifmap_demands)
        filter_demand = np.vstack(all_filter_demands)
        ofmap_demand = np.vstack(all_ofmap_demands)

        return {
            'ifmap_demand': ifmap_demand,
            'filter_demand': filter_demand,
            'ofmap_demand': ofmap_demand,
            'pe_mapping': pe_mapping,
            'total_cycles': global_cycle,
            'num_tiles': row_fold * col_fold
        }

    def compute_mapping_efficiency(self, operand_matrices):
        """Compute what percentage of PEs are utilized"""
        dims = operand_matrices['dimensions']
        Sr = dims['ofmap_pixels']
        Sc = dims['num_filters']

        utilized_pes = min(Sr, self.arr_h) * min(Sc, self.arr_w)
        total_pes = self.arr_h * self.arr_w

        return utilized_pes / total_pes
