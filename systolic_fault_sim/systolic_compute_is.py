"""
Input Stationary Dataflow Implementation
Adapted from SCALE-Sim

Each PE holds one input element (ifmap activation)
Weights stream through columns, outputs accumulate through rows
"""

import numpy as np


class SystolicComputeIS:
    """Input Stationary dataflow simulator"""

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
        Generate demand matrices from operand matrices using IS dataflow

        In IS dataflow:
        - Each PE holds ONE input activation (stationary)
        - Weights stream through from left to right
        - Outputs accumulate vertically

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
        T = dims['conv_window_size']  # Temporal (filter positions)

        # Compute folding
        # In IS: inputs are distributed spatially
        row_fold = int(np.ceil(Sr / self.arr_h))  # Spatial positions
        col_fold = int(np.ceil(Sc / self.arr_w))  # Output channels (same as OS/WS)

        all_ifmap_demands = []
        all_filter_demands = []
        all_ofmap_demands = []

        pe_mapping = {}  # Maps (cycle, pe_row, pe_col) -> address

        global_cycle = 0

        for fc in range(col_fold):
            for fr in range(row_fold):
                # Extract tile
                # In IS: each PE holds ifmap[spatial_pos, t]
                r_start = fr * self.arr_h
                r_end = min(r_start + self.arr_h, Sr)
                c_start = fc * self.arr_w
                c_end = min(c_start + self.arr_w, Sc)

                # IFMAP block: (Sr, T) indexed by [r_start:r_end, :]
                ifmap_block = ifmap_mat[r_start:r_end, :]  # (tile_h, T)

                # FILTER block: (T, Sc) indexed by [:, c_start:c_end]
                filter_block = filter_mat[:, c_start:c_end]  # (T, tile_w)

                # OFMAP block: (Sr, Sc) indexed by [r_start:r_end, c_start:c_end]
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

                # IS dataflow timing
                # Phase 1: Load inputs (arr_h cycles to fill PE rows)
                # Phase 2: Stream weights (T cycles, one filter position per cycle)
                # Phase 3: Drain outputs (arr_h - 1 cycles for systolic propagation)

                input_load_cycles = self.arr_h
                weight_stream_cycles = T
                output_drain_cycles = self.arr_h - 1

                num_cycles = input_load_cycles + weight_stream_cycles + output_drain_cycles

                tile_ifmap_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)
                tile_filter_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)
                tile_ofmap_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)

                # Phase 1: Input loading (cycles 0 to arr_h-1)
                # Inputs are loaded row by row
                for cycle in range(input_load_cycles):
                    row_to_load = cycle
                    if row_to_load < self.arr_h:
                        for c in range(self.arr_w):
                            pe_idx = row_to_load * self.arr_w + c

                            # Each PE loads ALL T positions for its spatial location
                            # But in cycle-by-cycle simulation, load one at a time
                            # Simplified: load first position, rest will come during streaming
                            tile_ifmap_demand[cycle, pe_idx] = ifmap_block[row_to_load, 0]

                            pe_mapping[(global_cycle + cycle, row_to_load, c)] = {
                                'ifmap': ifmap_block[row_to_load, 0]
                            }

                # Phase 2: Weight streaming (cycles arr_h to arr_h+T-1)
                # Each cycle, weights for one filter position stream through columns
                for t in range(weight_stream_cycles):
                    cycle = input_load_cycles + t

                    # FILTER: broadcast to all PEs in each column based on filter position t
                    for r in range(self.arr_h):
                        for c in range(self.arr_w):
                            pe_idx = r * self.arr_w + c

                            # Each PE at column c accesses filter[t, c]
                            tile_filter_demand[cycle, pe_idx] = filter_block[t, c]

                            # Also access corresponding ifmap position
                            if t < ifmap_block.shape[1]:
                                tile_ifmap_demand[cycle, pe_idx] = ifmap_block[r, t]

                            pe_mapping[(global_cycle + cycle, r, c)] = {
                                'ifmap': ifmap_block[r, t] if t < ifmap_block.shape[1] else -1,
                                'filter': filter_block[t, c]
                            }

                # Phase 3: Output accumulation and drain
                # Outputs flow vertically, accumulating partial sums
                output_ready_start = input_load_cycles + weight_stream_cycles

                for i in range(output_drain_cycles + 1):
                    cycle = output_ready_start + i
                    if cycle >= num_cycles:
                        break

                    # Outputs drain row by row
                    for r in range(min(i + 1, self.arr_h)):
                        for c in range(self.arr_w):
                            pe_idx = r * self.arr_w + c

                            # Each row produces output for its spatial position
                            if cycle == output_ready_start + r:
                                if r < ofmap_block.shape[0] and c < ofmap_block.shape[1]:
                                    tile_ofmap_demand[cycle, pe_idx] = ofmap_block[r, c]

                                    pe_mapping[(global_cycle + cycle, r, c)] = {
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

        # In IS: PEs utilized based on spatial dimensions
        utilized_pes = min(Sr, self.arr_h) * min(Sc, self.arr_w)
        total_pes = self.arr_h * self.arr_w

        return utilized_pes / total_pes
