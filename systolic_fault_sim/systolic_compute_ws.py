"""
Weight Stationary Dataflow Implementation
Adapted from SCALE-Sim

Each PE holds one weight element
Inputs stream through rows, outputs accumulate through columns
"""

import numpy as np


class SystolicComputeWS:
    """Weight Stationary dataflow simulator"""

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
        Generate demand matrices from operand matrices using WS dataflow

        In WS dataflow:
        - Each PE holds ONE weight (stationary)
        - Inputs stream through from top to bottom
        - Outputs accumulate horizontally

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
        # In WS: weights are distributed across rows
        row_fold = int(np.ceil(T / self.arr_h))  # Filter positions
        col_fold = int(np.ceil(Sc / self.arr_w))  # Output channels

        all_ifmap_demands = []
        all_filter_demands = []
        all_ofmap_demands = []

        pe_mapping = {}  # Maps (cycle, pe_row, pe_col) -> address

        global_cycle = 0

        for fc in range(col_fold):
            for fr in range(row_fold):
                # Extract tile
                # In WS: each PE holds filter[t, c]
                t_start = fr * self.arr_h
                t_end = min(t_start + self.arr_h, T)
                c_start = fc * self.arr_w
                c_end = min(c_start + self.arr_w, Sc)

                # Filter block: (T, Sc) indexed by [t_start:t_end, c_start:c_end]
                filter_block = filter_mat[t_start:t_end, c_start:c_end]  # (tile_h, tile_w)

                # IFMAP block: (Sr, T) - all spatial positions with relevant T
                ifmap_block = ifmap_mat[:, t_start:t_end]  # (Sr, tile_h)

                # OFMAP block: (Sr, Sc) - relevant spatial × channels
                ofmap_block = ofmap_mat[:, c_start:c_end]  # (Sr, tile_w)

                # Pad if needed
                tile_h = t_end - t_start
                tile_w = c_end - c_start

                if tile_h < self.arr_h:
                    pad_h = self.arr_h - tile_h
                    filter_block = np.vstack([filter_block, -np.ones((pad_h, tile_w), dtype=np.int32)])
                    ifmap_block = np.hstack([ifmap_block, -np.ones((Sr, pad_h), dtype=np.int32)])

                if tile_w < self.arr_w:
                    pad_w = self.arr_w - tile_w
                    filter_block = np.hstack([filter_block, -np.ones((self.arr_h, pad_w), dtype=np.int32)])
                    ofmap_block = np.hstack([ofmap_block, -np.ones((Sr, pad_w), dtype=np.int32)])

                # WS dataflow timing
                # Phase 1: Load weights (arr_h cycles to fill PE rows)
                # Phase 2: Stream inputs (Sr cycles, one spatial position per cycle)
                # Phase 3: Drain outputs
                #   - Initial drain: arr_w - 1 cycles for systolic propagation
                #   - Additional cycles needed to write all Sr outputs per column

                weight_load_cycles = self.arr_h
                input_stream_cycles = Sr

                # Calculate drain cycles: need enough cycles to write all Sr outputs
                # Each PE in a column writes Sr outputs, one per cycle
                # Column delay + Sr outputs
                output_drain_cycles = self.arr_w - 1 + Sr

                num_cycles = weight_load_cycles + input_stream_cycles + output_drain_cycles

                tile_ifmap_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)
                tile_filter_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)
                tile_ofmap_demand = -np.ones((num_cycles, self.arr_h * self.arr_w), dtype=np.int32)

                # Phase 1: Weight loading (cycles 0 to arr_h-1)
                # Weights are loaded row by row (from bottom to top to align with systolic flow)
                for cycle in range(weight_load_cycles):
                    row_to_load = self.arr_h - 1 - cycle  # Load from bottom up
                    for c in range(self.arr_w):
                        pe_idx = row_to_load * self.arr_w + c
                        tile_filter_demand[cycle, pe_idx] = filter_block[row_to_load, c]

                        pe_mapping[(global_cycle + cycle, row_to_load, c)] = {
                            'filter': filter_block[row_to_load, c]
                        }

                # Phase 2: Input streaming (cycles arr_h to arr_h+Sr-1)
                # Each cycle, one row of inputs enters from top
                # Outputs start accumulating
                for i in range(input_stream_cycles):
                    cycle = weight_load_cycles + i
                    spatial_idx = i

                    # IFMAP: broadcast to all PEs in each row based on their weight position
                    for r in range(self.arr_h):
                        for c in range(self.arr_w):
                            pe_idx = r * self.arr_w + c

                            # Each PE at row r accesses ifmap[spatial_idx, r] (corresponding to its weight position)
                            tile_ifmap_demand[cycle, pe_idx] = ifmap_block[spatial_idx, r]

                            pe_mapping[(global_cycle + cycle, r, c)] = {
                                'ifmap': ifmap_block[spatial_idx, r],
                                'filter': filter_block[r, c]
                            }

                # Phase 3: Output accumulation and drain
                # In WS: each PE in column c produces ALL Sr outputs for channel c
                # Each output needs a unique cycle to avoid overwriting
                output_start_cycle = weight_load_cycles + input_stream_cycles

                # Track which cycle each PE is at
                pe_output_cycle = {}
                for pe_idx in range(self.arr_h * self.arr_w):
                    pe_row = pe_idx // self.arr_w
                    pe_col = pe_idx % self.arr_w
                    # Outputs start draining with column-wise delay
                    pe_output_cycle[pe_idx] = output_start_cycle + pe_col

                # For each spatial position, assign outputs
                for s in range(Sr):
                    if s >= ofmap_block.shape[0]:
                        break

                    # For each channel (column)
                    for c in range(self.arr_w):
                        if c >= ofmap_block.shape[1]:
                            continue

                        # All PEs in this column participate in producing this output
                        for r in range(self.arr_h):
                            pe_idx = r * self.arr_w + c

                            output_cycle = pe_output_cycle[pe_idx]

                            if output_cycle < num_cycles:
                                # Mark this output
                                tile_ofmap_demand[output_cycle, pe_idx] = ofmap_block[s, c]

                                # Update pe_mapping
                                pe_mapping[(global_cycle + output_cycle, r, c)] = {
                                    'ofmap': ofmap_block[s, c]
                                }

                                # Move to next cycle for this PE's next output
                                pe_output_cycle[pe_idx] += 1

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
        T = dims['conv_window_size']
        Sc = dims['num_filters']

        # In WS: PEs utilized based on filter dimensions
        utilized_pes = min(T, self.arr_h) * min(Sc, self.arr_w)
        total_pes = self.arr_h * self.arr_w

        return utilized_pes / total_pes
