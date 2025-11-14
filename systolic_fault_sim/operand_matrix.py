"""
Operand Matrix Generator
Based on SCALE-Sim architecture but simplified for fault simulation

Generates address matrices for IFMAP, FILTER, and OFMAP
These matrices encode which memory addresses are accessed for each operation
"""

import numpy as np


class OperandMatrix:
    """Generate operand matrices for Conv and FC layers"""

    def __init__(self, layer_config):
        """
        Args:
            layer_config: Dict with layer parameters
                For Conv: input_shape, output_shape, kernel_size, stride, padding
                For FC: input_size, output_size
        """
        self.config = layer_config
        self.layer_type = layer_config.get('type', 'Conv')

        # Address offsets for different operands
        self.ifmap_offset = 0
        self.filter_offset = 10000000  # Large offset to separate address spaces
        self.ofmap_offset = 20000000

    def generate_matrices(self):
        """Generate all three operand matrices"""
        if self.layer_type == 'Conv':
            return self._generate_conv_matrices()
        else:  # FC
            return self._generate_fc_matrices()

    def _generate_conv_matrices(self):
        """Generate matrices for Conv layer"""
        Cin, Hin, Win = self.config['input_shape']
        Cout, Hout, Wout = self.config['output_shape']
        K = self.config['kernel_size'][0]
        stride = self.config['stride']
        padding = self.config['padding']

        # Dimensions
        ofmap_pixels = Hout * Wout
        conv_window_size = K * K * Cin
        num_filters = Cout

        # IFMAP matrix: (ofmap_pixels, conv_window_size)
        ifmap_matrix = np.zeros((ofmap_pixels, conv_window_size), dtype=np.int32)

        for i in range(ofmap_pixels):
            out_row = i // Wout
            out_col = i % Wout

            # Input position
            in_row_base = out_row * stride
            in_col_base = out_col * stride

            for j in range(conv_window_size):
                # Decompose j into (k_row, k_col, k_ch)
                k_row = j // (K * Cin)
                temp = j % (K * Cin)
                k_col = temp // Cin
                k_ch = temp % Cin

                # Actual input position
                in_row = in_row_base + k_row - padding
                in_col = in_col_base + k_col - padding

                # Check bounds
                if 0 <= in_row < Hin and 0 <= in_col < Win:
                    addr = (in_row * Win + in_col) * Cin + k_ch
                    ifmap_matrix[i, j] = self.ifmap_offset + addr
                else:
                    ifmap_matrix[i, j] = -1  # Padding

        # FILTER matrix: (conv_window_size, num_filters)
        filter_matrix = np.zeros((conv_window_size, num_filters), dtype=np.int32)

        for j in range(conv_window_size):
            for f in range(num_filters):
                addr = j * num_filters + f
                filter_matrix[j, f] = self.filter_offset + addr

        # OFMAP matrix: (ofmap_pixels, num_filters)
        ofmap_matrix = np.zeros((ofmap_pixels, num_filters), dtype=np.int32)

        for i in range(ofmap_pixels):
            for f in range(num_filters):
                addr = i * num_filters + f
                ofmap_matrix[i, f] = self.ofmap_offset + addr

        return {
            'ifmap': ifmap_matrix,
            'filter': filter_matrix,
            'ofmap': ofmap_matrix,
            'dimensions': {
                'ofmap_pixels': ofmap_pixels,
                'conv_window_size': conv_window_size,
                'num_filters': num_filters
            }
        }

    def _generate_fc_matrices(self):
        """Generate matrices for FC layer (treated as 1x1 conv)"""
        input_size = self.config['input_size']
        output_size = self.config['output_size']

        # FC as GEMM: output = input @ weights
        # Dimensions: (1, input_size) @ (input_size, output_size) = (1, output_size)

        # IFMAP matrix: (1, input_size) - single row
        ifmap_matrix = np.arange(input_size, dtype=np.int32).reshape(1, -1) + self.ifmap_offset

        # FILTER matrix: (input_size, output_size)
        filter_matrix = np.zeros((input_size, output_size), dtype=np.int32)
        for i in range(input_size):
            for o in range(output_size):
                filter_matrix[i, o] = self.filter_offset + i * output_size + o

        # OFMAP matrix: (1, output_size) - single row
        ofmap_matrix = np.arange(output_size, dtype=np.int32).reshape(1, -1) + self.ofmap_offset

        return {
            'ifmap': ifmap_matrix,
            'filter': filter_matrix,
            'ofmap': ofmap_matrix,
            'dimensions': {
                'ofmap_pixels': 1,
                'conv_window_size': input_size,
                'num_filters': output_size
            }
        }

    def map_address_to_pe(self, address, pe_mapping):
        """
        Find which PE accesses this address

        Args:
            address: Memory address
            pe_mapping: Mapping from (demand_matrix, cycle, pe_idx) -> address

        Returns:
            List of (cycle, pe_row, pe_col) tuples
        """
        accesses = []
        for (cycle, pe_row, pe_col), addr in pe_mapping.items():
            if addr == address:
                accesses.append((cycle, pe_row, pe_col))
        return accesses

    def get_output_dependencies(self, output_idx, operand_matrices):
        """
        Get all addresses that contribute to computing a specific output

        Args:
            output_idx: (out_pixel, out_filter) tuple
            operand_matrices: Result from generate_matrices()

        Returns:
            Dict with 'ifmap' and 'filter' address lists
        """
        out_pixel, out_filter = output_idx

        ifmap_addrs = operand_matrices['ifmap'][out_pixel, :]
        ifmap_addrs = ifmap_addrs[ifmap_addrs >= 0]  # Remove padding

        filter_addrs = operand_matrices['filter'][:, out_filter]

        return {
            'ifmap': ifmap_addrs.tolist(),
            'filter': filter_addrs.tolist()
        }
