"""
SCALE-Sim Operand Matrix Generation (Standalone Implementation)

This module implements operand matrix generation based on SCALE-Sim's logic.
Code is derived from reading SCALE-Sim-main/scalesim/compute/operand_matrix.py

Reference: SCALE-Sim operand_matrix.py lines 1-742
"""

import numpy as np
from typing import Tuple, Dict


class OperandMatrixGenerator:
    """
    Generates IFMAP, Filter, and OFMAP operand address matrices.

    Based on SCALE-Sim's operand_matrix class:
    - SCALE-Sim-main/scalesim/compute/operand_matrix.py

    Each matrix maps computation indices to memory addresses.
    """

    def __init__(self):
        """Initialize operand matrix generator."""
        # Layer parameters
        self.ifmap_h = 0
        self.ifmap_w = 0
        self.filter_h = 0
        self.filter_w = 0
        self.num_channels = 0
        self.num_filters = 0
        self.stride = 1
        self.padding = 0

        # Derived parameters
        self.ofmap_h = 0
        self.ofmap_w = 0
        self.ofmap_px_per_filt = 0  # ofmap_h * ofmap_w
        self.conv_window_size = 0   # filter_h * filter_w * num_channels

        # Memory offsets (from SCALE-Sim defaults)
        self.ifmap_offset = 0
        self.filter_offset = 10000000
        self.ofmap_offset = 20000000

        # Generated matrices
        self.ifmap_matrix = None
        self.filter_matrix = None
        self.ofmap_matrix = None

    def set_layer_params(self, ifmap_h: int, ifmap_w: int,
                        filter_h: int, filter_w: int,
                        num_channels: int, num_filters: int,
                        stride: int = 1, padding: int = 0):
        """
        Set layer parameters.

        Args:
            ifmap_h: Input feature map height
            ifmap_w: Input feature map width
            filter_h: Filter height
            filter_w: Filter width
            num_channels: Number of input channels
            num_filters: Number of output filters
            stride: Stride
            padding: Padding
        """
        self.ifmap_h = ifmap_h
        self.ifmap_w = ifmap_w
        self.filter_h = filter_h
        self.filter_w = filter_w
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding

        # Calculate output dimensions (with padding)
        self.ofmap_h = (ifmap_h + 2 * padding - filter_h) // stride + 1
        self.ofmap_w = (ifmap_w + 2 * padding - filter_w) // stride + 1
        self.ofmap_px_per_filt = self.ofmap_h * self.ofmap_w
        self.conv_window_size = filter_h * filter_w * num_channels

    def calc_ifmap_elem_addr(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """
        Calculate IFMAP element address.

        Based on SCALE-Sim operand_matrix.py lines 195-222

        Args:
            i: Output pixel index (0 to ofmap_px_per_filt-1)
            j: Convolution window position (0 to conv_window_size-1)

        Returns:
            IFMAP address or -1 if out of bounds
        """
        offset = self.ifmap_offset
        ifmap_rows = self.ifmap_h
        ifmap_cols = self.ifmap_w
        filter_col = self.filter_w
        r_stride = self.stride
        c_stride = self.stride
        Ew = self.ofmap_w
        channel = self.num_channels

        # Decompose output pixel index to (row, col)
        ofmap_row, ofmap_col = np.divmod(i, Ew)

        # Calculate input pixel location
        i_row, i_col = ofmap_row * r_stride, ofmap_col * c_stride

        # Base address for this output pixel's window
        window_addr = (i_row * ifmap_cols + i_col) * channel

        # Decompose window position to (filter_row, filter_col, channel)
        c_row, k = np.divmod(j, filter_col * channel)
        c_col, c_ch = np.divmod(k, channel)

        # Check bounds (padding is handled by returning -1)
        valid_indices = np.logical_and(c_row + i_row < ifmap_rows,
                                      c_col + i_col < ifmap_cols)

        # Initialize with -1 (invalid access)
        ifmap_px_addr = np.full(i.shape, -1, dtype='>i4')

        # Calculate valid addresses
        if valid_indices.any():
            internal_address = (c_row[valid_indices] * ifmap_cols +
                              c_col[valid_indices]) * channel + c_ch[valid_indices]
            ifmap_px_addr[valid_indices] = (internal_address +
                                           window_addr[valid_indices] + offset)

        return ifmap_px_addr

    def calc_filter_elem_addr(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """
        Calculate Filter element address.

        Based on SCALE-Sim operand_matrix.py lines 371-381

        Args:
            i: Window position (0 to conv_window_size-1)
            j: Filter number (0 to num_filters-1)

        Returns:
            Filter address
        """
        offset = self.filter_offset
        filter_row = self.filter_h
        filter_col = self.filter_w
        channel = self.num_channels

        internal_address = j * filter_row * filter_col * channel + i
        filter_px_addr = internal_address + offset

        return filter_px_addr

    def calc_ofmap_elem_addr(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """
        Calculate OFMAP element address.

        Based on SCALE-Sim operand_matrix.py lines 247-255

        Args:
            i: Output pixel index (0 to ofmap_px_per_filt-1)
            j: Filter number (0 to num_filters-1)

        Returns:
            OFMAP address
        """
        offset = self.ofmap_offset
        num_filt = self.num_filters

        internal_address = num_filt * i + j
        ofmap_px_addr = internal_address + offset

        return ofmap_px_addr

    def create_ifmap_matrix(self) -> np.ndarray:
        """
        Create IFMAP operand matrix.

        Based on SCALE-Sim operand_matrix.py lines 161-192

        Returns:
            IFMAP matrix of shape (ofmap_px_per_filt, conv_window_size)
            Each element is a memory address
        """
        row_indices = np.arange(self.ofmap_px_per_filt)
        col_indices = np.arange(self.conv_window_size)

        # Create 2D index arrays
        i, j = np.meshgrid(row_indices, col_indices, indexing='ij')

        # Calculate addresses
        self.ifmap_matrix = self.calc_ifmap_elem_addr(i, j)

        return self.ifmap_matrix

    def create_filter_matrix(self) -> np.ndarray:
        """
        Create Filter operand matrix.

        Based on SCALE-Sim operand_matrix.py lines 257-368

        Returns:
            Filter matrix of shape (conv_window_size, num_filters)
            Each element is a memory address
        """
        row_indices = np.expand_dims(np.arange(self.conv_window_size), axis=1)
        col_indices = np.arange(self.num_filters)

        self.filter_matrix = self.calc_filter_elem_addr(row_indices, col_indices)

        return self.filter_matrix

    def create_ofmap_matrix(self) -> np.ndarray:
        """
        Create OFMAP operand matrix.

        Based on SCALE-Sim operand_matrix.py lines 224-244

        Returns:
            OFMAP matrix of shape (ofmap_px_per_filt, num_filters)
            Each element is a memory address
        """
        row_indices = np.expand_dims(np.arange(self.ofmap_px_per_filt), axis=1)
        col_indices = np.arange(self.num_filters)

        self.ofmap_matrix = self.calc_ofmap_elem_addr(row_indices, col_indices)

        return self.ofmap_matrix

    def create_all_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create all operand matrices (IFMAP, Filter, OFMAP).

        Returns:
            Tuple of (ifmap_matrix, filter_matrix, ofmap_matrix)
        """
        self.create_ifmap_matrix()
        self.create_filter_matrix()
        self.create_ofmap_matrix()

        return self.ifmap_matrix, self.filter_matrix, self.ofmap_matrix

    def get_layer_info(self) -> Dict:
        """
        Get layer information summary.

        Returns:
            Dictionary with layer parameters and matrix dimensions
        """
        return {
            'ifmap_shape': (self.ifmap_h, self.ifmap_w, self.num_channels),
            'filter_shape': (self.filter_h, self.filter_w, self.num_channels, self.num_filters),
            'ofmap_shape': (self.ofmap_h, self.ofmap_w, self.num_filters),
            'ifmap_matrix_shape': (self.ofmap_px_per_filt, self.conv_window_size),
            'filter_matrix_shape': (self.conv_window_size, self.num_filters),
            'ofmap_matrix_shape': (self.ofmap_px_per_filt, self.num_filters),
            'stride': self.stride,
            'padding': self.padding,
        }


if __name__ == '__main__':
    print("Testing Operand Matrix Generator (SCALE-Sim based)\n")

    # Test Case 1: Simple 4x4 conv
    print("="*60)
    print("Test Case 1: Conv2d(3→2, kernel=3x3, stride=1, padding=1)")
    print("Input: 4x4, Output: 4x4")
    print("="*60)

    gen = OperandMatrixGenerator()
    gen.set_layer_params(
        ifmap_h=4, ifmap_w=4,
        filter_h=3, filter_w=3,
        num_channels=3, num_filters=2,
        stride=1, padding=1
    )

    ifmap_mat, filter_mat, ofmap_mat = gen.create_all_matrices()

    info = gen.get_layer_info()
    print("\nLayer Info:")
    for key, val in info.items():
        print(f"  {key}: {val}")

    print(f"\nIFMAP matrix shape: {ifmap_mat.shape}")
    print(f"  Min address: {np.min(ifmap_mat[ifmap_mat >= 0])}")
    print(f"  Max address: {np.max(ifmap_mat[ifmap_mat >= 0])}")
    print(f"  Invalid accesses: {np.sum(ifmap_mat < 0)} / {ifmap_mat.size}")

    print(f"\nFilter matrix shape: {filter_mat.shape}")
    print(f"  Min address: {np.min(filter_mat)}")
    print(f"  Max address: {np.max(filter_mat)}")

    print(f"\nOFMAP matrix shape: {ofmap_mat.shape}")
    print(f"  Min address: {np.min(ofmap_mat)}")
    print(f"  Max address: {np.max(ofmap_mat)}")
    print(f"  Total unique outputs: {len(np.unique(ofmap_mat))}")

    # Test Case 2: GTSRB Conv1
    print("\n" + "="*60)
    print("Test Case 2: GTSRB Conv1")
    print("Conv2d(3→32, kernel=3x3, stride=1, padding=1)")
    print("Input: 32x32, Output: 32x32")
    print("="*60)

    gen2 = OperandMatrixGenerator()
    gen2.set_layer_params(
        ifmap_h=32, ifmap_w=32,
        filter_h=3, filter_w=3,
        num_channels=3, num_filters=32,
        stride=1, padding=1
    )

    ifmap_mat2, filter_mat2, ofmap_mat2 = gen2.create_all_matrices()

    info2 = gen2.get_layer_info()
    print("\nLayer Info:")
    for key, val in info2.items():
        print(f"  {key}: {val}")

    print(f"\nIFMAP matrix shape: {ifmap_mat2.shape}")
    print(f"  Address range: {np.min(ifmap_mat2[ifmap_mat2 >= 0])} - {np.max(ifmap_mat2[ifmap_mat2 >= 0])}")

    print(f"\nFilter matrix shape: {filter_mat2.shape}")
    print(f"  Address range: {np.min(filter_mat2)} - {np.max(filter_mat2)}")

    print(f"\nOFMAP matrix shape: {ofmap_mat2.shape}")
    print(f"  Address range: {np.min(ofmap_mat2)} - {np.max(ofmap_mat2)}")
    print(f"  Total outputs: {ofmap_mat2.size}")

    print("\n✓ Operand matrix generation test complete!")
