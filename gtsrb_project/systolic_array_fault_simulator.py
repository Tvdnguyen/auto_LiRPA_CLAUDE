#!/usr/bin/env python3
"""
Systolic Array Fault Simulator

Mô phỏng ảnh hưởng của lỗi trên Systolic Array đến output tensor của DNN layers.

Chương trình này:
1. Cho phép chọn kích thước Systolic Array (e.g., 8x8 PE)
2. Chọn dataflow (Input Stationary, Output Stationary, Weight Stationary)
3. Chọn vùng PE bị lỗi
4. Chọn layer từ TrafficSignNet để simulate
5. Hiển thị vùng tensor bị ảnh hưởng (text + visualization)

References:
- "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs" (2017)
- "SCALE-Sim: Systolic CNN Accelerator Simulator" (2020)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traffic_sign_net import TrafficSignNet, TrafficSignNetSimple


class SystolicArraySimulator:
    """
    Simulate Systolic Array computation and fault propagation
    """

    def __init__(self, array_height, array_width, dataflow='OS'):
        """
        Initialize systolic array simulator

        Args:
            array_height: Number of PE rows
            array_width: Number of PE columns
            dataflow: 'IS' (Input Stationary), 'OS' (Output Stationary), 'WS' (Weight Stationary)
        """
        self.H = array_height  # PE rows
        self.W = array_width   # PE columns
        self.dataflow = dataflow.upper()

        if self.dataflow not in ['IS', 'OS', 'WS']:
            raise ValueError(f"Invalid dataflow: {dataflow}. Must be IS, OS, or WS")

        print(f"Initialized {self.H}×{self.W} Systolic Array with {self.dataflow} dataflow")

    def map_conv_layer(self, layer_config):
        """
        Map convolutional layer computation to systolic array

        Args:
            layer_config: Dict with keys:
                - input_shape: (Cin, Hin, Win)
                - output_shape: (Cout, Hout, Wout)
                - kernel_size: (K, K)
                - stride: int
                - padding: int

        Returns:
            mapping: Information about how computation maps to SA
        """
        Cin, Hin, Win = layer_config['input_shape']
        Cout, Hout, Wout = layer_config['output_shape']
        K, _ = layer_config['kernel_size']

        if self.dataflow == 'OS':
            # Output Stationary: Each PE computes one output element
            # PE accumulates partial sums for that output
            return self._map_conv_os(layer_config)

        elif self.dataflow == 'WS':
            # Weight Stationary: Each PE holds one weight
            # Inputs flow through, outputs accumulate
            return self._map_conv_ws(layer_config)

        elif self.dataflow == 'IS':
            # Input Stationary: Each PE holds one input activation
            # Weights flow through, outputs accumulate
            return self._map_conv_is(layer_config)

    def _map_conv_os(self, layer_config):
        """
        Output Stationary mapping for Conv layer

        Each PE computes one output activation.
        PE receives inputs and weights, accumulates partial sums.

        Tiling strategy:
        - Map Hout×Wout output spatial dimension to PE array
        - Loop over output channels Cout
        - Loop over input channels Cin and kernel K×K
        """
        Cin, Hin, Win = layer_config['input_shape']
        Cout, Hout, Wout = layer_config['output_shape']
        K, _ = layer_config['kernel_size']

        # Number of output activations per output channel
        n_outputs = Hout * Wout

        # Tile output spatial dimension to PE array
        tiles_h = int(np.ceil(Hout / self.H))
        tiles_w = int(np.ceil(Wout / self.W))

        mapping = {
            'dataflow': 'OS',
            'layer_type': 'Conv',
            'output_shape': (Cout, Hout, Wout),
            'tiles_h': tiles_h,
            'tiles_w': tiles_w,
            'pe_to_output': {}  # Maps (pe_row, pe_col) → list of (cout, hout, wout)
        }

        # Map each output position to PE
        for cout in range(Cout):
            for hout in range(Hout):
                for wout in range(Wout):
                    # Which tile does this output belong to?
                    tile_h = hout // self.H
                    tile_w = wout // self.W

                    # Position within tile
                    pe_row = hout % self.H
                    pe_col = wout % self.W

                    key = (pe_row, pe_col)
                    if key not in mapping['pe_to_output']:
                        mapping['pe_to_output'][key] = []

                    mapping['pe_to_output'][key].append((cout, hout, wout))

        return mapping

    def _map_conv_ws(self, layer_config):
        """
        Weight Stationary mapping for Conv layer

        Each PE holds one weight value.
        Inputs broadcast to multiple PEs.
        Each PE computes partial products for multiple outputs.

        Tiling strategy:
        - Map Cout×Cin to PE array (each PE holds one weight)
        - Spatial dimensions (Hout×Wout) are temporal
        """
        Cin, Hin, Win = layer_config['input_shape']
        Cout, Hout, Wout = layer_config['output_shape']
        K, _ = layer_config['kernel_size']

        # Tile output channels and input channels
        tiles_cout = int(np.ceil(Cout / self.H))
        tiles_cin = int(np.ceil(Cin / self.W))

        mapping = {
            'dataflow': 'WS',
            'layer_type': 'Conv',
            'output_shape': (Cout, Hout, Wout),
            'tiles_cout': tiles_cout,
            'tiles_cin': tiles_cin,
            'pe_to_output': {}
        }

        # Each PE is responsible for computing partial sums for specific (Cout, Cin) pair
        # across all spatial locations (Hout, Wout)
        for cout in range(Cout):
            for cin in range(Cin):
                pe_row = cout % self.H
                pe_col = cin % self.W

                key = (pe_row, pe_col)
                if key not in mapping['pe_to_output']:
                    mapping['pe_to_output'][key] = []

                # This PE affects all spatial outputs for this (cout, cin) pair
                for hout in range(Hout):
                    for wout in range(Wout):
                        mapping['pe_to_output'][key].append((cout, hout, wout))

        return mapping

    def _map_conv_is(self, layer_config):
        """
        Input Stationary mapping for Conv layer

        Each PE holds one input activation.
        Weights flow through PEs.
        Each PE computes contributions to multiple outputs.

        Tiling strategy:
        - Map input spatial dimension (Hin×Win) to PE array
        - Temporal: loop over Cout, Cin
        """
        Cin, Hin, Win = layer_config['input_shape']
        Cout, Hout, Wout = layer_config['output_shape']
        K, _ = layer_config['kernel_size']

        tiles_h = int(np.ceil(Hin / self.H))
        tiles_w = int(np.ceil(Win / self.W))

        mapping = {
            'dataflow': 'IS',
            'layer_type': 'Conv',
            'output_shape': (Cout, Hout, Wout),
            'tiles_h': tiles_h,
            'tiles_w': tiles_w,
            'pe_to_output': {}
        }

        # Each PE holds one input activation
        # It contributes to multiple outputs depending on kernel overlap
        stride = layer_config['stride']
        padding = layer_config['padding']

        for hin in range(Hin):
            for win in range(Win):
                pe_row = hin % self.H
                pe_col = win % self.W

                key = (pe_row, pe_col)
                if key not in mapping['pe_to_output']:
                    mapping['pe_to_output'][key] = []

                # Which outputs does this input contribute to?
                # Depends on kernel size and stride
                for cout in range(Cout):
                    for kh in range(K):
                        for kw in range(K):
                            # Output position that uses this input at kernel position (kh, kw)
                            hout = (hin + padding - kh) / stride
                            wout = (win + padding - kw) / stride

                            if hout >= 0 and hout < Hout and wout >= 0 and wout < Wout:
                                if hout == int(hout) and wout == int(wout):
                                    mapping['pe_to_output'][key].append((cout, int(hout), int(wout)))

        return mapping

    def map_fc_layer(self, layer_config):
        """
        Map fully connected layer computation to systolic array

        Args:
            layer_config: Dict with keys:
                - input_size: int
                - output_size: int

        Returns:
            mapping: Information about how computation maps to SA
        """
        input_size = layer_config['input_size']
        output_size = layer_config['output_size']

        if self.dataflow == 'OS':
            return self._map_fc_os(layer_config)
        elif self.dataflow == 'WS':
            return self._map_fc_ws(layer_config)
        elif self.dataflow == 'IS':
            return self._map_fc_is(layer_config)

    def _map_fc_os(self, layer_config):
        """
        Output Stationary mapping for FC layer

        Each PE computes one output neuron.
        """
        output_size = layer_config['output_size']

        mapping = {
            'dataflow': 'OS',
            'layer_type': 'FC',
            'output_shape': (output_size,),
            'pe_to_output': {}
        }

        # Map each output neuron to a PE
        for out_idx in range(output_size):
            pe_row = out_idx // self.W
            pe_col = out_idx % self.W

            if pe_row < self.H:  # Only if fits in array
                key = (pe_row, pe_col)
                if key not in mapping['pe_to_output']:
                    mapping['pe_to_output'][key] = []
                mapping['pe_to_output'][key].append((out_idx,))

        return mapping

    def _map_fc_ws(self, layer_config):
        """Weight Stationary mapping for FC layer"""
        input_size = layer_config['input_size']
        output_size = layer_config['output_size']

        mapping = {
            'dataflow': 'WS',
            'layer_type': 'FC',
            'output_shape': (output_size,),
            'pe_to_output': {}
        }

        # Each PE holds one weight W[out, in]
        # Contributes to partial sum of one output
        for out_idx in range(output_size):
            for in_idx in range(input_size):
                pe_row = out_idx % self.H
                pe_col = in_idx % self.W

                key = (pe_row, pe_col)
                if key not in mapping['pe_to_output']:
                    mapping['pe_to_output'][key] = []

                # This weight contributes to output[out_idx]
                if (out_idx,) not in mapping['pe_to_output'][key]:
                    mapping['pe_to_output'][key].append((out_idx,))

        return mapping

    def _map_fc_is(self, layer_config):
        """Input Stationary mapping for FC layer"""
        input_size = layer_config['input_size']
        output_size = layer_config['output_size']

        mapping = {
            'dataflow': 'IS',
            'layer_type': 'FC',
            'output_shape': (output_size,),
            'pe_to_output': {}
        }

        # Each PE holds one input activation
        # Contributes to all outputs
        for in_idx in range(input_size):
            pe_row = in_idx // self.W
            pe_col = in_idx % self.W

            if pe_row < self.H:
                key = (pe_row, pe_col)
                mapping['pe_to_output'][key] = [(out_idx,) for out_idx in range(output_size)]

        return mapping

    def get_faulty_outputs(self, mapping, faulty_pe_list):
        """
        Get list of output tensor elements affected by faulty PEs

        Args:
            mapping: Layer mapping from map_conv_layer or map_fc_layer
            faulty_pe_list: List of (pe_row, pe_col) tuples

        Returns:
            faulty_outputs: Set of tuples indicating affected output positions
        """
        faulty_outputs = set()

        for pe_row, pe_col in faulty_pe_list:
            key = (pe_row, pe_col)
            if key in mapping['pe_to_output']:
                outputs = mapping['pe_to_output'][key]
                for out_pos in outputs:
                    faulty_outputs.add(out_pos)

        return faulty_outputs

    def create_fault_mask(self, mapping, faulty_outputs):
        """
        Create boolean mask indicating which output elements are faulty

        Args:
            mapping: Layer mapping
            faulty_outputs: Set of faulty output positions

        Returns:
            mask: Boolean numpy array with same shape as output
        """
        output_shape = mapping['output_shape']
        layer_type = mapping['layer_type']

        if layer_type == 'Conv':
            Cout, Hout, Wout = output_shape
            mask = np.zeros((Cout, Hout, Wout), dtype=bool)

            for cout, hout, wout in faulty_outputs:
                if 0 <= cout < Cout and 0 <= hout < Hout and 0 <= wout < Wout:
                    mask[cout, hout, wout] = True

        elif layer_type == 'FC':
            output_size = output_shape[0]
            mask = np.zeros(output_size, dtype=bool)

            for (out_idx,) in faulty_outputs:
                if 0 <= out_idx < output_size:
                    mask[out_idx] = True

        return mask


class FaultVisualizer:
    """Visualize fault propagation in output tensors"""

    @staticmethod
    def print_affected_regions(mask, layer_name, layer_type):
        """
        Print text description of affected regions

        Args:
            mask: Boolean numpy array
            layer_name: Name of layer
            layer_type: 'Conv' or 'FC'
        """
        print("\n" + "="*80)
        print(f"AFFECTED REGIONS IN {layer_name} OUTPUT")
        print("="*80)

        if layer_type == 'Conv':
            Cout, Hout, Wout = mask.shape
            total_elements = Cout * Hout * Wout
            faulty_elements = np.sum(mask)

            print(f"\nOutput shape: (Channels={Cout}, Height={Hout}, Width={Wout})")
            print(f"Total elements: {total_elements}")
            print(f"Faulty elements: {faulty_elements} ({100*faulty_elements/total_elements:.2f}%)")

            # List affected regions by channel
            print(f"\nAffected regions by channel:")
            for c in range(Cout):
                channel_mask = mask[c, :, :]
                if np.any(channel_mask):
                    faulty_count = np.sum(channel_mask)
                    print(f"\n  Channel {c}: {faulty_count}/{Hout*Wout} elements faulty")

                    # Find bounding box
                    rows, cols = np.where(channel_mask)
                    if len(rows) > 0:
                        h_min, h_max = rows.min(), rows.max()
                        w_min, w_max = cols.min(), cols.max()
                        print(f"    Bounding box: H[{h_min}:{h_max+1}], W[{w_min}:{w_max+1}]")

                        # List specific positions if not too many
                        if faulty_count <= 20:
                            positions = [(h, w) for h, w in zip(rows, cols)]
                            print(f"    Positions: {positions}")

        elif layer_type == 'FC':
            output_size = mask.shape[0]
            faulty_elements = np.sum(mask)

            print(f"\nOutput shape: (Neurons={output_size})")
            print(f"Total elements: {output_size}")
            print(f"Faulty elements: {faulty_elements} ({100*faulty_elements/output_size:.2f}%)")

            # List faulty neurons
            faulty_indices = np.where(mask)[0]
            print(f"\nFaulty neuron indices:")
            if len(faulty_indices) <= 50:
                print(f"  {faulty_indices.tolist()}")
            else:
                print(f"  {faulty_indices[:25].tolist()} ... {faulty_indices[-25:].tolist()}")

        print("="*80)

    @staticmethod
    def visualize_conv_fault_mask(mask, layer_name, save_path=None, max_channels=16):
        """
        Visualize Conv layer fault mask using matplotlib

        Args:
            mask: Boolean array (Cout, Hout, Wout)
            layer_name: Name of layer
            save_path: Optional path to save figure
            max_channels: Maximum number of channels to display (default: 16)
        """
        Cout, Hout, Wout = mask.shape

        print(f"\n[DEBUG] Visualization:")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Channels: {Cout}, Height: {Hout}, Width: {Wout}")
        print(f"  Total faulty elements: {np.sum(mask)}/{mask.size}")

        # Limit number of channels to display if too many
        if Cout > max_channels:
            print(f"  Note: Displaying first {max_channels} channels (out of {Cout} total)")
            print(f"        Pattern is identical across all channels for OS dataflow")
            display_channels = max_channels
        else:
            display_channels = Cout

        # Determine grid layout for channels
        n_cols = min(4, display_channels)  # Max 4 columns for readability
        n_rows = int(np.ceil(display_channels / n_cols))

        # Increase figure size for better visibility (5 inches per channel)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), dpi=100)
        fig.suptitle(f'Fault Propagation in {layer_name} Output ({Hout}×{Wout} per channel)\n(Red=Faulty, White=OK)',
                     fontsize=16, fontweight='bold')

        if display_channels == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        # Colormap: white for OK, red for faulty
        cmap = ListedColormap(['white', 'red'])

        for c in range(display_channels):
            row = c // n_cols
            col = c % n_cols
            ax = axes[row, col]

            # Plot mask for this channel
            channel_mask = mask[c, :, :]

            # Use imshow with proper settings to show all pixels
            im = ax.imshow(channel_mask, cmap=cmap, vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')

            # Show grid lines to visualize each pixel
            ax.set_xticks(np.arange(-0.5, Wout, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, Hout, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

            # Add title with channel info and fault count
            faulty_count = np.sum(channel_mask)
            ax.set_title(f'Ch {c} ({faulty_count}/{Hout*Wout} faulty)', fontsize=10)

            # Show axes to see pixel coordinates
            ax.tick_params(labelsize=6)
            ax.set_xlabel('Width', fontsize=8)
            ax.set_ylabel('Height', fontsize=8)

        # Hide unused subplots
        for c in range(display_channels, n_rows * n_cols):
            row = c // n_cols
            col = c % n_cols
            axes[row, col].axis('off')

        # Add colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['OK', 'Faulty'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")

        plt.show()

    @staticmethod
    def visualize_fc_fault_mask(mask, layer_name, save_path=None):
        """
        Visualize FC layer fault mask

        Args:
            mask: Boolean array (output_size,)
            layer_name: Name of layer
            save_path: Optional path to save figure
        """
        output_size = mask.shape[0]

        # Reshape for visualization
        # Try to make it roughly square
        width = int(np.ceil(np.sqrt(output_size)))
        height = int(np.ceil(output_size / width))

        # Pad mask to fit grid
        padded_mask = np.zeros(width * height, dtype=bool)
        padded_mask[:output_size] = mask
        grid_mask = padded_mask.reshape(height, width)

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = ListedColormap(['white', 'red'])

        im = ax.imshow(grid_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Fault Propagation in {layer_name} Output\n(Red=Faulty, White=OK)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel(f'Total neurons: {output_size}, Faulty: {np.sum(mask)}')
        ax.axis('off')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['OK', 'Faulty'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")

        plt.show()


def get_layer_config_from_model(model, layer_idx):
    """
    Get layer configuration from TrafficSignNet

    Args:
        model: TrafficSignNet or TrafficSignNetSimple
        layer_idx: Layer index

    Returns:
        layer_config: Dict with layer parameters
        layer_name: Name of layer
        layer_type: 'Conv' or 'FC'
    """
    layers_info = model.get_layer_info()

    if layer_idx < 0 or layer_idx >= len(layers_info):
        raise ValueError(f"Invalid layer index: {layer_idx}")

    name, ltype, layer_obj, shape_str = layers_info[layer_idx]

    if ltype == 'Conv2d':
        # Extract parameters
        in_channels = layer_obj.in_channels
        out_channels = layer_obj.out_channels
        kernel_size = layer_obj.kernel_size[0]
        stride = layer_obj.stride[0]
        padding = layer_obj.padding[0]

        # For TrafficSignNet, we need to know input spatial size
        # Depends on layer position in network
        if layer_idx == 0:  # conv1
            Hin, Win = 32, 32
        elif layer_idx in [1, 2]:  # conv2, conv3
            Hin, Win = 16, 16
        elif layer_idx in [3, 4]:  # conv4, conv5
            Hin, Win = 8, 8
        elif layer_idx == 5:  # conv6
            Hin, Win = 4, 4
        else:
            Hin, Win = 32, 32  # default

        # Compute output size
        Hout = (Hin + 2*padding - kernel_size) // stride + 1
        Wout = (Win + 2*padding - kernel_size) // stride + 1

        layer_config = {
            'input_shape': (in_channels, Hin, Win),
            'output_shape': (out_channels, Hout, Wout),
            'kernel_size': (kernel_size, kernel_size),
            'stride': stride,
            'padding': padding
        }

        return layer_config, name, 'Conv'

    elif ltype == 'Linear':
        in_features = layer_obj.in_features
        out_features = layer_obj.out_features

        layer_config = {
            'input_size': in_features,
            'output_size': out_features
        }

        return layer_config, name, 'FC'

    else:
        raise ValueError(f"Unsupported layer type: {ltype}")


def interactive_simulation():
    """Interactive fault simulation session"""
    print("="*80)
    print(" "*20 + "SYSTOLIC ARRAY FAULT SIMULATOR")
    print("="*80)

    # Step 1: Configure Systolic Array
    print("\n[Step 1] Configure Systolic Array")
    print("-" * 80)

    while True:
        try:
            array_size = input("Enter array size (e.g., '8' for 8x8, or '16,8' for 16x8): ").strip()
            if ',' in array_size:
                h, w = map(int, array_size.split(','))
            else:
                h = w = int(array_size)

            if h > 0 and w > 0:
                break
            else:
                print("Size must be positive")
        except ValueError:
            print("Invalid input")

    while True:
        dataflow = input("Select dataflow (IS/OS/WS): ").strip().upper()
        if dataflow in ['IS', 'OS', 'WS']:
            break
        else:
            print("Invalid dataflow. Choose IS, OS, or WS")

    simulator = SystolicArraySimulator(h, w, dataflow)

    # Step 2: Select Layer
    print("\n[Step 2] Select Layer from TrafficSignNet")
    print("-" * 80)

    model = TrafficSignNet(num_classes=43)
    layers_info = model.get_layer_info()

    print(f"{'Index':>5} | {'Layer Name':^15} | {'Type':^10} | {'Output Shape':^20}")
    print("-"*80)
    for i, (name, ltype, layer, shape) in enumerate(layers_info):
        print(f"{i:>5} | {name:^15} | {ltype:^10} | {shape:^20}")

    while True:
        try:
            layer_idx = int(input("\nSelect layer index: "))
            layer_config, layer_name, layer_type = get_layer_config_from_model(model, layer_idx)
            print(f"Selected: {layer_name} ({layer_type})")
            break
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")

    # Step 3: Select Faulty PEs
    print("\n[Step 3] Select Faulty PE Region")
    print("-" * 80)
    print(f"PE Array: {h} rows × {w} columns")
    print("Enter PE coordinates (row,col) one per line. Enter 'done' when finished.")
    print("Or enter range like '0-2,0-3' for PEs in rows 0-2, cols 0-3")

    faulty_pe_list = []

    while True:
        pe_input = input("Faulty PE (or 'done'): ").strip()

        if pe_input.lower() == 'done':
            break

        try:
            if '-' in pe_input:
                # Range input
                row_part, col_part = pe_input.split(',')

                if '-' in row_part:
                    row_start, row_end = map(int, row_part.split('-'))
                else:
                    row_start = row_end = int(row_part)

                if '-' in col_part:
                    col_start, col_end = map(int, col_part.split('-'))
                else:
                    col_start = col_end = int(col_part)

                for r in range(row_start, row_end + 1):
                    for c in range(col_start, col_end + 1):
                        if 0 <= r < h and 0 <= c < w:
                            faulty_pe_list.append((r, c))

                print(f"  Added PEs: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")

            else:
                # Single PE
                r, c = map(int, pe_input.split(','))
                if 0 <= r < h and 0 <= c < w:
                    faulty_pe_list.append((r, c))
                    print(f"  Added PE ({r}, {c})")
                else:
                    print(f"  PE out of range!")

        except ValueError:
            print("Invalid format. Use 'row,col' or 'row_start-row_end,col_start-col_end'")

    faulty_pe_list = list(set(faulty_pe_list))  # Remove duplicates
    print(f"\nTotal faulty PEs: {len(faulty_pe_list)}")

    # Step 4: Simulate and Analyze
    print("\n[Step 4] Simulating Fault Propagation...")
    print("-" * 80)

    # Map layer to SA
    if layer_type == 'Conv':
        mapping = simulator.map_conv_layer(layer_config)
    else:  # FC
        mapping = simulator.map_fc_layer(layer_config)

    # Get faulty outputs
    faulty_outputs = simulator.get_faulty_outputs(mapping, faulty_pe_list)

    # Create fault mask
    fault_mask = simulator.create_fault_mask(mapping, faulty_outputs)

    # Step 5: Display Results
    print("\n[Step 5] Results")
    print("="*80)

    # Text output
    FaultVisualizer.print_affected_regions(fault_mask, layer_name, layer_type)

    # Visualization
    print("\n" + "="*80)
    print("Generating visualization...")
    print("="*80)

    save_path = f"fault_visualization_{layer_name}.png"

    if layer_type == 'Conv':
        FaultVisualizer.visualize_conv_fault_mask(fault_mask, layer_name, save_path)
    else:  # FC
        FaultVisualizer.visualize_fc_fault_mask(fault_mask, layer_name, save_path)

    print("\nSimulation completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Systolic Array Fault Simulator for DNN Layers'
    )
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')

    args = parser.parse_args()

    if args.interactive or len(sys.argv) == 1:
        interactive_simulation()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
