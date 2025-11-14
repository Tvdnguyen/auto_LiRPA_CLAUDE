"""
Main Fault Simulator
Integrates all components and provides user interface
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add parent paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))

from operand_matrix import OperandMatrix
from systolic_compute_os import SystolicComputeOS
from systolic_compute_ws import SystolicComputeWS
from systolic_compute_is import SystolicComputeIS
from fault_injector import FaultModel, FaultInjector

# Import TrafficSignNet
try:
    from traffic_sign_net import TrafficSignNet, TrafficSignNetSimple, TrafficSignNetNoDropout
except ImportError:
    print("Warning: Could not import TrafficSignNet. Layer config will be manual.")
    TrafficSignNet = None


class SystolicFaultSimulator:
    """Main simulator class"""

    def __init__(self, array_rows, array_cols, dataflow='OS'):
        """
        Args:
            array_rows: Number of PE rows
            array_cols: Number of PE columns
            dataflow: 'OS', 'WS', or 'IS' (only OS implemented for now)
        """
        self.arr_h = array_rows
        self.arr_w = array_cols
        self.dataflow = dataflow

        print(f"Initialized {array_rows}x{array_cols} Systolic Array")
        print(f"Dataflow: {dataflow}")

    def get_layer_config(self, model, layer_idx):
        """Get layer configuration from TrafficSignNet"""
        if TrafficSignNet is None:
            raise RuntimeError("TrafficSignNet not available")

        layers_info = model.get_layer_info()
        if layer_idx < 0 or layer_idx >= len(layers_info):
            raise ValueError(f"Invalid layer index: {layer_idx}")

        name, ltype, layer_obj, shape_str = layers_info[layer_idx]

        if ltype == 'Conv2d':
            # Get parameters
            in_channels = layer_obj.in_channels
            out_channels = layer_obj.out_channels
            kernel_size = layer_obj.kernel_size[0]
            stride = layer_obj.stride[0]
            padding = layer_obj.padding[0]

            # Determine input size based on layer position
            if layer_idx == 0:
                Hin, Win = 32, 32
            elif layer_idx in [1, 2]:
                Hin, Win = 16, 16
            elif layer_idx in [3, 4]:
                Hin, Win = 8, 8
            elif layer_idx == 5:
                Hin, Win = 4, 4
            else:
                Hin, Win = 32, 32

            # Compute output size
            Hout = (Hin + 2*padding - kernel_size) // stride + 1
            Wout = (Win + 2*padding - kernel_size) // stride + 1

            return {
                'type': 'Conv',
                'name': name,
                'input_shape': (in_channels, Hin, Win),
                'output_shape': (out_channels, Hout, Wout),
                'kernel_size': (kernel_size, kernel_size),
                'stride': stride,
                'padding': padding
            }

        elif ltype == 'Linear':
            return {
                'type': 'FC',
                'name': name,
                'input_size': layer_obj.in_features,
                'output_size': layer_obj.out_features
            }

        else:
            raise ValueError(f"Unsupported layer type: {ltype}")

    def simulate_layer(self, layer_config, faults):
        """
        Simulate a layer with faults

        Args:
            layer_config: Dict with layer parameters
            faults: List of FaultModel objects

        Returns:
            Dict with simulation results
        """
        print(f"\n{'='*80}")
        print(f"Simulating Layer: {layer_config.get('name', 'Unknown')}")
        print(f"{'='*80}")

        # Step 1: Generate operand matrices
        print("\n[1/5] Generating operand matrices...")
        op_gen = OperandMatrix(layer_config)
        operand_mats = op_gen.generate_matrices()

        dims = operand_mats['dimensions']
        print(f"  Dimensions: {dims}")

        # Step 2: Generate demand matrices
        print(f"\n[2/5] Generating demand matrices ({self.dataflow} dataflow)...")
        if self.dataflow == 'OS':
            compute_sim = SystolicComputeOS(self.arr_h, self.arr_w)
        elif self.dataflow == 'WS':
            compute_sim = SystolicComputeWS(self.arr_h, self.arr_w)
        elif self.dataflow == 'IS':
            compute_sim = SystolicComputeIS(self.arr_h, self.arr_w)
        else:
            raise NotImplementedError(f"Dataflow {self.dataflow} not implemented yet")

        demand_mats = compute_sim.generate_demand_matrices(operand_mats)
        print(f"  Total cycles: {demand_mats['total_cycles']}")
        print(f"  Number of tiles: {demand_mats['num_tiles']}")

        mapping_eff = compute_sim.compute_mapping_efficiency(operand_mats)
        print(f"  Mapping efficiency: {mapping_eff*100:.2f}%")

        # Step 3: Inject faults
        print(f"\n[3/5] Injecting {len(faults)} fault(s)...")

        # Display fault details
        for i, fault in enumerate(faults, 1):
            loc = fault.location
            pe_info = f"PE({loc['pe_row']},{loc['pe_col']})"
            comp_info = loc.get('component', 'entire_PE')
            timing_info = "permanent" if fault.timing['duration'] == float('inf') else \
                         f"cycles {fault.timing['start_cycle']}-{fault.timing['start_cycle']+fault.timing['duration']}"
            print(f"  [{i}] {pe_info} | {comp_info} | {fault.fault_type} | {timing_info}")

        injector = FaultInjector(faults)
        faulty_markers = injector.inject_into_demands(demand_mats)
        print(f"  Affected addresses: {len(faulty_markers['affected_addresses'])}")

        # Step 4: Trace fault propagation
        print("\n[4/5] Tracing fault propagation...")
        affected_outputs = injector.trace_fault_propagation(
            demand_mats, operand_mats, faulty_markers
        )
        print(f"  Affected outputs: {len(affected_outputs)}")

        # Step 5: Create fault mask
        print("\n[5/5] Creating fault mask...")
        fault_mask = injector.create_fault_mask(operand_mats, affected_outputs)

        stats = injector.compute_statistics(fault_mask, operand_mats)
        print(f"\nStatistics:")
        print(f"  Total outputs: {stats['total_outputs']}")
        print(f"  Affected outputs: {stats['affected_outputs']}")
        print(f"  Fault coverage: {stats['fault_coverage']*100:.2f}%")

        return {
            'layer_config': layer_config,
            'operand_matrices': operand_mats,
            'demand_matrices': demand_mats,
            'faulty_markers': faulty_markers,
            'fault_mask': fault_mask,
            'statistics': stats
        }

    def visualize_results(self, results, save_path=None):
        """Visualize fault impact on output tensor"""
        layer_config = results['layer_config']
        fault_mask = results['fault_mask']
        stats = results['statistics']

        layer_type = layer_config['type']
        layer_name = layer_config.get('name', 'Layer')

        if layer_type == 'Conv':
            self._visualize_conv(fault_mask, layer_name, stats, save_path)
        else:
            self._visualize_fc(fault_mask, layer_name, stats, save_path)

    def export_fault_report(self, results, save_path):
        """
        Export detailed fault impact report to text file

        Args:
            results: Simulation results from simulate_layer()
            save_path: Path to save the text report (e.g., 'fault_report.txt')
        """
        layer_config = results['layer_config']
        fault_mask = results['fault_mask']
        stats = results['statistics']
        layer_type = layer_config['type']
        layer_name = layer_config.get('name', 'Layer')

        with open(save_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(" " * 20 + "SYSTOLIC FAULT SIMULATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            # Layer information
            f.write(f"Layer: {layer_name} ({layer_type})\n")
            f.write(f"Dataflow: {self.dataflow}\n")
            f.write(f"Array Size: {self.arr_h}×{self.arr_w}\n")
            f.write("\n" + "-" * 80 + "\n\n")

            # Statistics
            f.write("STATISTICS:\n")
            f.write(f"  Total outputs: {stats['total_outputs']}\n")
            f.write(f"  Affected outputs: {stats['affected_outputs']}\n")
            f.write(f"  Fault coverage: {stats['fault_coverage']*100:.2f}%\n")
            f.write(f"  Number of faults: {stats['num_faults']}\n")
            f.write(f"  Affected addresses: {stats['affected_addresses']}\n")
            f.write("\n" + "-" * 80 + "\n\n")

            # Detailed affected regions
            if layer_type == 'Conv':
                self._write_conv_affected_regions(f, fault_mask, layer_config)
            else:
                self._write_fc_affected_regions(f, fault_mask, layer_config)

        print(f"Fault report exported to: {save_path}")

    def _write_conv_affected_regions(self, f, fault_mask, layer_config):
        """Write Conv layer affected regions to file"""
        num_outputs, num_filters = fault_mask.shape
        spatial_size = int(np.sqrt(num_outputs))

        f.write("AFFECTED REGIONS (Conv Layer):\n\n")

        # Reshape to spatial dimensions
        if spatial_size * spatial_size == num_outputs:
            fault_mask_spatial = fault_mask.reshape(spatial_size, spatial_size, num_filters)

            # Per-channel analysis
            f.write("Per-Channel Impact:\n")
            f.write("-" * 80 + "\n")
            for ch in range(num_filters):
                channel_mask = fault_mask_spatial[:, :, ch]
                affected_pixels = np.sum(channel_mask)

                if affected_pixels > 0:
                    f.write(f"\nChannel {ch}:\n")
                    f.write(f"  Affected pixels: {affected_pixels}/{num_outputs} ({affected_pixels/num_outputs*100:.2f}%)\n")

                    # Find bounding box
                    rows, cols = np.where(channel_mask)
                    if len(rows) > 0:
                        min_row, max_row = rows.min(), rows.max()
                        min_col, max_col = cols.min(), cols.max()
                        f.write(f"  Bounding box: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]\n")
                        f.write(f"  Box size: {max_row - min_row + 1}×{max_col - min_col + 1}\n")

                        # List affected coordinates
                        f.write(f"  Affected coordinates:\n")
                        affected_coords = list(zip(rows, cols))
                        # Group by rows for better readability
                        for row in range(min_row, max_row + 1):
                            row_coords = [c for r, c in affected_coords if r == row]
                            if row_coords:
                                f.write(f"    Row {row}: cols {sorted(row_coords)}\n")

            # Summary by spatial position across all channels
            f.write("\n" + "=" * 80 + "\n\n")
            f.write("Spatial Position Impact (across all channels):\n")
            f.write("-" * 80 + "\n")

            for row in range(spatial_size):
                for col in range(spatial_size):
                    affected_channels = np.sum(fault_mask_spatial[row, col, :])
                    if affected_channels > 0:
                        f.write(f"  Position ({row}, {col}): {affected_channels}/{num_filters} channels affected ({affected_channels/num_filters*100:.2f}%)\n")

        else:
            # Non-square spatial dimension
            f.write("Per-Channel Impact (flattened spatial):\n")
            f.write("-" * 80 + "\n")
            for ch in range(num_filters):
                channel_mask = fault_mask[:, ch]
                affected_pixels = np.sum(channel_mask)

                if affected_pixels > 0:
                    f.write(f"\nChannel {ch}:\n")
                    f.write(f"  Affected positions: {affected_pixels}/{num_outputs} ({affected_pixels/num_outputs*100:.2f}%)\n")

                    # List affected positions
                    affected_positions = np.where(channel_mask)[0]
                    f.write(f"  Affected indices: {list(affected_positions)}\n")

    def _write_fc_affected_regions(self, f, fault_mask, layer_config):
        """Write FC layer affected regions to file"""
        num_neurons = fault_mask.shape[0]

        f.write("AFFECTED NEURONS (FC Layer):\n\n")
        f.write(f"Total neurons: {num_neurons}\n")
        f.write(f"Affected neurons: {np.sum(fault_mask)}\n\n")

        f.write("Affected Neuron Indices:\n")
        f.write("-" * 80 + "\n")

        affected_indices = np.where(fault_mask.flatten())[0]

        # Group by ranges for compact representation
        if len(affected_indices) > 0:
            ranges = []
            start = affected_indices[0]
            end = start

            for idx in affected_indices[1:]:
                if idx == end + 1:
                    end = idx
                else:
                    ranges.append((start, end))
                    start = idx
                    end = idx
            ranges.append((start, end))

            for start, end in ranges:
                if start == end:
                    f.write(f"  Neuron {start}\n")
                else:
                    f.write(f"  Neurons {start}-{end} ({end - start + 1} neurons)\n")

    def _visualize_conv(self, fault_mask, layer_name, stats, save_path):
        """Visualize Conv layer fault mask"""
        num_outputs, num_filters = fault_mask.shape

        # Reshape fault mask to spatial dimensions
        # Assuming square spatial dimension
        spatial_size = int(np.sqrt(num_outputs))
        if spatial_size * spatial_size != num_outputs:
            print(f"Warning: Non-square spatial dimension {num_outputs}")
            spatial_size = num_outputs

        # Limit channels to display
        max_channels = 16
        display_channels = min(num_filters, max_channels)

        if num_filters > max_channels:
            print(f"Displaying first {max_channels} out of {num_filters} channels")

        # Create grid
        n_cols = min(4, display_channels)
        n_rows = int(np.ceil(display_channels / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), dpi=100)
        fig.suptitle(f'Fault Impact on {layer_name} Output\n'
                    f'({stats["affected_outputs"]}/{stats["total_outputs"]} outputs affected)',
                    fontsize=16, fontweight='bold')

        if display_channels == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        cmap = ListedColormap(['white', 'red'])

        for c in range(display_channels):
            row = c // n_cols
            col = c % n_cols
            ax = axes[row, col]

            # Get fault mask for this channel
            channel_mask = fault_mask[:, c].reshape(spatial_size, spatial_size)

            # Plot
            im = ax.imshow(channel_mask, cmap=cmap, vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')

            # Grid
            ax.set_xticks(np.arange(-0.5, spatial_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, spatial_size, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

            # Title
            faulty_count = np.sum(channel_mask)
            ax.set_title(f'Channel {c} ({faulty_count}/{num_outputs} faulty)', fontsize=10)

            # Axes
            ax.tick_params(labelsize=6)
            ax.set_xlabel('Width', fontsize=8)
            ax.set_ylabel('Height', fontsize=8)

        # Hide unused subplots
        for c in range(display_channels, n_rows * n_cols):
            row = c // n_cols
            col = c % n_cols
            axes[row, col].axis('off')

        # Colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['OK', 'Faulty'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")

        plt.show()

    def _visualize_fc(self, fault_mask, layer_name, stats, save_path):
        """Visualize FC layer fault mask"""
        num_outputs = fault_mask.shape[1]

        # Reshape for visualization
        width = int(np.ceil(np.sqrt(num_outputs)))
        height = int(np.ceil(num_outputs / width))

        padded_mask = np.zeros(width * height, dtype=bool)
        padded_mask[:num_outputs] = fault_mask[0, :]
        grid_mask = padded_mask.reshape(height, width)

        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = ListedColormap(['white', 'red'])

        im = ax.imshow(grid_mask, cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_title(f'Fault Impact on {layer_name} Output\n'
                    f'({stats["affected_outputs"]}/{stats["total_outputs"]} neurons affected)',
                    fontsize=14, fontweight='bold')
        ax.axis('off')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['OK', 'Faulty'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")

        plt.show()


def _get_component_choice():
    """Get user choice for faulty component"""
    component_map = {
        '1': 'MAC_unit',
        '2': 'accumulator_register',
        '3': 'input_register',
        '4': 'weight_register',
        '5': 'control_logic',
        '6': 'entire_PE'
    }

    while True:
        choice = input("  Select component (1-6) [default: 6]: ").strip() or '6'
        if choice in component_map:
            return component_map[choice]
        else:
            print("    Invalid choice. Please enter 1-6.")


def _get_fault_type_choice():
    """Get user choice for fault type"""
    print("\n  Fault Types:")
    print("    1. Stuck-at-0")
    print("    2. Stuck-at-1")
    print("    3. Bit-flip (random)")
    print("    4. Permanent (default)")

    fault_map = {
        '1': FaultModel.STUCK_AT_0,
        '2': FaultModel.STUCK_AT_1,
        '3': FaultModel.BIT_FLIP,
        '4': FaultModel.PERMANENT
    }

    while True:
        choice = input("  Select fault type (1-4) [default: 4]: ").strip() or '4'
        if choice in fault_map:
            return fault_map[choice]
        else:
            print("    Invalid choice. Please enter 1-4.")


def _compute_computation_cycles(dataflow, arr_h, arr_w, dims):
    """
    Compute the cycle range where computation actually happens

    Returns:
        (start_cycle, end_cycle, description)
    """
    Sr = dims['ofmap_pixels']
    T = dims['conv_window_size']

    if dataflow == 'OS':
        # OS: T cycles accumulation, then W-1 cycles drain
        comp_start = 0
        comp_end = T - 1
        desc = f"Accumulation: cycles 0-{T-1}, Output drain: cycles {T}-{T+arr_w-2}"
        return (comp_start, comp_end, desc)

    elif dataflow == 'WS':
        # WS: H cycles weight load, Sr cycles input stream, W-1 cycles drain
        weight_load = arr_h
        comp_start = weight_load
        comp_end = weight_load + Sr - 1
        desc = f"Weight load: cycles 0-{weight_load-1}, Input stream: cycles {comp_start}-{comp_end}, Drain: cycles {comp_end+1}-{comp_end+arr_w-1}"
        return (comp_start, comp_end, desc)

    elif dataflow == 'IS':
        # IS: H cycles input load, T cycles weight stream, H-1 cycles drain
        input_load = arr_h
        comp_start = input_load
        comp_end = input_load + T - 1
        desc = f"Input load: cycles 0-{input_load-1}, Weight stream: cycles {comp_start}-{comp_end}, Drain: cycles {comp_end+1}-{comp_end+arr_h-2}"
        return (comp_start, comp_end, desc)

    return (0, 0, "Unknown dataflow")


def _get_fault_timing(dataflow, arr_h, arr_w, dims):
    """Get fault timing configuration with guidance"""
    print("\n  Fault Duration:")
    print("    1. Permanent (active entire simulation)")
    print("    2. Transient (time-bounded)")

    while True:
        choice = input("  Select duration (1-2) [default: 1]: ").strip() or '1'
        if choice == '1':
            return {'start_cycle': 0, 'duration': float('inf')}
        elif choice == '2':
            # Show computation cycle ranges
            comp_start, comp_end, desc = _compute_computation_cycles(dataflow, arr_h, arr_w, dims)
            print(f"\n    💡 Computation Cycle Info ({dataflow} dataflow):")
            print(f"    {desc}")
            print(f"    ⚠️  Faults before cycle {comp_start} may have NO impact!")
            print(f"    ✅ Faults during cycles {comp_start}-{comp_end} WILL have impact\n")

            while True:
                try:
                    start = int(input("    Start cycle: "))
                    duration = int(input("    Duration (cycles): "))
                    if start >= 0 and duration > 0:
                        # Warn if fault is too early
                        if start + duration <= comp_start:
                            print(f"    ⚠️  WARNING: Fault ends at cycle {start+duration-1}, before computation starts at cycle {comp_start}")
                            print(f"    This fault will likely have NO impact on outputs!")
                            confirm = input("    Continue anyway? (y/n): ")
                            if confirm.lower() != 'y':
                                continue

                        return {'start_cycle': start, 'duration': duration}
                    else:
                        print("      Start must be ≥0, duration must be >0")
                except ValueError:
                    print("      Invalid input. Please enter integers.")
        else:
            print("    Invalid choice. Please enter 1 or 2.")


def interactive_simulation():
    """Interactive simulation interface"""
    print("="*80)
    print(" "*20 + "SYSTOLIC FAULT SIMULATOR")
    print(" "*15 + "(Based on SCALE-Sim Architecture)")
    print("="*80)

    # Step 1: Array configuration
    print("\n[Step 1] Configure Systolic Array")
    print("-" * 80)

    while True:
        try:
            size_input = input("Enter array size (e.g., '8' for 8x8): ").strip()
            if ',' in size_input:
                h, w = map(int, size_input.split(','))
            else:
                h = w = int(size_input)

            if h > 0 and w > 0:
                break
        except ValueError:
            print("Invalid input")

    # Select dataflow
    print("\nSelect dataflow:")
    print("  1. OS (Output Stationary) - Each PE accumulates one output")
    print("  2. WS (Weight Stationary) - Each PE holds one weight")
    print("  3. IS (Input Stationary) - Each PE holds one input")

    while True:
        dataflow_choice = input("Choose dataflow (1/2/3) [default: 1]: ").strip() or '1'
        if dataflow_choice == '1':
            dataflow = 'OS'
            break
        elif dataflow_choice == '2':
            dataflow = 'WS'
            break
        elif dataflow_choice == '3':
            dataflow = 'IS'
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    simulator = SystolicFaultSimulator(h, w, dataflow)

    # Step 2: Select layer
    print("\n[Step 2] Select Layer")
    print("-" * 80)

    model = TrafficSignNet(num_classes=43)
    layers_info = model.get_layer_info()

    print(f"{'Index':>5} | {'Name':^15} | {'Type':^10} | {'Shape':^20}")
    print("-"*80)
    for i, (name, ltype, _, shape) in enumerate(layers_info):
        print(f"{i:>5} | {name:^15} | {ltype:^10} | {shape:^20}")

    while True:
        try:
            layer_idx = int(input("\nSelect layer index: "))
            layer_config = simulator.get_layer_config(model, layer_idx)
            print(f"Selected: {layer_config['name']} ({layer_config['type']})")
            break
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")

    # Quick compute dimensions for timing guidance
    from operand_matrix import OperandMatrix
    op_gen_temp = OperandMatrix(layer_config)
    operand_mats_temp = op_gen_temp.generate_matrices()
    dims = operand_mats_temp['dimensions']

    # Step 3: Define faults
    print("\n[Step 3] Define Faults")
    print("-" * 80)
    print(f"PE Array: {h} rows × {w} columns")
    print("\nPE Components:")
    print("  1. MAC Unit (Multiply-Accumulate)")
    print("  2. Accumulator Register")
    print("  3. Input Register (IFMAP)")
    print("  4. Weight Register (FILTER)")
    print("  5. Control Logic")
    print("  6. Entire PE (all components)")

    faults = []

    while True:
        print("\n" + "-" * 80)
        pe_input = input("Enter faulty PE (row,col) or 'done': ").strip()

        if pe_input.lower() == 'done':
            break

        try:
            if '-' in pe_input:
                # Range mode
                print("\nRange mode: fault configuration will apply to all PEs in range")
                row_part, col_part = pe_input.split(',')

                if '-' in row_part:
                    r_start, r_end = map(int, row_part.split('-'))
                else:
                    r_start = r_end = int(row_part)

                if '-' in col_part:
                    c_start, c_end = map(int, col_part.split('-'))
                else:
                    c_start = c_end = int(col_part)

                # Get fault details once for all PEs in range
                component = _get_component_choice()
                fault_type = _get_fault_type_choice()
                fault_timing = _get_fault_timing(dataflow, h, w, dims)

                # Apply to all PEs in range
                num_added = 0
                for r in range(r_start, r_end + 1):
                    for c in range(c_start, c_end + 1):
                        if 0 <= r < h and 0 <= c < w:
                            fault = FaultModel(
                                fault_type=fault_type,
                                fault_location={
                                    'pe_row': r,
                                    'pe_col': c,
                                    'component': component
                                },
                                fault_timing=fault_timing
                            )
                            faults.append(fault)
                            num_added += 1

                print(f"  → Added {num_added} fault(s)")

            else:
                # Single PE mode
                r, c = map(int, pe_input.split(','))
                if not (0 <= r < h and 0 <= c < w):
                    print(f"  Error: PE ({r}, {c}) out of bounds!")
                    continue

                # Get fault details
                component = _get_component_choice()
                fault_type = _get_fault_type_choice()
                fault_timing = _get_fault_timing(dataflow, h, w, dims)

                fault = FaultModel(
                    fault_type=fault_type,
                    fault_location={
                        'pe_row': r,
                        'pe_col': c,
                        'component': component
                    },
                    fault_timing=fault_timing
                )
                faults.append(fault)
                print(f"  → Added {fault_type} fault at PE ({r}, {c}), {component}")

        except ValueError as e:
            print(f"  Invalid format: {e}")

    print(f"\n{'='*80}")
    print(f"Total faults defined: {len(faults)}")
    print(f"{'='*80}")

    # Step 4: Run simulation
    print("\n[Step 4] Running Simulation...")
    print("="*80)

    results = simulator.simulate_layer(layer_config, faults)

    # Step 5: Visualize
    print("\n[Step 5] Generating Visualization...")
    print("="*80)

    image_path = f"fault_impact_{layer_config['name']}.png"
    simulator.visualize_results(results, image_path)

    # Step 6: Export detailed text report
    print("\n[Step 6] Exporting Detailed Report...")
    print("="*80)

    report_path = f"fault_report_{layer_config['name']}.txt"
    simulator.export_fault_report(results, report_path)

    print("\n" + "="*80)
    print("Simulation completed!")
    print(f"  Visualization: {image_path}")
    print(f"  Detailed Report: {report_path}")
    print("="*80)


if __name__ == '__main__':
    interactive_simulation()
