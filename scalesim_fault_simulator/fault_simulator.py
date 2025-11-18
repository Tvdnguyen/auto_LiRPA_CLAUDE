"""
Main Fault Simulator (SCALE-Sim Based)
Interactive interface for systolic array fault simulation

Based on SCALE-Sim architecture with component-aware fault injection
"""

import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import our SCALE-Sim based modules
from scalesim_operand_matrix import OperandMatrixGenerator
from scalesim_dataflow_os import SystolicComputeOS
from scalesim_dataflow_ws import SystolicComputeWS
from scalesim_dataflow_is import SystolicComputeIS
from scalesim_fault_injector import ScaleSimFaultInjector, FaultModel
from fault_visualizer import FaultVisualizer

# Import TrafficSignNet for layer info (both old and new models)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_small_tensor_project'))

try:
    from traffic_sign_net import TrafficSignNet
except ImportError:
    print("Warning: Could not import TrafficSignNet (old model)")
    TrafficSignNet = None

try:
    from traffic_sign_net_small import TrafficSignNetSmall
except ImportError:
    print("Warning: Could not import TrafficSignNetSmall (new model)")
    TrafficSignNetSmall = None


class ScaleSimFaultSimulator:
    """Main fault simulator based on SCALE-Sim architecture."""

    def __init__(self, array_rows: int, array_cols: int, dataflow: str):
        """
        Initialize simulator.

        Args:
            array_rows: Number of PE rows
            array_cols: Number of PE columns
            dataflow: 'OS', 'WS', or 'IS'
        """
        self.arr_h = array_rows
        self.arr_w = array_cols
        self.dataflow = dataflow

        print(f"Initialized {array_rows}×{array_cols} Systolic Array")
        print(f"Dataflow: {dataflow}")

    def get_layer_config(self, model, layer_idx: int) -> Dict:
        """Extract layer configuration from model (works with both old and new models)."""
        layers_info = model.get_layer_info()
        if layer_idx < 0 or layer_idx >= len(layers_info):
            raise ValueError(f"Invalid layer index: {layer_idx}")

        name, ltype, layer_obj, shape_str = layers_info[layer_idx]

        # Determine input/output size dynamically by running a forward pass
        # This works for both TrafficSignNet and TrafficSignNetSmall
        import torch
        model.eval()

        # Start with 32x32 input (GTSRB standard)
        dummy_input = torch.randn(1, 3, 32, 32)

        # Get intermediate outputs up to the selected layer
        with torch.no_grad():
            x = dummy_input

            # Forward through model until we reach the target layer
            for i, (ln, lt, lobj, _) in enumerate(layers_info):
                if i == layer_idx:
                    # This is our target layer, x is the input
                    x_input = x

                    # Flatten if necessary for Linear layer
                    if lt == 'Linear' and len(x.shape) > 2:
                        x = x.view(x.size(0), -1)

                    # Compute output after this layer
                    x = lobj(x)
                    break
                else:
                    # Process layers before target layer
                    if lt in ['Conv2d', 'Conv']:
                        x = lobj(x)
                    elif lt == 'ReLU':
                        x = lobj(x)
                    elif lt == 'MaxPool2d':
                        x = lobj(x)
                    elif lt == 'Dropout':
                        # Skip dropout in eval mode
                        pass
                    elif lt == 'Linear':
                        # Flatten before FC layers
                        x = x.view(x.size(0), -1)
                        x = lobj(x)
                    else:
                        # Try to apply the layer
                        try:
                            x = lobj(x)
                        except:
                            pass

        if ltype in ['Conv2d', 'Conv']:
            in_channels = layer_obj.in_channels
            out_channels = layer_obj.out_channels
            kernel_size = layer_obj.kernel_size[0]
            stride = layer_obj.stride[0]
            padding = layer_obj.padding[0]

            Hin, Win = x_input.shape[2], x_input.shape[3]
            Hout, Wout = x.shape[2], x.shape[3]

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
            in_features = layer_obj.in_features
            out_features = layer_obj.out_features

            return {
                'type': 'Linear',
                'name': name,
                'input_shape': (in_features,),
                'output_shape': (out_features,),
                'kernel_size': (1, 1),  # Dummy for compatibility
                'stride': 1,
                'padding': 0
            }
        else:
            raise ValueError(f"Unsupported layer type: {ltype}. Only Conv2d/Conv/Linear layers are supported.")

    def simulate_layer(self, layer_config: Dict, faults: List[FaultModel]) -> Dict:
        """
        Run fault simulation for a layer.

        Args:
            layer_config: Layer configuration
            faults: List of fault models

        Returns:
            Simulation results dictionary
        """
        print(f"\n{'='*80}")
        print(f"Simulating Layer: {layer_config.get('name', 'Unknown')}")
        print(f"{'='*80}")

        # Step 1: Generate operand matrices
        print("\n[1/5] Generating operand matrices...")

        layer_type = layer_config['type']

        if layer_type == 'Linear':
            # For Linear layers, treat as 1x1 spatial operation
            # input_shape = (in_features,), output_shape = (out_features,)
            in_features = layer_config['input_shape'][0]
            out_features = layer_config['output_shape'][0]

            # Model as: in_features channels × 1×1 spatial -> out_features channels × 1×1 spatial
            in_ch = in_features
            Hin, Win = 1, 1
            out_ch = out_features
            Hout, Wout = 1, 1
            kernel_h, kernel_w = 1, 1

            print(f"  Linear layer: {in_features} -> {out_features}")
            print(f"  (Modeled as {in_ch} channels × 1×1 spatial)")
        else:
            # Conv layer
            in_ch, Hin, Win = layer_config['input_shape']
            out_ch, Hout, Wout = layer_config['output_shape']
            kernel_h, kernel_w = layer_config['kernel_size']

        gen = OperandMatrixGenerator()
        gen.set_layer_params(
            ifmap_h=Hin, ifmap_w=Win,
            filter_h=kernel_h, filter_w=kernel_w,
            num_channels=in_ch, num_filters=out_ch,
            stride=layer_config['stride'],
            padding=layer_config['padding']
        )

        ifmap_mat, filter_mat, ofmap_mat = gen.create_all_matrices()

        operand_mats = {
            'ifmap_matrix': ifmap_mat,
            'filter_matrix': filter_mat,
            'ofmap_matrix': ofmap_mat,
            'dimensions': gen.get_layer_info()
        }

        dims = gen.get_layer_info()
        print(f"  Operand matrix shapes:")
        print(f"    IFMAP: {dims['ifmap_matrix_shape']}")
        print(f"    Filter: {dims['filter_matrix_shape']}")
        print(f"    OFMAP: {dims['ofmap_matrix_shape']}")

        # Step 2: Generate demand matrices
        print(f"\n[2/5] Generating demand matrices ({self.dataflow} dataflow)...")

        if self.dataflow == 'OS':
            compute_sim = SystolicComputeOS(self.arr_h, self.arr_w)
        elif self.dataflow == 'WS':
            compute_sim = SystolicComputeWS(self.arr_h, self.arr_w)
        elif self.dataflow == 'IS':
            compute_sim = SystolicComputeIS(self.arr_h, self.arr_w)
        else:
            raise ValueError(f"Unknown dataflow: {self.dataflow}")

        demand_mats = compute_sim.generate_demand_matrices(operand_mats)

        print(f"  Total cycles: {demand_mats['total_cycles']}")
        print(f"  Cycles per tile: {demand_mats['cycles_per_tile']}")

        if self.dataflow == 'WS':
            print(f"    - Weight load: {demand_mats['weight_load_cycles']}")
            print(f"    - Input stream: {demand_mats['input_stream_cycles']}")
            print(f"    - Output drain: {demand_mats['output_drain_cycles']}")

        mapping_eff = compute_sim.compute_mapping_efficiency(operand_mats)
        print(f"  Mapping efficiency: {mapping_eff*100:.2f}%")

        # Step 3: Inject faults
        print(f"\n[3/5] Injecting {len(faults)} fault(s)...")

        for i, fault in enumerate(faults, 1):
            loc = fault.location
            pe_info = f"PE({loc['pe_row']},{loc['pe_col']})"
            comp_info = loc.get('component', 'entire_PE')
            timing_info = f"cycles {fault.timing['start_cycle']}-{fault.timing['start_cycle']+fault.timing['duration']-1}"
            print(f"  [{i}] {pe_info} | {comp_info} | {fault.fault_type} | {timing_info}")

        injector = ScaleSimFaultInjector(faults)
        faulty_markers = injector.inject_faults(demand_mats)

        print(f"  Affected addresses: {len(faulty_markers['affected_addresses'])}")

        # Step 4: Trace fault propagation
        print("\n[4/5] Tracing fault propagation...")

        affected_outputs = injector.trace_fault_propagation(
            demand_mats, operand_mats, faulty_markers
        )

        print(f"  Affected outputs: {len(affected_outputs)}")

        # Step 5: Map to tensor indices
        print("\n[5/5] Mapping to tensor indices...")

        exact_positions = injector.create_output_mapping(operand_mats, affected_outputs)

        # Count affected channels
        affected_channels = list(exact_positions.keys())
        print(f"  Affected channels: {len(affected_channels)}")
        if len(affected_channels) <= 10:
            print(f"    Channel IDs: {affected_channels}")

        # Compute statistics
        stats = injector.compute_statistics(affected_outputs, operand_mats)

        print(f"\n{'='*80}")
        print("STATISTICS:")
        print(f"  Total outputs: {stats['total_outputs']}")
        print(f"  Affected outputs: {stats['affected_outputs']}")
        print(f"  Fault coverage: {stats['fault_coverage']*100:.2f}%")
        print(f"{'='*80}")

        return {
            'layer_config': layer_config,
            'operand_matrices': operand_mats,
            'demand_matrices': demand_mats,
            'faulty_markers': faulty_markers,
            'affected_outputs': affected_outputs,
            'exact_positions': exact_positions,
            'statistics': stats
        }

    def export_fault_report(self, results: Dict, save_path: str):
        """Export detailed fault report to text file."""
        layer_config = results['layer_config']
        exact_positions = results['exact_positions']
        stats = results['statistics']

        with open(save_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write(" " * 20 + "SYSTOLIC FAULT SIMULATION REPORT\n")
            f.write(" " * 15 + "(SCALE-Sim Based Implementation)\n")
            f.write("=" * 80 + "\n\n")

            # Layer information
            f.write(f"Layer: {layer_config['name']} ({layer_config['type']})\n")
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

            # Per-channel details
            if layer_config['type'] == 'Linear':
                f.write("AFFECTED OUTPUTS (Per-Feature):\n\n")
                f.write("Note: For Linear layers, each 'channel' represents an output feature.\n\n")
            else:
                f.write("AFFECTED REGIONS (Per-Channel):\n\n")

            affected_channels = sorted(exact_positions.keys())

            for ch in affected_channels:
                rows_dict = exact_positions[ch]

                # Count total pixels for this channel
                total_pixels = sum(len(cols) for cols in rows_dict.values())

                if layer_config['type'] == 'Linear':
                    f.write(f"Output Feature {ch}:\n")
                    f.write(f"  Affected: {total_pixels > 0}\n")
                else:
                    f.write(f"Channel {ch}:\n")
                    f.write(f"  Affected pixels: {total_pixels}\n")

                    # Find bounding box
                    all_rows = list(rows_dict.keys())
                    all_cols = []
                    for cols in rows_dict.values():
                        all_cols.extend(cols)

                    if all_rows and all_cols:
                        min_row, max_row = min(all_rows), max(all_rows)
                        min_col, max_col = min(all_cols), max(all_cols)
                        f.write(f"  Bounding box: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]\n")

                    # List affected coordinates
                    f.write(f"  Affected coordinates:\n")
                    for row in sorted(rows_dict.keys()):
                        cols = sorted(rows_dict[row])
                        f.write(f"    Row {row}: cols {cols}\n")

                f.write("\n")

        print(f"Detailed report saved to: {save_path}")


def _get_component_choice() -> str:
    """Get user choice for faulty component."""
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


def _compute_computation_cycles(dataflow: str, arr_h: int, arr_w: int, dims: Dict):
    """
    Compute the cycle range where computation actually happens.

    Returns:
        (start_cycle, end_cycle, description)
    """
    Sr = dims['ofmap_shape'][0] * dims['ofmap_shape'][1]  # Total ofmap pixels
    T = dims['ifmap_matrix_shape'][1]  # Conv window size

    if dataflow == 'OS':
        comp_start = 0
        comp_end = T - 1
        desc = f"Accumulation: cycles 0-{T-1}, Output drain: cycles {T}-{T+arr_w-2}"
        return (comp_start, comp_end, desc)

    elif dataflow == 'WS':
        weight_load = arr_h
        comp_start = weight_load
        comp_end = weight_load + Sr - 1
        desc = f"Weight load: cycles 0-{weight_load-1}, Input stream: cycles {comp_start}-{comp_end}"
        return (comp_start, comp_end, desc)

    elif dataflow == 'IS':
        input_load = arr_h
        comp_start = input_load
        comp_end = input_load + T - 1
        desc = f"Input load: cycles 0-{input_load-1}, Weight stream: cycles {comp_start}-{comp_end}"
        return (comp_start, comp_end, desc)

    return (0, 0, "Unknown dataflow")


def _get_fault_timing(dataflow: str, arr_h: int, arr_w: int, dims: Dict) -> Dict:
    """Get fault timing from user with helpful guidance."""
    print("\n  Fault Timing (TRANSIENT - time-bounded):")

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
                    print(f"    ⚠️  WARNING: Fault ends at cycle {start+duration-1}, before computation starts!")
                    confirm = input("    Continue anyway? (y/n): ")
                    if confirm.lower() != 'y':
                        continue

                return {'start_cycle': start, 'duration': duration}
            else:
                print("      Start must be ≥0, duration must be >0")
        except ValueError:
            print("      Invalid input. Please enter integers.")


def interactive_simulation():
    """Main interactive simulation interface."""
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

    simulator = ScaleSimFaultSimulator(h, w, dataflow)

    # Step 2: Select model type
    print("\n[Step 2] Select Model")
    print("-" * 80)
    print("  1. TrafficSignNet (Original - 32/64/128 channels)")
    print("  2. TrafficSignNetSmall (New - 16/32/64 channels)")

    model = None
    while model is None:
        model_choice = input("\nChoose model (1/2) [default: 2]: ").strip() or '2'
        if model_choice == '1':
            if TrafficSignNet is None:
                print("  Error: TrafficSignNet not available")
                continue
            model = TrafficSignNet(num_classes=43)
            print("  Selected: TrafficSignNet (Original)")
            break
        elif model_choice == '2':
            if TrafficSignNetSmall is None:
                print("  Error: TrafficSignNetSmall not available")
                continue
            model = TrafficSignNetSmall(num_classes=43)
            print("  Selected: TrafficSignNetSmall (New)")
            break
        else:
            print("  Invalid choice. Please enter 1 or 2.")

    # Step 3: Select layer
    print("\n[Step 3] Select Layer")
    print("-" * 80)

    layers_info = model.get_layer_info()

    print(f"{'Index':>5} | {'Name':^15} | {'Type':^10} | {'Shape':^20}")
    print("-"*80)
    for i, (name, ltype, _, shape) in enumerate(layers_info):
        # Show Conv2d, Conv, and Linear layers
        if ltype in ['Conv2d', 'Conv', 'Linear']:
            print(f"{i:>5} | {name:^15} | {ltype:^10} | {shape:^20}")

    while True:
        try:
            layer_idx = int(input("\nSelect layer index: "))
            layer_config = simulator.get_layer_config(model, layer_idx)
            print(f"Selected: {layer_config['name']} ({layer_config['type']})")
            break
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")

    # Get dimensions for timing guidance
    in_ch, Hin, Win = layer_config['input_shape']
    out_ch, Hout, Wout = layer_config['output_shape']
    kernel_h, kernel_w = layer_config['kernel_size']

    gen_temp = OperandMatrixGenerator()
    gen_temp.set_layer_params(
        ifmap_h=Hin, ifmap_w=Win,
        filter_h=kernel_h, filter_w=kernel_w,
        num_channels=in_ch, num_filters=out_ch,
        stride=layer_config['stride'],
        padding=layer_config['padding']
    )
    dims = gen_temp.get_layer_info()

    # Step 4: Define faults
    print("\n[Step 4] Define Faults")
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

                # Get fault details once
                component = _get_component_choice()
                fault_timing = _get_fault_timing(dataflow, h, w, dims)

                # Apply to range
                num_added = 0
                for r in range(r_start, r_end + 1):
                    for c in range(c_start, c_end + 1):
                        if 0 <= r < h and 0 <= c < w:
                            fault = FaultModel(
                                fault_type=FaultModel.BIT_FLIP,
                                location={'pe_row': r, 'pe_col': c, 'component': component},
                                timing=fault_timing
                            )
                            faults.append(fault)
                            num_added += 1

                print(f"  → Added {num_added} fault(s)")

            else:
                # Single PE
                r, c = map(int, pe_input.split(','))
                if not (0 <= r < h and 0 <= c < w):
                    print(f"  Error: PE ({r}, {c}) out of bounds!")
                    continue

                component = _get_component_choice()
                fault_timing = _get_fault_timing(dataflow, h, w, dims)

                fault = FaultModel(
                    fault_type=FaultModel.BIT_FLIP,
                    location={'pe_row': r, 'pe_col': c, 'component': component},
                    timing=fault_timing
                )
                faults.append(fault)
                print(f"  → Added fault at PE ({r}, {c}), {component}")

        except ValueError as e:
            print(f"  Invalid format: {e}")

    if len(faults) == 0:
        print("\n⚠️  No faults defined. Exiting.")
        return

    print(f"\n{'='*80}")
    print(f"Total faults defined: {len(faults)}")
    print(f"{'='*80}")

    # Step 5: Run simulation
    print("\n[Step 5] Running Simulation...")
    print("="*80)

    results = simulator.simulate_layer(layer_config, faults)

    # Step 6: Visualize results
    print("\n[Step 6] Visualizing Results...")
    print("="*80)

    visualizer = FaultVisualizer()

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Build filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(faults) > 0:
        first_fault = faults[0]
        pe_info = f"PE{first_fault.location['pe_row']}-{first_fault.location['pe_col']}"
        component = first_fault.location.get('component', 'unknown')
        start_cycle = first_fault.timing['start_cycle']
        duration = first_fault.timing['duration']

        if len(faults) > 1:
            pe_info = f"PE_multi_{len(faults)}faults"
    else:
        pe_info = "no_faults"
        component = "none"
        start_cycle = 0
        duration = 0

    filename_base = f"{dataflow}_{h}x{w}_{layer_config['name']}_{pe_info}_{component}_cycle{start_cycle}dur{duration}_{timestamp}"
    report_path = os.path.join(results_dir, f"{filename_base}.txt")

    # Save text report
    simulator.export_fault_report(results, report_path)

    # Generate visualizations
    if results['statistics']['fault_coverage'] > 0:
        print("\nGenerating visualizations...")

        # 1. Affected channels visualization (red highlighted)
        channels_plot_path = os.path.join(results_dir, f"{filename_base}_channels.png")
        visualizer.visualize_affected_channels(
            results['exact_positions'],
            layer_config,
            save_path=channels_plot_path,
            max_channels=16
        )

        # 2. Summary plots
        summary_plot_path = os.path.join(results_dir, f"{filename_base}_summary.png")
        visualizer.create_summary_plot(
            results['exact_positions'],
            layer_config,
            results['statistics'],
            save_path=summary_plot_path
        )

        print(f"  Visualizations saved:")
        print(f"    - Channels: {channels_plot_path}")
        print(f"    - Summary: {summary_plot_path}")
    else:
        print("\n⚠️  No affected outputs - skipping visualization")

    print("\n" + "="*80)
    print("✓ Simulation completed!")
    print(f"  Report: {report_path}")
    print("="*80)

    # Show summary
    stats = results['statistics']
    if stats['fault_coverage'] > 0:
        print(f"\n✅ Fault coverage: {stats['fault_coverage']*100:.2f}%")
        print(f"   Affected {stats['affected_outputs']}/{stats['total_outputs']} outputs")
    else:
        print(f"\n⚠️  Fault coverage: 0%")
        print(f"   No outputs affected - check fault timing!")


if __name__ == '__main__':
    interactive_simulation()
