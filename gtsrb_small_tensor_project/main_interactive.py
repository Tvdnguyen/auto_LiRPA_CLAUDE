"""
Interactive main script for testing intermediate layer perturbations on GTSRB
This script provides a user-friendly interface to:
1. Load trained model
2. Select intermediate layer to perturb
3. Select region of tensor to perturb (batch, channel, spatial)
4. Compute bounds with perturbation
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import csv
import random
import re
from typing import List, Dict, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traffic_sign_net_small import TrafficSignNetSmall, TrafficSignNetSmallNoDropout

# Import NEW V2 version with element-wise epsilon approach
# Force reload to avoid cache issues
import importlib
if 'intermediate_bound_module_v2' in sys.modules:
    importlib.reload(sys.modules['intermediate_bound_module_v2'])
from intermediate_bound_module_v2 import IntermediateBoundedModuleV2 as IntermediateBoundedModule

# Now add gtsrb_project for other imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from collect_correct_samples import load_correct_indices
from efficient_c_matrix import create_efficient_c_matrix, compute_robustness_margin


def parse_fault_result_file(file_path: str) -> Dict:
    """
    Parse fault simulation result file to extract affected regions

    Args:
        file_path: Path to .txt result file from fault simulator

    Returns:
        Dictionary with:
            - layer_name: Name of affected layer
            - dataflow: Dataflow type (IS/OS/WS)
            - array_size: Array size (e.g., "8x8")
            - affected_channels: List of affected channel indices
            - regions: List of dicts with {channel_idx, height_slice, width_slice, exact_positions}
    """
    result = {
        'layer_name': None,
        'dataflow': None,
        'array_size': None,
        'affected_channels': [],
        'regions': []
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Parse header info
    for line in lines:
        if line.startswith('Layer:'):
            # "Layer: conv1 (Conv)"
            match = re.search(r'Layer:\s+(\w+)', line)
            if match:
                result['layer_name'] = match.group(1)

        elif line.startswith('Dataflow:'):
            # "Dataflow: OS"
            match = re.search(r'Dataflow:\s+(\w+)', line)
            if match:
                result['dataflow'] = match.group(1)

        elif line.startswith('Array Size:'):
            # "Array Size: 8×8"
            match = re.search(r'Array Size:\s+([\d×x]+)', line)
            if match:
                result['array_size'] = match.group(1).replace('×', 'x')

    # Parse affected channels and their bounding boxes
    current_channel = None
    current_bbox = None
    exact_positions = {}

    for line in lines:
        # Detect channel header: "Channel 1:"
        channel_match = re.match(r'^Channel (\d+):', line)
        if channel_match:
            # Save previous channel if exists
            if current_channel is not None and current_bbox is not None:
                result['regions'].append({
                    'channel_idx': current_channel,
                    'height_slice': (current_bbox['row_min'], current_bbox['row_max'] + 1),
                    'width_slice': (current_bbox['col_min'], current_bbox['col_max'] + 1),
                    'exact_positions': exact_positions
                })
                exact_positions = {}

            current_channel = int(channel_match.group(1))
            result['affected_channels'].append(current_channel)
            current_bbox = None

        # Detect bounding box: "  Bounding box: rows [0, 31], cols [1, 25]"
        bbox_match = re.search(r'Bounding box: rows \[(\d+), (\d+)\], cols \[(\d+), (\d+)\]', line)
        if bbox_match:
            current_bbox = {
                'row_min': int(bbox_match.group(1)),
                'row_max': int(bbox_match.group(2)),
                'col_min': int(bbox_match.group(3)),
                'col_max': int(bbox_match.group(4))
            }

        # Detect affected coordinates: "    Row 0: cols [1, 9, 17, 25]"
        coord_match = re.search(r'Row (\d+): cols \[([\d, ]+)\]', line)
        if coord_match:
            row = int(coord_match.group(1))
            cols = [int(c.strip()) for c in coord_match.group(2).split(',')]
            exact_positions[row] = cols

    # Save last channel
    if current_channel is not None and current_bbox is not None:
        result['regions'].append({
            'channel_idx': current_channel,
            'height_slice': (current_bbox['row_min'], current_bbox['row_max'] + 1),
            'width_slice': (current_bbox['col_min'], current_bbox['col_max'] + 1),
            'exact_positions': exact_positions
        })

    return result


def list_fault_result_files(results_dir: str) -> List[Tuple[str, str]]:
    """
    List all .txt fault result files in directory

    Args:
        results_dir: Path to results directory

    Returns:
        List of (filename, full_path) tuples
    """
    if not os.path.exists(results_dir):
        return []

    files = []
    for filename in sorted(os.listdir(results_dir)):
        if filename.endswith('.txt'):
            full_path = os.path.join(results_dir, filename)
            files.append((filename, full_path))

    return files


def display_fault_result_summary(parsed_result: Dict):
    """
    Display summary of parsed fault result

    Args:
        parsed_result: Result from parse_fault_result_file()
    """
    print(f"\n  Fault Simulation Result Summary:")
    print(f"    Layer: {parsed_result['layer_name']}")
    print(f"    Dataflow: {parsed_result['dataflow']}")
    print(f"    Array Size: {parsed_result['array_size']}")
    print(f"    Affected Channels: {parsed_result['affected_channels']}")
    print(f"    Number of Regions: {len(parsed_result['regions'])}")

    print(f"\n  Perturbation Regions:")
    for i, region in enumerate(parsed_result['regions'], 1):
        h_start, h_end = region['height_slice']
        w_start, w_end = region['width_slice']
        num_exact = sum(len(cols) for cols in region['exact_positions'].values())
        print(f"    Region {i}: Channel {region['channel_idx']}, "
              f"H({h_start}, {h_end}), W({w_start}, {w_end}) "
              f"[{num_exact} exact positions]")


def create_exact_mask_from_elements(elements: List[Tuple[int, int, int]], layer_output_shape: Tuple) -> torch.Tensor:
    """
    Create exact binary mask from list of (channel, row, col) elements

    Args:
        elements: List of (channel_idx, row, col) tuples
        layer_output_shape: Shape of layer output (B, C, H, W)

    Returns:
        Boolean mask tensor (True at affected positions, False elsewhere)
    """
    # Create False mask (MUST be bool for torch.where)
    mask = torch.zeros(layer_output_shape, dtype=torch.bool)

    batch_size, num_channels, height, width = layer_output_shape

    # Set mask to True at specified elements
    for channel_idx, row, col in elements:
        # Validate indices
        if channel_idx >= num_channels:
            print(f"  Warning: Channel {channel_idx} >= {num_channels}, skipping")
            continue
        if row >= height:
            print(f"  Warning: Row {row} >= {height}, skipping")
            continue
        if col >= width:
            print(f"  Warning: Col {col} >= {width}, skipping")
            continue

        # Set for batch 0 (assuming single batch)
        mask[0, channel_idx, row, col] = True

    return mask


class InteractiveTester:
    """Interactive testing interface for intermediate perturbations"""

    def __init__(self, model, checkpoint_path, data_dir, device='cuda', use_nodropout=True):
        """
        Initialize tester

        Args:
            model: Neural network model (for display/inference)
            checkpoint_path: Path to trained checkpoint
            data_dir: Path to GTSRB dataset
            device: Device to run on
            use_nodropout: If True, use no-dropout version for verification
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model for display (with dropout)
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        self.model = model.to(self.device)

        print(f"Model loaded successfully!")
        print(f"  Checkpoint accuracy: {checkpoint.get('test_acc', 'N/A'):.2f}%")

        # Create verification model (without dropout to avoid Patches issues)
        if use_nodropout and hasattr(model, '__class__'):
            if model.__class__.__name__ == 'TrafficSignNet':
                print("\nCreating no-dropout model for verification...")
                verification_model = TrafficSignNetNoDropout(num_classes=43)
                verification_model.load_from_dropout_checkpoint(checkpoint_path)
                verification_model.eval()
                verification_model = verification_model.to(self.device)
                print("No-dropout model created (avoids Dropout/Patches incompatibility)")
            elif model.__class__.__name__ == 'TrafficSignNetSmall':
                print("\nCreating no-dropout model for verification (TrafficSignNetSmall)...")
                verification_model = TrafficSignNetSmallNoDropout(num_classes=43)
                verification_model.load_from_dropout_checkpoint(checkpoint_path)
                verification_model.eval()
                verification_model = verification_model.to(self.device)
                print("No-dropout model created (avoids Dropout/Patches incompatibility)")
            else:
                verification_model = model
        else:
            verification_model = model

        # Create bounded module with verification model
        print("\nCreating bounded module...")
        dummy_input = torch.randn(1, 3, 32, 32).to(self.device)
        self.lirpa_model = IntermediateBoundedModule(
            verification_model,
            dummy_input,
            device=self.device
        )

        # Load dataset
        self.data_dir = data_dir
        self.test_dataset = GTSRBDataset(
            root_dir=data_dir,
            train=False,
            transform=get_gtsrb_transforms(train=False, img_size=32)
        )

        print(f"Dataset loaded: {len(self.test_dataset)} test samples")

    def show_model_layers(self):
        """Display all Conv and FC layers in the model"""
        print("\n" + "="*80)
        print("Available Layers for Perturbation (Conv and FC only)")
        print("="*80)

        # Get layer info from model if available
        if hasattr(self.model, 'get_layer_info'):
            layers_info = self.model.get_layer_info()
            print(f"{'Index':>5} | {'Layer Name':^15} | {'Type':^10} | {'Output Shape':^15}")
            print("-"*80)
            for i, (name, ltype, layer, shape) in enumerate(layers_info):
                print(f"{i:>5} | {name:^15} | {ltype:^10} | {shape:^15}")
        else:
            # Fallback: use bounded module structure
            layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])
            print(f"{'Index':>5} | {'Node Name':^40} | {'Type':^20}")
            print("-"*80)
            for i, (name, ltype) in enumerate(layers):
                print(f"{i:>5} | {name:^40} | {ltype:^20}")

        print("="*80)

    def get_layer_info(self, layer_idx):
        """
        Get detailed information about a specific layer

        Args:
            layer_idx: Layer index

        Returns:
            (node_name, layer_type, output_shape)
        """
        if hasattr(self.model, 'get_layer_info'):
            layers_info = self.model.get_layer_info()
            if 0 <= layer_idx < len(layers_info):
                name, ltype, layer, shape_str = layers_info[layer_idx]

                # We need the actual node name in the computation graph
                # Map model layer name to graph node name
                layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])

                # Try to find matching node
                for node_name, node_type in layers:
                    if name in node_name or node_name.endswith(name):
                        return node_name, ltype, shape_str

                # If not found, return first match
                if layer_idx < len(layers):
                    return layers[layer_idx][0], ltype, shape_str

        # Fallback
        layers = self.lirpa_model.get_layer_names(['Conv', 'Linear'])
        if 0 <= layer_idx < len(layers):
            node_name, node_type = layers[layer_idx]
            return node_name, node_type, "unknown"

        raise ValueError(f"Invalid layer index: {layer_idx}")

    def get_sample_from_class(self, class_id, sample_idx=None, correct_samples_dir='correct_samples'):
        """
        Get a random sample from a class (from correct_samples)

        Args:
            class_id: Class ID (0-42)
            sample_idx: If None, randomly select. If provided, use specific index.
            correct_samples_dir: Directory with correct sample indices

        Returns:
            image tensor, label, global_idx, sample_idx
        """
        # Load correct indices for this class
        try:
            correct_indices = load_correct_indices(correct_samples_dir, class_id)
        except FileNotFoundError:
            print(f"Warning: Correct samples file not found. Using any sample from class {class_id}")
            # Find any sample from this class
            correct_indices = [i for i, label in enumerate(self.test_dataset.labels)
                             if label == class_id]

        if len(correct_indices) == 0:
            raise ValueError(f"No samples found for class {class_id}")

        # IMPORTANT: Filter out invalid indices (in case dataset is smaller than expected)
        dataset_size = len(self.test_dataset)
        valid_indices = [idx for idx in correct_indices if idx < dataset_size]

        if len(valid_indices) < len(correct_indices):
            print(f"  Warning: Filtered {len(correct_indices) - len(valid_indices)} invalid indices")
            print(f"  Dataset size: {dataset_size}, Max index in CSV: {max(correct_indices)}")

        if len(valid_indices) == 0:
            raise ValueError(f"No valid samples found for class {class_id} (dataset too small?)")

        # Use valid indices only
        correct_indices = valid_indices

        # Random selection if sample_idx not provided
        if sample_idx is None:
            sample_idx = random.randint(0, len(correct_indices) - 1)
            print(f"  Randomly selected index {sample_idx} from {len(correct_indices)} correct samples")
        else:
            if sample_idx >= len(correct_indices):
                print(f"Warning: Sample index {sample_idx} out of range. Using index 0.")
                sample_idx = 0

        global_idx = correct_indices[sample_idx]
        image, label = self.test_dataset[global_idx]

        return image, label, global_idx, sample_idx

    def compute_clean_output(self, image):
        """
        Compute clean (unperturbed) output

        Args:
            image: Input image tensor

        Returns:
            output logits, predicted class
        """
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            pred_class = output.argmax(dim=1).item()

        return output, pred_class

    def compute_perturbed_bounds(
        self,
        image,
        node_name,
        epsilon,
        batch_idx=0,
        channel_idx=None,
        height_slice=None,
        width_slice=None,
        mask=None,
        method='backward'
    ):
        """
        Compute bounds with intermediate perturbation

        Args:
            image: Input image
            node_name: Node name to perturb
            epsilon: Perturbation magnitude
            batch_idx: Batch index
            channel_idx: Channel indices to perturb
            height_slice: Height slice (start, end)
            width_slice: Width slice (start, end)
            mask: Custom mask tensor (if provided, overrides channel/height/width specs)
            method: Bound computation method

        Returns:
            lower_bound, upper_bound
        """
        image = image.unsqueeze(0).to(self.device)

        # Clear previous perturbations
        self.lirpa_model.clear_intermediate_perturbations()

        # Forward pass to get intermediate outputs
        print("\nRunning forward pass...")
        with torch.no_grad():
            _ = self.lirpa_model(image)

        # Create masked perturbation
        print(f"Creating perturbation for node: {node_name}")
        print(f"  Epsilon: {epsilon}")

        if mask is not None:
            # Use custom mask
            num_perturbed = (mask > 0).sum().item()
            print(f"  Using custom mask: {num_perturbed} elements perturbed")

            perturbation = MaskedPerturbationLpNorm(
                eps=epsilon,
                norm=np.inf,
                mask=mask.to(self.device)
            )
        else:
            # Use slice-based specification
            print(f"  Batch: {batch_idx}")
            print(f"  Channel: {channel_idx}")
            print(f"  Height: {height_slice}")
            print(f"  Width: {width_slice}")

            perturbation = MaskedPerturbationLpNorm(
                eps=epsilon,
                norm=np.inf,
                batch_idx=batch_idx,
                channel_idx=channel_idx,
                height_slice=height_slice,
                width_slice=width_slice
            )

        # Register perturbation
        self.lirpa_model.register_intermediate_perturbation(node_name, perturbation)

        # Compute bounds
        print(f"\nComputing bounds using {method} method...")
        try:
            lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                x=image,
                method=method
            )

            return lb, ub

        except Exception as e:
            print(f"Error computing bounds: {e}")
            print("Trying with IBP method...")
            lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
                x=image,
                method='IBP'
            )
            return lb, ub

    def run_interactive(self):
        """Run interactive testing session"""
        print("\n" + "="*80)
        print(" "*20 + "INTERACTIVE PERTURBATION TESTER")
        print("="*80)

        while True:
            # Show layers
            self.show_model_layers()

            # Step 1: Select layer
            try:
                layer_idx = int(input("\nSelect layer index (or -1 to quit): "))
                if layer_idx == -1:
                    print("Exiting...")
                    break

                node_name, layer_type, shape_str = self.get_layer_info(layer_idx)
                print(f"\nSelected layer: {node_name}")
                print(f"  Type: {layer_type}")
                print(f"  Output shape: {shape_str}")

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}")
                continue

            # Step 2: Get sample to test
            try:
                class_id = int(input(f"\nSelect class ID (0-42): "))
                if not 0 <= class_id <= 42:
                    print("Class ID must be between 0 and 42")
                    continue

                # Automatically select random sample from correct_samples
                print(f"\nLoading random correct sample from class {class_id}...")
                image, label, global_idx, sample_idx = self.get_sample_from_class(class_id, sample_idx=None)

                print(f"\nLoaded sample:")
                print(f"  Class: {class_id}")
                print(f"  Sample index within class: {sample_idx}")
                print(f"  Global test set index: {global_idx}")
                print(f"  True label: {label}")

            except Exception as e:
                print(f"Error loading sample: {e}")
                continue

            # Step 3: Compute clean output
            print("\n" + "-"*80)
            print("Computing clean (unperturbed) output...")
            clean_output, pred_class = self.compute_clean_output(image)
            print(f"\nClean Output (Logits):")
            print(f"  Predicted class: {pred_class}")
            print(f"  Top-5 classes:")
            top5_vals, top5_idx = torch.topk(clean_output[0], 5)
            for i, (idx, val) in enumerate(zip(top5_idx, top5_vals)):
                marker = " ←" if idx == label else ""
                print(f"    {i+1}. Class {idx:2d}: {val.item():8.4f}{marker}")

            # Step 4: Configure perturbation regions (can add multiple)
            print("\n" + "-"*80)
            print("Configure Perturbation Regions:")
            print("  Options:")
            print("    1. Load from fault simulation result file")
            print("    2. Manual input - exact elements (add one by one, type 'done' to finish)")
            print("    3. Manual input - regions (specify channel + rectangular area)")

            # Determine if Conv or FC layer
            is_conv = 'Conv' in layer_type

            # Collect multiple regions
            perturbation_regions = []

            try:
                # Ask user for input method
                input_method = input("\n  Select input method (1, 2, or 3): ").strip()

                if input_method == '1':
                    # Load from fault result file
                    results_dir = os.path.join(os.path.dirname(__file__), '..', 'scalesim_fault_simulator', 'results')
                    fault_files = list_fault_result_files(results_dir)

                    if len(fault_files) == 0:
                        print(f"\n  No fault result files found in {results_dir}")
                        print("  Switching to manual input mode...")
                        input_method = '2'
                    else:
                        # Display available files
                        print(f"\n  Available fault simulation results ({len(fault_files)} files):")
                        for i, (filename, _) in enumerate(fault_files):
                            print(f"    {i+1}. {filename}")

                        # Let user select
                        file_idx = int(input(f"\n  Select file (1-{len(fault_files)}): ")) - 1
                        if 0 <= file_idx < len(fault_files):
                            selected_file = fault_files[file_idx][1]
                            print(f"\n  Loading: {fault_files[file_idx][0]}")

                            # Parse the file
                            fault_result = parse_fault_result_file(selected_file)

                            # Display summary
                            display_fault_result_summary(fault_result)

                            # Convert exact_positions to elements list
                            print("\n  Extracting exact elements from fault data...")
                            fault_elements = []
                            for region in fault_result['regions']:
                                channel_idx = region['channel_idx']
                                for row, cols in region['exact_positions'].items():
                                    for col in cols:
                                        fault_elements.append((channel_idx, row, col))

                            print(f"\n  Found {len(fault_elements)} exact affected elements:")

                            # Show first 20 elements as preview
                            preview_count = min(20, len(fault_elements))
                            print(f"\n  Preview (showing first {preview_count} of {len(fault_elements)}):")
                            for i, (ch, row, col) in enumerate(fault_elements[:preview_count]):
                                print(f"    {i+1}. Channel {ch}, Row {row}, Col {col}")

                            if len(fault_elements) > preview_count:
                                print(f"    ... and {len(fault_elements) - preview_count} more elements")

                            # Ask for confirmation
                            confirm = input("\n  Use these elements for perturbation? (y/n): ").strip().lower()
                            if confirm == 'y':
                                # Store as exact_elements type
                                perturbation_regions = [{
                                    'type': 'exact_elements',
                                    'elements': fault_elements
                                }]
                                print(f"\n  ✓ Loaded {len(fault_elements)} element(s) from fault simulation")
                            else:
                                print("  Cancelled. Switching to manual input mode...")
                                input_method = '2'
                        else:
                            print("  Invalid selection. Switching to manual input mode...")
                            input_method = '2'

                if input_method == '2':
                    # Manual input mode - element by element
                    print("\n  Manual Input Mode: Add exact elements one by one")
                    print("  Each element is specified by (channel, row, col)")
                    print("  Type 'done' when finished adding elements")

                    # Collect individual elements
                    perturbed_elements = []

                    while True:
                        elem_num = len(perturbed_elements) + 1
                        print(f"\n  Element #{elem_num}:")

                        if is_conv:
                            # Conv layer: ask for channel, row, col
                            try:
                                channel_input = input("    Channel index (or 'done'): ").strip()
                                if channel_input.lower() == 'done':
                                    print(f"  Finished adding elements. Total: {len(perturbed_elements)}")
                                    break

                                channel_idx = int(channel_input)

                                row_input = input("    Row index: ").strip()
                                row_idx = int(row_input)

                                col_input = input("    Column index: ").strip()
                                col_idx = int(col_input)

                                perturbed_elements.append((channel_idx, row_idx, col_idx))
                                print(f"    ✓ Added element: Channel {channel_idx}, Row {row_idx}, Col {col_idx}")

                            except ValueError as e:
                                print(f"    ✗ Invalid input: {e}")
                                continue

                        else:
                            # FC layer: ask for feature dimension
                            try:
                                feature_input = input("    Feature index (or 'done'): ").strip()
                                if feature_input.lower() == 'done':
                                    print(f"  Finished adding elements. Total: {len(perturbed_elements)}")
                                    break

                                feature_idx = int(feature_input)

                                perturbed_elements.append((feature_idx, 0, 0))  # FC: use (feature, 0, 0)
                                print(f"    ✓ Added element: Feature {feature_idx}")

                            except ValueError as e:
                                print(f"    ✗ Invalid input: {e}")
                                continue

                    # Store in perturbation_regions for later use
                    # Mark that we have exact elements (not regions)
                    perturbation_regions = [{
                        'type': 'exact_elements',
                        'elements': perturbed_elements
                    }]

                elif input_method == '3':
                    # Manual input mode - regions (channel + rectangular area)
                    print("\n  Manual Input Mode: Add regions one by one")
                    print("  Each region is specified by channel and rectangular area")
                    print("  Type 'done' when finished adding regions")

                    # Collect regions
                    region_list = []

                    while True:
                        region_num = len(region_list) + 1
                        print(f"\n  Region #{region_num}:")

                        if is_conv:
                            # Conv layer: ask for channel, height range, width range
                            try:
                                channel_input = input("    Channel index (or 'done'): ").strip()
                                if channel_input.lower() == 'done':
                                    print(f"  Finished adding regions. Total: {len(region_list)}")
                                    break

                                channel_idx = int(channel_input)

                                # Ask for height range
                                print("    Height range:")
                                h_start = int(input("      Start row: ").strip())
                                h_end = int(input("      End row (exclusive): ").strip())

                                # Ask for width range
                                print("    Width range:")
                                w_start = int(input("      Start col: ").strip())
                                w_end = int(input("      End col (exclusive): ").strip())

                                # Validate ranges
                                if h_start >= h_end:
                                    print(f"    ✗ Invalid height range: start ({h_start}) >= end ({h_end})")
                                    continue

                                if w_start >= w_end:
                                    print(f"    ✗ Invalid width range: start ({w_start}) >= end ({w_end})")
                                    continue

                                region_list.append({
                                    'channel_idx': channel_idx,
                                    'height_slice': (h_start, h_end),
                                    'width_slice': (w_start, w_end)
                                })

                                print(f"    ✓ Added region: Channel {channel_idx}, "
                                      f"Height [{h_start}, {h_end}), Width [{w_start}, {w_end})")
                                print(f"      (covers {(h_end - h_start) * (w_end - w_start)} elements)")

                            except ValueError as e:
                                print(f"    ✗ Invalid input: {e}")
                                continue

                        else:
                            # FC layer: ask for feature range
                            try:
                                start_input = input("    Start feature index (or 'done'): ").strip()
                                if start_input.lower() == 'done':
                                    print(f"  Finished adding regions. Total: {len(region_list)}")
                                    break

                                start_idx = int(start_input)
                                end_idx = int(input("    End feature index (exclusive): ").strip())

                                if start_idx >= end_idx:
                                    print(f"    ✗ Invalid range: start ({start_idx}) >= end ({end_idx})")
                                    continue

                                region_list.append({
                                    'channel_idx': list(range(start_idx, end_idx)),
                                    'height_slice': None,
                                    'width_slice': None
                                })

                                print(f"    ✓ Added region: Features [{start_idx}, {end_idx})")
                                print(f"      (covers {end_idx - start_idx} features)")

                            except ValueError as e:
                                print(f"    ✗ Invalid input: {e}")
                                continue

                    # Store regions
                    perturbation_regions = region_list

                # Get epsilon value
                epsilon = float(input("\n  Epsilon value: "))

                # Summary
                if len(perturbation_regions) > 0 and perturbation_regions[0].get('type') == 'exact_elements':
                    # Manual exact elements (option 2)
                    elements = perturbation_regions[0]['elements']
                    print(f"\n  Summary: {len(elements)} element(s) configured")
                    for i, (ch, row, col) in enumerate(elements, 1):
                        if is_conv:
                            print(f"    Element {i}: Channel {ch}, Row {row}, Col {col}")
                        else:
                            print(f"    Element {i}: Feature {ch}")
                elif len(perturbation_regions) > 0 and 'exact_positions' in perturbation_regions[0]:
                    # Fault file regions (option 1)
                    print(f"\n  Summary: {len(perturbation_regions)} region(s) from fault file")
                    for i, region in enumerate(perturbation_regions, 1):
                        num_exact = sum(len(cols) for cols in region['exact_positions'].values())
                        h_start, h_end = region['height_slice']
                        w_start, w_end = region['width_slice']
                        print(f"    Region {i}: Channel {region['channel_idx']}, "
                              f"H({h_start}, {h_end}), W({w_start}, {w_end}) [{num_exact} exact positions]")
                else:
                    # Manual regions (option 3)
                    print(f"\n  Summary: {len(perturbation_regions)} region(s) configured")
                    for i, region in enumerate(perturbation_regions, 1):
                        if is_conv:
                            h_start, h_end = region['height_slice']
                            w_start, w_end = region['width_slice']
                            num_elements = (h_end - h_start) * (w_end - w_start)
                            print(f"    Region {i}: Channel {region['channel_idx']}, "
                                  f"H[{h_start}, {h_end}), W[{w_start}, {w_end}) [{num_elements} elements]")
                        else:
                            num_features = len(region['channel_idx'])
                            print(f"    Region {i}: Features {region['channel_idx'][0]}-{region['channel_idx'][-1]} [{num_features} features]")

            except Exception as e:
                print(f"Invalid input: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Step 5: Compute bounds
            print("\n" + "-"*80)
            print("Computing bounds with perturbation...")

            try:
                # Check if we have exact elements (manual input or fault file)
                has_exact_elements = (len(perturbation_regions) > 0 and
                                     perturbation_regions[0].get('type') == 'exact_elements')

                if has_exact_elements:
                    # Manual exact elements
                    elements = perturbation_regions[0]['elements']
                    print(f"  Applying {len(elements)} exact element(s)...")

                    # Get actual layer shape by running forward pass
                    print("\n  Detecting layer output shape...")
                    image_batch = image.unsqueeze(0).to(self.device)

                    # Use a forward hook to capture the output shape
                    layer_shape_holder = [None]

                    def capture_shape(module, input, output):
                        if isinstance(output, torch.Tensor):
                            layer_shape_holder[0] = output.shape

                    # Get the actual PyTorch module from node name
                    # The node_name is from BoundedModule graph
                    # We need to find corresponding module in original model
                    hook_handle = None
                    try:
                        # Try to get node from lirpa_model
                        node = self.lirpa_model.get_node_by_name(node_name)

                        # Run forward pass with hook to capture shape
                        with torch.no_grad():
                            output = self.model(image_batch)

                        # For now, parse shape_str or run lirpa forward to get intermediate output
                        # Run lirpa forward pass and capture intermediate output
                        self.lirpa_model.clear_intermediate_perturbations()
                        with torch.no_grad():
                            _ = self.lirpa_model(image_batch)

                        # Try to get shape from node's output_shape attribute
                        if hasattr(node, 'output_shape') and node.output_shape is not None:
                            if isinstance(node.output_shape, (list, tuple)):
                                layer_shape = tuple(node.output_shape)
                            else:
                                layer_shape = node.output_shape
                        else:
                            # Parse from shape_str
                            # shape_str is like "(1, 32, 32, 32)" or "1x32x32x32"
                            import re
                            numbers = re.findall(r'\d+', shape_str)
                            if len(numbers) == 4:
                                layer_shape = tuple(int(n) for n in numbers)
                            elif len(numbers) == 3:
                                layer_shape = (1, int(numbers[0]), int(numbers[1]), int(numbers[2]))
                            else:
                                raise ValueError(f"Cannot parse shape from: {shape_str}")

                        print(f"  Detected layer shape: {layer_shape}")

                    except Exception as e:
                        print(f"  Warning: Could not auto-detect shape: {e}")
                        print("  Please provide layer output shape manually:")
                        num_channels = int(input("    Number of channels: "))
                        height = int(input("    Height: "))
                        width = int(input("    Width: "))
                        layer_shape = (1, num_channels, height, width)
                        print(f"  Using layer shape: {layer_shape}")

                    # Create exact mask
                    print(f"\n  Creating exact mask from {len(elements)} element(s)...")
                    exact_mask = create_exact_mask_from_elements(elements, layer_shape)
                    num_exact = exact_mask.sum().item()
                    total = exact_mask.numel()
                    print(f"  Exact mask: {num_exact}/{total} elements ({100*num_exact/total:.2f}%)")

                    # Compute bounds with exact mask
                    lb, ub = self.compute_perturbed_bounds(
                        image=image,
                        node_name=node_name,
                        epsilon=epsilon,
                        mask=exact_mask,
                        method='backward'
                    )

                elif len(perturbation_regions) == 1:
                    # Single region without exact positions - use slice specification
                    region = perturbation_regions[0]
                    lb, ub = self.compute_perturbed_bounds(
                        image=image,
                        node_name=node_name,
                        epsilon=epsilon,
                        batch_idx=0,
                        channel_idx=region['channel_idx'],
                        height_slice=region['height_slice'],
                        width_slice=region['width_slice'],
                        method='backward'
                    )
                else:
                    # Multiple regions without exact positions - combine using bounding box
                    print("  Note: Multiple regions - combining into single perturbation")

                    # Collect all channels
                    all_channels = [r['channel_idx'] for r in perturbation_regions]

                    # CRITICAL FIX: Remove duplicates to avoid MaskedPerturbationLpNorm issues
                    # Preserve order while removing duplicates
                    seen = set()
                    unique_channels = []
                    for ch in all_channels:
                        if ch not in seen:
                            seen.add(ch)
                            unique_channels.append(ch)

                    if len(unique_channels) < len(all_channels):
                        print(f"  Removed {len(all_channels) - len(unique_channels)} duplicate channels")
                        print(f"  Using unique channels: {unique_channels}")

                    # For spatial, use bounding box of all regions
                    if is_conv and perturbation_regions[0]['height_slice'] is not None:
                        all_h_starts = [r['height_slice'][0] for r in perturbation_regions]
                        all_h_ends = [r['height_slice'][1] for r in perturbation_regions]
                        all_w_starts = [r['width_slice'][0] for r in perturbation_regions]
                        all_w_ends = [r['width_slice'][1] for r in perturbation_regions]

                        combined_h_slice = (min(all_h_starts), max(all_h_ends))
                        combined_w_slice = (min(all_w_starts), max(all_w_ends))
                    else:
                        combined_h_slice = None
                        combined_w_slice = None

                    lb, ub = self.compute_perturbed_bounds(
                        image=image,
                        node_name=node_name,
                        epsilon=epsilon,
                        batch_idx=0,
                        channel_idx=unique_channels,  # Use unique channels instead of all_channels
                        height_slice=combined_h_slice,
                        width_slice=combined_w_slice,
                        method='backward'
                    )

                # Display results
                print("\n" + "="*80)
                print("RESULTS")
                print("="*80)

                # Clean output - show all 43 classes
                print(f"\nClean Output (no perturbation):")
                print(f"  Predicted class: {pred_class}")
                print(f"\n  Logits for all 43 classes:")
                print(f"  {'Class':>5} | {'Logit':>10} | {'Status'}")
                print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*20}")

                for cls in range(43):
                    logit_val = clean_output[0, cls].item()
                    status = ""
                    if cls == pred_class:
                        status = "PREDICTED"
                    if cls == label:
                        status += (" / " if status else "") + "TRUE LABEL"
                    print(f"  {cls:>5} | {logit_val:>10.4f} | {status}")

                # Compute robustness metrics
                pred_lb = lb[0, pred_class].item()
                true_lb = lb[0, label].item()

                # Find max upper bound among OTHER classes
                other_ub = ub[0].clone()
                other_ub[pred_class] = -float('inf')
                max_other_ub_val = other_ub.max().item()
                max_other_ub_cls = other_ub.argmax().item()

                is_robust = pred_lb > max_other_ub_val

                # Bounds table - show all 43 classes with highlighting
                print(f"\nBounds with Perturbation:")
                print(f"\n  {'Class':>5} | {'Lower Bound':>12} | {'Upper Bound':>12} | {'Width':>10} | {'Notes'}")
                print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*40}")

                for cls in range(43):
                    lb_val = lb[0, cls].item()
                    ub_val = ub[0, cls].item()
                    width = ub_val - lb_val

                    notes = []

                    # Mark predicted class
                    if cls == pred_class:
                        notes.append("PRED")

                    # Mark true label
                    if cls == label:
                        notes.append("TRUE")

                    # Highlight critical bounds for robustness verification
                    if cls == pred_class:
                        notes.append(f"LB={lb_val:.4f}")

                    if cls == max_other_ub_cls:
                        notes.append(f"MAX_OTHER_UB={ub_val:.4f}")

                    notes_str = " | ".join(notes)

                    print(f"  {cls:>5} | {lb_val:>12.6f} | {ub_val:>12.6f} | {width:>10.6f} | {notes_str}")

                # Robustness verification summary
                print("\n" + "="*80)
                print("ROBUSTNESS VERIFICATION")
                print("="*80)

                print(f"\nKey Bounds:")
                print(f"  Predicted class {pred_class}:")
                print(f"    Lower bound: {pred_lb:.6f}")
                print(f"  Max other class (class {max_other_ub_cls}):")
                print(f"    Upper bound: {max_other_ub_val:.6f}")
                print(f"  Margin: {pred_lb - max_other_ub_val:.6f}")

                if is_robust:
                    print(f"\n  ✓ Prediction is VERIFIED ROBUST")
                    print(f"    Predicted class {pred_class} lower bound ({pred_lb:.6f}) >")
                    print(f"    Max other class upper bound ({max_other_ub_val:.6f})")
                else:
                    print(f"\n  ✗ Prediction is NOT verified robust")
                    print(f"    Predicted class {pred_class} lower bound: {pred_lb:.6f}")
                    print(f"    Max other class upper bound: {max_other_ub_val:.6f}")
                    print(f"    Gap: {max_other_ub_val - pred_lb:.6f}")

                print("\n" + "="*80)

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()

            # Ask if continue
            cont = input("\nTest another configuration? (y/n): ")
            if cont.lower() != 'y':
                break

        print("\nThank you for using the Interactive Perturbation Tester!")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive testing of intermediate layer perturbations'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to GTSRB dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--correct_samples_dir', type=str,
                       default='correct_samples',
                       help='Directory with correct sample indices')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Create model (TrafficSignNetSmall is the default)
    model = TrafficSignNetSmall(num_classes=43)

    # Create tester
    tester = InteractiveTester(
        model=model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )

    # Run interactive session
    tester.run_interactive()


if __name__ == '__main__':
    main()
