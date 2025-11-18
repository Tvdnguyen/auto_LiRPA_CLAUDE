"""
SCA Automation Script (sca_au.py)
Automated script for testing intermediate layer perturbations from fault simulation results
Simplified version that only loads from fault simulation files (no manual input)
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
    Parse fault simulation result file to extract affected elements

    Args:
        file_path: Path to .txt result file from fault simulator

    Returns:
        Dictionary with:
            - layer_name: Name of affected layer
            - dataflow: Dataflow type (IS/OS/WS)
            - array_size: Array size (e.g., "8x8")
            - affected_channels: List of affected channel indices
            - exact_elements: List of (channel, row, col) tuples
    """
    result = {
        'layer_name': None,
        'dataflow': None,
        'array_size': None,
        'affected_channels': [],
        'exact_elements': []  # List of (channel, row, col) tuples
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

    # Parse affected elements per channel
    current_channel = None

    for line in lines:
        # Detect channel header: "Channel 1:"
        channel_match = re.match(r'^Channel (\d+):', line)
        if channel_match:
            current_channel = int(channel_match.group(1))
            if current_channel not in result['affected_channels']:
                result['affected_channels'].append(current_channel)

        # Detect affected coordinates: "    Row 0: cols [1, 9, 17, 25]"
        coord_match = re.search(r'Row (\d+): cols \[([^\]]+)\]', line)
        if coord_match and current_channel is not None:
            row = int(coord_match.group(1))
            cols_str = coord_match.group(2)
            cols = [int(c.strip()) for c in cols_str.split(',')]

            # Add each (channel, row, col) element
            for col in cols:
                result['exact_elements'].append((current_channel, row, col))

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


def create_exact_mask_from_elements(elements: List[Tuple[int, int, int]], layer_output_shape: Tuple) -> torch.Tensor:
    """
    Create exact binary mask from list of (channel, row, col) elements

    Args:
        elements: List of (channel_idx, row, col) tuples
        layer_output_shape: Shape of layer output - must be torch.Size or tuple

    Returns:
        Boolean mask tensor (True at affected positions, False elsewhere)
    """
    # Convert to tuple if torch.Size
    if hasattr(layer_output_shape, '__iter__'):
        layer_output_shape = tuple(layer_output_shape)

    print(f"  [DEBUG] Creating mask for shape: {layer_output_shape}")
    print(f"  [DEBUG] Shape length: {len(layer_output_shape)}")

    # Create False mask (MUST be bool for torch.where)
    mask = torch.zeros(layer_output_shape, dtype=torch.bool)

    if len(layer_output_shape) != 4:
        raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {layer_output_shape}")

    batch_size, num_channels, height, width = layer_output_shape
    print(f"  [DEBUG] Mask dimensions: B={batch_size}, C={num_channels}, H={height}, W={width}")

    # Set mask to True at specified elements
    skipped = 0
    for channel_idx, row, col in elements:
        # Validate indices
        if channel_idx >= num_channels:
            skipped += 1
            if skipped <= 5:  # Only print first 5 warnings
                print(f"  Warning: Channel {channel_idx} >= {num_channels}, skipping")
            continue
        if row >= height:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: Row {row} >= {height}, skipping")
            continue
        if col >= width:
            skipped += 1
            if skipped <= 5:
                print(f"  Warning: Col {col} >= {width}, skipping")
            continue

        # Set for batch 0 (assuming single batch)
        mask[0, channel_idx, row, col] = True

    if skipped > 5:
        print(f"  Warning: Skipped {skipped} total out-of-bounds elements")

    return mask


class AutomatedTester:
    """Automated testing interface for intermediate perturbations from fault files"""

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
            if model.__class__.__name__ == 'TrafficSignNetSmall':
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
        mask=None,
        method='backward'
    ):
        """
        Compute bounds with intermediate perturbation using exact mask

        Args:
            image: Input image
            node_name: Node name to perturb
            epsilon: Perturbation magnitude
            mask: Custom mask tensor
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
            print(f"  Using exact element mask: {num_perturbed} elements perturbed")
            print(f"  [DEBUG] Mask shape: {mask.shape}")
            print(f"  [DEBUG] Mask dtype: {mask.dtype}")
            print(f"  [DEBUG] Mask device: {mask.device}")

            # Get node's actual output to verify shape match
            node = self.lirpa_model.get_node_by_name(node_name)
            if hasattr(node, 'forward_value') and node.forward_value is not None:
                actual_shape = node.forward_value.shape
                print(f"  [DEBUG] Node forward_value shape: {actual_shape}")
                if mask.shape != actual_shape:
                    print(f"  [ERROR] Shape mismatch!")
                    print(f"    Mask shape: {mask.shape}")
                    print(f"    Expected (node output): {actual_shape}")
                    raise ValueError(f"Mask shape {mask.shape} does not match node output shape {actual_shape}")

            perturbation = MaskedPerturbationLpNorm(
                eps=epsilon,
                norm=np.inf,
                mask=mask.to(self.device)
            )
            print(f"  [DEBUG] Perturbation created successfully")
        else:
            raise ValueError("Mask is required for automated testing")

        # Register perturbation
        self.lirpa_model.register_intermediate_perturbation(node_name, perturbation)

        # Compute bounds
        print(f"\nComputing bounds using {method} method...")
        lb, ub = self.lirpa_model.compute_bounds_with_intermediate_perturbation(
            x=image,
            method=method
        )

        return lb, ub

    def run_automated(self):
        """Run automated testing session"""
        print("\n" + "="*80)
        print(" "*20 + "SCA AUTOMATION TESTER")
        print(" "*15 + "(Fault Simulation File Reader)")
        print("="*80)

        # Show layers
        self.show_model_layers()

        # Step 1: Select layer
        try:
            layer_idx = int(input("\nSelect layer index (or -1 to quit): "))
            if layer_idx == -1:
                print("Exiting...")
                return

            node_name, layer_type, shape_str = self.get_layer_info(layer_idx)
            print(f"\nSelected layer: {node_name}")
            print(f"  Type: {layer_type}")
            print(f"  Output shape: {shape_str}")

        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}")
            return

        # Step 2: Get sample to test
        try:
            class_id = int(input(f"\nSelect class ID (0-42): "))
            if not 0 <= class_id <= 42:
                print("Class ID must be between 0 and 42")
                return

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
            return

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

        # Step 4: Load fault simulation file
        print("\n" + "-"*80)
        print("Load Fault Simulation Result File:")

        results_dir = os.path.join(os.path.dirname(__file__), '..', 'scalesim_fault_simulator', 'results')
        fault_files = list_fault_result_files(results_dir)

        if len(fault_files) == 0:
            print(f"\n  No fault result files found in {results_dir}")
            return

        # Display available files
        print(f"\n  Available fault simulation results ({len(fault_files)} files):")
        for i, (filename, _) in enumerate(fault_files):
            print(f"    {i+1}. {filename}")

        # Let user select
        try:
            file_idx = int(input(f"\n  Select file (1-{len(fault_files)}): ")) - 1
            if not 0 <= file_idx < len(fault_files):
                print("  Invalid selection.")
                return

            selected_file = fault_files[file_idx][1]
            print(f"\n  Loading: {fault_files[file_idx][0]}")

            # Parse the file
            fault_result = parse_fault_result_file(selected_file)

            # Display extracted elements
            print(f"\n  Fault Simulation Result:")
            print(f"    Layer: {fault_result['layer_name']}")
            print(f"    Dataflow: {fault_result['dataflow']}")
            print(f"    Array Size: {fault_result['array_size']}")
            print(f"    Affected Channels: {fault_result['affected_channels']}")
            print(f"    Total Elements: {len(fault_result['exact_elements'])}")

            # Show extracted elements (first 20 as preview)
            elements = fault_result['exact_elements']

            # Validate that we have elements
            if len(elements) == 0:
                print("\n  ✗ ERROR: No elements extracted from fault file!")
                print("  The file may not contain affected coordinates.")
                return

            print(f"\n  Extracted Elements (Channel, Row, Col):")
            preview_count = min(20, len(elements))
            print(f"  Showing first {preview_count} of {len(elements)} elements:")
            for i, (ch, row, col) in enumerate(elements[:preview_count]):
                print(f"    {i+1:3d}. Channel={ch:2d}, Row={row:2d}, Col={col:2d}")

            if len(elements) > preview_count:
                print(f"    ... and {len(elements) - preview_count} more elements")

            # Ask for confirmation
            confirm = input("\n  Use these elements for perturbation? (y/n): ").strip().lower()
            if confirm != 'y':
                print("  Cancelled.")
                return

            # Get epsilon value
            epsilon = float(input("\n  Epsilon value: "))

            print(f"\n  Summary:")
            print(f"    Total elements: {len(elements)}")
            print(f"    Epsilon: {epsilon}")

        except Exception as e:
            print(f"Invalid input: {e}")
            import traceback
            traceback.print_exc()
            return

        # Step 5: Compute bounds
        print("\n" + "-"*80)
        print("Computing bounds with perturbation...")

        try:
            # Get node to determine layer shape
            print("\n  Detecting layer output shape...")

            # Try to get node from lirpa_model
            node = self.lirpa_model.get_node_by_name(node_name)
            print(f"  Node found: {node_name}")
            print(f"  Node type: {type(node).__name__}")

            # Get shape from node attributes WITHOUT running forward pass yet
            # (forward pass will be run inside compute_perturbed_bounds)
            if hasattr(node, 'output_shape') and node.output_shape is not None:
                if isinstance(node.output_shape, (list, tuple)):
                    layer_shape = tuple(node.output_shape)
                else:
                    layer_shape = node.output_shape
                print(f"  ✓ Shape from node.output_shape: {layer_shape}")
            else:
                # Parse from shape_str
                print(f"  Parsing shape from string: {shape_str}")
                numbers = re.findall(r'\d+', shape_str)
                print(f"  Extracted numbers: {numbers}")
                if len(numbers) == 4:
                    layer_shape = tuple(int(n) for n in numbers)
                elif len(numbers) == 3:
                    layer_shape = (1, int(numbers[0]), int(numbers[1]), int(numbers[2]))
                elif len(numbers) == 2:
                    layer_shape = (1, int(numbers[0]))
                else:
                    raise ValueError(f"Cannot parse shape from: {shape_str}")
                print(f"  ✓ Detected layer shape from parsing: {layer_shape}")

            print(f"  ✓ Using layer shape: {layer_shape}")
            print(f"    Format: (batch={layer_shape[0]}, channels={layer_shape[1]}, height={layer_shape[2] if len(layer_shape) > 2 else 'N/A'}, width={layer_shape[3] if len(layer_shape) > 3 else 'N/A'})")

            # Validate elements against layer shape
            print(f"\n  Validating {len(elements)} elements against layer shape...")
            if len(layer_shape) == 4:
                _, C, H, W = layer_shape
                max_ch = max(e[0] for e in elements)
                max_row = max(e[1] for e in elements)
                max_col = max(e[2] for e in elements)
                print(f"    Layer capacity: C={C}, H={H}, W={W}")
                print(f"    Max indices in elements: ch={max_ch}, row={max_row}, col={max_col}")

                if max_ch >= C or max_row >= H or max_col >= W:
                    print(f"  ✗ ERROR: Element indices exceed layer dimensions!")
                    print(f"    This may indicate a mismatch between fault file layer and selected layer.")
                    return

            print(f"  ✓ All elements are within valid range")

            # Analyze the elements to suggest input method
            from collections import defaultdict
            channel_elements = defaultdict(list)
            for ch, row, col in elements:
                channel_elements[ch].append((row, col))

            print(f"\n  Analysis of extracted elements:")
            print(f"    Total elements: {len(elements)}")
            print(f"    Affected channels: {len(channel_elements)}")

            # Compute bounding box for each channel
            for ch in sorted(channel_elements.keys()):
                positions = channel_elements[ch]
                rows = [r for r, c in positions]
                cols = [r for r, c in positions]
                min_row, max_row = min(rows), max(rows)
                min_col, max_col = min(cols), max(cols)
                bounding_box_size = (max_row - min_row + 1) * (max_col - min_col + 1)
                coverage = 100.0 * len(positions) / bounding_box_size if bounding_box_size > 0 else 0

                print(f"      Channel {ch}: {len(positions)} elements, "
                      f"Bounding box [{min_row}:{max_row+1}, {min_col}:{max_col+1}] "
                      f"({bounding_box_size} total, {coverage:.1f}% coverage)")

            # Ask user how to proceed
            print(f"\n  How to apply perturbation?")
            print(f"    1. Use EXACT elements from file (may cause errors)")
            print(f"    2. Use BOUNDING BOX (regions, like manual option 3 - RECOMMENDED)")

            method_choice = input(f"\n  Select method (1 or 2): ").strip()

            if method_choice == "1":
                # Use exact elements - create mask
                print(f"\n  Creating exact mask from {len(elements)} elements...")
                exact_mask = create_exact_mask_from_elements(elements, layer_shape)
                num_exact = exact_mask.sum().item()
                total = exact_mask.numel()
                print(f"  ✓ Exact mask created: {num_exact}/{total} elements ({100*num_exact/total:.2f}%)")

                if num_exact == 0:
                    print(f"  ✗ ERROR: Mask has 0 elements! Check element coordinates.")
                    return

                # Compute bounds with exact mask
                lb, ub = self.compute_perturbed_bounds(
                    image=image,
                    node_name=node_name,
                    epsilon=epsilon,
                    mask=exact_mask,
                    method='backward'
                )

            elif method_choice == "2":
                # Use bounding box - convert to slice specification
                print(f"\n  Converting to bounding box specification...")

                channel_list = sorted(channel_elements.keys())

                # Find global bounding box across all positions
                all_rows = [row for ch, row, col in elements]
                all_cols = [col for ch, row, col in elements]
                height_min, height_max = min(all_rows), max(all_rows)
                width_min, width_max = min(all_cols), max(all_cols)

                print(f"\n  Using slice specification:")
                print(f"    Channels: {channel_list}")
                print(f"    Height: [{height_min}, {height_max+1})")
                print(f"    Width: [{width_min}, {width_max+1})")

                bounding_box_size = len(channel_list) * (height_max - height_min + 1) * (width_max - width_min + 1)
                print(f"    Bounding box size: {bounding_box_size} elements (includes {bounding_box_size - len(elements)} extra elements)")

                # Compute bounds using slice specification (like manual option 3)
                lb, ub = self.compute_perturbed_bounds(
                    image=image,
                    node_name=node_name,
                    epsilon=epsilon,
                    batch_idx=0,
                    channel_idx=channel_list,
                    height_slice=(height_min, height_max + 1),
                    width_slice=(width_min, width_max + 1),
                    method='backward'
                )

            else:
                print(f"  Invalid choice. Exiting.")
                return

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

        print("\nThank you for using the SCA Automation Tester!")


def main():
    parser = argparse.ArgumentParser(
        description='Automated testing of intermediate layer perturbations from fault files'
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
    tester = AutomatedTester(
        model=model,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        device=args.device
    )

    # Run automated session
    tester.run_automated()


if __name__ == '__main__':
    main()
