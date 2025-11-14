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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from traffic_sign_net import TrafficSignNet, TrafficSignNetSimple, TrafficSignNetNoDropout
from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms
from masked_perturbation import MaskedPerturbationLpNorm
from intermediate_bound_module import IntermediateBoundedModule
from collect_correct_samples import load_correct_indices


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

            # Step 4: Configure perturbation
            print("\n" + "-"*80)
            print("Configure Perturbation Region:")

            # Determine if Conv or FC layer
            is_conv = 'Conv' in layer_type

            try:
                if is_conv:
                    # Conv layer: ask for channel, height, width
                    channel_input = input("  Channel index (or 'all', or comma-separated list): ")
                    if channel_input.lower() == 'all':
                        channel_idx = None
                    elif ',' in channel_input:
                        channel_idx = [int(x.strip()) for x in channel_input.split(',')]
                    else:
                        channel_idx = int(channel_input)

                    height_input = input("  Height slice (start,end) or 'all': ")
                    if height_input.lower() == 'all':
                        height_slice = None
                    else:
                        h_start, h_end = map(int, height_input.split(','))
                        height_slice = (h_start, h_end)

                    width_input = input("  Width slice (start,end) or 'all': ")
                    if width_input.lower() == 'all':
                        width_slice = None
                    else:
                        w_start, w_end = map(int, width_input.split(','))
                        width_slice = (w_start, w_end)

                else:
                    # FC layer: ask for feature dimensions
                    feature_input = input("  Feature indices (comma-separated or 'all'): ")
                    if feature_input.lower() == 'all':
                        channel_idx = None
                    elif ',' in feature_input:
                        channel_idx = [int(x.strip()) for x in feature_input.split(',')]
                    else:
                        channel_idx = int(feature_input)

                    height_slice = None
                    width_slice = None

                epsilon = float(input("  Epsilon value: "))

            except Exception as e:
                print(f"Invalid input: {e}")
                continue

            # Step 5: Compute bounds
            print("\n" + "-"*80)
            print("Computing bounds with perturbation...")

            try:
                lb, ub = self.compute_perturbed_bounds(
                    image=image,
                    node_name=node_name,
                    epsilon=epsilon,
                    batch_idx=0,
                    channel_idx=channel_idx,
                    height_slice=height_slice,
                    width_slice=width_slice,
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
    parser.add_argument('--model', type=str, default='full',
                       choices=['full', 'simple'],
                       help='Model architecture')
    parser.add_argument('--correct_samples_dir', type=str,
                       default='correct_samples',
                       help='Directory with correct sample indices')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Create model
    if args.model == 'full':
        model = TrafficSignNet(num_classes=43)
    else:
        model = TrafficSignNetSimple(num_classes=43)

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
