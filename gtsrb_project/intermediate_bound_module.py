"""
Extended BoundedModule to support intermediate layer perturbations
"""
import torch
import torch.nn as nn
import sys
import os
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

# Add auto_LiRPA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from masked_perturbation import MaskedPerturbationLpNorm


class IntermediateBoundedModule(BoundedModule):
    """
    Extended BoundedModule that supports perturbations on intermediate layers

    Key features:
    - Register perturbations on specific intermediate layer outputs
    - Support masked perturbations (spatial regions, channels, etc.)
    - Compute bounds from intermediate perturbation points
    """

    def __init__(self, model, global_input, *args, **kwargs):
        """
        Initialize IntermediateBoundedModule

        Args:
            model: PyTorch model
            global_input: Dummy input for tracing
            *args, **kwargs: Additional arguments for BoundedModule
        """
        super().__init__(model, global_input, *args, **kwargs)

        # Storage for intermediate perturbations
        self.intermediate_perturbations: Dict[str, MaskedPerturbationLpNorm] = {}

        # Storage for intermediate layer outputs (forward values)
        self.intermediate_outputs: Dict[str, torch.Tensor] = {}

    def register_intermediate_perturbation(
        self,
        node_name: str,
        perturbation: MaskedPerturbationLpNorm
    ):
        """
        Register a perturbation to be applied at an intermediate layer

        Args:
            node_name: Name of the node/layer in the computation graph
            perturbation: MaskedPerturbationLpNorm object
        """
        # Verify node exists
        if node_name not in self._modules:
            available_nodes = list(self._modules.keys())
            raise ValueError(
                f"Node '{node_name}' not found in model. "
                f"Available nodes: {available_nodes}"
            )

        # Store perturbation
        self.intermediate_perturbations[node_name] = perturbation
        print(f"Registered perturbation for node: {node_name}")

    def clear_intermediate_perturbations(self):
        """Clear all intermediate perturbations"""
        self.intermediate_perturbations.clear()
        self.intermediate_outputs.clear()

    def get_node_by_name(self, node_name: str):
        """Get node object by name"""
        if node_name not in self._modules:
            raise ValueError(f"Node '{node_name}' not found")
        return self._modules[node_name]

    def compute_bounds_with_intermediate_perturbation(
        self,
        x=None,
        aux=None,
        C=None,
        method='backward',
        return_full=False,
        **kwargs
    ):
        """
        Compute bounds with intermediate layer perturbations

        This is the main method that handles intermediate perturbations.

        Args:
            x: Input data (can be None if no input perturbation)
            aux: Auxiliary information
            C: Specification matrix
            method: Bound computation method ('backward', 'forward', 'IBP')
            return_full: Whether to return full bound information

        Returns:
            Lower and upper bounds at output layer
        """
        if not self.intermediate_perturbations:
            # No intermediate perturbations, use standard bound computation
            return self.compute_bounds(x=x, aux=aux, C=C, method=method, **kwargs)

        # Step 1: Forward pass to get intermediate layer outputs
        print("Step 1: Computing forward values...")
        if x is not None:
            _ = self.forward(x)
        else:
            # If no input provided, we need to have run forward pass before
            if not hasattr(self, '_forward_value'):
                raise ValueError("Must provide input x or run forward pass first")

        # Step 2: Collect intermediate outputs for perturbed nodes
        print("Step 2: Collecting intermediate layer outputs...")
        for node_name in self.intermediate_perturbations.keys():
            node = self.get_node_by_name(node_name)
            if hasattr(node, 'forward_value') and node.forward_value is not None:
                self.intermediate_outputs[node_name] = node.forward_value.detach()
                print(f"  Collected output from {node_name}: shape {node.forward_value.shape}")
            else:
                raise ValueError(
                    f"Node '{node_name}' does not have forward_value. "
                    f"Make sure forward pass has been executed."
                )

        # Step 3: Apply perturbations to intermediate nodes
        print("Step 3: Applying perturbations to intermediate nodes...")
        for node_name, perturbation in self.intermediate_perturbations.items():
            node = self.get_node_by_name(node_name)
            intermediate_output = self.intermediate_outputs[node_name]

            # Initialize perturbation bounds
            bounds, center, aux_new = perturbation.init(
                intermediate_output,
                aux=aux,
                forward=(method == 'forward')
            )

            # Mark node as perturbed
            node.perturbed = True

            # Store bounds in the node
            # We're treating this intermediate node as a "perturbed input"
            node.lower = bounds.lower
            node.upper = bounds.upper

            # For backward mode, we need to set interval
            node.interval = (bounds.lower, bounds.upper)

            print(f"  Applied perturbation to {node_name}")
            print(f"    Original shape: {intermediate_output.shape}")
            print(f"    Perturbed elements: {(bounds.lower != bounds.upper).sum().item()}")

        # Step 4: Compute bounds from intermediate nodes to output
        print("Step 4: Computing bounds from intermediate layers to output...")

        # We need to propagate bounds from the perturbed nodes to the output
        # This requires modifying the bound computation to start from intermediate nodes

        if method == 'IBP' or method == 'interval':
            # Use Interval Bound Propagation
            return self._compute_bounds_IBP_from_intermediate()

        elif method == 'backward' or method == 'CROWN':
            # Use backward LiRPA (CROWN)
            return self._compute_bounds_backward_from_intermediate(C=C, **kwargs)

        elif method == 'forward':
            # Use forward LiRPA
            return self._compute_bounds_forward_from_intermediate(C=C, **kwargs)

        else:
            raise ValueError(f"Unsupported method: {method}")

    def _compute_bounds_IBP_from_intermediate(self):
        """
        Compute bounds using IBP starting from intermediate perturbed nodes

        Returns:
            lb, ub: Lower and upper bounds at output
        """
        # Find all perturbed nodes
        perturbed_nodes = [
            self.get_node_by_name(name)
            for name in self.intermediate_perturbations.keys()
        ]

        # We need to propagate intervals from these nodes to the output
        # Auto_LiRPA's IBP starts from input nodes, so we need to adapt

        # For simplicity, we'll use the interval_propagate method
        # First, mark ancestors of perturbed nodes as clean (not perturbed)
        # Then propagate from perturbed nodes forward

        # Get output node (usually the last node)
        output_node = self._modules[self.final_name]

        # Use recursive interval propagation
        def propagate_interval(node):
            """Recursively propagate intervals"""
            # If this node is perturbed, use its bounds
            if node.name in self.intermediate_perturbations:
                return node.lower, node.upper

            # If this node has already computed interval, return it
            if hasattr(node, 'interval') and node.interval is not None:
                return node.interval

            # Otherwise, compute from inputs
            if hasattr(node, 'interval_propagate'):
                node.interval = node.interval_propagate()
                return node.interval
            else:
                # Fallback: use forward value as both bounds (no perturbation)
                if hasattr(node, 'forward_value'):
                    return node.forward_value, node.forward_value
                else:
                    return None, None

        # Propagate to output
        lower, upper = propagate_interval(output_node)

        return lower, upper

    def _compute_bounds_backward_from_intermediate(self, C=None, **kwargs):
        """
        Compute bounds using backward mode (CROWN) from intermediate nodes

        Args:
            C: Specification matrix

        Returns:
            lb, ub: Lower and upper bounds
        """
        # The key challenge: backward mode usually starts from output and propagates to input
        # With intermediate perturbations, we need to:
        # 1. Propagate backward from output to the perturbed intermediate node
        # 2. Concretize bounds at the perturbed node using its perturbation spec

        # Find the perturbed node(s)
        perturbed_node_names = list(self.intermediate_perturbations.keys())

        if len(perturbed_node_names) > 1:
            print("Warning: Multiple intermediate perturbations detected.")
            print("Bounds will be computed considering all perturbations.")

        # Use standard backward bound computation
        # But we need to modify the concretization step

        # Call compute_bounds with bound_lower=True and bound_upper=True
        # Set the starting point (node_start) to the intermediate node

        # For now, use a simplified approach:
        # Compute bounds using standard method but starting from intermediate node

        try:
            # Method 1: Try using standard compute_bounds
            # This works if the perturbation has been properly initialized
            lb, ub = self.compute_bounds(
                x=None,  # No input perturbation
                C=C,
                method='backward',
                bound_lower=True,
                bound_upper=True,
                **kwargs
            )
            return lb, ub

        except Exception as e:
            print(f"Error in standard compute_bounds: {e}")
            print("Falling back to IBP method...")
            return self._compute_bounds_IBP_from_intermediate()

    def _compute_bounds_forward_from_intermediate(self, C=None, **kwargs):
        """
        Compute bounds using forward mode from intermediate nodes

        Returns:
            lb, ub: Lower and upper bounds
        """
        # Forward mode propagates bounds layer by layer
        # We've already initialized bounds at intermediate nodes

        try:
            lb, ub = self.compute_bounds(
                x=None,
                C=C,
                method='forward',
                **kwargs
            )
            return lb, ub
        except Exception as e:
            print(f"Error in forward mode: {e}")
            print("Falling back to IBP method...")
            return self._compute_bounds_IBP_from_intermediate()

    def get_layer_names(self, layer_types=['Conv', 'Linear']) -> List[Tuple[str, str]]:
        """
        Get names of all layers of specified types

        Args:
            layer_types: List of layer type strings to include

        Returns:
            List of (node_name, layer_type) tuples
        """
        layers = []
        for name, node in self._modules.items():
            node_type = type(node).__name__
            # Check if any of the layer_types is in the node_type
            if any(lt in node_type for lt in layer_types):
                layers.append((name, node_type))
        return layers

    def print_model_structure(self, show_shapes=True):
        """
        Print the model structure showing all nodes

        Args:
            show_shapes: Whether to show output shapes
        """
        print("\n" + "="*70)
        print("Model Structure (Computation Graph)")
        print("="*70)

        for i, (name, node) in enumerate(self._modules.items()):
            node_type = type(node).__name__
            shape_str = ""

            if show_shapes and hasattr(node, 'output_shape'):
                if node.output_shape is not None:
                    shape_str = f" | Shape: {node.output_shape}"

            print(f"{i:3d}. {name:40s} | Type: {node_type:20s}{shape_str}")

        print("="*70)


if __name__ == '__main__':
    # Test the IntermediateBoundedModule
    print("Testing IntermediateBoundedModule...")

    # Create a simple test model
    class SimpleConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(32 * 16 * 16, 10)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    # Create model and bounded module
    model = SimpleConvNet()
    dummy_input = torch.randn(1, 3, 32, 32)

    print("\nCreating IntermediateBoundedModule...")
    lirpa_model = IntermediateBoundedModule(model, dummy_input)

    # Print structure
    lirpa_model.print_model_structure()

    # Get Conv and FC layers
    conv_fc_layers = lirpa_model.get_layer_names(['Conv', 'Linear'])
    print(f"\nFound {len(conv_fc_layers)} Conv/Linear layers:")
    for name, ltype in conv_fc_layers:
        print(f"  {name:40s} | {ltype}")

    print("\nTest completed successfully!")
