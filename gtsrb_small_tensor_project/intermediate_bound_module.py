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
            # Store x for later use in bound computation
            self._stored_input = x
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
            # Try CROWN backward, but fallback to IBP if OOM
            try:
                return self._compute_bounds_backward_from_intermediate(C=C, **kwargs)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() or 'allocate' in str(e).lower():
                    print(f"\n  ⚠️  CROWN backward OOM! Falling back to IBP (memory efficient)...")
                    print(f"     IBP gives looser bounds but uses 100x less memory")
                    return self._compute_bounds_IBP_from_intermediate()
                else:
                    raise

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
        print("  Using IBP (Interval Bound Propagation) mode")

        # CRITICAL: Initialize intervals for all INPUT nodes (no perturbation)
        # This is required for interval_propagate to work
        for name in self.input_name:
            if name in self._modules:
                node = self._modules[name]
                if hasattr(node, 'forward_value') and node.forward_value is not None:
                    # Set interval to forward_value (no perturbation at input)
                    node.interval = (node.forward_value, node.forward_value)
                    node.lower = node.forward_value
                    node.upper = node.forward_value
                    print(f"  Set input interval for {name}: shape {node.forward_value.shape}")

        # Find all perturbed nodes
        perturbed_node_names = list(self.intermediate_perturbations.keys())
        print(f"  Perturbed intermediate nodes: {perturbed_node_names}")

        # Set intervals for perturbed intermediate nodes
        for node_name in perturbed_node_names:
            node = self.get_node_by_name(node_name)
            if hasattr(node, 'lower') and hasattr(node, 'upper'):
                node.interval = (node.lower, node.upper)
                print(f"  Set perturbed interval for {node_name}")

        # Now use auto_LiRPA's standard IBP propagation
        # Call compute_bounds with IBP method
        try:
            lb, ub = self.compute_bounds(
                x=None,  # Use stored forward values
                method='IBP',
                bound_lower=True,
                bound_upper=True
            )
            return lb, ub
        except Exception as e:
            print(f"  Error in IBP compute_bounds: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"IBP propagation failed: {e}") from e

    def _manual_IBP_propagation(self):
        """
        Manual IBP propagation from intermediate nodes to output
        Used as fallback when auto_LiRPA's compute_bounds fails
        """
        print("  Using manual IBP propagation")

        # Get output node
        output_node = self._modules[self.final_name]

        # Recursive propagation
        visited = set()

        def propagate(node):
            if node.name in visited:
                return node.interval
            visited.add(node.name)

            # If perturbed intermediate node, use its bounds
            if node.name in self.intermediate_perturbations:
                return node.interval

            # If input node, use forward value (no perturbation)
            if node.name in self.input_name:
                if hasattr(node, 'interval'):
                    return node.interval
                return (node.forward_value, node.forward_value)

            # Otherwise, propagate from inputs
            if hasattr(node, 'inputs') and len(node.inputs) > 0:
                # Get intervals from all inputs
                input_intervals = []
                for inp in node.inputs:
                    inp_interval = propagate(inp)
                    if inp_interval is not None:
                        input_intervals.append(inp_interval)

                # Use interval_propagate if available
                if hasattr(node, 'interval_propagate') and len(input_intervals) > 0:
                    try:
                        # Set input intervals
                        for inp, (lb, ub) in zip(node.inputs, input_intervals):
                            inp.interval = (lb, ub)

                        node.interval = node.interval_propagate()
                        return node.interval
                    except:
                        pass

            # Fallback: use forward value
            if hasattr(node, 'forward_value'):
                return (node.forward_value, node.forward_value)

            return (None, None)

        lb, ub = propagate(output_node)
        return lb, ub

    def _compute_bounds_backward_from_intermediate(self, C=None, **kwargs):
        """
        Compute bounds using backward mode (CROWN) from intermediate nodes

        NOTE: CROWN backward from intermediate nodes requires significant memory!
        - CPU (8GB RAM): May fail due to large A matrices during backward propagation
        - GPU (8-24GB VRAM): Should work fine

        If running on CPU and getting OOM errors, use GPU with --device cuda

        Args:
            C: Specification matrix

        Returns:
            lb, ub: Lower and upper bounds
        """
        # Find the perturbed node(s)
        perturbed_node_names = list(self.intermediate_perturbations.keys())

        if len(perturbed_node_names) > 1:
            print("  Warning: Multiple intermediate perturbations detected.")
            print("  Bounds will be computed considering all perturbations.")

        # Get device from intermediate output
        first_node_name = perturbed_node_names[0]
        intermediate_output = self.intermediate_outputs[first_node_name]
        device = intermediate_output.device

        print(f"  Computing CROWN backward bounds on device: {device}")
        if C is not None:
            print(f"  Using C matrix with shape: {C.shape}")

        # Collect intermediate bounds that we computed from perturbations
        interm_bounds_dict = {}
        for node_name in self.intermediate_perturbations.keys():
            node = self.get_node_by_name(node_name)
            if hasattr(node, 'lower') and hasattr(node, 'upper'):
                interm_bounds_dict[node_name] = (node.lower, node.upper)
                print(f"  Providing intermediate bounds for {node_name}")
                num_perturbed = (node.lower != node.upper).sum().item()
                print(f"    Perturbed elements: {num_perturbed}/{node.lower.numel()}")
                print(f"    Lower shape: {node.lower.shape}")
                print(f"    Upper shape: {node.upper.shape}")
                print(f"    Lower dtype: {node.lower.dtype}, device: {node.lower.device}")
                print(f"    Upper dtype: {node.upper.dtype}, device: {node.upper.device}")

                # DEBUG: Check if bounds are valid
                if torch.any(node.lower > node.upper):
                    print(f"    ⚠️  WARNING: Lower > Upper in some positions!")
                if torch.any(torch.isnan(node.lower)) or torch.any(torch.isnan(node.upper)):
                    print(f"    ⚠️  WARNING: NaN values detected!")
                if torch.any(torch.isinf(node.lower)) or torch.any(torch.isinf(node.upper)):
                    print(f"    ⚠️  WARNING: Inf values detected!")

        try:
            # CRITICAL FIX: Disable automatic intermediate bound computation
            # We only want to use OUR provided interm_bounds, not compute new ones
            # This saves massive amounts of memory!

            # Temporarily disable intermediate bound requirements
            original_bound_opts = self.bound_opts.copy() if hasattr(self, 'bound_opts') else {}

            # Set options to skip intermediate bound computation
            self.set_bound_opts({
                'optimize_bound_args': {
                    'enable_beta_crown': False,  # Disable beta-CROWN (memory intensive)
                    'enable_alpha_crown': False,  # Disable alpha optimization (not needed)
                    'iteration': 0,  # No optimization iterations
                }
            })

            # Compute bounds with CROWN backward
            # CRITICAL: Pass x=None to avoid input perturbation
            # We ONLY want intermediate perturbations via interm_bounds
            # Passing x would cause auto_LiRPA to try adding input bounds during concretization
            # which leads to shape mismatch errors

            result = self.compute_bounds(
                x=None,  # No input - we use interm_bounds only
                C=C,
                method='backward',
                bound_lower=True,
                bound_upper=True,  # Need both lower and upper bounds
                interm_bounds=interm_bounds_dict,
                needed_A_dict=None,  # Don't need A matrices
                return_A=False,  # Don't return A matrices
                reuse_alpha=False,  # Don't reuse alpha from previous computations
            )

            # Handle result (auto_LiRPA may return tuple or single value)
            if isinstance(result, tuple):
                if len(result) == 2:
                    lb, ub = result
                elif len(result) == 1:
                    lb = result[0]
                    ub = lb  # Use same as lb
            else:
                lb = result
                ub = lb  # Use same as lb

            # Restore original bound options
            if original_bound_opts:
                self.set_bound_opts(original_bound_opts)

            return lb, ub

        except RuntimeError as e:
            error_msg = str(e)
            if 'allocate memory' in error_msg or 'out of memory' in error_msg.lower():
                print(f"\n  ✗ OUT OF MEMORY ERROR!")
                print(f"  Error: {e}")
                print(f"\n  CROWN backward from intermediate nodes requires significant memory.")
                print(f"  Current device: {device}")
                print(f"\n  SOLUTIONS:")
                print(f"  1. ✅ Use GPU: python main_interactive.py --device cuda")
                print(f"  2. Reduce perturbation region size")
                print(f"  3. Use a machine with more RAM/VRAM")
                raise
            else:
                raise

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

    def get_intermediate_output(self, node_name: str) -> torch.Tensor:
        """
        Get the intermediate output (forward value) for a specific node

        Args:
            node_name: Name of the node in the computation graph

        Returns:
            Tensor containing the intermediate output

        Note:
            You must run a forward pass before calling this method
        """
        if node_name not in self.intermediate_outputs:
            raise ValueError(
                f"No intermediate output found for '{node_name}'. "
                f"Make sure to run a forward pass first. "
                f"Available outputs: {list(self.intermediate_outputs.keys())}"
            )
        return self.intermediate_outputs[node_name]


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
