"""
Intermediate Layer Perturbation - Version 2
NEW APPROACH: Use auto_LiRPA's standard workflow with element-wise epsilon

Key idea:
1. Forward pass (clean) from input → intermediate layer
2. Apply perturbation with element-wise epsilon:
   - Selected elements: epsilon = user_epsilon
   - Other elements: epsilon = 0
3. Use standard CROWN backward from intermediate → output

This follows auto_LiRPA's native design instead of hacking it!
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from auto_LiRPA import BoundedModule
from auto_LiRPA.perturbations import PerturbationLpNorm
import sys
import os

# Add gtsrb_project to path for masked perturbation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gtsrb_project'))
from masked_perturbation import MaskedPerturbationLpNorm


class IntermediateBoundedModuleV2(BoundedModule):
    """
    Extended BoundedModule that supports perturbations on intermediate layers

    NEW APPROACH: Treats intermediate layer as a perturbed input
    """

    def __init__(self, model, dummy_input, **kwargs):
        """
        Initialize intermediate perturbation support

        Args:
            model: PyTorch model
            dummy_input: Example input for tracing
        """
        super().__init__(model, dummy_input, **kwargs)

        # Storage for intermediate perturbations and outputs
        self.intermediate_perturbations = {}
        self.intermediate_outputs = {}

    def set_intermediate_perturbation(
        self,
        node_name: str,
        perturbation: MaskedPerturbationLpNorm
    ):
        """
        Set perturbation for an intermediate layer

        Args:
            node_name: Name of the intermediate node to perturb
            perturbation: MaskedPerturbationLpNorm object with element-wise epsilon
        """
        if node_name not in self._modules:
            raise ValueError(f"Node '{node_name}' not found in model")

        self.intermediate_perturbations[node_name] = perturbation
        print(f"Set intermediate perturbation for node: {node_name}")

    def clear_intermediate_perturbations(self):
        """Clear all intermediate perturbations"""
        self.intermediate_perturbations.clear()
        self.intermediate_outputs.clear()

    def register_intermediate_perturbation(
        self,
        node_name: str,
        perturbation: MaskedPerturbationLpNorm
    ):
        """
        Alias for set_intermediate_perturbation() for backward compatibility

        Args:
            node_name: Name of the intermediate node to perturb
            perturbation: MaskedPerturbationLpNorm object with element-wise epsilon
        """
        return self.set_intermediate_perturbation(node_name, perturbation)

    def compute_bounds_from_intermediate(
        self,
        x: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        method: str = 'backward'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounds with intermediate layer perturbation

        NEW WORKFLOW:
        1. Clean forward pass: input → intermediate layer
        2. Apply element-wise epsilon perturbation to intermediate output
        3. Standard CROWN backward: intermediate → output

        Args:
            x: Input tensor (clean, no perturbation)
            C: Specification matrix for output bounds
            method: Bound computation method ('backward' recommended)

        Returns:
            lb, ub: Lower and upper bounds at output
        """
        if not self.intermediate_perturbations:
            raise ValueError("No intermediate perturbations set. Use set_intermediate_perturbation() first")

        if len(self.intermediate_perturbations) != 1:
            raise NotImplementedError("Currently only support single intermediate perturbation")

        # Get the perturbed node
        node_name = list(self.intermediate_perturbations.keys())[0]
        perturbation = self.intermediate_perturbations[node_name]

        print(f"\n{'='*80}")
        print(f"Computing bounds with intermediate perturbation")
        print(f"{'='*80}")

        # Step 1: Clean forward pass to get intermediate output
        print(f"\nStep 1: Clean forward pass from input → {node_name}")
        with torch.no_grad():
            _ = self.forward(x)

        node = self.get_node_by_name(node_name)
        if not hasattr(node, 'forward_value') or node.forward_value is None:
            raise RuntimeError(f"Node {node_name} has no forward_value")

        intermediate_output = node.forward_value.detach()
        print(f"  Intermediate output shape: {intermediate_output.shape}")

        # Step 2: Apply element-wise epsilon perturbation
        print(f"\nStep 2: Apply element-wise epsilon perturbation")
        print(f"  Creating epsilon tensor (user_eps for selected, 0 for others)...")

        # Use perturbation.init() to get bounds with element-wise epsilon
        bounds, center, _ = perturbation.init(intermediate_output, forward=False)

        lower = bounds.lower
        upper = bounds.upper

        num_perturbed = (lower != upper).sum().item()
        total = lower.numel()
        print(f"  ✓ Perturbation applied:")
        print(f"    Selected elements: {num_perturbed}/{total} ({100*num_perturbed/total:.2f}%)")
        print(f"    Non-selected elements: {total - num_perturbed} (epsilon=0)")

        # Step 3: Run CROWN backward from intermediate layer
        print(f"\nStep 3: CROWN backward from {node_name} → output")

        # KEY INSIGHT: We need to tell auto_LiRPA to use these bounds
        # as "input bounds" for the sub-computation from intermediate to output

        # Set bounds on the node
        node.lower = lower
        node.upper = upper
        node.interval = (lower, upper)
        node.perturbed = True

        # Now compute bounds using standard auto_LiRPA workflow
        # Pass the intermediate bounds via interm_bounds parameter
        interm_bounds = {node_name: (lower, upper)}

        print(f"  Calling compute_bounds with interm_bounds for {node_name}...")

        try:
            result = self.compute_bounds(
                x=None,  # No input perturbation
                C=C,
                method=method,
                bound_lower=True,
                bound_upper=True,
                interm_bounds=interm_bounds,  # Provide intermediate bounds
                return_A=False
            )

            # Parse result
            if isinstance(result, tuple):
                lb, ub = result if len(result) == 2 else (result[0], result[0])
            else:
                lb = ub = result

            print(f"  ✓ Bounds computed successfully")
            print(f"    Output lower bound shape: {lb.shape}")
            print(f"    Output upper bound shape: {ub.shape}")

            return lb, ub

        except Exception as e:
            print(f"\n  ✗ Error in compute_bounds: {e}")
            import traceback
            traceback.print_exc()
            raise

    def compute_bounds_with_intermediate_perturbation(
        self,
        x: Optional[torch.Tensor] = None,
        aux: Optional[torch.Tensor] = None,
        C: Optional[torch.Tensor] = None,
        method: str = 'backward',
        return_full: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Alias for compute_bounds_from_intermediate() for backward compatibility

        This method name matches the old API but uses the new V2 implementation

        Args:
            x: Input tensor (clean, no perturbation)
            aux: Auxiliary information (unused in V2)
            C: Specification matrix
            method: Bound computation method ('backward', 'forward', 'IBP')
            return_full: Whether to return full bound information (unused in V2)
            **kwargs: Additional arguments

        Returns:
            lb, ub: Lower and upper bounds at output
        """
        return self.compute_bounds_from_intermediate(x=x, C=C, method=method)

    def get_node_by_name(self, node_name: str):
        """Get node by name"""
        if node_name not in self._modules:
            raise ValueError(f"Node '{node_name}' not found")
        return self._modules[node_name]

    def get_layer_names(self, layer_types=['Conv', 'Linear']) -> List[Tuple[str, str]]:
        """Get names of layers of specified types"""
        layers = []
        for name, node in self._modules.items():
            node_type = type(node).__name__
            if any(lt in node_type for lt in layer_types):
                layers.append((name, node_type))
        return layers

    def print_model_structure(self, show_shapes=True):
        """
        Print the model structure showing all nodes

        Args:
            show_shapes: Whether to show output shapes
        """
        print("\nModel Structure:")
        print("-" * 80)
        for name, node in self._modules.items():
            node_type = type(node).__name__
            if show_shapes and hasattr(node, 'output_shape'):
                shape = node.output_shape
                print(f"  {name:40s} {node_type:20s} {str(shape)}")
            else:
                print(f"  {name:40s} {node_type}")
        print("-" * 80)

    def compute_perturbed_bounds(
        self,
        x: torch.Tensor,
        node_name: str,
        epsilon: float,
        mask: Optional[torch.Tensor] = None,
        batch_idx: Optional[int] = None,
        channel_idx: Optional[List[int]] = None,
        height_slice: Optional[Tuple[int, int]] = None,
        width_slice: Optional[Tuple[int, int]] = None,
        method: str = 'backward'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute bounds with intermediate layer perturbation

        Convenient wrapper that creates MaskedPerturbationLpNorm and calls
        compute_bounds_from_intermediate()

        Args:
            x: Input tensor (clean)
            node_name: Intermediate node to perturb
            epsilon: Perturbation magnitude
            mask: Optional boolean mask (if provided, overrides other params)
            batch_idx: Batch index to perturb
            channel_idx: Channel indices to perturb (list)
            height_slice: Height slice (start, end)
            width_slice: Width slice (start, end)
            method: Bound computation method

        Returns:
            lb, ub: Output bounds
        """
        import numpy as np

        # Create perturbation with element-wise epsilon
        perturbation = MaskedPerturbationLpNorm(
            eps=epsilon,
            norm=np.inf,
            mask=mask,
            batch_idx=batch_idx,
            channel_idx=channel_idx,
            height_slice=height_slice,
            width_slice=width_slice
        )

        # Set perturbation
        self.set_intermediate_perturbation(node_name, perturbation)

        # Compute bounds
        lb, ub = self.compute_bounds_from_intermediate(x=x, C=None, method=method)

        # Clear perturbation
        self.intermediate_perturbations.clear()

        return lb, ub


if __name__ == '__main__':
    print("Testing IntermediateBoundedModuleV2...")

    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.fc = nn.Linear(32 * 8 * 8, 10)

        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleModel()
    model.eval()

    dummy_input = torch.randn(1, 3, 8, 8)

    # Wrap with V2
    lirpa_model = IntermediateBoundedModuleV2(model, dummy_input)

    # List layers
    print("\nAvailable layers:")
    for name, type_name in lirpa_model.get_layer_names():
        print(f"  {name}: {type_name}")

    print("\nTest passed!")
