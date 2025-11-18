"""
Masked Perturbation for Intermediate Layers
Extends auto_LiRPA to support perturbations on specific regions of intermediate layer outputs
"""
import torch
import numpy as np
import sys
import os

# Add auto_LiRPA to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.linear_bound import LinearBound
from auto_LiRPA.utils import eyeC


class MaskedPerturbationLpNorm(PerturbationLpNorm):
    """
    Masked Lp-norm perturbation for intermediate layer outputs

    Allows perturbation on only a subset of tensor elements specified by a mask.
    Useful for adding perturbations to specific regions/channels of intermediate layers.
    """

    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None, eps_min=0,
                 mask=None, batch_idx=None, channel_idx=None,
                 height_slice=None, width_slice=None):
        """
        Args:
            eps: Perturbation magnitude
            norm: Lp norm (default: np.inf for L-infinity)
            x_L: Optional explicit lower bounds
            x_U: Optional explicit upper bounds
            eps_min: Minimum epsilon
            mask: Boolean tensor mask specifying which elements to perturb
            batch_idx: Batch index to perturb (int or list)
            channel_idx: Channel index to perturb (int or list)
            height_slice: Height slice (tuple: start, end) or list of indices
            width_slice: Width slice (tuple: start, end) or list of indices
        """
        super().__init__(eps=eps, norm=norm, x_L=x_L, x_U=x_U, eps_min=eps_min)

        # Store mask specification
        self.mask = mask
        self.batch_idx = batch_idx
        self.channel_idx = channel_idx
        self.height_slice = height_slice
        self.width_slice = width_slice

        # Will be set during init()
        self.full_mask = None

    def _create_mask_from_spec(self, shape):
        """
        Create full mask tensor from specifications

        Args:
            shape: Target tensor shape (B, C, H, W) or (B, D)

        Returns:
            Boolean mask tensor
        """
        # Start with all False (no perturbation)
        mask = torch.zeros(shape, dtype=torch.bool)

        # Handle different tensor shapes
        if len(shape) == 4:  # Conv layer output: (B, C, H, W)
            batch_size, channels, height, width = shape

            # Determine batch indices
            if self.batch_idx is None:
                batch_indices = list(range(batch_size))
            elif isinstance(self.batch_idx, int):
                batch_indices = [self.batch_idx]
            else:
                batch_indices = self.batch_idx

            # Determine channel indices
            if self.channel_idx is None:
                channel_indices = list(range(channels))
            elif isinstance(self.channel_idx, int):
                channel_indices = [self.channel_idx]
            else:
                channel_indices = self.channel_idx

            # Determine height range
            if self.height_slice is None:
                height_range = list(range(height))
            elif isinstance(self.height_slice, tuple) and len(self.height_slice) == 2:
                height_range = list(range(self.height_slice[0], self.height_slice[1]))
            else:
                height_range = self.height_slice

            # Determine width range
            if self.width_slice is None:
                width_range = list(range(width))
            elif isinstance(self.width_slice, tuple) and len(self.width_slice) == 2:
                width_range = list(range(self.width_slice[0], self.width_slice[1]))
            else:
                width_range = self.width_slice

            # Set mask to True for specified regions
            for b in batch_indices:
                for c in channel_indices:
                    for h in height_range:
                        for w in width_range:
                            mask[b, c, h, w] = True

        elif len(shape) == 2:  # FC layer output: (B, D)
            batch_size, dim = shape

            # For FC layers, use batch_idx and channel_idx as feature indices
            if self.batch_idx is None:
                batch_indices = list(range(batch_size))
            elif isinstance(self.batch_idx, int):
                batch_indices = [self.batch_idx]
            else:
                batch_indices = self.batch_idx

            # For FC, channel_idx represents feature dimensions
            if self.channel_idx is None:
                feature_indices = list(range(dim))
            elif isinstance(self.channel_idx, int):
                feature_indices = [self.channel_idx]
            else:
                feature_indices = self.channel_idx

            for b in batch_indices:
                for f in feature_indices:
                    mask[b, f] = True
        else:
            raise ValueError(f"Unsupported shape: {shape}")

        return mask

    def get_input_bounds(self, x, A):
        """
        Get input bounds with mask applied

        Args:
            x: Center value (unperturbed output)
            A: Bound matrix

        Returns:
            x_L, x_U: Lower and upper bounds with mask applied
        """
        # Create mask if not already created
        if self.full_mask is None:
            self.full_mask = self._create_mask_from_spec(x.shape).to(x.device)

        # Start with center value (no perturbation)
        if self.x_L is not None and self.x_U is not None:
            x_L = self.x_L.clone()
            x_U = self.x_U.clone()
        else:
            x_L = x.clone()
            x_U = x.clone()

        # Apply perturbation only to masked elements
        if self.norm == np.inf:
            # For L-infinity, directly add/subtract eps
            x_L = torch.where(self.full_mask, x - self.eps, x)
            x_U = torch.where(self.full_mask, x + self.eps, x)
        else:
            # For other norms, more complex handling needed
            # For now, we support L-infinity primarily
            raise NotImplementedError(f"Masked perturbation for L{self.norm} norm not yet implemented")

        return x_L, x_U

    def init(self, x, aux=None, forward=False):
        """
        Initialize bounds for masked perturbation

        IMPORTANT: For masked intermediate perturbations, we ALWAYS use backward mode
        (CROWN backward) to avoid issues with sparse identity matrices in forward mode.

        Args:
            x: Intermediate layer output tensor
            aux: Auxiliary information
            forward: Whether using forward mode (IGNORED - always use backward)

        Returns:
            LinearBound object, center, aux
        """
        # Create mask
        if self.mask is not None:
            self.full_mask = self.mask.to(x.device)
        else:
            self.full_mask = self._create_mask_from_spec(x.shape).to(x.device)

        # Get masked bounds
        x_L, x_U = self.get_input_bounds(x, None)

        # ALWAYS use backward mode for masked perturbations
        # Setting lw=None, uw=None forces CROWN backward bound propagation
        # This avoids dimension mismatch issues with sparse identity matrices
        #
        # Reason: When only a few elements are perturbed (sparse mask),
        # forward mode creates a nearly-zero identity matrix which causes:
        # 1. Numerical instability
        # 2. Dimension mismatch in matrix operations
        # 3. Incorrect bound computation
        #
        # Backward mode (CROWN) only needs x_L and x_U bounds, which works
        # correctly with any sparsity level.
        return LinearBound(None, None, None, None, x_L, x_U), x, aux

    def __repr__(self):
        mask_info = "full" if self.mask is None else "masked"
        return f'MaskedPerturbationLpNorm(norm={self.norm}, eps={self.eps}, {mask_info})'


def create_region_mask(shape, batch_idx=None, channel_idx=None,
                       height_slice=None, width_slice=None):
    """
    Helper function to create a boolean mask for a specific region

    Args:
        shape: Tensor shape (B, C, H, W) or (B, D)
        batch_idx: Batch index (int or list)
        channel_idx: Channel index (int or list)
        height_slice: Height slice (tuple or list)
        width_slice: Width slice (tuple or list)

    Returns:
        Boolean mask tensor
    """
    ptb = MaskedPerturbationLpNorm(
        eps=0,  # Dummy value
        batch_idx=batch_idx,
        channel_idx=channel_idx,
        height_slice=height_slice,
        width_slice=width_slice
    )
    return ptb._create_mask_from_spec(shape)


if __name__ == '__main__':
    # Test masked perturbation
    print("Testing MaskedPerturbationLpNorm...")

    # Test Conv layer output
    print("\n1. Testing Conv layer output (B=2, C=64, H=8, W=8)")
    x_conv = torch.randn(2, 64, 8, 8)

    # Perturb only channel 0, height 2-4, width 3-6 for batch 0
    ptb_conv = MaskedPerturbationLpNorm(
        eps=0.1,
        norm=np.inf,
        batch_idx=0,
        channel_idx=0,
        height_slice=(2, 4),
        width_slice=(3, 6)
    )

    bounds, center, aux = ptb_conv.init(x_conv, forward=False)
    print(f"  Center shape: {center.shape}")
    print(f"  Lower bound shape: {bounds.lower.shape}")
    print(f"  Upper bound shape: {bounds.upper.shape}")
    print(f"  Perturbed elements: {(bounds.lower != bounds.upper).sum().item()}")

    # Test FC layer output
    print("\n2. Testing FC layer output (B=2, D=256)")
    x_fc = torch.randn(2, 256)

    # Perturb only features 10-20 for batch 0
    ptb_fc = MaskedPerturbationLpNorm(
        eps=0.1,
        norm=np.inf,
        batch_idx=0,
        channel_idx=list(range(10, 20))
    )

    bounds, center, aux = ptb_fc.init(x_fc, forward=False)
    print(f"  Center shape: {center.shape}")
    print(f"  Lower bound shape: {bounds.lower.shape}")
    print(f"  Upper bound shape: {bounds.upper.shape}")
    print(f"  Perturbed elements: {(bounds.lower != bounds.upper).sum().item()}")

    # Test create_region_mask
    print("\n3. Testing create_region_mask helper")
    mask = create_region_mask(
        shape=(1, 32, 16, 16),
        batch_idx=0,
        channel_idx=[0, 1, 2],
        height_slice=(5, 10),
        width_slice=(5, 10)
    )
    print(f"  Mask shape: {mask.shape}")
    print(f"  True elements: {mask.sum().item()}")
    print(f"  Expected: {3 * 5 * 5} (3 channels × 5 height × 5 width)")

    print("\nAll tests completed!")
