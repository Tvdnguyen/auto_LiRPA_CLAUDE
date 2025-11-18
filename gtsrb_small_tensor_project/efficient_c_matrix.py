"""
Efficient C matrix construction for verification

Reduces memory usage by only verifying necessary constraints instead of all-vs-all
"""
import torch


def create_efficient_c_matrix(predicted_class, num_classes=43, device='cpu'):
    """
    Create efficient C matrix for verifying predicted class

    Instead of all-vs-all verification (C × (C-1) constraints),
    only verify: predicted_class - other_class for each other class

    Memory reduction: (C × (C-1), C) → (C-1, C)
    For 43 classes: (1806, 43) → (42, 43) = 43x smaller!

    Args:
        predicted_class: Index of predicted class
        num_classes: Total number of classes (default: 43 for GTSRB)
        device: Device to create matrix on ('cpu' or 'cuda')

    Returns:
        C matrix of shape (1, num_classes-1, num_classes)
        where C[0, i] represents: predicted_class - other_class[i]

    Note:
        The batch dimension (first dimension = 1) is REQUIRED by auto_LiRPA.
        auto_LiRPA expects C shape: (batch, spec, num_classes)
    """
    if not 0 <= predicted_class < num_classes:
        raise ValueError(f"predicted_class must be in [0, {num_classes-1}], got {predicted_class}")

    # Create C matrix: (1, num_classes-1, num_classes) with batch dimension
    C = torch.zeros(1, num_classes - 1, num_classes, device=device, dtype=torch.float32)

    row_idx = 0
    for cls in range(num_classes):
        if cls != predicted_class:
            C[0, row_idx, predicted_class] = 1.0   # +1 for predicted class
            C[0, row_idx, cls] = -1.0               # -1 for other class
            row_idx += 1

    return C


def compute_robustness_margin(lb_diff, ub_diff=None):
    """
    Compute robustness margin from bounds

    Args:
        lb_diff: Lower bounds of (predicted - others)
        ub_diff: Upper bounds (optional, not used for margin)

    Returns:
        margin: Minimum margin (worst case)
        is_robust: True if all lb_diff > 0
        worst_class_idx: Index of class with smallest margin
    """
    # lb_diff shape: (batch, num_classes-1) or (num_classes-1,)
    if lb_diff.dim() > 1:
        lb_diff = lb_diff.squeeze(0)

    # Find minimum margin
    min_margin = lb_diff.min().item()
    worst_idx = lb_diff.argmin().item()

    # Robust if all differences are positive
    is_robust = min_margin > 0

    return min_margin, is_robust, worst_idx


if __name__ == '__main__':
    print("Testing efficient C matrix creation...")

    # Test 1: Create C matrix
    predicted = 5
    num_classes = 43

    C = create_efficient_c_matrix(predicted, num_classes)

    print(f"\nTest 1: C matrix creation")
    print(f"  Predicted class: {predicted}")
    print(f"  Num classes: {num_classes}")
    print(f"  C matrix shape: {C.shape}")
    print(f"  Expected shape: (1, {num_classes-1}, {num_classes})")

    assert C.shape == (1, num_classes - 1, num_classes), "Shape mismatch!"

    # Verify structure
    print(f"\n  Verifying C matrix structure...")
    for i in range(num_classes - 1):
        # Each row should have exactly 2 non-zero elements
        non_zero = (C[0, i] != 0).sum().item()
        assert non_zero == 2, f"Row {i} should have 2 non-zero elements, got {non_zero}"

        # Should have +1 at predicted and -1 at some other class
        assert C[0, i, predicted].item() == 1.0, f"Row {i} should have +1 at predicted class"

        # Find the -1 position
        other_class = (C[0, i] == -1.0).nonzero(as_tuple=True)[0].item()
        assert other_class != predicted, f"Row {i} has -1 at predicted class!"

    print(f"  ✓ C matrix structure correct")

    # Test 2: Memory comparison
    print(f"\nTest 2: Memory comparison")

    # All-vs-all C matrix
    C_full_size = num_classes * (num_classes - 1) * num_classes * 4  # float32
    C_full_mb = C_full_size / (1024 ** 2)

    # Efficient C matrix
    C_efficient_size = (num_classes - 1) * num_classes * 4
    C_efficient_mb = C_efficient_size / (1024 ** 2)

    reduction = C_full_size / C_efficient_size

    print(f"  All-vs-all C: ({num_classes * (num_classes-1)}, {num_classes}) = {C_full_mb:.4f} MB")
    print(f"  Efficient C:  ({num_classes-1}, {num_classes}) = {C_efficient_mb:.4f} MB")
    print(f"  Reduction: {reduction:.1f}x smaller")

    # Test 3: Robustness margin computation
    print(f"\nTest 3: Robustness margin")

    # Simulate bounds
    lb_diff = torch.randn(num_classes - 1)
    lb_diff[0] = 0.5   # Smallest positive margin
    lb_diff[10] = -0.2  # One negative (not robust)

    margin, is_robust, worst_idx = compute_robustness_margin(lb_diff)

    print(f"  Simulated bounds: {lb_diff.shape}")
    print(f"  Minimum margin: {margin:.4f}")
    print(f"  Is robust: {is_robust}")
    print(f"  Worst class index: {worst_idx}")

    assert is_robust == False, "Should not be robust (has negative margin)"
    assert margin < 0, "Margin should be negative"

    print(f"\n✓ All tests passed!")
