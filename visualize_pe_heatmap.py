#!/usr/bin/env python3
"""
PE Array Sensitivity Heatmap Visualization

Visualizes PE array sensitivity with assumption that center PEs are more sensitive than edge PEs.
Lower epsilon = higher sensitivity (more impact on DNN output)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


def generate_center_sensitive_heatmap(array_size=8, center_epsilon=0.01, edge_epsilon=0.5):
    """
    Generate heatmap where center PEs are more sensitive (lower epsilon)

    Args:
        array_size: Size of PE array (default: 8 for 8x8)
        center_epsilon: Epsilon value for center PEs (lower = more sensitive)
        edge_epsilon: Epsilon value for edge PEs (higher = less sensitive)

    Returns:
        heatmap: 2D numpy array with epsilon values
    """
    heatmap = np.zeros((array_size, array_size))
    center = array_size / 2.0

    for row in range(array_size):
        for col in range(array_size):
            # Calculate distance from center (normalized to [0, 1])
            distance = np.sqrt((row - center + 0.5)**2 + (col - center + 0.5)**2)
            max_distance = np.sqrt(2 * (center)**2)
            normalized_distance = distance / max_distance

            # Linear interpolation: center has low epsilon, edge has high epsilon
            epsilon = center_epsilon + (edge_epsilon - center_epsilon) * normalized_distance
            heatmap[row, col] = epsilon

    return heatmap


def generate_gaussian_sensitive_heatmap(array_size=8, center_epsilon=0.01, edge_epsilon=0.5, sigma=1.5):
    """
    Generate heatmap with Gaussian distribution (smoother transition)

    Args:
        array_size: Size of PE array
        center_epsilon: Epsilon at center
        edge_epsilon: Epsilon at edges
        sigma: Standard deviation for Gaussian (controls falloff)

    Returns:
        heatmap: 2D numpy array with epsilon values
    """
    heatmap = np.zeros((array_size, array_size))
    center = array_size / 2.0

    for row in range(array_size):
        for col in range(array_size):
            # Gaussian falloff from center
            distance_sq = (row - center + 0.5)**2 + (col - center + 0.5)**2
            gaussian = np.exp(-distance_sq / (2 * sigma**2))

            # Invert: center = low epsilon (sensitive), edge = high epsilon (resilient)
            epsilon = edge_epsilon - (edge_epsilon - center_epsilon) * gaussian
            heatmap[row, col] = epsilon

    return heatmap


def plot_pe_heatmap(heatmap, title="PE Array Sensitivity Heatmap",
                    save_path="pe_sensitivity_heatmap.png", show_values=True):
    """
    Plot PE array heatmap

    Args:
        heatmap: 2D numpy array with epsilon values
        title: Plot title
        save_path: Path to save figure
        show_values: Whether to show epsilon values in cells
    """
    array_size = heatmap.shape[0]

    fig, ax = plt.subplots(figsize=(10, 9))

    # Create custom colormap: Red (sensitive/low epsilon) -> Yellow -> Green (resilient/high epsilon)
    cmap = plt.cm.RdYlGn

    # Plot heatmap
    im = ax.imshow(heatmap, cmap=cmap, aspect='auto',
                   vmin=np.min(heatmap), vmax=np.max(heatmap),
                   interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Max Robust Epsilon (ε)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Set title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('PE Column', fontsize=13, fontweight='bold')
    ax.set_ylabel('PE Row', fontsize=13, fontweight='bold')

    # Set ticks
    ax.set_xticks(np.arange(array_size))
    ax.set_yticks(np.arange(array_size))
    ax.set_xticklabels(np.arange(array_size))
    ax.set_yticklabels(np.arange(array_size))

    # Add grid
    ax.set_xticks(np.arange(array_size) - 0.5, minor=True)
    ax.set_yticks(np.arange(array_size) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)

    # Show epsilon values in cells
    if show_values:
        for row in range(array_size):
            for col in range(array_size):
                epsilon = heatmap[row, col]

                # Choose text color based on background
                text_color = 'white' if epsilon < 0.25 else 'black'

                ax.text(col, row, f'{epsilon:.3f}',
                       ha="center", va="center",
                       color=text_color, fontsize=9, fontweight='bold')

    # Add annotations
    ax.text(array_size/2 - 0.5, -0.7, 'Red = Sensitive (Low ε)',
            ha='center', fontsize=11, color='red', fontweight='bold')
    ax.text(array_size/2 - 0.5, array_size + 0.2, 'Green = Resilient (High ε)',
            ha='center', fontsize=11, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {save_path}")
    plt.close()


def plot_comparison(heatmap_linear, heatmap_gaussian, save_path="pe_sensitivity_comparison.png"):
    """
    Plot side-by-side comparison of linear and Gaussian distributions
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    array_size = heatmap_linear.shape[0]

    # Linear distribution
    im1 = axes[0].imshow(heatmap_linear, cmap=plt.cm.RdYlGn, aspect='auto',
                         vmin=0, vmax=0.5, interpolation='nearest')
    axes[0].set_title('Linear Distance Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PE Column', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('PE Row', fontsize=12, fontweight='bold')
    axes[0].set_xticks(np.arange(array_size))
    axes[0].set_yticks(np.arange(array_size))
    axes[0].grid(which="minor", color="black", linestyle='-', linewidth=1)
    axes[0].set_xticks(np.arange(array_size) - 0.5, minor=True)
    axes[0].set_yticks(np.arange(array_size) - 0.5, minor=True)

    # Add values
    for row in range(array_size):
        for col in range(array_size):
            eps = heatmap_linear[row, col]
            color = 'white' if eps < 0.25 else 'black'
            axes[0].text(col, row, f'{eps:.3f}', ha="center", va="center",
                        color=color, fontsize=8, fontweight='bold')

    # Gaussian distribution
    im2 = axes[1].imshow(heatmap_gaussian, cmap=plt.cm.RdYlGn, aspect='auto',
                         vmin=0, vmax=0.5, interpolation='nearest')
    axes[1].set_title('Gaussian Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PE Column', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('PE Row', fontsize=12, fontweight='bold')
    axes[1].set_xticks(np.arange(array_size))
    axes[1].set_yticks(np.arange(array_size))
    axes[1].grid(which="minor", color="black", linestyle='-', linewidth=1)
    axes[1].set_xticks(np.arange(array_size) - 0.5, minor=True)
    axes[1].set_yticks(np.arange(array_size) - 0.5, minor=True)

    # Add values
    for row in range(array_size):
        for col in range(array_size):
            eps = heatmap_gaussian[row, col]
            color = 'white' if eps < 0.25 else 'black'
            axes[1].text(col, row, f'{eps:.3f}', ha="center", va="center",
                        color=color, fontsize=8, fontweight='bold')

    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Max Robust Epsilon (ε)', rotation=270, labelpad=20,
                   fontsize=12, fontweight='bold')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison to: {save_path}")
    plt.close()


def print_statistics(heatmap, distribution_name):
    """Print statistics about the heatmap"""
    print(f"\n{'='*60}")
    print(f"{distribution_name} - Statistics")
    print(f"{'='*60}")
    print(f"Array size: {heatmap.shape[0]}×{heatmap.shape[1]}")
    print(f"Min epsilon (most sensitive):    {np.min(heatmap):.4f}")
    print(f"Max epsilon (most resilient):    {np.max(heatmap):.4f}")
    print(f"Mean epsilon:                    {np.mean(heatmap):.4f}")
    print(f"Median epsilon:                  {np.median(heatmap):.4f}")
    print(f"Std deviation:                   {np.std(heatmap):.4f}")

    # Find most sensitive PE (lowest epsilon)
    min_idx = np.unravel_index(np.argmin(heatmap), heatmap.shape)
    print(f"\nMost sensitive PE:               PE({min_idx[0]}, {min_idx[1]}) with ε={heatmap[min_idx]:.4f}")

    # Find most resilient PE (highest epsilon)
    max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    print(f"Most resilient PE:               PE({max_idx[0]}, {max_idx[1]}) with ε={heatmap[max_idx]:.4f}")

    # Center vs corner comparison
    center = heatmap.shape[0] // 2
    center_avg = np.mean(heatmap[center-1:center+1, center-1:center+1])

    corners = [
        heatmap[0, 0], heatmap[0, -1],
        heatmap[-1, 0], heatmap[-1, -1]
    ]
    corner_avg = np.mean(corners)

    print(f"\nCenter PEs average:              {center_avg:.4f}")
    print(f"Corner PEs average:              {corner_avg:.4f}")
    print(f"Sensitivity ratio (corner/center): {corner_avg/center_avg:.2f}x")
    print(f"{'='*60}")


def main():
    """Generate and visualize PE array sensitivity heatmaps"""

    print("\n" + "="*60)
    print("PE ARRAY SENSITIVITY HEATMAP VISUALIZATION")
    print("="*60)
    print("\nAssumption: Center PEs are more sensitive than edge PEs")
    print("  - Lower epsilon (red) = More sensitive")
    print("  - Higher epsilon (green) = More resilient")
    print()

    # Configuration
    array_size = 8
    center_epsilon = 0.01   # Very sensitive (low epsilon)
    edge_epsilon = 0.50     # Less sensitive (high epsilon)

    print(f"Configuration:")
    print(f"  Array size: {array_size}×{array_size}")
    print(f"  Center epsilon: {center_epsilon:.3f} (high sensitivity)")
    print(f"  Edge epsilon:   {edge_epsilon:.3f} (low sensitivity)")

    # Generate heatmaps
    print(f"\nGenerating heatmaps...")
    heatmap_linear = generate_center_sensitive_heatmap(
        array_size=array_size,
        center_epsilon=center_epsilon,
        edge_epsilon=edge_epsilon
    )

    heatmap_gaussian = generate_gaussian_sensitive_heatmap(
        array_size=array_size,
        center_epsilon=center_epsilon,
        edge_epsilon=edge_epsilon,
        sigma=1.5
    )

    # Print statistics
    print_statistics(heatmap_linear, "Linear Distribution")
    print_statistics(heatmap_gaussian, "Gaussian Distribution")

    # Plot individual heatmaps
    print(f"\nGenerating visualizations...")
    plot_pe_heatmap(
        heatmap_linear,
        title="PE Array Sensitivity (Linear Distance)",
        save_path="pe_sensitivity_linear.png",
        show_values=True
    )

    plot_pe_heatmap(
        heatmap_gaussian,
        title="PE Array Sensitivity (Gaussian Distribution)",
        save_path="pe_sensitivity_gaussian.png",
        show_values=True
    )

    # Plot comparison
    plot_comparison(heatmap_linear, heatmap_gaussian,
                   save_path="pe_sensitivity_comparison.png")

    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  1. pe_sensitivity_linear.png      - Linear distance distribution")
    print("  2. pe_sensitivity_gaussian.png    - Gaussian distribution")
    print("  3. pe_sensitivity_comparison.png  - Side-by-side comparison")
    print()


if __name__ == "__main__":
    main()
