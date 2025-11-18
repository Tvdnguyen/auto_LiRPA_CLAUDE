"""
Fault Impact Visualizer

Visualizes affected channels with red highlighting for faulty elements.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Dict
import os


class FaultVisualizer:
    """Visualize fault impact on output tensor."""

    def __init__(self):
        self.cmap = ListedColormap(['white', 'red'])

    def visualize_affected_channels(self, exact_positions: Dict,
                                    layer_config: Dict,
                                    save_path: str = None,
                                    max_channels: int = 16):
        """
        Visualize affected channels with faulty elements highlighted in red.

        Args:
            exact_positions: Dict {channel_idx: {row: [cols]}}
            layer_config: Layer configuration
            save_path: Path to save figure
            max_channels: Maximum channels to display
        """
        out_ch, Hout, Wout = layer_config['output_shape']

        # Get affected channels
        affected_channels = sorted(exact_positions.keys())

        if len(affected_channels) == 0:
            print("No affected channels to visualize")
            return

        # Limit display
        display_channels = affected_channels[:max_channels]
        num_display = len(display_channels)

        if len(affected_channels) > max_channels:
            print(f"Displaying {max_channels} out of {len(affected_channels)} affected channels")

        # Create grid
        n_cols = min(4, num_display)
        n_rows = int(np.ceil(num_display / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), dpi=100)

        # Handle single subplot case
        if num_display == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Fault Impact on {layer_config["name"]} Output\n'
                    f'{len(affected_channels)} channels affected (showing {num_display})',
                    fontsize=16, fontweight='bold')

        # Plot each channel
        for idx, ch in enumerate(display_channels):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            # Create mask for this channel
            channel_mask = np.zeros((Hout, Wout), dtype=bool)

            rows_dict = exact_positions[ch]
            for h, cols in rows_dict.items():
                for w in cols:
                    if 0 <= h < Hout and 0 <= w < Wout:
                        channel_mask[h, w] = True

            # Count faulty pixels
            faulty_count = np.sum(channel_mask)
            total_pixels = Hout * Wout

            # Plot
            im = ax.imshow(channel_mask, cmap=self.cmap, vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')

            # Grid
            ax.set_xticks(np.arange(-0.5, Wout, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, Hout, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

            # Title
            ax.set_title(f'Channel {ch}\n{faulty_count}/{total_pixels} faulty ({faulty_count/total_pixels*100:.1f}%)',
                        fontsize=10)

            # Labels
            ax.set_xlabel('Width', fontsize=8)
            ax.set_ylabel('Height', fontsize=8)
            ax.tick_params(labelsize=6)

        # Hide unused subplots
        for idx in range(num_display, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')

        # Colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['OK', 'Faulty'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to: {save_path}")

        plt.show()

    def create_summary_plot(self, exact_positions: Dict,
                           layer_config: Dict,
                           statistics: Dict,
                           save_path: str = None):
        """
        Create summary visualization with:
        1. Per-channel fault count bar chart
        2. Spatial heatmap (aggregated across channels)

        Args:
            exact_positions: Dict {channel_idx: {row: [cols]}}
            layer_config: Layer configuration
            statistics: Statistics dict
            save_path: Path to save figure
        """
        out_ch, Hout, Wout = layer_config['output_shape']

        fig = plt.figure(figsize=(14, 6))

        # Subplot 1: Per-channel fault count
        ax1 = plt.subplot(1, 2, 1)

        affected_channels = sorted(exact_positions.keys())
        channel_counts = []

        for ch in affected_channels:
            rows_dict = exact_positions[ch]
            count = sum(len(cols) for cols in rows_dict.values())
            channel_counts.append(count)

        ax1.bar(range(len(affected_channels)), channel_counts, color='red', alpha=0.7)
        ax1.set_xlabel('Affected Channel Index', fontsize=12)
        ax1.set_ylabel('Number of Faulty Pixels', fontsize=12)
        ax1.set_title(f'Fault Count per Channel\n{len(affected_channels)} channels affected',
                     fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Set x-axis labels
        if len(affected_channels) <= 20:
            ax1.set_xticks(range(len(affected_channels)))
            ax1.set_xticklabels([f'{ch}' for ch in affected_channels], rotation=45)
        else:
            ax1.set_xlabel('Affected Channel Index (sampled)', fontsize=12)

        # Subplot 2: Spatial heatmap (aggregated)
        ax2 = plt.subplot(1, 2, 2)

        spatial_heatmap = np.zeros((Hout, Wout), dtype=int)

        for ch, rows_dict in exact_positions.items():
            for h, cols in rows_dict.items():
                for w in cols:
                    if 0 <= h < Hout and 0 <= w < Wout:
                        spatial_heatmap[h, w] += 1

        im = ax2.imshow(spatial_heatmap, cmap='Reds', aspect='equal', interpolation='nearest')
        ax2.set_title(f'Spatial Fault Density\n(aggregated across {len(affected_channels)} channels)',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Width', fontsize=12)
        ax2.set_ylabel('Height', fontsize=12)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Number of affected channels', fontsize=10)

        # Add grid
        ax2.set_xticks(np.arange(-0.5, Wout, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, Hout, 1), minor=True)
        ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)

        plt.suptitle(f'{layer_config["name"]}: Fault Coverage = {statistics["fault_coverage"]*100:.2f}%',
                    fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Summary plot saved to: {save_path}")

        plt.show()


if __name__ == '__main__':
    print("Fault Visualizer Module")
    print("Use visualize_affected_channels() to visualize fault impact")
