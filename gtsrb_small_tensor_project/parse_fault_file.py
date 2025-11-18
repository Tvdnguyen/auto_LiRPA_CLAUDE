"""
Parse fault simulation result files to extract affected elements
For use with main_interactive.py and main_interactive_ver2.py
"""
import os
import re
from typing import List, Dict, Tuple


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
            - exact_elements: List of (channel, row, col) tuples for ALL affected positions
    """
    result = {
        'layer_name': None,
        'dataflow': None,
        'array_size': None,
        'affected_channels': [],
        'regions': [],
        'exact_elements': []  # NEW: flat list of all (channel, row, col)
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

                # Add to exact_elements list
                for row, cols in exact_positions.items():
                    for col in cols:
                        result['exact_elements'].append((current_channel, row, col))

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
        coord_match = re.search(r'Row (\d+): cols \[([^\]]+)\]', line)
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

        # Add to exact_elements list
        for row, cols in exact_positions.items():
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
    print(f"    Total Exact Elements: {len(parsed_result['exact_elements'])}")

    print(f"\n  Perturbation Regions:")
    for i, region in enumerate(parsed_result['regions'], 1):
        h_start, h_end = region['height_slice']
        w_start, w_end = region['width_slice']
        num_exact = sum(len(cols) for cols in region['exact_positions'].values())
        print(f"    Region {i}: Channel {region['channel_idx']}, "
              f"H({h_start}, {h_end}), W({w_start}, {w_end}) "
              f"[{num_exact} exact positions]")


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python parse_fault_file.py <fault_result_file.txt>")
        sys.exit(1)

    fault_file = sys.argv[1]

    if not os.path.exists(fault_file):
        print(f"Error: File not found: {fault_file}")
        sys.exit(1)

    print(f"Parsing fault result file: {fault_file}")
    result = parse_fault_result_file(fault_file)

    display_fault_result_summary(result)

    print(f"\n  First 20 exact elements:")
    for i, (ch, row, col) in enumerate(result['exact_elements'][:20], 1):
        print(f"    {i}. Channel {ch}, Row {row}, Col {col}")

    if len(result['exact_elements']) > 20:
        print(f"    ... and {len(result['exact_elements']) - 20} more elements")
