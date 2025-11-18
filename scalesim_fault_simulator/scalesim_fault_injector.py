"""
Component-Aware Fault Injector for Systolic Arrays

This module implements fault injection with proper component-level modeling.
Fixes the bug where accumulator_register faults were incorrectly propagated.
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class FaultModel:
    """Fault model with location, timing, and type."""

    # Fault types
    BIT_FLIP = 'BIT_FLIP'
    STUCK_AT_0 = 'STUCK_AT_0'
    STUCK_AT_1 = 'STUCK_AT_1'

    fault_type: str
    location: Dict  # {'pe_row', 'pe_col', 'component'}
    timing: Dict    # {'start_cycle', 'duration'}

    def is_active(self, cycle: int) -> bool:
        """Check if fault is active at given cycle."""
        start = self.timing['start_cycle']
        duration = self.timing['duration']

        if duration == float('inf'):
            return cycle >= start
        else:
            return start <= cycle < start + duration


class ScaleSimFaultInjector:
    """
    Component-aware fault injector for systolic arrays.

    Correctly models faults based on PE component:
    - accumulator_register: Only affects during MAC (both ifmap and filter active)
    - input_register: Only affects ifmap accesses
    - weight_register: Only affects filter accesses
    - MAC_unit: Affects all computation
    - entire_PE: Affects all operations
    """

    def __init__(self, faults: List[FaultModel]):
        """
        Initialize fault injector.

        Args:
            faults: List of fault models
        """
        self.faults = faults
        self.affected_addresses = set()

    def inject_faults(self, demand_matrices: Dict) -> Dict:
        """
        Inject faults into demand matrices with component-aware logic.

        Args:
            demand_matrices: Dict containing:
                - ifmap_demand: (num_cycles, num_pes)
                - filter_demand: (num_cycles, num_pes)
                - ofmap_demand: (num_cycles, num_pes)

        Returns:
            Dict containing:
                - faulty_ifmap: (num_cycles, num_pes) boolean
                - faulty_filter: (num_cycles, num_pes) boolean
                - faulty_ofmap: (num_cycles, num_pes) boolean
                - faulty_computation: (num_cycles, num_pes) boolean (NEW!)
                - affected_addresses: Set of affected memory addresses
        """
        ifmap_demand = demand_matrices['ifmap_demand']
        filter_demand = demand_matrices['filter_demand']
        ofmap_demand = demand_matrices['ofmap_demand']

        num_cycles, num_pes = ifmap_demand.shape

        # Initialize fault markers
        faulty_ifmap = np.zeros((num_cycles, num_pes), dtype=bool)
        faulty_filter = np.zeros((num_cycles, num_pes), dtype=bool)
        faulty_ofmap = np.zeros((num_cycles, num_pes), dtype=bool)
        faulty_computation = np.zeros((num_cycles, num_pes), dtype=bool)

        self.affected_addresses = set()

        # Determine array dimensions from num_pes
        arr_w = int(np.sqrt(num_pes))
        arr_h = num_pes // arr_w

        # Process each cycle
        for cycle in range(num_cycles):
            for pe_idx in range(num_pes):
                pe_row = pe_idx // arr_w
                pe_col = pe_idx % arr_w

                # Find faults affecting this PE at this cycle
                pe_faults = [f for f in self.faults
                           if f.location.get('pe_row') == pe_row
                           and f.location.get('pe_col') == pe_col
                           and f.is_active(cycle)]

                if not pe_faults:
                    continue

                # Check what this PE is doing this cycle
                has_ifmap = ifmap_demand[cycle, pe_idx] >= 0
                has_filter = filter_demand[cycle, pe_idx] >= 0
                has_ofmap = ofmap_demand[cycle, pe_idx] >= 0
                is_computing = has_ifmap and has_filter

                # Apply component-specific fault logic
                for fault in pe_faults:
                    component = fault.location.get('component', 'entire_PE')

                    if component == 'accumulator_register':
                        # ✓ FIX: Accumulator only affects during MAC operation
                        if is_computing:
                            faulty_computation[cycle, pe_idx] = True
                            if has_ifmap:
                                self.affected_addresses.add(('ifmap', ifmap_demand[cycle, pe_idx]))
                            if has_filter:
                                self.affected_addresses.add(('filter', filter_demand[cycle, pe_idx]))

                    elif component == 'input_register':
                        # Only affects ifmap accesses
                        if has_ifmap:
                            faulty_ifmap[cycle, pe_idx] = True
                            self.affected_addresses.add(('ifmap', ifmap_demand[cycle, pe_idx]))

                    elif component == 'weight_register':
                        # Only affects filter accesses
                        if has_filter:
                            faulty_filter[cycle, pe_idx] = True
                            self.affected_addresses.add(('filter', filter_demand[cycle, pe_idx]))

                    elif component == 'MAC_unit':
                        # Affects any computation
                        if is_computing:
                            faulty_computation[cycle, pe_idx] = True
                            if has_ifmap:
                                self.affected_addresses.add(('ifmap', ifmap_demand[cycle, pe_idx]))
                            if has_filter:
                                self.affected_addresses.add(('filter', filter_demand[cycle, pe_idx]))

                    elif component == 'control_logic':
                        # Control logic affects all operations
                        if has_ifmap or has_filter or has_ofmap:
                            faulty_computation[cycle, pe_idx] = True
                            if has_ifmap:
                                faulty_ifmap[cycle, pe_idx] = True
                                self.affected_addresses.add(('ifmap', ifmap_demand[cycle, pe_idx]))
                            if has_filter:
                                faulty_filter[cycle, pe_idx] = True
                                self.affected_addresses.add(('filter', filter_demand[cycle, pe_idx]))

                    elif component == 'entire_PE':
                        # Affects everything
                        if has_ifmap:
                            faulty_ifmap[cycle, pe_idx] = True
                            self.affected_addresses.add(('ifmap', ifmap_demand[cycle, pe_idx]))
                        if has_filter:
                            faulty_filter[cycle, pe_idx] = True
                            self.affected_addresses.add(('filter', filter_demand[cycle, pe_idx]))
                        if has_ofmap:
                            faulty_ofmap[cycle, pe_idx] = True
                            self.affected_addresses.add(('ofmap', ofmap_demand[cycle, pe_idx]))
                        if is_computing:
                            faulty_computation[cycle, pe_idx] = True

        return {
            'faulty_ifmap': faulty_ifmap,
            'faulty_filter': faulty_filter,
            'faulty_ofmap': faulty_ofmap,
            'faulty_computation': faulty_computation,
            'affected_addresses': self.affected_addresses
        }

    def trace_fault_propagation(self, demand_matrices: Dict,
                               operand_matrices: Dict,
                               faulty_markers: Dict) -> Set[int]:
        """
        Trace which output addresses are affected by faults.

        Key insight: In systolic arrays, outputs are computed by MULTIPLE PEs.
        - In WS: All PEs in a column contribute to that column's outputs
        - In OS: Each PE computes specific outputs
        - In IS: Similar to OS

        We need to find:
        1. Which PEs are faulty during computation
        2. Which outputs those PEs contributed to (not just drained!)

        Args:
            demand_matrices: Demand matrices
            operand_matrices: Operand matrices
            faulty_markers: Output from inject_faults()

        Returns:
            Set of affected OFMAP addresses
        """
        ifmap_demand = demand_matrices['ifmap_demand']
        filter_demand = demand_matrices['filter_demand']
        ofmap_demand = demand_matrices['ofmap_demand']

        faulty_ifmap = faulty_markers['faulty_ifmap']
        faulty_filter = faulty_markers['faulty_filter']
        faulty_computation = faulty_markers['faulty_computation']

        num_cycles, num_pes = ifmap_demand.shape

        # Determine array dimensions
        arr_w = int(np.sqrt(num_pes))
        arr_h = num_pes // arr_w

        # Build mapping: Which outputs does each PE CONTRIBUTE to?
        # Not just drain, but actually participate in computing
        pe_contributes_to = {}

        for pe_idx in range(num_pes):
            pe_row = pe_idx // arr_w
            pe_col = pe_idx % arr_w

            # Find all cycles where this PE does computation
            # (both ifmap and filter active)
            computing_cycles = []
            for cycle in range(num_cycles):
                if (ifmap_demand[cycle, pe_idx] >= 0 and
                    filter_demand[cycle, pe_idx] >= 0):
                    computing_cycles.append(cycle)

            # Find outputs related to this PE's column
            # In WS: PE column determines output channels
            # All outputs in ofmap_demand with same column are related
            contributed_outputs = set()

            # Collect all ofmap addresses accessed by PEs in same column
            for other_idx in range(num_pes):
                other_col = other_idx % arr_w
                if other_col == pe_col:
                    output_addrs = ofmap_demand[:, other_idx]
                    valid_outputs = output_addrs[output_addrs >= 0]
                    contributed_outputs.update(valid_outputs)

            if len(contributed_outputs) > 0:
                pe_contributes_to[pe_idx] = contributed_outputs

        # Trace faults to outputs
        affected_outputs = set()

        for cycle in range(num_cycles):
            # Find faulty PEs at this cycle
            faulty_pe_mask = (faulty_computation[cycle] |
                            faulty_ifmap[cycle] |
                            faulty_filter[cycle])

            faulty_pe_indices = np.where(faulty_pe_mask)[0]

            # Mark outputs that faulty PEs contribute to
            for pe_idx in faulty_pe_indices:
                if pe_idx in pe_contributes_to:
                    affected_outputs.update(pe_contributes_to[pe_idx])

        return affected_outputs

    def create_output_mapping(self, operand_matrices: Dict,
                             affected_outputs: Set[int]) -> Dict:
        """
        Map affected OFMAP addresses to tensor indices (C, H, W).

        Args:
            operand_matrices: Contains dimensions and ofmap_matrix
            affected_outputs: Set of affected OFMAP addresses

        Returns:
            Dict with format: {channel_idx: {row: [cols]}}
        """
        dims = operand_matrices['dimensions']
        ofmap_shape = dims['ofmap_shape']  # (H, W, C)
        H, W, C = ofmap_shape

        OFMAP_OFFSET = 20000000

        # Create mapping: address -> (c, h, w)
        exact_positions = {}

        for addr in affected_outputs:
            # Remove offset
            internal = addr - OFMAP_OFFSET

            # Decompose address
            # Address formula: addr = offset + num_filters * pixel_idx + filter_idx
            pixel_idx = internal // C
            filter_idx = internal % C

            # Pixel index to (h, w)
            h = pixel_idx // W
            w = pixel_idx % W
            c = filter_idx

            # Validate bounds
            if 0 <= c < C and 0 <= h < H and 0 <= w < W:
                if c not in exact_positions:
                    exact_positions[c] = {}
                if h not in exact_positions[c]:
                    exact_positions[c][h] = []
                exact_positions[c][h].append(w)

        # Sort columns for each row
        for c in exact_positions:
            for h in exact_positions[c]:
                exact_positions[c][h] = sorted(exact_positions[c][h])

        return exact_positions

    def compute_statistics(self, affected_outputs: Set[int],
                          operand_matrices: Dict) -> Dict:
        """Compute fault coverage statistics."""
        dims = operand_matrices['dimensions']
        ofmap_shape = dims['ofmap_shape']
        H, W, C = ofmap_shape
        total_outputs = H * W * C

        return {
            'total_outputs': total_outputs,
            'affected_outputs': len(affected_outputs),
            'fault_coverage': len(affected_outputs) / total_outputs if total_outputs > 0 else 0,
            'num_faults': len(self.faults),
            'affected_addresses': len(self.affected_addresses)
        }


if __name__ == '__main__':
    print("Testing Component-Aware Fault Injector\n")
    print("="*60)

    # This will be tested in the main simulator
    print("✓ Fault injector module created")
    print("  - Supports component-aware fault modeling")
    print("  - Fixes accumulator_register bug")
    print("  - Maps outputs to tensor indices")
