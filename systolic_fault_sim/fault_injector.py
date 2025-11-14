"""
Fault Injection Module
Injects faults into systolic array computation and tracks their impact
"""

import numpy as np


class FaultModel:
    """Define different types of faults"""

    STUCK_AT_0 = 'stuck_at_0'
    STUCK_AT_1 = 'stuck_at_1'
    BIT_FLIP = 'bit_flip'
    TRANSIENT = 'transient'
    PERMANENT = 'permanent'

    def __init__(self, fault_type, fault_location, fault_timing=None):
        """
        Args:
            fault_type: Type of fault (STUCK_AT_0, BIT_FLIP, etc.)
            fault_location: Dict with 'component', 'pe_row', 'pe_col'
            fault_timing: Dict with 'start_cycle', 'duration' (None = permanent)
        """
        self.fault_type = fault_type
        self.location = fault_location
        self.timing = fault_timing or {'start_cycle': 0, 'duration': float('inf')}

    def is_active(self, cycle):
        """Check if fault is active at given cycle"""
        start = self.timing['start_cycle']
        duration = self.timing['duration']
        return start <= cycle < start + duration

    def apply_to_value(self, value, cycle):
        """Apply fault to a data value"""
        if not self.is_active(cycle):
            return value

        if self.fault_type == self.STUCK_AT_0:
            return 0
        elif self.fault_type == self.STUCK_AT_1:
            return 1  # Or max value depending on data type
        elif self.fault_type == self.BIT_FLIP:
            # Flip random bit
            if isinstance(value, (int, np.integer)):
                return value ^ (1 << np.random.randint(0, 32))
            else:
                return value  # Can't bit flip non-integer
        else:
            return value


class FaultInjector:
    """Inject faults into demand matrices and track impact"""

    def __init__(self, faults):
        """
        Args:
            faults: List of FaultModel objects
        """
        self.faults = faults
        self.affected_addresses = set()
        self.affected_outputs = set()

    def inject_into_demands(self, demand_matrices):
        """
        Mark which memory accesses are affected by faults

        Args:
            demand_matrices: Output from systolic_compute_*.generate_demand_matrices()

        Returns:
            Dict with faulty demand markers
        """
        ifmap_demand = demand_matrices['ifmap_demand']
        filter_demand = demand_matrices['filter_demand']
        ofmap_demand = demand_matrices['ofmap_demand']
        pe_mapping = demand_matrices['pe_mapping']

        num_cycles, num_pes = ifmap_demand.shape
        arr_w = int(np.sqrt(num_pes))  # Assume square for simplicity
        arr_h = num_pes // arr_w

        # Track faulty accesses
        faulty_ifmap = np.zeros_like(ifmap_demand, dtype=bool)
        faulty_filter = np.zeros_like(filter_demand, dtype=bool)
        faulty_ofmap = np.zeros_like(ofmap_demand, dtype=bool)

        for cycle in range(num_cycles):
            for pe_idx in range(num_pes):
                pe_row = pe_idx // arr_w
                pe_col = pe_idx % arr_w

                # Check if this PE has faults
                pe_faults = [f for f in self.faults
                            if f.location.get('pe_row') == pe_row
                            and f.location.get('pe_col') == pe_col
                            and f.is_active(cycle)]

                if pe_faults:
                    # Mark accesses as faulty
                    if ifmap_demand[cycle, pe_idx] >= 0:
                        faulty_ifmap[cycle, pe_idx] = True
                        self.affected_addresses.add(('ifmap', ifmap_demand[cycle, pe_idx]))

                    if filter_demand[cycle, pe_idx] >= 0:
                        faulty_filter[cycle, pe_idx] = True
                        self.affected_addresses.add(('filter', filter_demand[cycle, pe_idx]))

                    if ofmap_demand[cycle, pe_idx] >= 0:
                        faulty_ofmap[cycle, pe_idx] = True
                        self.affected_addresses.add(('ofmap', ofmap_demand[cycle, pe_idx]))

        return {
            'faulty_ifmap': faulty_ifmap,
            'faulty_filter': faulty_filter,
            'faulty_ofmap': faulty_ofmap,
            'affected_addresses': self.affected_addresses
        }

    def trace_fault_propagation(self, demand_matrices, operand_matrices, faulty_markers):
        """
        Trace how faults propagate through computation to output

        Args:
            demand_matrices: Demand matrices
            operand_matrices: Original operand matrices
            faulty_markers: Output from inject_into_demands()

        Returns:
            Set of affected output addresses
        """
        ofmap_mat = operand_matrices['ofmap']
        ifmap_demand = demand_matrices['ifmap_demand']
        filter_demand = demand_matrices['filter_demand']
        ofmap_demand = demand_matrices['ofmap_demand']

        faulty_ifmap = faulty_markers['faulty_ifmap']
        faulty_filter = faulty_markers['faulty_filter']
        faulty_ofmap = faulty_markers['faulty_ofmap']

        # Track which PEs have faults during computation
        affected_outputs = set()

        num_cycles, num_pes = faulty_ifmap.shape

        # For each PE, track which output it eventually produces
        # by looking at ALL cycles where this PE writes output
        pe_to_outputs = {}  # Maps pe_idx → set of output addresses

        for pe_idx in range(num_pes):
            output_addrs = ofmap_demand[:, pe_idx]
            valid_outputs = output_addrs[output_addrs >= 0]
            if len(valid_outputs) > 0:
                pe_to_outputs[pe_idx] = set(valid_outputs)

        # Now trace faults: if a PE has faulty computation,
        # mark ALL outputs it produces as affected
        for cycle in range(num_cycles):
            # Check which PEs have faulty ifmap or filter accesses this cycle
            faulty_pe_mask = faulty_ifmap[cycle] | faulty_filter[cycle]
            faulty_pe_indices = np.where(faulty_pe_mask)[0]

            # For each faulty PE, mark its outputs as affected
            for pe_idx in faulty_pe_indices:
                if pe_idx in pe_to_outputs:
                    affected_outputs.update(pe_to_outputs[pe_idx])

            # Also handle direct ofmap faults
            if np.any(faulty_ofmap[cycle]):
                ofmap_addrs = ofmap_demand[cycle]
                valid_addrs = ofmap_addrs[ofmap_addrs >= 0]
                affected_outputs.update(valid_addrs)

        self.affected_outputs = affected_outputs
        return affected_outputs

    def create_fault_mask(self, operand_matrices, affected_outputs):
        """
        Create boolean mask for output tensor

        Args:
            operand_matrices: Original operand matrices
            affected_outputs: Set of affected output addresses

        Returns:
            Boolean array matching output shape
        """
        ofmap_mat = operand_matrices['ofmap']
        fault_mask = np.zeros_like(ofmap_mat, dtype=bool)

        for addr in affected_outputs:
            # Find positions in ofmap matrix
            positions = np.where(ofmap_mat == addr)
            if len(positions[0]) > 0:
                fault_mask[positions] = True

        return fault_mask

    def compute_statistics(self, fault_mask, operand_matrices):
        """Compute fault impact statistics"""
        dims = operand_matrices['dimensions']

        total_outputs = dims['ofmap_pixels'] * dims['num_filters']
        affected_outputs = np.sum(fault_mask)

        return {
            'total_outputs': total_outputs,
            'affected_outputs': int(affected_outputs),
            'fault_coverage': affected_outputs / total_outputs if total_outputs > 0 else 0,
            'num_faults': len(self.faults),
            'affected_addresses': len(self.affected_addresses)
        }
