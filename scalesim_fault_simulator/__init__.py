"""
SCALE-Sim Based Fault Simulator Package

This package implements systolic array fault simulation based on SCALE-Sim architecture.
Includes component-aware fault injection and support for OS/WS/IS dataflows.
"""

from .scalesim_operand_matrix import OperandMatrixGenerator
from .scalesim_dataflow_os import SystolicComputeOS
from .scalesim_dataflow_ws import SystolicComputeWS
from .scalesim_dataflow_is import SystolicComputeIS
from .scalesim_fault_injector import ScaleSimFaultInjector, FaultModel
from .fault_simulator import ScaleSimFaultSimulator
from .fault_visualizer import FaultVisualizer

__all__ = [
    'OperandMatrixGenerator',
    'SystolicComputeOS',
    'SystolicComputeWS',
    'SystolicComputeIS',
    'ScaleSimFaultInjector',
    'FaultModel',
    'ScaleSimFaultSimulator',
    'FaultVisualizer'
]

__version__ = '1.0.0'
__author__ = 'Claude (based on SCALE-Sim)'
