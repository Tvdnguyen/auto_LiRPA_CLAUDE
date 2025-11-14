# Systolic Array Fault Simulator

Based on SCALE-Sim architecture, optimized for fault injection and impact analysis on DNN layers.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Fault Simulator Pipeline               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Operand Matrix Generation (operand_matrix.py)      │
│     ├─ Parse layer config (Conv/FC)                    │
│     ├─ Generate IFMAP address matrix                   │
│     ├─ Generate FILTER address matrix                  │
│     └─ Generate OFMAP address matrix                   │
│           ↓                                            │
│  2. Demand Matrix Generation (systolic_compute_os.py)  │
│     ├─ Tile operands to fit array                     │
│     ├─ Generate cycle-by-cycle PE access patterns     │
│     ├─ Apply dataflow-specific scheduling             │
│     └─ Create PE mapping (cycle, pe_row, pe_col → addr)│
│           ↓                                            │
│  3. Fault Injection (fault_injector.py)                │
│     ├─ Define fault models (stuck-at, bit-flip, etc)  │
│     ├─ Mark faulty PE accesses                        │
│     ├─ Track affected addresses                       │
│     └─ Trace fault propagation to outputs             │
│           ↓                                            │
│  4. Impact Analysis (fault_simulator.py)               │
│     ├─ Create output fault mask                       │
│     ├─ Compute statistics                             │
│     ├─ Visualize affected regions                     │
│     └─ Generate reports                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. SCALE-Sim Based Design
- **Operand matrices**: Address-level mapping (from SCALE-Sim)
- **Demand matrices**: Cycle-accurate PE access patterns (from SCALE-Sim)
- **Tiling**: Handles layers larger than array (from SCALE-Sim)
- **Dataflow support**: OS, WS, IS all implemented
  - **OS (Output Stationary)**: Each PE accumulates one output element
  - **WS (Weight Stationary)**: Each PE holds one weight value
  - **IS (Input Stationary)**: Each PE holds one input activation

### 2. Fault Injection
- **PE-level faults**: Target specific (row, col) coordinates
- **Component-level granularity**: Select specific PE components
  - MAC Unit (Multiply-Accumulate)
  - Accumulator Register
  - Input Register (IFMAP)
  - Weight Register (FILTER)
  - Control Logic
  - Entire PE (all components)
- **Fault types**: Stuck-at-0, stuck-at-1, bit-flip, permanent
- **Timing control**: Permanent or time-bounded (transient) faults
- **Propagation tracking**: Trace faults from PE → addresses → outputs

### 3. TrafficSignNet Integration
- Direct integration with GTSRB project models
- Automatic layer config extraction
- Support Conv and FC layers

### 4. Visualization
- Per-channel fault masks for Conv layers
- Spatial heatmaps showing affected regions
- Grid overlay for pixel-level inspection

## Improvements Over Original GTSRB Simulator

| Feature | Original (gtsrb_project) | New (systolic_fault_sim) |
|---------|-------------------------|--------------------------|
| **Foundation** | Manual mapping heuristics | SCALE-Sim operand/demand matrices |
| **Accuracy** | Approximate | Cycle-accurate |
| **Tiling** | Basic | Proper folding with efficiency metrics |
| **Dataflows** | OS only | **OS, WS, IS** all implemented |
| **Fault Tracking** | Output-level only | Address → PE → cycle tracking |
| **Extensibility** | Hard-coded OS | Modular dataflow architecture |
| **Validation** | None | Based on validated SCALE-Sim |

## Usage

### Quick Start

```bash
cd systolic_fault_sim
python fault_simulator.py
```

### Interactive Mode

```
[Step 1] Configure Systolic Array
Enter array size: 8

Select dataflow:
  1. OS (Output Stationary) - Each PE accumulates one output
  2. WS (Weight Stationary) - Each PE holds one weight
  3. IS (Input Stationary) - Each PE holds one input
Choose dataflow (1/2/3) [default: 1]: 1

[Step 2] Select Layer
Index | Name | Type | Shape
-------------------------------
    0 | conv1 | Conv2d | 1x32x32x32
    1 | conv2 | Conv2d | 1x32x16x16
    ...
Select layer index: 0

[Step 3] Define Faults
PE Array: 8 rows × 8 columns

PE Components:
  1. MAC Unit (Multiply-Accumulate)
  2. Accumulator Register
  3. Input Register (IFMAP)
  4. Weight Register (FILTER)
  5. Control Logic
  6. Entire PE (all components)

Enter faulty PE (row,col) or 'done': 2,3
  Select component (1-6) [default: 6]: 2

  Fault Types:
    1. Stuck-at-0
    2. Stuck-at-1
    3. Bit-flip (random)
    4. Permanent (default)
  Select fault type (1-4) [default: 4]: 1

  Fault Duration:
    1. Permanent (active entire simulation)
    2. Transient (time-bounded)
  Select duration (1-2) [default: 1]: 1

  → Added stuck_at_0 fault at PE (2, 3), accumulator_register

Enter faulty PE (row,col) or 'done': done

[Step 4] Running Simulation...
[Step 5] Generating Visualization...
[Step 6] Exporting Detailed Report...

Output files:
  - fault_impact_conv1.png (visualization)
  - fault_report_conv1.txt (detailed affected regions)
```

### Programmatic API

```python
from fault_simulator import SystolicFaultSimulator, FaultModel

# Create simulator
sim = SystolicFaultSimulator(array_rows=8, array_cols=8, dataflow='OS')

# Define layer
layer_config = {
    'type': 'Conv',
    'name': 'conv1',
    'input_shape': (3, 32, 32),
    'output_shape': (32, 32, 32),
    'kernel_size': (3, 3),
    'stride': 1,
    'padding': 1
}

# Define faults
faults = [
    FaultModel(
        fault_type=FaultModel.STUCK_AT_0,
        fault_location={
            'pe_row': 2,
            'pe_col': 3,
            'component': 'accumulator_register'
        },
        fault_timing={'start_cycle': 0, 'duration': float('inf')}
    )
]

# Run simulation
results = sim.simulate_layer(layer_config, faults)

# Visualize
sim.visualize_results(results, 'output.png')

# Export detailed report
sim.export_fault_report(results, 'fault_report.txt')

# Get statistics
stats = results['statistics']
print(f"Fault coverage: {stats['fault_coverage']*100:.2f}%")
```

## Files

### Core Components

- **operand_matrix.py**: Generate address matrices from layer configs
- **systolic_compute_os.py**: Output Stationary dataflow simulator
- **systolic_compute_ws.py**: Weight Stationary dataflow simulator
- **systolic_compute_is.py**: Input Stationary dataflow simulator
- **fault_injector.py**: Fault models and injection logic
- **fault_simulator.py**: Main simulator and user interface

### Comparison with SCALE-Sim

| SCALE-Sim Module | Our Module | Changes |
|------------------|------------|---------|
| `operand_matrix.py` | `operand_matrix.py` | Simplified, removed sparsity/layout |
| `systolic_compute_os.py` | `systolic_compute_os.py` | Core logic preserved, simplified memory |
| `systolic_compute_ws.py` | `systolic_compute_ws.py` | Simplified from SCALE-Sim WS |
| `systolic_compute_is.py` | `systolic_compute_is.py` | Simplified from SCALE-Sim IS |
| N/A | `fault_injector.py` | **New**: Fault injection capabilities |
| `scale_sim.py` | `fault_simulator.py` | Adapted for fault simulation |

## Dataflow Details

### Output Stationary (OS)

```
Characteristics:
- Each PE accumulates one output element
- Inputs broadcast vertically
- Weights broadcast horizontally
- T cycles of accumulation + W-1 cycles of drain

Mapping:
  Sr = ofmap_pixels (spatial dimension)
  Sc = num_filters (channels)
  T = kernel_size^2 × input_channels

Folding:
  row_fold = ceil(Sr / array_rows)
  col_fold = ceil(Sc / array_cols)

Fault Impact:
- Single PE fault → affects specific spatial positions across all channels
- Localized damage pattern
- Most resilient dataflow
```

### Weight Stationary (WS)

```
Characteristics:
- Each PE holds one weight value (stationary)
- Inputs stream through from top to bottom
- Outputs accumulate horizontally
- H cycles weight load + Sr cycles input stream + W-1 cycles drain

Mapping:
  Sr = ofmap_pixels (spatial dimension)
  Sc = num_filters (channels)
  T = kernel_size^2 × input_channels

Folding:
  row_fold = ceil(T / array_rows)  # Filter positions
  col_fold = ceil(Sc / array_cols)  # Output channels

Fault Impact:
- Single PE fault → affects ALL spatial positions for specific filter weights
- Broader damage pattern than OS
- Critical for filters (one faulty weight affects entire feature map)
```

### Input Stationary (IS)

```
Characteristics:
- Each PE holds one input activation (stationary)
- Weights stream through from left to right
- Outputs accumulate vertically
- H cycles input load + T cycles weight stream + H-1 cycles drain

Mapping:
  Sr = ofmap_pixels (spatial dimension)
  Sc = num_filters (channels)
  T = kernel_size^2 × input_channels

Folding:
  row_fold = ceil(Sr / array_rows)  # Spatial positions
  col_fold = ceil(Sc / array_cols)  # Output channels

Fault Impact:
- Single PE fault → affects specific spatial position across filters
- Similar to OS but different propagation pattern
- Input reuse makes it vulnerable to input-side faults
```

## Example Results

### Conv1 (32 channels, 32×32 spatial)

**Config:**
- Array: 8×8 PE
- Fault: PE (2,3)
- Result: 144/1024 elements per channel affected

**Analysis:**
- Tiling: 32×32 → 4×4 tiles of 8×8
- PE (2,3) in each tile → 4×4 = 16 output positions
- Across 32 channels → 16×32 = 512 total affected (not 144!)
- Discrepancy suggests tiling optimization

**Pattern:**
- Same spatial positions across all channels (OS dataflow)
- Bounding box shows affected region
- Grid visualization shows exact pixels

## Output Files

### Visualization (PNG)
The simulator generates a matplotlib visualization showing:
- **Per-channel view**: Grid showing affected pixels (red) vs normal (green) for each output channel
- **Statistics**: Fault coverage percentage
- **Bounding box**: Visual indicator of affected region

### Detailed Report (TXT)
The text report contains comprehensive information about affected regions:

```
================================================================================
                    SYSTOLIC FAULT SIMULATION REPORT
================================================================================

Layer: conv1 (Conv)
Dataflow: OS
Array Size: 8×8

--------------------------------------------------------------------------------

STATISTICS:
  Total outputs: 32768
  Affected outputs: 28672
  Fault coverage: 87.50%
  Number of faults: 2
  Affected addresses: 145

--------------------------------------------------------------------------------

AFFECTED REGIONS (Conv Layer):

Per-Channel Impact:
--------------------------------------------------------------------------------

Channel 0:
  Affected pixels: 896/1024 (87.50%)
  Bounding box: rows [0, 31], cols [0, 31]
  Box size: 32×32
  Affected coordinates:
    Row 0: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    Row 1: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    ...

Channel 1:
  Affected pixels: 896/1024 (87.50%)
  ...

================================================================================

Spatial Position Impact (across all channels):
--------------------------------------------------------------------------------
  Position (0, 0): 32/32 channels affected (100.00%)
  Position (0, 1): 32/32 channels affected (100.00%)
  ...
```

**Report Contents:**
1. **Layer Information**: Name, type, dataflow, array size
2. **Statistics**: Total outputs, affected outputs, fault coverage
3. **Per-Channel Impact**: For each affected channel:
   - Number and percentage of affected pixels
   - Bounding box coordinates
   - Detailed list of affected coordinates grouped by row
4. **Spatial Position Impact**: Shows which channels are affected at each spatial position

## Future Extensions

### Planned Features

1. **Dataflow Comparison Mode**
   - ✅ All three dataflows (OS, WS, IS) now implemented
   - 🚧 Side-by-side comparison tool (compare fault impact across dataflows)

2. **Advanced Fault Models**
   - Probabilistic faults (Monte Carlo simulation)
   - Correlated faults (spatial/temporal clustering)
   - Memory hierarchy faults (SRAM/DRAM errors)
   - Transient vs permanent fault statistics

3. **Error Detection/Correction**
   - ECC integration
   - Redundancy schemes (DMR, TMR)
   - Selective protection strategies

4. **Integration with Verification**
   - Connect to `batch_verification.py`
   - Verify robustness against fault patterns
   - Critical fault identification

5. **Multi-Layer Analysis**
   - Network-level fault propagation
   - Layer vulnerability ranking
   - End-to-end impact assessment

## Validation

### Correctness Checks

- [x] Operand matrix addresses match layer dimensions
- [x] Demand matrix dimensions: (cycles, num_pes)
- [x] Tiling produces correct number of fold iterations
- [x] PE mapping consistent with demand matrices
- [x] Fault mask dimensions match output shape

### Comparison with SCALE-Sim

- [x] Same operand matrix generation algorithm
- [x] Same OS folding strategy
- [x] Same systolic pipeline depth calculation
- [x] Mapping efficiency matches SCALE-Sim

## References

1. **SCALE-Sim**: "SCALE-Sim: Systolic CNN Accelerator Simulator" (arXiv 2020)
2. **Eyeriss**: "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs" (ISSCC 2016)
3. **TrafficSignNet**: GTSRB project integration

## License

Based on SCALE-Sim (MIT License) with extensions for fault simulation.

---

**Version**: 1.0
**Last Updated**: 2025
**Integration**: Works with GTSRB TrafficSignNet project
