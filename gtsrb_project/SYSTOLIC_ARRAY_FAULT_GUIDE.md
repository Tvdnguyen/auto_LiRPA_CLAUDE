# Systolic Array Fault Simulator - User Guide

## Tổng Quan

**Systolic Array Fault Simulator** mô phỏng ảnh hưởng của lỗi phần cứng (hardware faults) trên Systolic Array đến output tensor của các layer trong DNN.

### Mục đích

Hiểu được:
- Lỗi ở PE nào → ảnh hưởng đến vùng nào của output tensor
- Dataflow khác nhau → fault propagation khác nhau như thế nào
- Layer nào sensitive hơn với hardware faults

### Applications

- ✅ Fault-tolerant DNN design
- ✅ Selective protection strategies
- ✅ Reliability analysis
- ✅ Error correction placement
- ✅ Safety-critical system design

---

## 1. Quick Start

```bash
# Activate environment
source gtsrb_env/bin/activate

# Run simulator
bash run_fault_simulator.sh

# Or directly
python systolic_array_fault_simulator.py --interactive
```

---

## 2. Interactive Workflow

### Step 1: Configure Systolic Array

```
[Step 1] Configure Systolic Array
--------------------------------------------------------------------------------
Enter array size (e.g., '8' for 8x8, or '16,8' for 16x8): 8
Select dataflow (IS/OS/WS): OS
```

**Array Size:**
- Square: `8` → 8×8 PE array
- Rectangle: `16,8` → 16×8 PE array (16 rows, 8 columns)

**Dataflow Options:**

| Dataflow | Full Name | Description |
|----------|-----------|-------------|
| **OS** | Output Stationary | Mỗi PE tính một output activation |
| **WS** | Weight Stationary | Mỗi PE giữ một weight |
| **IS** | Input Stationary | Mỗi PE giữ một input activation |

---

### Step 2: Select Layer

```
[Step 2] Select Layer from TrafficSignNet
--------------------------------------------------------------------------------
Index |   Layer Name    |    Type    |    Output Shape
--------------------------------------------------------------------------------
    0 |      conv1      |   Conv2d   |     1x32x32x32
    1 |      conv2      |   Conv2d   |     1x32x16x16
    2 |      conv3      |   Conv2d   |     1x64x16x16
    3 |      conv4      |   Conv2d   |     1x64x8x8
    4 |      conv5      |   Conv2d   |     1x128x8x8
    5 |      conv6      |   Conv2d   |     1x128x4x4
    6 |       fc1       |   Linear   |       1x512
    7 |       fc2       |   Linear   |       1x256
    8 |       fc3       |   Linear   |        1x43

Select layer index: 2
Selected: conv3 (Conv)
```

**Chọn layer nào?**
- **Conv layers (0-5):** Spatial output, interesting fault patterns
- **FC layers (6-8):** Vector output, simpler patterns

---

### Step 3: Select Faulty PE Region

```
[Step 3] Select Faulty PE Region
--------------------------------------------------------------------------------
PE Array: 8 rows × 8 columns
Enter PE coordinates (row,col) one per line. Enter 'done' when finished.
Or enter range like '0-2,0-3' for PEs in rows 0-2, cols 0-3

Faulty PE (or 'done'): 0,0
  Added PE (0, 0)
Faulty PE (or 'done'): 1,1
  Added PE (1, 1)
Faulty PE (or 'done'): 2-3,2-3
  Added PEs: rows [2:3], cols [2:3]
Faulty PE (or 'done'): done

Total faulty PEs: 6
```

**Input Formats:**

1. **Single PE:** `row,col`
   ```
   0,0     → PE at row 0, col 0
   3,5     → PE at row 3, col 5
   ```

2. **Range of PEs:** `row_start-row_end,col_start-col_end`
   ```
   0-2,0-2     → PEs in rows [0,1,2], cols [0,1,2] (9 PEs total)
   4-7,0-7     → Bottom half of 8×8 array (32 PEs)
   ```

**Common Fault Patterns:**

```
Single PE failure:        0,0

Corner region (2×2):      0-1,0-1
                          □ □ . . . . . .
                          □ □ . . . . . .
                          . . . . . . . .

Diagonal:                 0,0  1,1  2,2  3,3
                          □ . . . . . . .
                          . □ . . . . . .
                          . . □ . . . . .
                          . . . □ . . . .

Row failure:              0-0,0-7
                          □ □ □ □ □ □ □ □
                          . . . . . . . .

Column failure:           0-7,0-0
                          □ . . . . . . .
                          □ . . . . . . .
                          □ . . . . . . .
```

---

### Step 4-5: Results

**Text Output:**
```
================================================================================
AFFECTED REGIONS IN conv3 OUTPUT
================================================================================

Output shape: (Channels=64, Height=16, Width=16)
Total elements: 16384
Faulty elements: 1024 (6.25%)

Affected regions by channel:

  Channel 0: 16/256 elements faulty
    Bounding box: H[0:4], W[0:4]
    Positions: [(0, 0), (0, 1), (1, 0), (1, 1), ...]

  Channel 1: 16/256 elements faulty
    Bounding box: H[0:4], W[0:4]
    ...
================================================================================
```

**Visualization:**

Mỗi channel được vẽ thành một subplot:
- **Trắng:** Output elements OK
- **Đỏ:** Output elements bị ảnh hưởng bởi faulty PEs

Saved to: `fault_visualization_conv3.png`

---

## 3. Dataflow Comparison

### 3.1. Output Stationary (OS)

**Concept:**
- Mỗi PE accumulate một output element
- Input và weight stream vào PE
- Mỗi PE tính đầy đủ một output

**Fault Impact:**
```
Faulty PE (r, c) → Affects specific output positions

For Conv: PE (r,c) computes outputs at positions (*, h, w)
          where h = tile_h*H + r, w = tile_w*W + c

For FC:   PE (r,c) computes output neuron index = r*W + c
```

**Example:**
```
8×8 Array, OS dataflow, Conv3 (64×16×16)

Faulty PE: (0, 0)
Affected:
  - Channel 0: position (0, 0)
  - Channel 1: position (0, 0)
  - ...
  - Channel 63: position (0, 0)

→ Same spatial position across all channels
```

**Visualization Pattern:**
- Sparse, regular pattern
- Each faulty PE → Small number of outputs
- Localized spatial damage

---

### 3.2. Weight Stationary (WS)

**Concept:**
- Mỗi PE giữ một weight cố định
- Inputs broadcast đến nhiều PEs
- Mỗi PE tính partial products cho nhiều outputs

**Fault Impact:**
```
Faulty PE (r, c) → Affects ALL spatial positions for specific (Cout, Cin) pair

For Conv: PE (r,c) holds weight W[cout, cin, k, k]
          Affects all (cout, h, w) for all h, w

For FC:   PE (r,c) holds weight W[out, in]
          Affects output[out] only
```

**Example:**
```
8×8 Array, WS dataflow, Conv3 (64×16×16)

Faulty PE: (0, 0)
Affected:
  - Channel 0: ALL 16×16 positions

→ Entire channel corrupted!
```

**Visualization Pattern:**
- Dense, channel-wide damage
- Most severe fault impact
- Whole channels can be corrupted

---

### 3.3. Input Stationary (IS)

**Concept:**
- Mỗi PE giữ một input activation
- Weights flow qua PEs
- Mỗi PE contribute đến nhiều outputs (kernel overlap)

**Fault Impact:**
```
Faulty PE (r, c) → Affects outputs that use this input

For Conv: PE (r,c) holds input I[cin, h, w]
          Affects outputs where kernel overlaps this position

For FC:   PE (r,c) holds input I[in]
          Affects ALL outputs (full connection)
```

**Example:**
```
8×8 Array, IS dataflow, Conv3 (64×16×16)

Faulty PE: (0, 0)
Affected:
  - Multiple spatial positions (kernel size dependent)
  - All channels

→ Scattered spatial damage across channels
```

**Visualization Pattern:**
- Medium density
- Scattered spatial pattern
- Depends on kernel size and stride

---

## 4. Use Cases

### 4.1. Single PE Fault Analysis

**Question:** Which dataflow is most resilient to single PE failure?

```bash
# Test with OS
Array: 8x8
Dataflow: OS
Layer: conv3
Faulty PE: 0,0

# Test with WS
Array: 8x8
Dataflow: WS
Layer: conv3
Faulty PE: 0,0

# Test with IS
Array: 8x8
Dataflow: IS
Layer: conv3
Faulty PE: 0,0

# Compare faulty element percentages
```

**Expected Result:**
- OS: Lowest impact (localized)
- WS: Highest impact (channel-wide)
- IS: Medium impact (scattered)

---

### 4.2. Critical PE Identification

**Question:** Which PEs are most critical?

```bash
# Test corner PE
Faulty PE: 0,0

# Test center PE
Faulty PE: 4,4

# Test edge PE
Faulty PE: 0,4

# Compare impact
```

**Analysis:**
- All PEs equally important in OS
- Some PEs more critical in WS/IS (depends on mapping)

---

### 4.3. Layer Sensitivity Analysis

**Question:** Which layer is most sensitive to faults?

```bash
# Test conv1 (32×32×32 output)
Layer: 0
Faulty PE: 0-1,0-1

# Test conv3 (64×16×16 output)
Layer: 2
Faulty PE: 0-1,0-1

# Test fc1 (512 output)
Layer: 6
Faulty PE: 0-1,0-1

# Compare faulty percentage
```

**Insight:**
- Larger output → lower fault percentage (same faulty PEs)
- Early layers (large outputs) more resilient
- Final layers (small outputs) more vulnerable

---

### 4.4. Fault Coverage Analysis

**Question:** How many PEs can fail before X% output corrupted?

```bash
# Gradually increase faulty PE region
1 PE:     0,0
4 PEs:    0-1,0-1
9 PEs:    0-2,0-2
16 PEs:   0-3,0-3

# Record faulty output percentage
# Plot curve
```

---

### 4.5. Protection Strategy Design

**Question:** Where to add redundancy/error correction?

**Scenario 1: Protect critical outputs**
```bash
# Identify which PEs affect final classification layer
Layer: fc3 (output layer, 43 classes)
Dataflow: OS

# Each PE computes specific output classes
# Protect PEs computing top-K likely classes
```

**Scenario 2: Protect critical channels**
```bash
# In WS dataflow, each PE maps to specific channels
# Identify important channels (feature importance analysis)
# Add ECC to PEs computing those channels
```

---

## 5. Understanding Output

### 5.1. Text Output Interpretation

```
Channel 5: 64/256 elements faulty
  Bounding box: H[0:8], W[0:8]
```

**Meaning:**
- Out of 256 elements in channel 5 (16×16 spatial)
- 64 elements are corrupted (25%)
- Corrupted region spans H[0:8], W[0:8] (top-left quadrant)

**Implications:**
- If this channel detects important features → serious impact
- If this region corresponds to important image area → serious impact
- May need protection for these PEs

---

### 5.2. Visualization Interpretation

**Example: Conv Layer**

```
Ch 0  Ch 1  Ch 2  Ch 3  Ch 4  Ch 5  Ch 6  Ch 7
[▓░░]  [░░░]  [▓▓░]  [░░░]  [░░░]  [▓▓▓]  [░░░]  [░░░]
 ...    ...    ...    ...    ...    ...    ...    ...

Ch 8  Ch 9  Ch 10 Ch 11 Ch 12 Ch 13 Ch 14 Ch 15
[░▓░]  [░░░]  [░░░]  [▓░░]  [░░░]  [░░▓]  [░░░]  [░░░]
```

Where:
- `░` = White = OK elements
- `▓` = Red = Faulty elements

**Pattern Analysis:**

**OS Dataflow:** Sparse red dots, same position across channels
```
Ch 0: [▓░░░]    Ch 1: [▓░░░]    Ch 2: [▓░░░]
      [░░░░]          [░░░░]          [░░░░]
      [░░░░]          [░░░░]          [░░░░]

→ Top-left corner affected across all channels
```

**WS Dataflow:** Entire channels red
```
Ch 0: [▓▓▓▓]    Ch 1: [░░░░]    Ch 2: [▓▓▓▓]
      [▓▓▓▓]          [░░░░]          [▓▓▓▓]
      [▓▓▓▓]          [░░░░]          [▓▓▓▓]

→ Channels 0 and 2 completely corrupted
```

**IS Dataflow:** Scattered red regions
```
Ch 0: [░▓░░]    Ch 1: [░▓░░]    Ch 2: [░▓░░]
      [▓▓▓░]          [▓▓▓░]          [▓▓▓░]
      [░▓░░]          [░▓░░]          [░▓░░]

→ Scattered pattern, similar across channels
```

---

## 6. Example Session

```bash
$ bash run_fault_simulator.sh

========================================================================
                    SYSTOLIC ARRAY FAULT SIMULATOR
========================================================================

[Step 1] Configure Systolic Array
--------------------------------------------------------------------------------
Enter array size (e.g., '8' for 8x8, or '16,8' for 16x8): 8
Select dataflow (IS/OS/WS): OS
Initialized 8×8 Systolic Array with OS dataflow

[Step 2] Select Layer from TrafficSignNet
--------------------------------------------------------------------------------
Index |   Layer Name    |    Type    |    Output Shape
--------------------------------------------------------------------------------
    0 |      conv1      |   Conv2d   |     1x32x32x32
    1 |      conv2      |   Conv2d   |     1x32x16x16
    2 |      conv3      |   Conv2d   |     1x64x16x16
    3 |      conv4      |   Conv2d   |     1x64x8x8
    4 |      conv5      |   Conv2d   |     1x128x8x8
    5 |      conv6      |   Conv2d   |     1x128x4x4
    6 |       fc1       |   Linear   |       1x512
    7 |       fc2       |   Linear   |       1x256
    8 |       fc3       |   Linear   |        1x43

Select layer index: 2
Selected: conv3 (Conv)

[Step 3] Select Faulty PE Region
--------------------------------------------------------------------------------
PE Array: 8 rows × 8 columns
Enter PE coordinates (row,col) one per line. Enter 'done' when finished.
Or enter range like '0-2,0-3' for PEs in rows 0-2, cols 0-3

Faulty PE (or 'done'): 0-1,0-1
  Added PEs: rows [0:1], cols [0:1]
Faulty PE (or 'done'): done

Total faulty PEs: 4

[Step 4] Simulating Fault Propagation...
--------------------------------------------------------------------------------

[Step 5] Results
================================================================================

================================================================================
AFFECTED REGIONS IN conv3 OUTPUT
================================================================================

Output shape: (Channels=64, Height=16, Width=16)
Total elements: 16384
Faulty elements: 256 (1.56%)

Affected regions by channel:

  Channel 0: 4/256 elements faulty
    Bounding box: H[0:2], W[0:2]
    Positions: [(0, 0), (0, 1), (1, 0), (1, 1)]

  Channel 1: 4/256 elements faulty
    Bounding box: H[0:2], W[0:2]
    Positions: [(0, 0), (0, 1), (1, 0), (1, 1)]

  ... (channels 2-63 similar)

================================================================================

================================================================================
Generating visualization...
================================================================================

Visualization saved to: fault_visualization_conv3.png

Simulation completed!
```

---

## 7. Advanced Topics

### 7.1. Tiling and Temporal Loops

**Large Layers Don't Fit in SA:**

For conv3 (64×16×16 output), 8×8 SA must tile:
- Spatial tiling: 16×16 output tiled into 2×2 tiles of 8×8
- Channel tiling: 64 channels processed sequentially

**Impact:**
- Same PE computes multiple outputs across tiles
- Fault affects outputs in multiple tiles
- More outputs affected than naive expectation

**Example:**
```
OS dataflow, 8×8 SA, conv3

PE (0, 0) computes:
  Tile (0,0): output (*, 0, 0)  ← Affected
  Tile (0,1): output (*, 0, 8)  ← Affected
  Tile (1,0): output (*, 8, 0)  ← Affected
  Tile (1,1): output (*, 8, 8)  ← Affected

→ 4 spatial positions affected, not just 1!
```

Current simulator **accounts for tiling** in mapping.

---

### 7.2. Comparison with Other Accelerators

| Accelerator Type | Fault Propagation | Resilience Strategy |
|------------------|-------------------|---------------------|
| **Systolic Array** | Structured, predictable | Spatial redundancy, PE-level ECC |
| **GPU** | Thread-level, scattered | Thread re-execution, checkpointing |
| **FPGA** | Configuration-dependent | TMR, partial reconfiguration |
| **Neuromorphic** | Spike-based, temporal | Redundant synapses, noise tolerance |

Systolic Array advantage: **Predictable fault mapping** → easier to protect critical regions

---

### 7.3. Connecting to Verification

**Integration with `batch_verification.py`:**

```python
# Idea: Combine fault simulation + verification

# 1. Run fault simulator
#    → Get fault mask for layer output

# 2. Apply fault mask as perturbation
#    → Use MaskedPerturbation with fault_mask

# 3. Run verification
#    → Check if network robust to this fault pattern

# 4. Identify critical faults
#    → Which fault patterns break verification?
```

**Example workflow:**
```bash
# Step 1: Simulate fault
python systolic_array_fault_simulator.py
# Select conv3, faulty PEs (0-1,0-1)
# Get fault mask

# Step 2: Convert fault mask → perturbation config
# Fault mask: channels [0-63], H[0:2], W[0:2]

# Step 3: Run batch verification with this perturbation
python batch_verification.py
# Select conv3
# Channels: 0-63
# Height: 0,2
# Width: 0,2
# Epsilon: large (to model complete corruption)

# Step 4: Analyze
# How many samples remain verified despite hardware fault?
```

---

## 8. Limitations

### Current Limitations:

1. **Simplified Dataflow Models**
   - Real accelerators have complex mappings
   - Many optimizations not modeled (e.g., data reuse)

2. **No Timing Information**
   - Doesn't model when faults occur
   - Assumes persistent faults

3. **Binary Fault Model**
   - Fault = complete output corruption
   - Reality: soft errors, value distortion

4. **Single Layer at a Time**
   - Doesn't propagate faults across layers
   - End-to-end impact not modeled

### Future Extensions:

- ✅ Multi-layer fault propagation
- ✅ Probabilistic fault models
- ✅ Timing-aware simulation
- ✅ Power/energy estimation
- ✅ Fault injection experiments

---

## 9. References

### Papers:

1. **Eyeriss:** "Eyeriss: An Energy-Efficient Reconfigurable Accelerator for Deep CNNs" (ISSCC 2016)
2. **SCALE-Sim:** "SCALE-Sim: Systolic CNN Accelerator Simulator" (arXiv 2020)
3. **Fault Tolerance:** "Understanding the Resilience of Deep Neural Networks to Faults in Systolic Arrays" (DATE 2020)
4. **Protection:** "Selective Protection of DNN Accelerators Against Soft Errors" (IISWC 2019)

### Related Tools:

- **SCALE-Sim:** Cycle-accurate SA simulator
- **Timeloop:** Data-centric mapping exploration
- **MAGNet:** Fault injection for DNNs

---

## 10. Troubleshooting

### Issue 1: Matplotlib Not Installed

```
ModuleNotFoundError: No module named 'matplotlib'
```

**Solution:**
```bash
pip install matplotlib
```

### Issue 2: All Outputs Affected

**Possible causes:**
- Too many faulty PEs
- WS dataflow (inherently more impactful)
- Small layer output

**Debug:**
- Try fewer faulty PEs
- Try OS dataflow
- Try larger layer (e.g., conv1 vs conv6)

### Issue 3: No Visualization Window

**On remote server:**
```bash
# Save figure without display
# Modify code to use plt.savefig() without plt.show()
```

---

**Version:** 1.0
**Last Updated:** 2025
**Related Files:**
- `systolic_array_fault_simulator.py`: Main simulator
- `run_fault_simulator.sh`: Runner script
- Survey paper: "A Survey of Design and Optimization for Systolic Array-based DNN Accelerators"
