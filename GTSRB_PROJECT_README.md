# GTSRB Systolic Array PE Sensitivity Analysis

Integrated PE sensitivity analysis combining systolic array fault simulation with DNN verification for GTSRB traffic sign recognition.

## Overview

This project analyzes the sensitivity of Processing Elements (PEs) in systolic arrays by:
1. Simulating faults in specific PE positions
2. Mapping fault impact to Conv1 layer activations
3. Using Auto-LiRPA to verify robustness under perturbations
4. Ranking PEs by their impact on DNN robustness (lower epsilon = more critical PE)

## Quick Setup on New Server

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd auto_LiRPA_CLAUDE
```

### 2. Create Virtual Environment

```bash
python3 -m venv gtsrb_env
source gtsrb_env/bin/activate  # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install torch torchvision
pip install auto-lirpa
pip install numpy pandas matplotlib Pillow
```

### 4. Download GTSRB Dataset

Download the GTSRB test dataset and extract to:
```
gtsrb_project/data/GTSRB_data/
```

Expected structure:
```
gtsrb_project/data/GTSRB_data/
├── Test/
│   ├── GT-final_test.csv
│   └── 00000.ppm, 00001.ppm, ... (12,630 images)
```

**Dataset source**: [GTSRB at INI](https://benchmark.ini.rub.de/gtsrb_dataset.html)

Download command:
```bash
cd gtsrb_project/data
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
unzip GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_GT.zip
mv GTSRB/Final_Test GTSRB_data/Test
```

### 5. Verify Setup

```bash
python3 debug_dataset.py
```

Expected output: `✓ Dataset loaded: 12630 samples`

## Project Structure

```
auto_LiRPA_CLAUDE/
├── gtsrb_project/                    # GTSRB model and dataset
│   ├── traffic_sign_net.py          # CNN model (3 conv + 3 fc)
│   ├── gtsrb_dataset.py              # Dataset loader
│   ├── main_interactive.py           # Interactive perturbation testing
│   ├── correct_samples/              # Pre-computed correct samples ✓
│   │   └── class_XX_correct_indices.csv (43 files)
│   ├── checkpoints/
│   │   └── traffic_sign_net_full.pth # Trained model (17MB) ✓
│   └── data/
│       └── GTSRB_data/               # Dataset (download separately)
│
├── systolic_fault_sim/               # Systolic array fault simulator
│   ├── fault_simulator.py           # Fault simulation engine
│   ├── operand_matrix.py            # IS dataflow operand generation
│   └── fault_model.py                # Transient bit-flip models
│
├── integrated_pe_sensitivity_analysis_low_memory.py  # Main analyzer ⭐
├── test_single_pe.py                 # Test single PE (RAM-efficient)
├── run_all_pes_sequential.sh         # Batch process all 64 PEs
├── test_memory_fix.sh                # Quick verification test
│
├── MEMORY_FIX_EXPLANATION.md         # Details on optimization
└── MAIN_INTERACTIVE_UPDATE.md        # Random sample selection docs
```

## Usage

### 1. Interactive Testing (Manual Exploration)

Manually test perturbations on specific layers and regions:

```bash
cd gtsrb_project
python3 main_interactive.py \
    --data_dir data/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net_full.pth
```

**Interactive flow**:
1. Select layer (0=Conv1, 1=Conv2, etc.)
2. **Enter class ID** → program randomly selects a correct sample
3. Configure perturbation region (channel, height, width)
4. Enter epsilon value
5. View verification results (robust/not robust)

### 2. PE Sensitivity Analysis

#### Option A: Test Single PE (For Low RAM Systems)

Test one PE at a time (RAM-safe, ~30-60 seconds per PE):

```bash
python3 test_single_pe.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --pe_row 0 --pe_col 0 \
    --class_id 0 --test_idx 0 \
    --duration 1 --tolerance 0.05 --epsilon_max 0.3
```

**Parameters**:
- `--pe_row`, `--pe_col`: PE position (0-7 for 8×8 array)
- `--class_id`: GTSRB class (0-42)
- `--test_idx`: Sample index within class correct_samples
- `--duration`: Fault duration in cycles (1 = minimal memory)
- `--tolerance`: Binary search tolerance (0.05 = faster, less precise)
- `--epsilon_max`: Maximum epsilon to search (0.3 = faster)

**Output**: `pe_<row>_<col>_result_<timestamp>.csv`

#### Option B: Test All PEs (Sequential, High RAM Recommended)

Run all 64 PEs sequentially (~30-60 minutes on 16GB+ RAM):

```bash
./run_all_pes_sequential.sh
```

**Output**:
- Individual results: `pe_results_<timestamp>/pe_<row>_<col>_result_*.csv`
- Combined: `pe_results_<timestamp>/combined_results.csv`

#### Option C: Quick Memory Test

Verify the memory optimization works:

```bash
./test_memory_fix.sh
```

## Configuration

### Systolic Array
- **Dataflow**: IS (Input Stationary) only
- **Array sizes**: 8×8 (default), 16×16 supported
- **Fault location**: Accumulator register
- **Fault type**: Transient bit-flip
- **Target layer**: Conv1 only
- **Critical timing**: Weight stream phase at cycles [H, H+T-1]

### Memory Optimization
- **Duration**: 1 cycle (reduced from 10)
- **Tolerance**: 0.05 (vs 0.001 for high precision)
- **Epsilon max**: 0.3 (vs 1.0 full range)
- **Batch size**: 1 sample at a time
- **Perturbation**: Bounding box slicing (10-20x faster than loops)

## Output Format

### Single PE Result CSV

```csv
pe_row,pe_col,max_epsilon,num_affected_channels,num_affected_spatial,global_idx,class_idx,class_id,true_label,pred_class,array_size
0,0,0.1234,12,45,243,0,0,0,0,8
```

**Fields**:
- `pe_row`, `pe_col`: PE position in systolic array
- `max_epsilon`: Maximum epsilon for verified robustness (⚠️ **lower = more critical PE**)
- `num_affected_channels`: Channels impacted by PE fault
- `num_affected_spatial`: Spatial positions (H×W) impacted
- `global_idx`: Test dataset index
- `class_idx`, `class_id`: Sample info
- `array_size`: 8 or 16

### Combined Results Analysis

Use `combined_results.csv` to:
1. **Rank PEs by sensitivity**: Sort by `max_epsilon` (ascending)
2. **Create heatmap**: Visualize PE criticality on 8×8 grid
3. **Fault tolerance**: Identify redundant vs critical PEs
4. **Spatial analysis**: Correlate PE position with impact

Example Python analysis:
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
df = pd.read_csv('pe_results_<timestamp>/combined_results.csv')

# Create 8x8 heatmap
heatmap = np.zeros((8, 8))
for _, row in df.iterrows():
    heatmap[row['pe_row'], row['pe_col']] = row['max_epsilon']

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=0.3)
plt.colorbar(label='Max Epsilon (lower = more critical)')
plt.title('PE Sensitivity Heatmap')
plt.xlabel('PE Column')
plt.ylabel('PE Row')
plt.savefig('pe_heatmap.png', dpi=300)
```

## System Requirements

### Minimum (Single PE Test)
- **RAM**: 8GB
- **CPU**: 4 cores
- **Time per PE**: 30-60 seconds
- **Storage**: 2GB (code + checkpoint + correct_samples)

### Recommended (Full Analysis)
- **RAM**: 16GB+
- **CPU**: 8+ cores
- **Time for 64 PEs**: 30-60 minutes
- **Storage**: 5GB (+ GTSRB dataset ~500MB)

### High Performance
- **RAM**: 32GB+
- **CPU**: 16+ cores
- Can reduce `tolerance` to 0.001 for high precision
- Can increase `epsilon_max` to 1.0 for full range
- Time for 64 PEs: ~15-20 minutes

## Troubleshooting

### 1. Dataset Not Found
```
Error: Found 0 samples of class 0
```
**Solution**: Ensure data_dir is `gtsrb_project/data/GTSRB_data` (includes the `GTSRB_data` folder)

### 2. Process Killed (Out of Memory)
```
zsh: killed python3 test_single_pe.py
```
**Solutions**:
- Close all other applications
- Restart computer to free RAM
- Reduce `--epsilon_max` to 0.2 or 0.1
- Increase `--tolerance` to 0.1
- Ensure using `test_single_pe.py` not the full analyzer

### 3. Checkpoint Error
```
_pickle.UnpicklingError: invalid load key, '\x0a'
```
**Solutions**:
- Re-clone repository: `git clone --depth 1 <repo-url>`
- Verify checkpoint size: `ls -lh gtsrb_project/checkpoints/traffic_sign_net_full.pth` (should be ~17MB)
- Check file integrity: `file gtsrb_project/checkpoints/traffic_sign_net_full.pth` (should be "data")

### 4. Correct Samples Missing
```
FileNotFoundError: correct_samples/class_00_correct_indices.csv
```
**Solution**: Verify git clone included `gtsrb_project/correct_samples/` directory with 43 CSV files

### 5. Import Errors
```
ModuleNotFoundError: No module named 'auto_LiRPA'
```
**Solution**: Ensure virtual environment is activated and dependencies installed:
```bash
source gtsrb_env/bin/activate
pip install auto-lirpa torch numpy
```

## Key Features

### 1. Memory-Efficient Perturbation Creation
Uses **bounding box slicing** instead of nested loops:
- Old: O(H×W×C) iterations → 10GB+ RAM
- New: O(1) slice operations → <6GB RAM
- See [MEMORY_FIX_EXPLANATION.md](MEMORY_FIX_EXPLANATION.md)

### 2. Random Sample Selection
Automatically selects random correctly-classified samples from pre-computed lists:
- Ensures testing on correct predictions only
- Improves coverage vs fixed indices
- See [MAIN_INTERACTIVE_UPDATE.md](MAIN_INTERACTIVE_UPDATE.md)

### 3. Exact Fault-to-Activation Mapping
- Maps PE faults to Conv1 activation regions precisely
- Accounts for IS dataflow weight streaming timing
- Uses bounding box over-approximation (conservative)

### 4. Verified Robustness Bounds
- Auto-LiRPA backward bounds (α-CROWN)
- Robustness criterion: `pred_lb > max(other_ub)`
- Binary search for maximum verified epsilon

## Citation

If you use this code, please cite:

**Auto-LiRPA**:
```bibtex
@inproceedings{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  booktitle={NeurIPS},
  year={2020}
}
```

**GTSRB Dataset**:
```bibtex
@inproceedings{stallkamp2012man,
  title={Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition},
  author={Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  booktitle={Neural networks},
  year={2012}
}
```

## License

MIT License - See LICENSE file for details

## Related Work

- [Auto-LiRPA Library](https://github.com/Verified-Intelligence/auto_LiRPA)
- [α,β-CROWN Verifier](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [GTSRB Dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html)

## Contact

For issues or questions:
- Open an issue on GitHub
- Check [MEMORY_FIX_EXPLANATION.md](MEMORY_FIX_EXPLANATION.md) for RAM issues
- Check [MAIN_INTERACTIVE_UPDATE.md](MAIN_INTERACTIVE_UPDATE.md) for usage
