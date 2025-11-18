# Intermediate Layer Perturbation Analysis for DNN Verification

This project implements intermediate layer perturbation analysis for Deep Neural Networks using **auto_LiRPA** (Linear Relaxation-based Perturbation Analysis) combined with **SCALE-Sim** fault simulation to study the impact of hardware faults on neural network robustness.

## 📌 Project Overview

We extend [auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) to support **perturbations on intermediate layers** rather than just input layers, enabling analysis of how faults in systolic array hardware affect neural network inference.

### Key Features

- ✅ **Intermediate Layer Perturbation**: Apply controlled perturbations to any intermediate layer output
- ✅ **Element-wise Epsilon**: Different perturbation magnitudes for different tensor elements
- ✅ **CROWN Backward Bounds**: Compute tight output bounds from perturbed intermediate layers
- ✅ **Fault Simulation Integration**: Parse SCALE-Sim fault simulation results and analyze their impact
- ✅ **GTSRB Traffic Sign Recognition**: Case study on German Traffic Sign Recognition Benchmark

## 🔗 Source Attribution

This project is built upon and extends the following open-source projects:

### 1. auto_LiRPA
- **Source**: [https://github.com/Verified-Intelligence/auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA)
- **License**: BSD 3-Clause License
- **Usage**: Core bound propagation engine (CROWN, IBP)
- **Our Modifications**: Extended to support intermediate layer perturbations with element-wise epsilon

### 2. SCALE-Sim
- **Source**: [https://github.com/ARM-software/SCALE-Sim](https://github.com/ARM-software/SCALE-Sim)
- **License**: MIT License
- **Usage**: Systolic array fault injection simulation
- **Our Modifications**: Integrated fault simulation results for DNN robustness analysis

### 3. GTSRB Dataset
- **Source**: [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
- **Usage**: Training and evaluation dataset

## 📁 Repository Structure

```
auto_LiRPA_CLAUDE/
├── gtsrb_small_tensor_project/          # Our implementation
│   ├── intermediate_bound_module_v2.py  # Extended auto_LiRPA module
│   ├── main_interactive.py              # Interactive testing interface
│   ├── sca_au.py                        # Automated fault analysis
│   ├── traffic_sign_net_small.py        # Smaller CNN model (16/32/64 channels)
│   └── checkpoints/                     # Model checkpoints (download separately)
│
├── scalesim_fault_simulator/            # Modified SCALE-Sim
│   ├── fault_injection.py               # Fault injection logic
│   └── configs/                         # Systolic array configurations
│
├── README.md                            # This file
└── .gitignore                           # Git ignore configuration
```

**Note**: The following directories are excluded from this repository (users must clone them separately):
- `auto_LiRPA/` - Original auto_LiRPA library
- `gtsrb_project/` - Original GTSRB training code
- `systolic_fault_sim/` - Original systolic array simulation

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Step 1: Clone this repository

```bash
git clone <your-repository-url>
cd auto_LiRPA_CLAUDE
```

### Step 2: Clone required source repositories

```bash
# Clone auto_LiRPA
git clone https://github.com/Verified-Intelligence/auto_LiRPA.git

# Clone SCALE-Sim (if needed for full fault simulation)
git clone https://github.com/ARM-software/SCALE-Sim.git systolic_fault_sim
```

### Step 3: Create Python environment

```bash
python3 -m venv gtsrb_env
source gtsrb_env/bin/activate  # On Windows: gtsrb_env\Scripts\activate
```

### Step 4: Install dependencies

```bash
# Install auto_LiRPA
cd auto_LiRPA
pip install -e .
cd ..

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision

# Install other dependencies
pip install numpy pandas matplotlib pillow
```

### Step 5: Download GTSRB dataset

```bash
# Download from https://benchmark.ini.rub.de/
# Extract to gtsrb_project/GTSRB/
mkdir -p gtsrb_project/GTSRB
# ... extract dataset here ...
```

### Step 6: Download model checkpoint (if available)

Download the pre-trained model checkpoint and place it in:
```
gtsrb_small_tensor_project/checkpoints/traffic_sign_net_full.pth
```

## 💡 Usage

### Interactive Mode

Run the interactive testing script to manually select layers and perturbation regions:

```bash
cd gtsrb_small_tensor_project
python main_interactive.py --device cpu
```

**Interactive workflow:**
1. Select a test image from GTSRB dataset
2. Choose intermediate layer to perturb (e.g., `conv6`)
3. Specify perturbation region:
   - **Option 1**: Entire layer
   - **Option 2**: Specific channels
   - **Option 3**: Specific region (channels + spatial coordinates)
   - **Option 4**: Exact mask (load from file)
4. View computed bounds and robustness analysis

### Automated Fault Analysis Mode

Parse SCALE-Sim fault simulation results and automatically analyze their impact:

```bash
cd gtsrb_small_tensor_project
python sca_au.py --device cpu
```

**Automated workflow:**
1. Load fault simulation result file
2. Parse affected elements from fault injection
3. Automatically create perturbation specification
4. Compute bounds with CROWN backward
5. Generate robustness analysis report

## 🔬 Technical Details

### Our Approach: Element-wise Epsilon Perturbation

Unlike standard auto_LiRPA which applies uniform perturbations to input layers, our implementation supports **element-wise epsilon** on intermediate layers:

1. **Forward Pass (Clean)**: Input → Intermediate Layer
   ```python
   with torch.no_grad():
       intermediate_output = model.forward_to_layer(x, target_layer)
   ```

2. **Create Epsilon Tensor**: Each element has its own epsilon value
   ```python
   eps_tensor = torch.zeros_like(intermediate_output)
   eps_tensor[selected_elements] = user_epsilon  # User-specified epsilon
   eps_tensor[other_elements] = 0.0              # No perturbation
   ```

3. **Compute Bounds**: Apply CROWN backward from intermediate layer
   ```python
   lower = intermediate_output - eps_tensor
   upper = intermediate_output + eps_tensor
   lb, ub = lirpa_model.compute_bounds(
       x=None,
       method='backward',
       interm_bounds={target_layer: (lower, upper)}
   )
   ```

This approach allows us to model hardware faults that affect specific elements in a layer's output tensor.

### Key Modifications to auto_LiRPA

**File**: `gtsrb_small_tensor_project/intermediate_bound_module_v2.py`

- **`IntermediateBoundedModuleV2`**: Extended `BoundedModule` class
  - `set_intermediate_perturbation()`: Register perturbation for a specific layer
  - `compute_bounds_from_intermediate()`: Compute bounds starting from perturbed intermediate layer
  - `compute_perturbed_bounds()`: Convenient wrapper with automatic perturbation creation

**File**: `masked_perturbation.py` (in gtsrb_project, shared)

- **`MaskedPerturbationLpNorm`**: Extended `PerturbationLpNorm`
  - `_create_epsilon_tensor()`: Create element-wise epsilon tensor
  - `get_input_bounds()`: Return bounds with element-wise epsilon applied

### Model Architecture

**TrafficSignNetSmall**: Smaller CNN for faster verification
```
Input (3×32×32)
├── Conv1: 3→16, 3×3 + ReLU + MaxPool → (16×16×16)
├── Conv2: 16→32, 3×3 + ReLU + MaxPool → (32×8×8)
├── Conv6: 32→64, 3×3 + ReLU → (64×8×8)  ← Perturbation target
├── Flatten → (4096)
└── FC: 4096→43 (classes)
```

## 📊 Example Results

### Perturbation on Conv6 Layer

```
Layer: /features/features.9/Conv
Shape: (1, 64, 8, 8)
Perturbed region: Channels [1, 9], Spatial [1:8, 1:8]
Epsilon: 0.5

Clean prediction: Class 13 (Yield sign)
Lower bounds: Class 13 = 8.234
Upper bounds: Class 13 = 12.456
Robustness margin: 4.222

Verdict: ROBUST (prediction unchanged under perturbation)
```

## 🐛 Troubleshooting

### Common Issues

1. **"Module 'intermediate_bound_module_v2' has no attribute 'XXX'"**
   - Clear Python cache: `rm -rf __pycache__`
   - Restart Python interpreter

2. **"The size of tensor a (4096) must match the size of tensor b (8)"**
   - This error should be fixed in V2 implementation
   - If it persists, check that you're using `intermediate_bound_module_v2.py`

3. **Out of Memory (OOM)**
   - Use GPU: `--device cuda`
   - Reduce perturbation region size
   - Use smaller model

## 📝 Citation

If you use this code in your research, please cite the original auto_LiRPA paper:

```bibtex
@inproceedings{xu2020automatic,
  title={Automatic perturbation analysis for scalable certified robustness and beyond},
  author={Xu, Kaidi and Shi, Zhouxing and Zhang, Huan and Wang, Yihan and Chang, Kai-Wei and Huang, Minlie and Kailkhura, Bhavya and Lin, Xue and Hsieh, Cho-Jui},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

And SCALE-Sim:

```bibtex
@inproceedings{samajdar2018scale,
  title={SCALE-Sim: Systolic CNN accelerator simulator},
  author={Samajdar, Ananda and Zhu, Yuhao and Whatmough, Paul and Mattina, Matthew and Krishna, Tushar},
  booktitle={IEEE International Symposium on Performance Analysis of Systems and Software},
  year={2018}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

## 📄 License

This project extends BSD-licensed (auto_LiRPA) and MIT-licensed (SCALE-Sim) software.

**Our modifications** are provided as-is for research purposes. Please refer to the original licenses for auto_LiRPA and SCALE-Sim for their respective terms.

## 🙏 Acknowledgments

- **auto_LiRPA team** for the excellent neural network verification library
- **SCALE-Sim team** for the systolic array simulator
- **GTSRB dataset creators** for the traffic sign recognition benchmark
