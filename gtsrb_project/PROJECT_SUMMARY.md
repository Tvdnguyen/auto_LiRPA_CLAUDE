# Tổng Kết Dự Án GTSRB với Intermediate Layer Perturbations

## Tổng Quan

Dự án này thực hiện việc **thêm nhiễu loạn (perturbation) vào các tensor output của các layer trung gian** trong mô hình nhận dạng biển báo giao thông GTSRB, sử dụng thư viện auto_LiRPA để tính toán bounds.

### Điểm Mới So Với auto_LiRPA Gốc

1. ✅ **Perturbation trên intermediate layers** thay vì chỉ input layer
2. ✅ **Masked perturbation** - chỉ nhiễu loạn một vùng cụ thể của tensor
3. ✅ **Interactive interface** - dễ dàng test và thử nghiệm
4. ✅ **Complete pipeline** - từ training đến verification

---

## Danh Sách Files Đã Tạo

### 1. Core Implementation Files

#### `gtsrb_dataset.py` (190 dòng)
**Mục đích**: Load và preprocess GTSRB dataset

**Chức năng chính**:
- `GTSRBDataset`: PyTorch Dataset class cho GTSRB
- `get_gtsrb_transforms()`: Data augmentation và normalization
- `get_gtsrb_dataloaders()`: Tạo train/test dataloaders

**Cách sử dụng**:
```python
from gtsrb_dataset import get_gtsrb_dataloaders

train_loader, test_loader = get_gtsrb_dataloaders(
    root_dir='/path/to/GTSRB',
    batch_size=128
)
```

---

#### `traffic_sign_net.py` (189 dòng)
**Mục đích**: Định nghĩa kiến trúc CNN cho nhận dạng biển báo

**Models**:
- `TrafficSignNet`: Full model với 6 Conv layers + 3 FC layers (~1.4M params)
- `TrafficSignNetSimple`: Simplified model cho testing nhanh

**Kiến trúc TrafficSignNet**:
```
Conv1(3→32) → Conv2(32→32) → Pool → Dropout
↓
Conv3(32→64) → Conv4(64→64) → Pool → Dropout
↓
Conv5(64→128) → Conv6(128→128) → Pool → Dropout
↓
FC1(2048→512) → FC2(512→256) → FC3(256→43)
```

**Method đặc biệt**:
- `get_layer_info()`: Trả về list các Conv/FC layers để user chọn

---

#### `masked_perturbation.py` (354 dòng)
**Mục đích**: **CORE INNOVATION** - Implement masked perturbation cho intermediate layers

**Class chính**: `MaskedPerturbationLpNorm`

**Kế thừa từ**: `PerturbationLpNorm` (auto_LiRPA)

**Parameters mới**:
- `batch_idx`: Chọn batch nào để perturb
- `channel_idx`: Chọn channel/feature nào
- `height_slice`: Vùng height (cho Conv)
- `width_slice`: Vùng width (cho Conv)

**Methods quan trọng**:
- `_create_mask_from_spec()`: Tạo boolean mask từ specs
- `get_input_bounds()`: Tính x_L, x_U với mask applied
- `init()`: Initialize bounds cho forward/backward mode

**Ví dụ sử dụng**:
```python
# Perturb only channels 0-2, pixels (5-10, 5-10) of batch 0
ptb = MaskedPerturbationLpNorm(
    eps=0.1,
    norm=np.inf,
    batch_idx=0,
    channel_idx=[0, 1, 2],
    height_slice=(5, 10),
    width_slice=(5, 10)
)

# Apply to intermediate tensor
bounds, center, aux = ptb.init(intermediate_tensor, forward=False)
```

---

#### `intermediate_bound_module.py` (422 dòng)
**Mục đích**: **CORE INNOVATION** - Extend BoundedModule để hỗ trợ intermediate perturbations

**Class chính**: `IntermediateBoundedModule`

**Kế thừa từ**: `BoundedModule` (auto_LiRPA)

**Thuộc tính mới**:
- `intermediate_perturbations`: Dictionary lưu perturbations cho từng node
- `intermediate_outputs`: Dictionary lưu forward values

**Methods quan trọng**:
- `register_intermediate_perturbation(node_name, perturbation)`: Đăng ký perturbation
- `compute_bounds_with_intermediate_perturbation()`: **MAIN METHOD** - tính bounds
- `_compute_bounds_IBP_from_intermediate()`: IBP method
- `_compute_bounds_backward_from_intermediate()`: CROWN method
- `get_layer_names()`: Lấy danh sách Conv/Linear layers
- `print_model_structure()`: In cấu trúc graph

**Workflow**:
```python
# 1. Create bounded module
lirpa_model = IntermediateBoundedModule(model, dummy_input)

# 2. Forward pass để lấy intermediate outputs
_ = lirpa_model(input_image)

# 3. Đăng ký perturbation cho layer cụ thể
lirpa_model.register_intermediate_perturbation(node_name, perturbation)

# 4. Tính bounds
lb, ub = lirpa_model.compute_bounds_with_intermediate_perturbation(
    x=input_image,
    method='backward'
)
```

**Cơ chế hoạt động**:
1. Forward pass thu thập intermediate outputs
2. Áp dụng perturbation vào node được chọn
3. Mark node là "perturbed" và set bounds
4. Propagate bounds từ perturbed node đến output
5. Concretize để có lower/upper bounds

---

### 2. Training & Inference Scripts

#### `train_gtsrb.py` (235 dòng)
**Mục đích**: Training script với early stopping và checkpointing

**Chức năng**:
- Train model đến >90% accuracy
- Cosine annealing scheduler
- Save best checkpoint
- Early stopping khi đạt 90%

**Arguments**:
```bash
--data_dir       # Path to GTSRB
--model          # full or simple
--epochs         # Number of epochs (default: 50)
--batch_size     # Batch size (default: 128)
--lr             # Learning rate (default: 0.001)
--save_path      # Checkpoint path
```

**Output**: Checkpoint file `.pth` chứa:
- `model_state_dict`
- `optimizer_state_dict`
- `test_acc`, `test_loss`
- `epoch`

---

#### `collect_correct_samples.py` (205 dòng)
**Mục đích**: Inference và thu thập indices của correctly classified samples

**Chức năng**:
- Run inference trên test set
- Track correct predictions per class
- Save indices to CSV files

**Output**:
- `correct_samples/class_XX_correct_indices.csv` (43 files)
- `correct_samples/summary.csv`

**Tại sao cần file này?**
Để interactive testing có thể load samples đã biết chắc được phân loại đúng.

---

#### `main_interactive.py` (583 dòng)
**Mục đích**: **MAIN PROGRAM** - Interactive testing interface

**Class chính**: `InteractiveTester`

**Workflow tương tác**:

1. **Show layers**: Hiển thị tất cả Conv/FC layers
   ```python
   show_model_layers()
   ```

2. **Select layer**: User chọn layer index
   ```python
   node_name, layer_type, shape = get_layer_info(layer_idx)
   ```

3. **Select sample**: User chọn class và sample
   ```python
   image, label, idx = get_sample_from_class(class_id, sample_idx)
   ```

4. **Clean output**: Hiển thị prediction không có perturbation
   ```python
   output, pred_class = compute_clean_output(image)
   ```

5. **Configure perturbation**: User nhập:
   - Channel indices (Conv) hoặc feature indices (FC)
   - Height/width slices (Conv)
   - Epsilon value

6. **Compute bounds**: Tính và hiển thị bounds
   ```python
   lb, ub = compute_perturbed_bounds(...)
   ```

7. **Verification**: Kiểm tra robustness
   ```
   if pred_lb > max(other_ub):
       "VERIFIED ROBUST"
   else:
       "NOT verified robust"
   ```

**User Experience**:
- Prompts rõ ràng
- Hiển thị kết quả dễ đọc
- Top-5 predictions
- Robustness verification
- Loop để test nhiều configurations

---

### 3. Documentation & Setup Files

#### `SETUP_GUIDE.md` (615 dòng)
**Mục đích**: Hướng dẫn chi tiết bằng Tiếng Việt

**Nội dung**:
1. Yêu cầu hệ thống
2. Cài đặt môi trường ảo (venv/conda)
3. Cài đặt PyTorch (GPU/CPU)
4. Cài đặt auto_LiRPA
5. Tải GTSRB dataset
6. Chạy từng bước pipeline
7. Troubleshooting

**Đặc điểm**:
- Hướng dẫn từng bước rất chi tiết
- Ví dụ lệnh cụ thể
- Giải thích lỗi thường gặp
- Dễ hiểu cho người mới

---

#### `README.md` (527 dòng)
**Mục đích**: Overview và quick start guide

**Nội dung**:
1. Tính năng chính
2. Cấu trúc project
3. Quick start
4. Ví dụ sử dụng interactive mode
5. Kiến trúc model
6. Chi tiết kỹ thuật
7. Troubleshooting

**Đặc điểm**:
- Ngắn gọn, súc tích
- Code examples
- Workflow diagrams
- Technical details

---

#### `requirements.txt`
**Mục đích**: Python dependencies

**Nội dung**:
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scipy>=1.5.0
Pillow>=8.0.0
tqdm>=4.50.0
pandas>=1.1.0
```

---

#### `test_installation.py` (126 dòng)
**Mục đích**: Kiểm tra cài đặt

**Tests**:
1. ✓ Import all libraries
2. ✓ PyTorch version và CUDA
3. ✓ Model creation
4. ✓ Masked perturbation
5. ✓ Intermediate bounded module
6. ✓ Dataset loader (optional)

**Cách chạy**:
```bash
python test_installation.py [data_dir]
```

---

#### `run_all.sh` (260 dòng)
**Mục đích**: Automated pipeline runner

**Chức năng**:
- Check requirements
- Run training
- Run sample collection
- Offer to start interactive mode

**Usage**:
```bash
bash run_all.sh /path/to/GTSRB_data --model full --epochs 50
```

**Options**:
- `--model [full|simple]`
- `--epochs N`
- `--batch-size N`
- `--device [cuda|cpu]`
- `--skip-training`
- `--skip-collection`

---

#### `PROJECT_SUMMARY.md` (File này)
**Mục đích**: Tổng hợp và giải thích toàn bộ project

---

## Kiến Trúc Tổng Thể

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│                   (main_interactive.py)                      │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ├──> Select Layer
                  ├──> Select Sample (từ correct_samples/)
                  ├──> Configure Perturbation
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              IntermediateBoundedModule                       │
│          (intermediate_bound_module.py)                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Forward pass → collect intermediate outputs     │    │
│  │  2. Apply MaskedPerturbation to selected node       │    │
│  │  3. Mark node as perturbed, set bounds              │    │
│  │  4. Propagate bounds to output (IBP/CROWN)          │    │
│  │  5. Return lower_bound, upper_bound                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────┬───────────────────────────────────────────────┘
              │
              ├──> MaskedPerturbationLpNorm
              │    (masked_perturbation.py)
              │
              └──> BoundedModule (auto_LiRPA)
                   └──> Bound Computation
                        ├── IBP
                        ├── CROWN (Backward)
                        └── Forward LiRPA
```

---

## Workflow Hoàn Chỉnh

### Phase 1: Preparation

```bash
# 1. Tạo môi trường
python -m venv gtsrb_env
source gtsrb_env/bin/activate

# 2. Cài đặt
pip install torch torchvision
cd .. && pip install -e .
cd gtsrb_project && pip install -r requirements.txt

# 3. Kiểm tra
python test_installation.py

# 4. Tải GTSRB dataset
# (xem SETUP_GUIDE.md)
```

### Phase 2: Training

```bash
# Training model
python train_gtsrb.py \
    --data_dir ~/Documents/GTSRB_data \
    --model full \
    --epochs 50 \
    --batch_size 128 \
    --save_path checkpoints/traffic_sign_net.pth

# Kết quả: checkpoint với >90% accuracy
```

### Phase 3: Sample Collection

```bash
# Collect correctly classified samples
python collect_correct_samples.py \
    --data_dir ~/Documents/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full \
    --output_dir correct_samples

# Kết quả: 43 CSV files với indices
```

### Phase 4: Interactive Testing

```bash
# Start interactive mode
python main_interactive.py \
    --data_dir ~/Documents/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full \
    --device cuda
```

**Interactive Steps**:
1. Chọn layer (e.g., conv3)
2. Chọn class và sample (e.g., class 5, sample 0)
3. Xem clean output
4. Configure perturbation:
   - Channels: 0,1,2
   - Height: 5,10
   - Width: 5,10
   - Epsilon: 0.1
5. Xem bounds và verification result

---

## Điểm Đặc Biệt

### 1. Masked Perturbation
- **Vấn đề**: auto_LiRPA gốc chỉ perturb toàn bộ input tensor
- **Giải pháp**: MaskedPerturbationLpNorm cho phép chọn vùng cụ thể
- **Implementation**: Boolean mask → selective x_L/x_U

### 2. Intermediate Layer Support
- **Vấn đề**: auto_LiRPA gốc chỉ perturb input layer
- **Giải pháp**: IntermediateBoundedModule treat intermediate node như "perturbed input"
- **Implementation**:
  - Forward pass collect outputs
  - Inject perturbation at node
  - Propagate bounds từ node đó

### 3. User-Friendly Interface
- **Vấn đề**: auto_LiRPA cần nhiều code để setup
- **Giải pháp**: Interactive CLI với prompts
- **Features**:
  - Auto-detect layers
  - Load correct samples
  - Display results clearly
  - Verification checking

---

## Code Quality & Best Practices

### 1. Modularity
- Mỗi file có chức năng rõ ràng
- Classes well-organized
- Easy to extend

### 2. Documentation
- Docstrings cho mọi class/function
- Comments giải thích logic
- README và guides chi tiết

### 3. Error Handling
- Try-catch blocks
- Informative error messages
- Fallback methods (e.g., IBP nếu CROWN fails)

### 4. User Experience
- Clear prompts
- Progress indicators (tqdm)
- Colored output (for bash script)
- Helpful error messages

---

## Ví Dụ Sử Dụng Cụ Thể

### Example 1: Perturb toàn bộ một channel

```python
# Load model
lirpa_model = IntermediateBoundedModule(model, dummy_input)
_ = lirpa_model(test_image)

# Perturb entire channel 5 of conv3 output
perturbation = MaskedPerturbationLpNorm(
    eps=0.1,
    norm=np.inf,
    batch_idx=0,
    channel_idx=5,
    height_slice=None,  # All heights
    width_slice=None    # All widths
)

lirpa_model.register_intermediate_perturbation('/conv3', perturbation)
lb, ub = lirpa_model.compute_bounds_with_intermediate_perturbation(
    x=test_image,
    method='backward'
)
```

### Example 2: Perturb một vùng nhỏ của nhiều channels

```python
# Perturb center 5x5 region of channels 0-9
perturbation = MaskedPerturbationLpNorm(
    eps=0.05,
    norm=np.inf,
    batch_idx=0,
    channel_idx=list(range(10)),  # Channels 0-9
    height_slice=(6, 11),         # Center 5 pixels
    width_slice=(6, 11)           # Center 5 pixels
)

lirpa_model.register_intermediate_perturbation('/conv4', perturbation)
lb, ub = lirpa_model.compute_bounds_with_intermediate_perturbation(
    x=test_image,
    method='backward'
)
```

### Example 3: Perturb FC layer features

```python
# Perturb features 100-199 of fc1 output
perturbation = MaskedPerturbationLpNorm(
    eps=0.2,
    norm=np.inf,
    batch_idx=0,
    channel_idx=list(range(100, 200))  # Features 100-199
)

lirpa_model.register_intermediate_perturbation('/fc1', perturbation)
lb, ub = lirpa_model.compute_bounds_with_intermediate_perturbation(
    x=test_image,
    method='backward'
)
```

---

## Technical Insights

### 1. Bound Propagation với Intermediate Perturbations

**Challenge**: Auto_LiRPA assumes perturbation ở input layer

**Solution**:
1. Treat intermediate perturbed node như "new input"
2. Set `node.perturbed = True`
3. Set `node.interval = (lower, upper)` với masked bounds
4. Propagate chỉ từ node này về sau

**Tradeoff**:
- ✓ Flexible: có thể perturb bất kỳ layer nào
- ✗ Complexity: phải track multiple perturbed nodes
- ✗ Bounds có thể loose hơn input perturbation

### 2. Masked Perturbation Implementation

**Challenge**: auto_LiRPA concretize giả định uniform perturbation

**Solution**:
1. Tạo x_L, x_U với mask applied
2. Unperturbed elements: x_L = x_U = forward_value
3. Perturbed elements: x_L = x - eps, x_U = x + eps

**Benefits**:
- ✓ Tighter bounds (chỉ perturb subset)
- ✓ More realistic (targeted attacks)
- ✓ Flexible spatial/channel selection

### 3. Interactive Testing Design

**Design Principles**:
1. **Discoverability**: Show available options
2. **Guidance**: Clear prompts with examples
3. **Feedback**: Show intermediate results
4. **Verification**: Automatically check robustness

**User Flow**:
```
Show layers → Select → Show sample → Clean output →
Configure region → Compute bounds → Verify → Repeat
```

---

## Performance Considerations

### Memory Usage
- **Forward pass**: O(model_size)
- **Bound computation**: O(model_size × perturbation_size)
- **Optimization**:
  - Use sparse perturbations
  - Reduce batch size
  - Use IBP for faster (but looser) bounds

### Computation Time
- **IBP**: Fast (~seconds)
- **CROWN**: Medium (~10s-1min)
- **Optimization**:
  - Cache forward values
  - Reuse bounds for same perturbation
  - Use GPU acceleration

### Scalability
- **Small perturbations**: Very efficient
- **Large perturbations**: May need optimization
- **Multiple layers**: Can compound complexity

---

## Future Enhancements

### Possible Extensions

1. **Multiple simultaneous perturbations**
   - Perturb nhiều layers cùng lúc
   - Tính combined bounds

2. **Adaptive epsilon**
   - Tự động tìm epsilon max để robust
   - Binary search over epsilon

3. **Visualization**
   - Visualize perturbed regions
   - Heatmap of sensitivity
   - Bound evolution across layers

4. **Optimization**
   - Sparse matrix operations
   - GPU-optimized kernels
   - Parallel bound computation

5. **Analysis tools**
   - Sensitivity analysis per region
   - Layer importance ranking
   - Robustness certification

---

## Troubleshooting Guide

### Common Issues

#### 1. "CUDA out of memory"
**Solution**: Giảm batch_size hoặc dùng CPU

#### 2. "Node not found"
**Cause**: Node name không đúng
**Solution**: Dùng `print_model_structure()` để xem tên chính xác

#### 3. "Bounds too loose"
**Cause**: IBP method hoặc perturbation quá lớn
**Solution**: Dùng CROWN method hoặc giảm epsilon

#### 4. "Not verified robust"
**Explanation**: Bounds overlap, không thể chứng minh
**Action**: Giảm epsilon hoặc perturb ít elements hơn

---

## Kết Luận

Dự án này thành công implement:

✅ **Masked perturbations** cho intermediate layers
✅ **Interactive testing** interface
✅ **Complete pipeline** từ training đến verification
✅ **Comprehensive documentation**

**Key Innovations**:
1. MaskedPerturbationLpNorm - flexible region selection
2. IntermediateBoundedModule - intermediate layer support
3. Interactive CLI - user-friendly testing

**Applications**:
- Robustness analysis
- Layer sensitivity study
- Targeted adversarial analysis
- Neural network understanding

---

**Tác giả**: Claude Code (Anthropic)
**Ngày tạo**: 2025
**License**: BSD 3-Clause (following auto_LiRPA)
