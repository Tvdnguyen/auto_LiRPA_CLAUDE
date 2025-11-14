# GTSRB Traffic Sign Recognition với Intermediate Layer Perturbations

Dự án này thực hiện phân tích nhiễu loạn trên các layer trung gian (intermediate layers) cho bài toán nhận dạng biển báo giao thông sử dụng dataset GTSRB và thư viện auto_LiRPA.

## Tính Năng Chính

✅ **Nhận dạng biển báo**: Mô hình CNN đạt >90% độ chính xác trên GTSRB
✅ **Nhiễu loạn layer trung gian**: Thêm perturbation vào output của Conv/FC layers
✅ **Masked perturbation**: Chỉ nhiễu loạn vùng cụ thể (batch, channel, spatial region)
✅ **Tính toán bounds**: Sử dụng LiRPA/CROWN để tính bounds có thể chứng minh được
✅ **Giao diện tương tác**: CLI thân thiện để test và thử nghiệm

## Cấu Trúc Dự Án

```
gtsrb_project/
├── gtsrb_dataset.py              # Loader cho GTSRB dataset
├── traffic_sign_net.py           # Kiến trúc CNN model
├── train_gtsrb.py                # Script training model
├── collect_correct_samples.py    # Thu thập samples phân loại đúng
├── masked_perturbation.py        # Implementation masked perturbation
├── intermediate_bound_module.py  # Extended BoundedModule
├── main_interactive.py           # Giao diện test tương tác
├── test_installation.py          # Kiểm tra cài đặt
├── requirements.txt              # Python dependencies
├── SETUP_GUIDE.md               # Hướng dẫn chi tiết (Tiếng Việt)
└── README.md                    # File này
```

## Bắt Đầu Nhanh (Quick Start)

### 1. Cài Đặt Môi Trường

```bash
# Tạo môi trường ảo
python -m venv gtsrb_env
source gtsrb_env/bin/activate  # Linux/Mac
# hoặc: gtsrb_env\Scripts\activate  # Windows

# Cài đặt PyTorch (ví dụ cho CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài đặt auto_LiRPA
cd ..
pip install -e .

# Cài đặt dependencies
cd gtsrb_project
pip install -r requirements.txt
```

**Hướng dẫn chi tiết**: Xem [SETUP_GUIDE.md](SETUP_GUIDE.md)

### 2. Kiểm Tra Cài Đặt

```bash
python test_installation.py
```

### 3. Tải GTSRB Dataset

Tải và giải nén:
- Training: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
- Test images: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
- Test GT: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip

Cấu trúc thư mục:
```
GTSRB_data/
├── Train/
│   ├── 00000/ ... 00042/
└── Test/
    ├── GT-final_test.csv
    └── *.ppm files
```

### 4. Training Model

```bash
python train_gtsrb.py \
    --data_dir ~/Documents/GTSRB_data \
    --model full \
    --epochs 50 \
    --batch_size 128 \
    --save_path checkpoints/traffic_sign_net.pth
```

**Kết quả mong đợi**: Test accuracy > 90%

### 5. Thu Thập Correct Samples

```bash
python collect_correct_samples.py \
    --data_dir ~/Documents/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full \
    --output_dir correct_samples
```

Tạo 43 file CSV chứa indices của samples được phân loại đúng theo từng class.

### 6. Interactive Testing

```bash
python main_interactive.py \
    --data_dir ~/Documents/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full \
    --device cuda
```

## Sử Dụng Interactive Mode

Khi chạy `main_interactive.py`, chương trình sẽ hỏi từng bước:

### a. Chọn Layer để nhiễu loạn

```
Available Layers for Perturbation (Conv and FC only)
================================================================================
Index | Layer Name      | Type       | Output Shape
--------------------------------------------------------------------------------
    0 | conv1           | Conv2d     | 32x32x32
    1 | conv2           | Conv2d     | 32x32x32
    2 | conv3           | Conv2d     | 16x16x64
    3 | conv4           | Conv2d     | 16x16x64
    4 | conv5           | Conv2d     | 8x8x128
    5 | conv6           | Conv2d     | 8x8x128
    6 | fc1             | Linear     | 512
    7 | fc2             | Linear     | 256
    8 | fc3             | Linear     | 43
================================================================================

Select layer index (or -1 to quit): 2
```

### b. Chọn Class và Sample

```
Select class ID (0-42): 5
Select sample index within class (default 0): 0
```

### c. Xem Output Clean

```
Clean Output (Logits):
  Predicted class: 5
  Top-5 classes:
    1. Class  5: 12.3456 ←
    2. Class  3:  8.1234
    3. Class  2:  7.9876
    ...
```

### d. Cấu hình vùng nhiễu loạn

**Cho Conv layer:**
```
Configure Perturbation Region:
  Channel index (or 'all', or comma-separated list): 0,1,2
  Height slice (start,end) or 'all': 5,10
  Width slice (start,end) or 'all': 5,10
  Epsilon value: 0.1
```

**Cho FC layer:**
```
Configure Perturbation Region:
  Feature indices (comma-separated or 'all'): 10,20,30,40,50
  Epsilon value: 0.1
```

### e. Kết quả

```
RESULTS
================================================================================

Clean Output (no perturbation):
  Predicted class: 5
  Logit for true class 5: 12.3456

Bounds with Perturbation:
  Lower bound shape: torch.Size([1, 43])
  Upper bound shape: torch.Size([1, 43])

  Top-5 Lower Bounds:
    1. Class  5: 11.2345 ←
    2. Class  3:  7.8901
    ...

  Top-5 Upper Bounds:
    1. Class  5: 13.4567 ←
    2. Class  3:  8.5432
    ...

  Bounds for true class 5:
    Lower: 11.2345
    Upper: 13.4567
    Width:  2.2222

  ✓ Prediction is VERIFIED ROBUST
    Predicted class 5 lower bound (11.2345) >
    All other classes upper bounds (max: 8.5432)
================================================================================
```

## Kiến Trúc Model

### TrafficSignNet (Full)

```
Input: 3×32×32
│
├─ Conv1: 3→32, ReLU
├─ Conv2: 32→32, ReLU
├─ MaxPool 2×2, Dropout(0.2)
│
├─ Conv3: 32→64, ReLU
├─ Conv4: 64→64, ReLU
├─ MaxPool 2×2, Dropout(0.2)
│
├─ Conv5: 64→128, ReLU
├─ Conv6: 128→128, ReLU
├─ MaxPool 2×2, Dropout(0.3)
│
├─ Flatten
├─ FC1: 2048→512, ReLU, Dropout(0.5)
├─ FC2: 512→256, ReLU, Dropout(0.5)
└─ FC3: 256→43 (output)
```

**Tham số**: ~1.4M parameters
**Độ chính xác**: >90% trên GTSRB test set

## Chi Tiết Kỹ Thuật

### 1. Masked Perturbation

Class `MaskedPerturbationLpNorm` cho phép nhiễu loạn chỉ một vùng cụ thể:

```python
perturbation = MaskedPerturbationLpNorm(
    eps=0.1,                    # Độ lớn nhiễu loạn
    norm=np.inf,                # L-infinity norm
    batch_idx=0,                # Batch nào
    channel_idx=[0, 1, 2],      # Channels nào
    height_slice=(5, 10),       # Vùng height (5-9)
    width_slice=(5, 10)         # Vùng width (5-9)
)
```

**Đối với Conv layers** (B, C, H, W):
- `batch_idx`: Chỉ số batch
- `channel_idx`: Chỉ số channel
- `height_slice`: Vùng chiều cao
- `width_slice`: Vùng chiều rộng

**Đối với FC layers** (B, D):
- `batch_idx`: Chỉ số batch
- `channel_idx`: Chỉ số feature dimension

### 2. Intermediate Layer Perturbation

Workflow:

1. **Forward Pass**: Tính toán outputs của tất cả các layers
2. **Inject Perturbation**: Áp dụng masked perturbation vào layer được chọn
3. **Bound Propagation**: Tính bounds từ layer đó đến output
4. **Verification**: Kiểm tra xem prediction có robust không

### 3. Phương pháp tính Bounds

- **IBP** (Interval Bound Propagation): Nhanh nhưng bounds rộng
- **CROWN** (Backward LiRPA): Bounds chặt hơn, chậm hơn
- **Forward LiRPA**: Propagate bounds từng layer

### 4. Implementation

**IntermediateBoundedModule** extends `BoundedModule`:
```python
# Đăng ký perturbation cho layer
lirpa_model.register_intermediate_perturbation(node_name, perturbation)

# Tính bounds
lb, ub = lirpa_model.compute_bounds_with_intermediate_perturbation(
    x=input_image,
    method='backward'
)
```

## Các File Scripts

### 1. `gtsrb_dataset.py`
- Load GTSRB dataset
- Preprocessing và augmentation
- Dataloader creation

### 2. `traffic_sign_net.py`
- Định nghĩa model architecture
- TrafficSignNet (full) và TrafficSignNetSimple
- Method `get_layer_info()` để list các layers

### 3. `train_gtsrb.py`
- Training script với early stopping
- Cosine annealing scheduler
- Save best checkpoint

### 4. `collect_correct_samples.py`
- Inference trên test set
- Thu thập indices của correctly classified samples
- Lưu vào CSV files per class

### 5. `masked_perturbation.py`
- Class `MaskedPerturbationLpNorm`
- Hỗ trợ masked/partial perturbations
- Helper function `create_region_mask()`

### 6. `intermediate_bound_module.py`
- Class `IntermediateBoundedModule`
- Extend `BoundedModule` cho intermediate perturbations
- Methods cho bound computation

### 7. `main_interactive.py`
- Interactive CLI interface
- User-friendly prompts
- Display results và verification

### 8. `test_installation.py`
- Kiểm tra cài đặt
- Test các components chính
- Quick sanity checks

## Troubleshooting

### GPU Out of Memory
```bash
# Giảm batch size
python train_gtsrb.py --batch_size 64
```

### Slow Training
```bash
# Dùng simple model
python train_gtsrb.py --model simple

# Hoặc giảm epochs
python train_gtsrb.py --epochs 30
```

### Dataset Not Found
```bash
# Kiểm tra đường dẫn
ls ~/Documents/GTSRB_data/Train
ls ~/Documents/GTSRB_data/Test

# Cung cấp đường dẫn đúng
python train_gtsrb.py --data_dir /correct/path/to/GTSRB_data
```

### Import Error
```bash
# Đảm bảo môi trường ảo được kích hoạt
source gtsrb_env/bin/activate

# Cài lại auto_LiRPA
cd .. && pip install -e .

# Cài lại dependencies
pip install -r requirements.txt
```

## Yêu Cầu Hệ Thống

- **Python**: 3.7+
- **PyTorch**: 1.9.0+
- **CUDA**: 10.2+ (optional, cho GPU)
- **RAM**: 8GB+ (khuyến nghị 16GB)
- **Storage**: 5GB+ (cho dataset và checkpoints)

## Tài Liệu Tham Khảo

- **GTSRB Dataset**: http://benchmark.ini.rub.de/
- **auto_LiRPA**: https://github.com/Verified-Intelligence/auto_LiRPA
- **CROWN Paper**: https://arxiv.org/abs/1811.00866
- **LiRPA**: https://arxiv.org/abs/2002.12920

## License

Dự án này mở rộng auto_LiRPA được phát hành theo BSD 3-Clause License.

## Liên Hệ & Đóng Góp

Nếu có câu hỏi hoặc phát hiện lỗi, vui lòng tạo issue hoặc liên hệ trực tiếp.

---

**Chúc bạn thành công với dự án!** 🚀
