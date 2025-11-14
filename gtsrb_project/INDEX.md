# Danh Mục Files - GTSRB Intermediate Perturbation Project

## Tổng Quan
Dự án gồm **13 files** được tổ chức theo chức năng

---

## 📋 Danh Sách Files

### 🔧 Core Implementation (3 files)

| File | Lines | Mục đích | Độ ưu tiên |
|------|-------|----------|------------|
| `masked_perturbation.py` | 354 | **CORE**: Masked perturbation cho intermediate layers | ⭐⭐⭐ |
| `intermediate_bound_module.py` | 422 | **CORE**: Extended BoundedModule | ⭐⭐⭐ |
| `traffic_sign_net.py` | 189 | Model architecture | ⭐⭐ |

### 📊 Data & Training (2 files)

| File | Lines | Mục đích | Độ ưu tiên |
|------|-------|----------|------------|
| `gtsrb_dataset.py` | 190 | GTSRB dataset loader | ⭐⭐ |
| `train_gtsrb.py` | 235 | Training script | ⭐⭐ |

### 🎯 Main Programs (3 files)

| File | Lines | Mục đích | Độ ưu tiên |
|------|-------|----------|------------|
| `main_interactive.py` | 583 | **MAIN PROGRAM**: Interactive testing | ⭐⭐⭐ |
| `collect_correct_samples.py` | 205 | Collect correctly classified samples | ⭐⭐ |
| `test_installation.py` | 126 | Installation verification | ⭐ |

### 📚 Documentation (4 files)

| File | Lines | Mục đích | Đọc đầu tiên |
|------|-------|----------|--------------|
| `README.md` | 527 | Quick start guide | 1️⃣ |
| `SETUP_GUIDE.md` | 615 | Chi tiết cài đặt (Tiếng Việt) | 2️⃣ |
| `PROJECT_SUMMARY.md` | 750 | Tổng hợp và giải thích chi tiết | 3️⃣ |
| `INDEX.md` | - | File này | - |

### ⚙️ Config & Automation (2 files)

| File | Lines | Mục đích |
|------|-------|----------|
| `requirements.txt` | 10 | Python dependencies |
| `run_all.sh` | 260 | Automated pipeline script |

---

## 🚀 Hướng Dẫn Bắt Đầu

### 1️⃣ Đọc Tài Liệu (10 phút)
```
README.md          → Hiểu tổng quan
SETUP_GUIDE.md     → Cách cài đặt
PROJECT_SUMMARY.md → Chi tiết kỹ thuật
```

### 2️⃣ Cài Đặt Môi Trường (30 phút)
```bash
# Tạo môi trường ảo
python -m venv gtsrb_env
source gtsrb_env/bin/activate

# Cài đặt dependencies
pip install torch torchvision
cd .. && pip install -e .
cd gtsrb_project && pip install -r requirements.txt

# Kiểm tra
python test_installation.py
```

### 3️⃣ Tải Dataset (10 phút)
Xem hướng dẫn trong `SETUP_GUIDE.md` section "Tải Dataset GTSRB"

### 4️⃣ Training (30-60 phút)
```bash
python train_gtsrb.py \
    --data_dir /path/to/GTSRB \
    --model full \
    --epochs 50
```

### 5️⃣ Collect Samples (5 phút)
```bash
python collect_correct_samples.py \
    --data_dir /path/to/GTSRB \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full
```

### 6️⃣ Interactive Testing (∞)
```bash
python main_interactive.py \
    --data_dir /path/to/GTSRB \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full
```

---

## 📖 Hướng Dẫn Đọc Code

### Nếu bạn muốn hiểu cách hoạt động:

**Bước 1**: Đọc class đơn giản trước
```
traffic_sign_net.py         → Model definition
gtsrb_dataset.py           → Data loading
```

**Bước 2**: Đọc core innovations
```
masked_perturbation.py     → Masked perturbation mechanism
intermediate_bound_module.py → Intermediate layer support
```

**Bước 3**: Đọc main program
```
main_interactive.py        → How everything works together
```

### Nếu bạn muốn sửa/mở rộng:

**Thêm model mới**:
- Edit: `traffic_sign_net.py`
- Add new class, implement `get_layer_info()`

**Thêm perturbation type**:
- Edit: `masked_perturbation.py`
- Extend `MaskedPerturbationLpNorm`

**Thêm bound computation method**:
- Edit: `intermediate_bound_module.py`
- Add method `_compute_bounds_XXX_from_intermediate()`

**Thay đổi UI**:
- Edit: `main_interactive.py`
- Modify `InteractiveTester` class

---

## 🎯 Use Cases

### Use Case 1: Test một layer cụ thể
```
Files cần: main_interactive.py + model + dataset
Workflow: Run interactive → Select layer → Test
```

### Use Case 2: Training model mới
```
Files cần: train_gtsrb.py + gtsrb_dataset.py + traffic_sign_net.py
Workflow: Modify model → Run training → Save checkpoint
```

### Use Case 3: Batch testing nhiều configs
```
Files cần: intermediate_bound_module.py + masked_perturbation.py
Workflow: Write script → Loop over configs → Collect results
```

### Use Case 4: Research experiment
```
Files cần: All implementation files
Workflow: Extend classes → Run experiments → Analyze
```

---

## 🔍 File Dependencies

```
main_interactive.py
├── intermediate_bound_module.py
│   ├── masked_perturbation.py
│   │   └── auto_LiRPA.perturbations
│   └── auto_LiRPA.BoundedModule
├── traffic_sign_net.py
├── gtsrb_dataset.py
└── collect_correct_samples.py

train_gtsrb.py
├── traffic_sign_net.py
└── gtsrb_dataset.py

collect_correct_samples.py
├── traffic_sign_net.py
└── gtsrb_dataset.py
```

---

## 📊 Thống Kê

### Lines of Code
- **Implementation**: ~2,400 lines
- **Documentation**: ~1,900 lines
- **Total**: ~4,300 lines

### File Types
- Python scripts: 8 files
- Markdown docs: 4 files
- Config: 2 files

### Complexity
- **Core Innovation**: Medium-High (masked_perturbation, intermediate_bound_module)
- **Main Program**: Medium (main_interactive)
- **Utilities**: Low (dataset, training, collection)

---

## ⚠️ Important Notes

### Files bạn NÊN đọc kỹ:
1. ⭐⭐⭐ `masked_perturbation.py` - Core innovation
2. ⭐⭐⭐ `intermediate_bound_module.py` - Core innovation
3. ⭐⭐⭐ `main_interactive.py` - Main program
4. ⭐⭐ `SETUP_GUIDE.md` - Để cài đặt đúng

### Files bạn CÓ THỂ bỏ qua nếu chỉ dùng:
- `train_gtsrb.py` - Nếu đã có checkpoint
- `collect_correct_samples.py` - Nếu đã có CSV files
- `test_installation.py` - Sau khi đã test xong
- `run_all.sh` - Nếu prefer manual commands

### Files bạn PHẢI đọc để extend:
- `masked_perturbation.py` - Để add perturbation types mới
- `intermediate_bound_module.py` - Để add bound methods mới
- `PROJECT_SUMMARY.md` - Để hiểu architecture

---

## 🛠️ Modification Guide

### Thêm Model Mới

**Edit**: `traffic_sign_net.py`

```python
class MyNewNet(nn.Module):
    def __init__(self, num_classes=43):
        super().__init__()
        # Define layers...

    def forward(self, x):
        # Define forward pass...

    def get_layer_info(self):
        # Return layer info for UI
        return [
            ('layer_name', 'layer_type', layer_obj, 'shape_info'),
            ...
        ]
```

### Thêm Perturbation Type Mới

**Edit**: `masked_perturbation.py`

```python
class MyCustomPerturbation(MaskedPerturbationLpNorm):
    def __init__(self, eps, **kwargs):
        super().__init__(eps, **kwargs)
        # Add custom parameters

    def get_input_bounds(self, x, A):
        # Custom bound computation
        pass
```

### Thêm Bound Method Mới

**Edit**: `intermediate_bound_module.py`

```python
def _compute_bounds_my_method_from_intermediate(self, **kwargs):
    """
    New bound computation method
    """
    # Implement your method
    return lower_bound, upper_bound
```

Sau đó update `compute_bounds_with_intermediate_perturbation()`:
```python
elif method == 'my_method':
    return self._compute_bounds_my_method_from_intermediate(**kwargs)
```

---

## 🐛 Debugging Tips

### Issue: Import errors
**Check**:
- Môi trường ảo có được activate không?
- auto_LiRPA có được install không? (`pip install -e ..`)

### Issue: Model structure không khớp
**Check**:
- `lirpa_model.print_model_structure()` để xem node names
- Node name có đúng không?

### Issue: Bounds quá rộng
**Try**:
- Giảm epsilon
- Giảm vùng perturb (ít elements hơn)
- Dùng CROWN thay vì IBP

### Issue: Out of memory
**Try**:
- Giảm batch_size
- Dùng CPU thay vì GPU
- Perturb ít elements hơn

---

## 📞 Quick Reference

### Chạy Training
```bash
python train_gtsrb.py --data_dir /path/to/GTSRB --model full
```

### Chạy Interactive
```bash
python main_interactive.py --data_dir /path/to/GTSRB --checkpoint checkpoints/traffic_sign_net.pth --model full
```

### Chạy Full Pipeline
```bash
bash run_all.sh /path/to/GTSRB --model full --epochs 50
```

### Test Installation
```bash
python test_installation.py /path/to/GTSRB
```

---

## 🎓 Learning Path

### Beginner (Chỉ dùng)
1. Đọc `README.md`
2. Follow `SETUP_GUIDE.md`
3. Run `run_all.sh`
4. Use `main_interactive.py`

### Intermediate (Hiểu cách hoạt động)
1. Đọc `README.md` + `PROJECT_SUMMARY.md`
2. Đọc `traffic_sign_net.py`
3. Đọc `gtsrb_dataset.py`
4. Đọc `main_interactive.py`
5. Experiment với different configs

### Advanced (Modify & Extend)
1. Đọc tất cả docs
2. Đọc `masked_perturbation.py` kỹ
3. Đọc `intermediate_bound_module.py` kỹ
4. Study auto_LiRPA source code
5. Implement extensions

---

## 📁 Cấu Trúc Thư Mục Sau Khi Chạy

```
gtsrb_project/
├── [CODE FILES]
│   ├── gtsrb_dataset.py
│   ├── traffic_sign_net.py
│   ├── train_gtsrb.py
│   ├── collect_correct_samples.py
│   ├── masked_perturbation.py
│   ├── intermediate_bound_module.py
│   ├── main_interactive.py
│   └── test_installation.py
│
├── [DOCS]
│   ├── README.md
│   ├── SETUP_GUIDE.md
│   ├── PROJECT_SUMMARY.md
│   └── INDEX.md (file này)
│
├── [CONFIG]
│   ├── requirements.txt
│   └── run_all.sh
│
├── [GENERATED - after training]
│   ├── checkpoints/
│   │   └── traffic_sign_net.pth
│   ├── correct_samples/
│   │   ├── class_00_correct_indices.csv
│   │   ├── class_01_correct_indices.csv
│   │   ├── ... (43 files)
│   │   └── summary.csv
│   └── logs/
│       └── run_YYYYMMDD_HHMMSS.log
```

---

## ✅ Checklist

### Trước khi chạy:
- [ ] Python 3.7+ installed
- [ ] Môi trường ảo created & activated
- [ ] PyTorch installed (with CUDA if available)
- [ ] auto_LiRPA installed (`pip install -e ..`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] GTSRB dataset downloaded and extracted
- [ ] Directory structure verified

### Sau khi training:
- [ ] Checkpoint file exists
- [ ] Test accuracy > 90%
- [ ] 43 CSV files in correct_samples/
- [ ] Summary.csv shows reasonable accuracy per class

### Trước khi test:
- [ ] Checkpoint loaded successfully
- [ ] Dataset accessible
- [ ] Correct samples loaded
- [ ] GPU available (optional)

---

## 🎉 Hoàn Thành!

Bạn đã có đầy đủ thông tin về project. Giờ có thể:
1. ✅ Cài đặt môi trường
2. ✅ Training model
3. ✅ Chạy interactive testing
4. ✅ Mở rộng và customize

**Good luck với research! 🚀**

---

**Version**: 1.0
**Last Updated**: 2025
**Total Files**: 13
**Total Lines**: ~4,300
