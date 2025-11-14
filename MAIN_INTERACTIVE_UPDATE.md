# Cập nhật main_interactive.py - Random Sample Selection

## Thay đổi

Đã sửa `gtsrb_project/main_interactive.py` để **tự động chọn ngẫu nhiên** một sample từ `correct_samples` thay vì yêu cầu user nhập index thủ công.

## Chi tiết thay đổi

### 1. Import thêm module `random`
```python
import random
```

### 2. Sửa function `get_sample_from_class()`

**Trước:**
- Nhận `sample_idx` với giá trị mặc định = 0
- Luôn sử dụng index được chỉ định
- Return: `(image, label, global_idx)`

**Sau:**
- Nhận `sample_idx` với giá trị mặc định = `None`
- Nếu `sample_idx = None`: **tự động chọn ngẫu nhiên** từ correct_samples
- Nếu `sample_idx` được chỉ định: sử dụng index đó (cho backward compatibility)
- Return: `(image, label, global_idx, sample_idx)`

```python
def get_sample_from_class(self, class_id, sample_idx=None, correct_samples_dir='correct_samples'):
    """
    Get a random sample from a class (from correct_samples)

    Args:
        class_id: Class ID (0-42)
        sample_idx: If None, randomly select. If provided, use specific index.
        correct_samples_dir: Directory with correct sample indices

    Returns:
        image tensor, label, global_idx, sample_idx
    """
    # Load correct indices
    correct_indices = load_correct_indices(correct_samples_dir, class_id)

    # Random selection if sample_idx not provided
    if sample_idx is None:
        sample_idx = random.randint(0, len(correct_indices) - 1)
        print(f"  Randomly selected index {sample_idx} from {len(correct_indices)} correct samples")

    global_idx = correct_indices[sample_idx]
    image, label = self.test_dataset[global_idx]

    return image, label, global_idx, sample_idx
```

### 3. Sửa interactive flow

**Trước:**
```python
class_id = int(input(f"\nSelect class ID (0-42): "))
sample_idx = int(input(f"Select sample index within class (default 0): ") or "0")

image, label, global_idx = self.get_sample_from_class(class_id, sample_idx)
```

**Sau:**
```python
class_id = int(input(f"\nSelect class ID (0-42): "))

# Automatically select random sample from correct_samples
print(f"\nLoading random correct sample from class {class_id}...")
image, label, global_idx, sample_idx = self.get_sample_from_class(class_id, sample_idx=None)

print(f"\nLoaded sample:")
print(f"  Class: {class_id}")
print(f"  Sample index within class: {sample_idx}")
print(f"  Global test set index: {global_idx}")
print(f"  True label: {label}")
```

## Cách sử dụng

### Interactive mode (chỉ hỏi class ID)

```bash
cd gtsrb_project
python3 main_interactive.py \
    --data_dir data/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net_full.pth
```

**Flow:**
1. User chọn layer
2. **User chỉ nhập class ID (0-42)**
3. **Program tự động random chọn 1 sample từ correct_samples của class đó**
4. User cấu hình perturbation
5. Xem kết quả

### Ví dụ output

```
Select layer index (or -1 to quit): 0

Selected layer: /19
  Type: Conv2d
  Output shape: [32, 30, 30]

Select class ID (0-42): 15

Loading random correct sample from class 15...
  Randomly selected index 23 from 60 correct samples

Loaded sample:
  Class: 15
  Sample index within class: 23
  Global test set index: 5472
  True label: 15
```

## Lợi ích

✅ **Đơn giản hơn**: User chỉ cần nhập class ID, không cần biết có bao nhiêu samples

✅ **Ngẫu nhiên hóa**: Mỗi lần chạy test sample khác nhau → coverage tốt hơn

✅ **An toàn**: Vẫn chỉ test trên **correct_samples** (samples được classify đúng)

✅ **Backward compatible**: Vẫn có thể chỉ định index cụ thể nếu cần (cho automated scripts)

## Notes

- Function `get_sample_from_class()` vẫn hỗ trợ truyền `sample_idx` cụ thể cho các scripts khác
- Random seed không được set → mỗi lần chạy sẽ random khác nhau
- Nếu cần reproducibility, có thể thêm `random.seed()` trong code
