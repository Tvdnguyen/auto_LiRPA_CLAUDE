# GTSRB Dataset Structure Guide

## Tổng Quan

GTSRB dataset sau khi giải nén có thể có nhiều cấu trúc khác nhau. Code đã được cập nhật để **tự động detect và hỗ trợ tất cả các cấu trúc phổ biến**.

---

## ✅ Các Cấu Trúc Được Hỗ Trợ

### Option A: Test images trong subfolder `Images/` (Phổ biến nhất)

```
data/GTSRB_data/
├── Train/
│   ├── 00000/
│   │   ├── GT-00000.csv
│   │   └── *.ppm files
│   ├── 00001/
│   └── ... (43 folders: 00000 to 00042)
└── Test/
    ├── GT-final_test.csv
    └── Images/              ← Subfolder
        ├── 00000.ppm
        ├── 00001.ppm
        └── ... (12,630 files)
```

**Đây là cấu trúc bạn đang có:**
`/Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE/gtsrb_project/data/GTSRB_data/Test/Images/`

### Option B: Test images trực tiếp trong `Test/`

```
data/GTSRB_data/
├── Train/
│   └── ... (như trên)
└── Test/
    ├── GT-final_test.csv
    ├── 00000.ppm
    ├── 00001.ppm
    └── ... (12,630 files)
```

---

## 🔧 Code Updates

### 1. `gtsrb_dataset.py` - Dataset Loader

**Đã cập nhật** `_load_test_data()` method để tự động tìm test images ở nhiều vị trí:

```python
def _load_test_data(self):
    # Try multiple locations:
    # 1. Test/Filename (as in CSV)
    # 2. Test/Images/Filename
    # 3. Test/basename(Filename)
    # 4. Test/Images/basename(Filename)
```

**Kết quả**: Dataset loader hoạt động với cả Option A và Option B mà không cần sửa gì!

### 2. `setup_dataset.sh` - Automated Setup Script

**Đã cập nhật** để giữ nguyên cấu trúc `Test/Images/`:

```bash
# Keeps Images subfolder structure
mv GTSRB/Final_Test/Images ../Test/
```

**Verify** cả 2 locations:

```bash
TEST_COUNT=$(ls ../Test/Images/*.ppm 2>/dev/null | wc -l)
```

### 3. `run_all.sh` - Pipeline Runner

**Đã cập nhật** validation để check cả 2 locations:

```bash
# Check both locations
TEST_IMG_COUNT=$(ls "$DATA_DIR/Test"/*.ppm 2>/dev/null | wc -l)
TEST_IMG_SUBDIR_COUNT=$(ls "$DATA_DIR/Test/Images"/*.ppm 2>/dev/null | wc -l)

# Accept if either has images
if [ "$TEST_IMG_COUNT" -eq 0 ] && [ "$TEST_IMG_SUBDIR_COUNT" -eq 0 ]; then
    print_error "No test images found"
fi
```

**Kết quả**: Script chạy thành công với cả 2 cấu trúc!

---

## 🎯 Sử Dụng

### Với Cấu Trúc Hiện Tại (Test/Images/)

```bash
# Không cần làm gì! Chỉ cần chạy:
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE/gtsrb_project

# Test dataset loader
python gtsrb_dataset.py data/GTSRB_data

# Nếu thấy:
# Loaded 39209 training images
# Loaded 12630 test images
# => Thành công!

# Chạy pipeline
bash run_all.sh data/GTSRB_data --model full
```

### Nếu Muốn Flatten Structure (Optional)

```bash
cd data/GTSRB_data/Test

# Di chuyển ảnh ra ngoài
mv Images/*.ppm ./

# Xóa folder Images
rmdir Images

# Test lại
cd ../../..
python gtsrb_dataset.py data/GTSRB_data
```

---

## 🔍 Troubleshooting

### Issue 1: "Warning: Image not found"

**Nguyên nhân**: Paths trong GT CSV không khớp với file structure

**Giải pháp**: Code đã handle vấn đề này. Nếu vẫn gặp lỗi:

```bash
# Check GT CSV format
head -5 data/GTSRB_data/Test/GT-final_test.csv

# Check actual files
ls data/GTSRB_data/Test/Images/ | head -5
# hoặc
ls data/GTSRB_data/Test/*.ppm | head -5
```

### Issue 2: "Invalid GTSRB directory structure"

**Giải pháp**:

```bash
# Verify structure
ls -la data/GTSRB_data/
ls -la data/GTSRB_data/Train/ | head
ls -la data/GTSRB_data/Test/

# Should see:
# Train/ with 43 subdirectories (00000 to 00042)
# Test/ with GT-final_test.csv
# Test/Images/ with .ppm files OR Test/ with .ppm files
```

### Issue 3: "No test images found"

**Kiểm tra**:

```bash
# Count images in both locations
ls data/GTSRB_data/Test/*.ppm 2>/dev/null | wc -l
ls data/GTSRB_data/Test/Images/*.ppm 2>/dev/null | wc -l

# Should see 12630 in one of them
```

---

## 📊 Verification Checklist

Sau khi setup, verify:

- [ ] `data/GTSRB_data/Train/` exists
- [ ] Train has 43 subdirectories (00000 to 00042)
- [ ] Each train subfolder has GT CSV and .ppm files
- [ ] `data/GTSRB_data/Test/GT-final_test.csv` exists
- [ ] Test images exist in either:
  - [ ] `data/GTSRB_data/Test/Images/*.ppm` (12,630 files), OR
  - [ ] `data/GTSRB_data/Test/*.ppm` (12,630 files)
- [ ] Dataset loader works: `python gtsrb_dataset.py data/GTSRB_data`
- [ ] Shows: "Loaded 39209 training images" and "Loaded 12630 test images"

---

## 🎉 Kết Luận

**Cấu trúc hiện tại của bạn (Test/Images/) hoàn toàn OK!**

Code đã được update để:
- ✅ Tự động detect cả Option A và Option B
- ✅ Không cần di chuyển files
- ✅ Hoạt động ngay lập tức

Chỉ cần chạy:
```bash
bash run_all.sh data/GTSRB_data --model full
```

---

**Last Updated**: 2025 (sau khi fix dataset structure issue)
