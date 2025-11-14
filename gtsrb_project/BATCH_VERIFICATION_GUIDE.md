# Batch Verification Guide

## Tổng Quan

**Batch Verification** là chương trình mới để test robustness của **TẤT CẢ samples** trong một class cụ thể, thay vì test từng sample một như `main_interactive.py`.

### So Sánh với Interactive Mode

| Feature | main_interactive.py | batch_verification.py |
|---------|---------------------|----------------------|
| **Scope** | Test 1 sample tại 1 thời điểm | Test toàn bộ class |
| **Output** | In kết quả ra terminal | Lưu CSV + summary file |
| **Statistics** | Không có | ✓ Verified/Not verified counts |
| **Use case** | Khám phá, debug | Thu thập số liệu thống kê |
| **Interaction** | Nhiều tương tác | Config 1 lần, chạy hết |

---

## 1. Cách Sử Dụng

### 1.1. Quick Start

```bash
# Activate virtual environment
source gtsrb_env/bin/activate

# Run batch verification
bash run_batch_verification.sh \
    data/GTSRB_data \
    checkpoints/traffic_sign_net_full.pth
```

**Program sẽ hỏi:**
1. Chọn layer index (ví dụ: 2 cho conv2)
2. Chọn class ID (0-42)
3. Chọn channels (ví dụ: `0,1,2` hoặc `0-5` hoặc Enter để perturb all)
4. Chọn height slice (ví dụ: `5,10` cho [5:10])
5. Chọn width slice (ví dụ: `5,10`)
6. Nhập epsilon (ví dụ: `0.1`)
7. Confirm (y/n)

**Output:**
```
verification_results/
├── class_05_verification_20250109_143022.csv     ← Chi tiết từng sample
├── class_05_summary_20250109_143022.txt          ← Tóm tắt thống kê
└── ... (các lần chạy khác)
```

---

### 1.2. Manual Usage

```bash
python batch_verification.py \
    --data_dir data/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net_full.pth \
    --model full \
    --device cuda \
    --correct_samples_dir correct_samples \
    --output_dir verification_results
```

Sau đó trả lời các câu hỏi configuration như trên.

---

## 2. Workflow Chi Tiết

### Bước 1: Preparation

Đảm bảo đã có:
- ✅ Trained checkpoint (đã training xong)
- ✅ Correct samples CSV files (đã chạy `collect_correct_samples.py`)
- ✅ Virtual environment activated

```bash
# Check files
ls checkpoints/*.pth
ls correct_samples/*.csv

# Should see:
# checkpoints/traffic_sign_net_full.pth
# correct_samples/class_00_correct_indices.csv
# correct_samples/class_01_correct_indices.csv
# ... (43 files)
```

### Bước 2: Configure Verification

Khi chạy program, bạn cần config:

#### **2.1. Select Layer**

```
Available Layers for Perturbation
================================================================================
Index |   Layer Name    |    Type    |  Output Shape
--------------------------------------------------------------------------------
    0 |      conv1      |   Conv2d   |   1x32x32x32
    1 |      conv2      |   Conv2d   |   1x32x16x16
    2 |      conv3      |   Conv2d   |   1x64x16x16
    3 |      conv4      |   Conv2d   |   1x64x8x8
    4 |      conv5      |   Conv2d   |   1x128x8x8
    5 |      conv6      |   Conv2d   |   1x128x4x4
    6 |       fc1       |   Linear   |     1x512
    7 |       fc2       |   Linear   |     1x256
    8 |       fc3       |   Linear   |      1x43
================================================================================

Select layer index to perturb: 2
```

**Chọn layer nào?**
- Conv layers (0-5): Test robustness của features
- FC layers (6-8): Test robustness của representations

#### **2.2. Select Class**

```
Select class ID (0-42): 5
```

GTSRB có 43 classes (0-42). Chọn class muốn test.

#### **2.3. Configure Perturbation Region**

**Option A: Perturb specific channels**
```
Channel indices (e.g., '0,1,2' or '0-5'): 0,1,2
  Selected channels: [0, 1, 2]
```

**Option B: Perturb all channels** (recommended đầu tiên)
```
Channel indices (e.g., '0,1,2' or '0-5'): [Enter]
  All channels will be perturbed
```

**Height and Width:**
```
Height slice (e.g., '5,10' for [5:10]): 5,10
  Height slice: (5, 10)

Width slice (e.g., '5,10' for [5:10]): 5,10
  Width slice: (5, 10)
```

**Epsilon:**
```
Epsilon (perturbation magnitude, e.g., 0.1): 0.1
```

Giá trị nhỏ (0.01-0.1) cho perturbations nhẹ.

### Bước 3: Run Verification

Sau khi confirm, program sẽ:
1. Load tất cả samples correctly classified trong class
2. Verify từng sample với config đã chọn
3. Hiển thị progress bar
4. Lưu kết quả vào CSV

**Progress bar:**
```
Loading 127 samples from class 5...

Verifying: 100%|████████████████████| 127/127 [00:45<00:00,  2.81it/s]
```

### Bước 4: View Results

**Terminal output:**
```
================================================================================
VERIFICATION RESULTS
================================================================================
Class: 5
Total samples: 127
  ✓ Verified robust: 89 (70.1%)
  ✗ Not verified: 35 (27.6%)
  ⚠ Errors: 0
  ⚠ Incorrect predictions: 3
================================================================================
Detailed results saved to: verification_results/class_05_verification_20250109_143022.csv
```

**CSV file format:**
```csv
sample_idx,global_idx,class_id,verified,reason,clean_logit,lower_bound,upper_bound,margin
0,234,5,True,verified,12.3456,10.2345,14.5678,2.1234
1,456,5,False,not_verified,11.2345,8.9012,13.5678,-0.5432
2,789,5,True,verified,15.6789,13.4567,17.8901,3.4567
...
```

**Summary file:**
```
VERIFICATION SUMMARY
================================================================================

Class: 5
Timestamp: 20250109_143022

Configuration:
  Node: /conv3
  Epsilon: 0.1
  Batch: 0
  Channels: [0, 1, 2]
  Height slice: (5, 10)
  Width slice: (5, 10)
  Method: backward

Results:
  Total samples: 127
  Verified robust: 89 (70.1%)
  Not verified: 35 (27.6%)
  Errors: 0
  Incorrect predictions: 3
```

---

## 3. Analyzing Results

### 3.1. Analyze Single Run

```bash
python analyze_verification_results.py \
    --file verification_results/class_05_verification_20250109_143022.csv
```

**Output:**
```
================================================================================
STATISTICS FOR CLASS 5
================================================================================

Total samples: 127
  ✓ Verified robust:      89 ( 70.1%)
  ✗ Not verified:         35 ( 27.6%)
  ⚠ Errors:                0
  ⚠ Incorrect pred:        3

Margin Statistics:
  Average margin:       2.3456
  Min margin:          -1.2345
  Max margin:           5.6789
  Positive margins:       89
  Negative margins:       35
================================================================================

First 5 Verified Samples:
  Idx | Global | Clean   | LB      | UB      | Margin
------------------------------------------------------------------------
    0 |    234 | 12.3456 | 10.2345 | 14.5678 |  2.1234
    2 |    789 | 15.6789 | 13.4567 | 17.8901 |  3.4567
  ...

First 5 Not Verified Samples:
  Idx | Global | Clean   | LB      | UB      | Margin
------------------------------------------------------------------------
    1 |    456 | 11.2345 |  8.9012 | 13.5678 | -0.5432
  ...
```

### 3.2. Analyze Multiple Runs

Nếu chạy verification cho nhiều classes:

```bash
python analyze_verification_results.py --results_dir verification_results
```

**Output:**
```
Found 3 verification result files

Analyzing: class_05_verification_20250109_143022.csv
... (statistics for class 5)

Analyzing: class_13_verification_20250109_144533.csv
... (statistics for class 13)

Analyzing: class_28_verification_20250109_145844.csv
... (statistics for class 28)

================================================================================
OVERALL STATISTICS
================================================================================
Total samples: 387
  ✓ Verified robust:     267 ( 69.0%)
  ✗ Not verified:        112 ( 28.9%)
  ⚠ Errors:                0
  ⚠ Incorrect pred:        8
================================================================================

PER-CLASS SUMMARY
================================================================================
 Class |  Total | Verified |   Rate | Avg Margin
--------------------------------------------------------------------------------
     5 |    127 |       89 |  70.1% |     2.3456
    13 |    145 |      103 |  71.0% |     2.5678
    28 |    115 |       75 |  65.2% |     1.9876
================================================================================
```

---

## 4. Use Cases

### 4.1. Compare Different Layers

**Question:** Layer nào robust hơn?

```bash
# Test class 5 at conv2
python batch_verification.py ...
# Select layer: 1 (conv2)
# Select class: 5
# ... configure ...

# Test class 5 at conv3
python batch_verification.py ...
# Select layer: 2 (conv3)
# Select class: 5
# ... same config ...

# Compare results
python analyze_verification_results.py --results_dir verification_results
```

### 4.2. Compare Different Epsilon Values

**Question:** Epsilon nhỏ hơn → verified rate cao hơn?

```bash
# Test với epsilon = 0.05
# ... configure epsilon: 0.05 ...

# Test với epsilon = 0.1
# ... configure epsilon: 0.1 ...

# Test với epsilon = 0.2
# ... configure epsilon: 0.2 ...

# Compare
```

### 4.3. Compare Different Regions

**Question:** Perturb center region vs corner region?

```bash
# Test center region
# Height: 6,10
# Width: 6,10

# Test corner region
# Height: 0,4
# Width: 0,4

# Compare
```

### 4.4. Per-Class Robustness Analysis

**Question:** Class nào robust nhất?

```bash
# Run for each class
for class_id in {0..42}; do
    python batch_verification.py ...
    # Select class: $class_id
    # ... same config for all ...
done

# Analyze all
python analyze_verification_results.py --results_dir verification_results
```

---

## 5. Understanding Results

### 5.1. Verified vs Not Verified

**Verified Robust:**
```
Sample được verify robust nếu:
  lower_bound(true_class) > upper_bound(all_other_classes)

Nghĩa là: Dù perturb thế nào, model vẫn predict đúng true class
```

**Not Verified:**
```
Không thể verify robust vì:
  lower_bound(true_class) ≤ upper_bound(some_other_class)

Nghĩa là: Có thể tồn tại perturbation làm model predict sai
(nhưng không chắc chắn có tồn tại, chỉ là không verify được)
```

### 5.2. Margin

```python
margin = lower_bound(true_class) - max(upper_bound(other_classes))

margin > 0  →  Verified robust (chắc chắn đúng)
margin ≤ 0  →  Not verified (không chắc)
margin >> 0 →  Very robust (rất chắc chắn)
```

### 5.3. Verified Rate

```
Verified Rate = (Verified Samples) / (Total Samples) × 100%

High rate (>80%): Layer này robust với perturbation config này
Medium rate (50-80%): Một số samples robust, một số không
Low rate (<50%): Hầu hết samples không verify được
```

---

## 6. Tips & Best Practices

### 6.1. Start Small

```bash
# Đầu tiên: Test 1 class với config đơn giản
- Perturb all channels
- Small epsilon (0.05)
- Small region (vài pixels)

# Sau đó: Tăng dần complexity
```

### 6.2. Reasonable Configurations

**Good:**
- Epsilon: 0.01 - 0.1 (small perturbations)
- Region: 5×5 hoặc 10×10 pixels
- Channels: 1-3 channels hoặc all

**Bad:**
- Epsilon: > 0.5 (quá lớn, không realistic)
- Region: Toàn bộ spatial dimension (bounds quá rộng)
- Perturb tất cả channels + tất cả spatial (vô nghĩa)

### 6.3. Interpret Results Carefully

**Verified ≠ Completely Safe**
- Chỉ chứng minh robust với perturbation type đã config
- Không chứng minh robust với attack khác

**Not Verified ≠ Vulnerable**
- Chỉ là không verify được
- Có thể vẫn robust nhưng bounds quá loose

### 6.4. Save Configurations

Lưu lại config của từng run để reproduce:

```bash
# Ghi config vào file
cat > config_experiment1.txt <<EOF
Layer: conv3 (index 2)
Class: 5
Channels: all
Height: 5,10
Width: 5,10
Epsilon: 0.1
Method: backward
EOF
```

---

## 7. Troubleshooting

### Issue 1: Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Option 1: Use CPU
python batch_verification.py ... --device cpu

# Option 2: Reduce number of samples (modify CSV)
# Option 3: Smaller perturbation region
```

### Issue 2: Slow Verification

**Typical speed:**
- CUDA: ~2-5 samples/second
- CPU: ~0.2-0.5 samples/second

**For 100 samples:**
- CUDA: ~20-50 seconds
- CPU: ~3-8 minutes

**Speed up:**
- Use CUDA
- Smaller perturbation region → tighter bounds → faster convergence

### Issue 3: All Samples Not Verified

**Possible causes:**
1. Epsilon quá lớn
2. Perturb region quá rộng
3. Layer này inherently not robust

**Debug:**
```bash
# Try smaller epsilon
Epsilon: 0.01  (thay vì 0.1)

# Try smaller region
Height: 8,9  (chỉ 1×1 pixel)

# Try different layer
```

### Issue 4: CSV File Not Found

**Error:**
```
FileNotFoundError: CSV not found: correct_samples/class_05_correct_indices.csv
```

**Solution:**
```bash
# Run collection first
python collect_correct_samples.py \
    --data_dir data/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net_full.pth \
    --model full
```

---

## 8. Example Session

```bash
$ bash run_batch_verification.sh data/GTSRB_data checkpoints/traffic_sign_net_full.pth

========================================================================
              GTSRB Batch Verification Runner
========================================================================

Configuration:
  Data directory: data/GTSRB_data
  Checkpoint: checkpoints/traffic_sign_net_full.pth
  Model: full
  Device: cuda

[INFO] Starting batch verification...

Using device: cuda
Loading model from checkpoints/traffic_sign_net_full.pth...
Model loaded successfully!
  Checkpoint accuracy: 95.67%

Creating no-dropout model for verification...
No-dropout model created

Creating bounded module...
Dataset loaded: 12630 test samples

================================================================================
                         BATCH VERIFICATION
================================================================================

Available Layers for Perturbation
================================================================================
Index |   Layer Name    |    Type    |  Output Shape
--------------------------------------------------------------------------------
    0 |      conv1      |   Conv2d   |   1x32x32x32
    1 |      conv2      |   Conv2d   |   1x32x16x16
    2 |      conv3      |   Conv2d   |   1x64x16x16
    3 |      conv4      |   Conv2d   |   1x64x8x8
    4 |      conv5      |   Conv2d   |   1x128x8x8
    5 |      conv6      |   Conv2d   |   1x128x4x4
    6 |       fc1       |   Linear   |     1x512
    7 |       fc2       |   Linear   |     1x256
    8 |       fc3       |   Linear   |      1x43
================================================================================

Select layer index to perturb: 2
Selected: /conv3 (Conv2d, shape: 1x64x16x16)

Select class ID (0-42): 5

Configure perturbation region:
(Press Enter to skip, will perturb all)
Channel indices (e.g., '0,1,2' or '0-5'): 0,1,2
  Selected channels: [0, 1, 2]
Height slice (e.g., '5,10' for [5:10]): 5,10
  Height slice: (5, 10)
Width slice (e.g., '5,10' for [5:10]): 5,10
  Width slice: (5, 10)

Epsilon (perturbation magnitude, e.g., 0.1): 0.1

================================================================================
Configuration Summary:
  Class: 5
  Layer: /conv3
  Channels: [0, 1, 2]
  Height: (5, 10)
  Width: (5, 10)
  Epsilon: 0.1
  Method: backward (CROWN)
================================================================================

Proceed with verification? (y/n): y

Loading 127 samples from class 5...

Verifying 127 samples from class 5...
Configuration:
  Node: /conv3
  Epsilon: 0.1
  Batch: 0
  Channels: [0, 1, 2]
  Height: (5, 10)
  Width: (5, 10)
  Method: backward

Verifying: 100%|████████████████████| 127/127 [00:45<00:00,  2.81it/s]

Saving results to verification_results/class_05_verification_20250109_143022.csv...

================================================================================
VERIFICATION RESULTS
================================================================================
Class: 5
Total samples: 127
  ✓ Verified robust: 89 (70.1%)
  ✗ Not verified: 35 (27.6%)
  ⚠ Errors: 0
  ⚠ Incorrect predictions: 3
================================================================================
Detailed results saved to: verification_results/class_05_verification_20250109_143022.csv

Batch verification completed!
[INFO] Batch verification completed successfully!

Results saved in verification_results/
```

---

## 9. Files Generated

```
verification_results/
├── class_05_verification_20250109_143022.csv
│   ├── Column: sample_idx (index trong class)
│   ├── Column: global_idx (index trong test set)
│   ├── Column: class_id (5)
│   ├── Column: verified (True/False)
│   ├── Column: reason (verified/not_verified/error/incorrect_prediction)
│   ├── Column: clean_logit (logit của true class, no perturbation)
│   ├── Column: lower_bound (lower bound với perturbation)
│   ├── Column: upper_bound (upper bound với perturbation)
│   └── Column: margin (lower_bound - max_other_upper_bound)
│
└── class_05_summary_20250109_143022.txt
    ├── Configuration
    ├── Statistics
    └── Timestamp
```

---

## 10. Next Steps

Sau khi chạy batch verification:

1. **Analyze results**
   ```bash
   python analyze_verification_results.py
   ```

2. **Compare configurations**
   - Different layers
   - Different epsilon values
   - Different perturbation regions

3. **Investigate failure cases**
   - Xem samples nào not verified
   - Visualize những samples đó
   - Hiểu tại sao không verify được

4. **Write paper/report**
   - Use statistics từ CSV files
   - Create tables/figures
   - Draw conclusions

---

**Version:** 1.0
**Last Updated:** 2025
**Related Files:**
- `batch_verification.py`: Main implementation
- `run_batch_verification.sh`: Convenient runner script
- `analyze_verification_results.py`: Analysis tool
