# Hướng Dẫn Cài Đặt Môi Trường và Chạy Dự Án GTSRB

## Mục Lục
1. [Yêu Cầu Hệ Thống](#yêu-cầu-hệ-thống)
2. [Cài Đặt Môi Trường Ảo](#cài-đặt-môi-trường-ảo)
3. [Cài Đặt Thư Viện](#cài-đặt-thư-viện)
4. [Tải Dataset GTSRB](#tải-dataset-gtsrb)
5. [Chạy Dự Án](#chạy-dự-án)
6. [Kiểm Tra Cài Đặt](#kiểm-tra-cài-đặt)

---

## Yêu Cầu Hệ Thống

### Phần Cứng
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **GPU**: Khuyến nghị có NVIDIA GPU với CUDA support
- **Ổ Cứng**: Tối thiểu 5GB trống

### Phần Mềm
- **Python**: 3.7 hoặc cao hơn (khuyến nghị Python 3.8 hoặc 3.9)
- **CUDA**: 10.2 hoặc cao hơn (nếu dùng GPU)
- **pip**: Công cụ quản lý package của Python

---

## Cài Đặt Môi Trường Ảo

### Bước 1: Kiểm tra Python

Mở terminal/command prompt và kiểm tra version Python:

```bash
python --version
# hoặc
python3 --version
```

Nếu chưa có Python, tải tại: https://www.python.org/downloads/

### Bước 2: Di chuyển đến thư mục dự án

```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE
```

### Bước 3: Tạo môi trường ảo

#### Cách 1: Sử dụng venv (khuyến nghị)

```bash
# Tạo môi trường ảo tên là 'gtsrb_env'
python -m venv gtsrb_env

# hoặc nếu dùng python3
python3 -m venv gtsrb_env
```

#### Cách 2: Sử dụng conda (nếu bạn đã cài Anaconda/Miniconda)

```bash
# Tạo môi trường conda
conda create -n gtsrb_env python=3.9
```

### Bước 4: Kích hoạt môi trường ảo

#### Trên Linux/MacOS (venv):
```bash
source gtsrb_env/bin/activate
```

#### Trên Windows (venv):
```bash
# Command Prompt
gtsrb_env\Scripts\activate.bat

# PowerShell
gtsrb_env\Scripts\Activate.ps1
```

#### Với Conda:
```bash
conda activate gtsrb_env
```

Sau khi kích hoạt, bạn sẽ thấy `(gtsrb_env)` ở đầu dòng lệnh:
```
(gtsrb_env) user@computer:~$
```

---

## Cài Đặt Thư Viện

### Bước 1: Cập nhật pip

```bash
pip install --upgrade pip
```

### Bước 2: Cài đặt PyTorch

#### Với GPU (CUDA):

Truy cập https://pytorch.org/ để chọn lệnh phù hợp với hệ thống của bạn.

Ví dụ cho CUDA 11.3:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

Ví dụ cho CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Chỉ CPU (không có GPU):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Bước 3: Cài đặt auto_LiRPA

```bash
# Từ thư mục auto_LiRPA_CLAUDE
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE
pip install -e .
```

Lệnh `pip install -e .` sẽ cài đặt auto_LiRPA ở chế độ "editable", cho phép bạn sửa code mà không cần cài lại.

### Bước 4: Cài đặt các thư viện còn lại

```bash
cd gtsrb_project
pip install -r requirements.txt
```

### Bước 5: Xác nhận cài đặt thành công

```bash
# Kiểm tra PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Kiểm tra auto_LiRPA
python -c "import auto_LiRPA; print('auto_LiRPA installed successfully')"

# Kiểm tra các thư viện khác
python -c "import torchvision; import numpy; import PIL; import tqdm; print('All dependencies OK')"
```

---

## Tải Dataset GTSRB

### Bước 1: Tạo thư mục cho dataset

**Bạn có 2 options để đặt dataset:**

#### Option 1: Trong thư mục project (Khuyến nghị)
```bash
# Vào thư mục project
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE/gtsrb_project

# Tạo thư mục data
mkdir -p data/GTSRB_data
cd data/GTSRB_data
```

**Cấu trúc kết quả:**
```
gtsrb_project/
├── data/
│   └── GTSRB_data/      ← Dataset ở đây
│       ├── Train/
│       └── Test/
├── gtsrb_dataset.py
└── ...
```

**Khi chạy script:**
```bash
python train_gtsrb.py --data_dir data/GTSRB_data
```

#### Option 2: Trong thư mục Documents
```bash
# Tạo thư mục ở Documents
mkdir -p ~/Documents/GTSRB_data
cd ~/Documents/GTSRB_data
```

**Cấu trúc kết quả:**
```
/Users/springbaby/Documents/
├── GTSRB_data/          ← Dataset ở đây
│   ├── Train/
│   └── Test/
└── Nguyen/
    └── PHD/...
```

**Khi chạy script:**
```bash
python train_gtsrb.py --data_dir ~/Documents/GTSRB_data
```

**💡 Khuyến nghị:** Dùng Option 1 để dễ quản lý và backup.

### Bước 2: Tải xuống dataset

#### Cách 1: Tải thủ công từ website

Truy cập các link sau và tải về:

1. **Training set**:
   - Link: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
   - Size: ~300MB

2. **Test set**:
   - Images: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
   - Size: ~90MB

3. **Test annotations**:
   - Link: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
   - Size: ~1KB

#### Cách 2: Sử dụng wget (Linux/Mac)

```bash
# Training data
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip

# Test images
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

# Test annotations
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
```

#### Cách 3: Sử dụng curl (Mac)

```bash
curl -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip
curl -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip
curl -O https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
```

### Bước 3: Giải nén dataset

```bash
# Giải nén training data
unzip GTSRB-Training_fixed.zip

# Giải nén test images
unzip GTSRB_Final_Test_Images.zip

# Giải nén test annotations
unzip GTSRB_Final_Test_GT.zip
```

### Bước 4: Tổ chức lại cấu trúc thư mục

Sau khi giải nén, cấu trúc thư mục có thể có 2 dạng (cả 2 đều được hỗ trợ):

#### Cấu trúc Option A (Phổ biến - Test images trong subfolder):
```
GTSRB_data/
├── Train/
│   ├── 00000/
│   │   ├── GT-00000.csv
│   │   ├── 00000_00000.ppm
│   │   └── ...
│   ├── 00001/
│   └── ... (43 classes: 00000 to 00042)
└── Test/
    ├── GT-final_test.csv
    └── Images/              ← Subfolder chứa ảnh test
        ├── 00000.ppm
        ├── 00001.ppm
        └── ... (12,630 images)
```

#### Cấu trúc Option B (Ảnh trực tiếp trong Test/):
```
GTSRB_data/
├── Train/
│   └── ... (như trên)
└── Test/
    ├── GT-final_test.csv
    ├── 00000.ppm           ← Ảnh trực tiếp trong Test/
    ├── 00001.ppm
    └── ... (12,630 images)
```

**✅ Lưu ý quan trọng:**

Dataset loader đã được cập nhật để **tự động detect cả 2 cấu trúc**. Bạn **KHÔNG CẦN** di chuyển files!

Code sẽ tự động tìm ảnh test ở:
1. `Test/Images/*.ppm` (Option A)
2. `Test/*.ppm` (Option B)

Nếu bạn muốn chuyển từ Option A sang Option B (optional):

```bash
# Chỉ làm nếu muốn flatten structure
cd data/GTSRB_data/Test
mv Images/*.ppm ./
rmdir Images
```

### Bước 5: Kiểm tra dataset

```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE/gtsrb_project

# Test dataset loader
python gtsrb_dataset.py ~/Documents/GTSRB_data
```

Output mong đợi:
```
Testing GTSRB dataset loader...
Loaded 39209 training images
Loaded 12630 test images
Train batches: 1226
Test batches: 395
Batch shape: torch.Size([32, 3, 32, 32])
Labels shape: torch.Size([32])
Label range: 0-42
```

---

## Chạy Dự Án

### Bước 1: Training Model

Quay lại thư mục gtsrb_project và training model:

```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE/gtsrb_project

# Tạo thư mục cho checkpoints
mkdir -p checkpoints

# Training với full model
python train_gtsrb.py \
    --data_dir ~/Documents/GTSRB_data \
    --model full \
    --epochs 50 \
    --batch_size 128 \
    --lr 0.001 \
    --save_path checkpoints/traffic_sign_net.pth
```

**Tham số:**
- `--data_dir`: Đường dẫn đến thư mục GTSRB_data
- `--model`: `full` (chính xác cao) hoặc `simple` (train nhanh hơn)
- `--epochs`: Số epoch (mặc định 50)
- `--batch_size`: Batch size (giảm xuống nếu GPU hết RAM)
- `--lr`: Learning rate

**Training time:**
- Với GPU: ~15-30 phút
- Với CPU: ~2-4 giờ

**Kết quả mong đợi:** Test accuracy > 90%

### Bước 2: Collect Correct Samples

Sau khi training xong, chạy inference để thu thập các samples được phân loại đúng:

```bash
python collect_correct_samples.py \
    --data_dir ~/Documents/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full \
    --output_dir correct_samples \
    --batch_size 128
```

Output:
- Tạo thư mục `correct_samples/`
- 43 file CSV: `class_00_correct_indices.csv` đến `class_42_correct_indices.csv`
- 1 file `summary.csv` với thống kê

### Bước 3: Interactive Testing

Chạy chương trình test tương tác:

```bash
python main_interactive.py \
    --data_dir ~/Documents/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full \
    --correct_samples_dir correct_samples \
    --device cuda
```

Nếu không có GPU, dùng `--device cpu`

#### Workflow trong Interactive Mode:

1. **Chọn layer**: Chương trình hiển thị danh sách các layer Conv và FC
   ```
   Select layer index (or -1 to quit): 2
   ```

2. **Chọn class và sample**:
   ```
   Select class ID (0-42): 5
   Select sample index within class (default 0): 0
   ```

3. **Xem output clean** (không có perturbation)

4. **Cấu hình perturbation**:
   - Cho Conv layer:
     ```
     Channel index (or 'all', or comma-separated list): 0,1,2
     Height slice (start,end) or 'all': 5,10
     Width slice (start,end) or 'all': 5,10
     Epsilon value: 0.1
     ```

   - Cho FC layer:
     ```
     Feature indices (comma-separated or 'all'): 10,11,12,13,14
     Epsilon value: 0.1
     ```

5. **Xem kết quả**: Bounds và verification result

---

## Kiểm Tra Cài Đặt

### Script kiểm tra nhanh

Tạo file `test_installation.py`:

```python
#!/usr/bin/env python
"""Quick installation test"""

print("Testing installation...")

# Test 1: Import libraries
print("\n1. Testing imports...")
try:
    import torch
    import torchvision
    import numpy
    import PIL
    import tqdm
    import auto_LiRPA
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

# Test 2: PyTorch version
print("\n2. Checking PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")

# Test 3: Test model creation
print("\n3. Testing model creation...")
try:
    from traffic_sign_net import TrafficSignNet
    model = TrafficSignNet(num_classes=43)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 43), f"Wrong output shape: {y.shape}"
    print("   ✓ Model works correctly")
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    exit(1)

# Test 4: Test masked perturbation
print("\n4. Testing masked perturbation...")
try:
    from masked_perturbation import MaskedPerturbationLpNorm
    import numpy as np

    ptb = MaskedPerturbationLpNorm(
        eps=0.1,
        norm=np.inf,
        batch_idx=0,
        channel_idx=0,
        height_slice=(0, 5),
        width_slice=(0, 5)
    )

    x_test = torch.randn(1, 32, 8, 8)
    bounds, center, aux = ptb.init(x_test, forward=False)

    assert bounds.lower.shape == x_test.shape
    assert bounds.upper.shape == x_test.shape
    print("   ✓ Masked perturbation works correctly")
except Exception as e:
    print(f"   ✗ Perturbation test failed: {e}")
    exit(1)

# Test 5: Test intermediate bounded module
print("\n5. Testing intermediate bounded module...")
try:
    from intermediate_bound_module import IntermediateBoundedModule
    import torch.nn as nn

    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.fc = nn.Linear(16*32*32, 10)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)

    simple_model = SimpleNet()
    dummy = torch.randn(1, 3, 32, 32)

    lirpa_model = IntermediateBoundedModule(simple_model, dummy)
    print("   ✓ Intermediate bounded module works correctly")
except Exception as e:
    print(f"   ✗ Bounded module test failed: {e}")
    exit(1)

print("\n" + "="*50)
print("All tests passed! Installation is correct.")
print("="*50)
```

Chạy script:

```bash
python test_installation.py
```

---

## Xử Lý Lỗi Thường Gặp

### Lỗi 1: ImportError: No module named 'torch'

**Nguyên nhân**: PyTorch chưa được cài đặt hoặc môi trường ảo chưa được kích hoạt

**Giải pháp**:
```bash
# Kích hoạt môi trường ảo
source gtsrb_env/bin/activate  # Linux/Mac
# hoặc
gtsrb_env\Scripts\activate.bat  # Windows

# Cài lại PyTorch
pip install torch torchvision
```

### Lỗi 2: CUDA out of memory

**Nguyên nhân**: GPU không đủ RAM

**Giải pháp**:
```bash
# Giảm batch size
python train_gtsrb.py --batch_size 64  # hoặc 32
```

### Lỗi 3: ModuleNotFoundError: No module named 'auto_LiRPA'

**Nguyên nhân**: auto_LiRPA chưa được cài đặt

**Giải pháp**:
```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE
pip install -e .
```

### Lỗi 4: FileNotFoundError: GTSRB dataset not found

**Nguyên nhân**: Đường dẫn dataset không đúng

**Giải pháp**:
```bash
# Kiểm tra đường dẫn
ls ~/Documents/GTSRB_data/Train
ls ~/Documents/GTSRB_data/Test

# Sửa đường dẫn trong lệnh
python train_gtsrb.py --data_dir /path/to/your/GTSRB_data
```

### Lỗi 5: Permission denied khi tạo môi trường ảo

**Nguyên nhân**: Không có quyền ghi trong thư mục

**Giải pháp**:
```bash
# Tạo môi trường ảo ở thư mục home
cd ~
python -m venv gtsrb_env
source gtsrb_env/bin/activate
```

---

## Gỡ Cài Đặt

Nếu muốn gỡ bỏ môi trường và bắt đầu lại:

```bash
# Thoát môi trường ảo
deactivate

# Xóa thư mục môi trường ảo
rm -rf gtsrb_env

# Hoặc với conda
conda env remove -n gtsrb_env
```

---

## Hỗ Trợ

Nếu gặp vấn đề:

1. Kiểm tra lại các bước cài đặt
2. Chạy `test_installation.py` để xác định lỗi
3. Kiểm tra log messages
4. Đảm bảo Python version >= 3.7

---

## Tóm Tắt Các Lệnh Quan Trọng

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv gtsrb_env
source gtsrb_env/bin/activate

# Cài đặt thư viện
pip install torch torchvision
cd auto_LiRPA_CLAUDE && pip install -e .
cd gtsrb_project && pip install -r requirements.txt

# Tải GTSRB dataset (tải thủ công hoặc dùng wget)

# Training
python train_gtsrb.py --data_dir ~/Documents/GTSRB_data --model full

# Collect samples
python collect_correct_samples.py --data_dir ~/Documents/GTSRB_data --checkpoint checkpoints/traffic_sign_net.pth --model full

# Interactive testing
python main_interactive.py --data_dir ~/Documents/GTSRB_data --checkpoint checkpoints/traffic_sign_net.pth --model full
```

---

**Chúc bạn thành công!**
