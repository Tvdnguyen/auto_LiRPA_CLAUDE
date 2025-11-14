# Quick Start - Cheat Sheet

## 🚀 Setup trong 5 phút

### 1. Tạo môi trường
```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE/gtsrb_project

# Tạo venv
python -m venv gtsrb_env

# Kích hoạt (Mac/Linux)
source gtsrb_env/bin/activate

# Kích hoạt (Windows)
# gtsrb_env\Scripts\activate
```

### 2. Cài đặt
```bash
# PyTorch
pip install torch torchvision

# auto_LiRPA
cd ..
pip install -e .
cd gtsrb_project

# Dependencies
pip install -r requirements.txt
```

### 3. Test
```bash
python test_installation.py
```

---

## 📦 Tải Dataset

### Tạo thư mục
```bash
# OPTION 1: Trong project (khuyến nghị)
mkdir -p data/GTSRB_data
cd data/GTSRB_data

# OPTION 2: Trong Documents
# mkdir -p ~/Documents/GTSRB_data
# cd ~/Documents/GTSRB_data
```

### Tải files
```bash
# Training set
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip

# Test images
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip

# Test labels
wget https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip
```

### Giải nén
```bash
unzip GTSRB-Training_fixed.zip
unzip GTSRB_Final_Test_Images.zip
unzip GTSRB_Final_Test_GT.zip

# Quay lại project
cd ../..  # hoặc cd /path/to/gtsrb_project
```

---

## 🎯 Chạy Pipeline

### Full auto (khuyến nghị)
```bash
bash run_all.sh data/GTSRB_data --model full --epochs 50
```

### Từng bước

#### Bước 1: Training
```bash
python train_gtsrb.py \
    --data_dir data/GTSRB_data \
    --model full \
    --epochs 50 \
    --batch_size 128
```

#### Bước 2: Collect samples
```bash
python collect_correct_samples.py \
    --data_dir data/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full
```

#### Bước 3: Interactive testing
```bash
python main_interactive.py \
    --data_dir data/GTSRB_data \
    --checkpoint checkpoints/traffic_sign_net.pth \
    --model full
```

---

## 💡 Interactive Mode - Example Session

```
>>> Select layer index: 2
    Selected: /conv3 (Conv2d, 16x16x64)

>>> Select class ID (0-42): 5
>>> Select sample index: 0
    Loaded sample from class 5

>>> Clean output: Predicted class 5, logit 12.34

>>> Channel index: 0,1,2
>>> Height slice: 5,10
>>> Width slice: 5,10
>>> Epsilon: 0.1

>>> Computing bounds...
    Lower bound for class 5: 11.23
    Upper bound for class 5: 13.45
    ✓ VERIFIED ROBUST
```

---

## 📍 Đường Dẫn Quan Trọng

### Với Option 1 (trong project)
```bash
DATA_DIR="data/GTSRB_data"
CHECKPOINT="checkpoints/traffic_sign_net.pth"
CORRECT_SAMPLES="correct_samples"
```

### Với Option 2 (trong Documents)
```bash
DATA_DIR="~/Documents/GTSRB_data"
CHECKPOINT="checkpoints/traffic_sign_net.pth"
CORRECT_SAMPLES="correct_samples"
```

---

## 🔧 Troubleshooting

### Out of memory
```bash
# Giảm batch size
python train_gtsrb.py --batch_size 64
```

### No GPU
```bash
# Dùng CPU
python main_interactive.py --device cpu
```

### Import errors
```bash
# Re-install
pip install -e .. --force-reinstall
pip install -r requirements.txt --force-reinstall
```

### Dataset not found
```bash
# Check path
ls data/GTSRB_data/Train
ls data/GTSRB_data/Test

# Verify structure
python gtsrb_dataset.py data/GTSRB_data
```

---

## 📋 Checklist

### Pre-flight
- [ ] Python 3.7+ installed
- [ ] venv created and activated
- [ ] PyTorch installed
- [ ] auto_LiRPA installed
- [ ] Dependencies installed
- [ ] test_installation.py passes

### Dataset
- [ ] GTSRB downloaded
- [ ] Files unzipped
- [ ] Train/ folder has 43 subdirectories
- [ ] Test/ folder has GT-final_test.csv
- [ ] Dataset loader works

### Training
- [ ] Model trains without errors
- [ ] Test accuracy > 90%
- [ ] Checkpoint saved
- [ ] CSV files generated

### Testing
- [ ] Checkpoint loads
- [ ] Interactive mode starts
- [ ] Can select layers
- [ ] Bounds compute successfully

---

## 🎓 Learning Curve

### Just use it (15 min)
1. Follow this cheat sheet
2. Run `bash run_all.sh`
3. Use interactive mode

### Understand it (1 hour)
1. Read README.md
2. Read PROJECT_SUMMARY.md
3. Browse the code

### Extend it (1 day)
1. Read all docs
2. Study masked_perturbation.py
3. Study intermediate_bound_module.py
4. Implement your extension

---

## 🌟 Pro Tips

1. **Use full paths** to avoid confusion:
   ```bash
   python train_gtsrb.py --data_dir $(pwd)/data/GTSRB_data
   ```

2. **Save logs** for later analysis:
   ```bash
   python train_gtsrb.py ... 2>&1 | tee training.log
   ```

3. **Use tmux/screen** for long training:
   ```bash
   tmux new -s training
   python train_gtsrb.py ...
   # Ctrl+B, D to detach
   # tmux attach -t training to reattach
   ```

4. **Test with simple model first**:
   ```bash
   python train_gtsrb.py --model simple --epochs 10
   ```

5. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## 📞 Quick Reference Commands

```bash
# Activate env
source gtsrb_env/bin/activate

# Deactivate env
deactivate

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test dataset
python gtsrb_dataset.py data/GTSRB_data

# Test model
python traffic_sign_net.py

# Full pipeline
bash run_all.sh data/GTSRB_data

# Interactive
python main_interactive.py --data_dir data/GTSRB_data --checkpoint checkpoints/traffic_sign_net.pth --model full
```

---

## 🎉 You're Ready!

Nếu tất cả các bước trên work, bạn đã sẵn sàng để:
- ✅ Train models
- ✅ Test perturbations
- ✅ Analyze robustness
- ✅ Extend the code

**Have fun! 🚀**
