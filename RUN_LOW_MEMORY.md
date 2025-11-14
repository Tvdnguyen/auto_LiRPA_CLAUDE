# Hướng dẫn chạy với RAM thấp (8GB)

## Vấn đề hiện tại
Chương trình bị killed do thiếu RAM ngay cả với các tối ưu hóa.

## Giải pháp 1: Chạy với tham số tối thiểu

```bash
python3 integrated_pe_sensitivity_analysis_low_memory.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --class_id 0 \
    --test_idx 0 \
    --array_size 8 \
    --duration 1 \
    --tolerance 0.05 \
    --batch_size 4 \
    --save_every 4 \
    --epsilon_max 0.3
```

**Giải thích các tham số giảm RAM:**
- `--tolerance 0.05`: Tăng từ 0.01 → giảm số lần binary search từ 6-7 xuống còn 4-5 lần
- `--batch_size 4`: Giảm từ 8 → cleanup thường xuyên hơn
- `--save_every 4`: Save sớm hơn để tránh mất data
- `--epsilon_max 0.3`: Giảm từ 1.0 → ít iteration hơn trong binary search

## Giải pháp 2: Test từng PE một (an toàn nhất)

Thay vì test cả 64 PEs, test từng PE và save kết quả:

```bash
# Test PE (0,0)
python3 test_single_pe.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --pe_row 0 --pe_col 0

# Test PE (0,1)
python3 test_single_pe.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --pe_row 0 --pe_col 1

# ... (tiếp tục cho các PE khác)
```

## Giải pháp 3: Chạy trên máy có nhiều RAM hơn

Nếu có thể, chạy trên máy có ≥16GB RAM với tham số đầy đủ:

```bash
python3 integrated_pe_sensitivity_analysis_low_memory.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --class_id 0 \
    --test_idx 0 \
    --array_size 8 \
    --duration 2 \
    --tolerance 0.001 \
    --batch_size 16
```

## Monitoring RAM

Trong khi chạy, monitor RAM trong terminal khác:

```bash
# macOS
top -pid $(pgrep -f integrated_pe_sensitivity)

# Hoặc
watch -n 1 "ps aux | grep python | grep integrated"
```

## Ước tính RAM usage

| Configuration | RAM Peak | Thời gian | PEs/giờ |
|--------------|----------|-----------|---------|
| tolerance=0.05, batch=4 | ~5-6 GB | ~30 min | ~120 |
| tolerance=0.01, batch=8 | ~10 GB | ~25 min | ~150 |
| tolerance=0.001, batch=16 | ~14 GB | ~20 min | ~200 |

## Nếu vẫn bị killed

1. **Tắt tất cả ứng dụng khác** trước khi chạy
2. **Restart máy** để free RAM
3. **Chạy script test_single_pe.py** (sẽ tạo ở dưới)
4. **Giảm epsilon_max xuống 0.1 hoặc 0.2**
