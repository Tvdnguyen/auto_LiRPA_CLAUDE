# Hướng Dẫn Upload Lên GitHub

## Phương Pháp 1: Sử Dụng GitHub Desktop (Dễ Nhất)

### Bước 1: Cài Đặt GitHub Desktop
- Download: https://desktop.github.com/
- Cài đặt và đăng nhập bằng GitHub account

### Bước 2: Add Repository
1. Mở GitHub Desktop
2. **File → Add Local Repository**
3. Click **Choose...** và chọn folder:
   ```
   /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE
   ```
4. Nếu thấy thông báo "This directory does not appear to be a Git repository":
   - Click **"Create a Repository"**
   - Repository name: `auto_LiRPA_CLAUDE`
   - Click **Create Repository**

### Bước 3: Commit Changes
1. Trong GitHub Desktop, bạn sẽ thấy tất cả files trong tab "Changes"
2. Ở dưới cùng:
   - **Summary**: Nhập "Initial commit"
   - **Description**: Nhập mô tả (hoặc để trống)
3. Click **Commit to main**

### Bước 4: Publish to GitHub
1. Click **Publish repository** (nút ở trên)
2. Chọn:
   - **Name**: auto_LiRPA_CLAUDE
   - **Description**: Auto-LiRPA GTSRB & Systolic Array Fault Analysis
   - **Keep this code private**: Bỏ chọn nếu muốn public, chọn nếu muốn private
3. Click **Publish Repository**

✅ **Xong!** Repository đã được upload lên GitHub!

---

## Phương Pháp 2: Sử Dụng Command Line

### Bước 1: Mở Terminal
```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE
```

### Bước 2: Chạy Upload Script
```bash
chmod +x upload_to_github.sh
./upload_to_github.sh
```

### Bước 3: Tạo GitHub Repository
1. Mở browser, vào: https://github.com/new
2. **Repository name**: `auto_LiRPA_CLAUDE`
3. **Description**: `Auto-LiRPA GTSRB & Systolic Array Fault Analysis`
4. **Public** hoặc **Private**: Chọn theo ý bạn
5. **❌ KHÔNG chọn**: "Initialize this repository with a README"
6. Click **Create repository**

### Bước 4: Link và Push
Sau khi tạo repo trên GitHub, chạy các lệnh sau (thay YOUR_USERNAME bằng GitHub username của bạn):

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/auto_LiRPA_CLAUDE.git
git push -u origin main
```

**Ví dụ**: Nếu username là `springbaby`:
```bash
git remote add origin https://github.com/springbaby/auto_LiRPA_CLAUDE.git
git push -u origin main
```

### Bước 5: Nhập GitHub Credentials
- Khi được yêu cầu, nhập GitHub username và password
- **Lưu ý**: Từ 2021, GitHub yêu cầu dùng **Personal Access Token** thay vì password
  - Tạo token tại: https://github.com/settings/tokens
  - Chọn scope: `repo`
  - Copy token và dùng làm password

✅ **Xong!** Repository đã được upload!

---

## Phương Pháp 3: Kéo Thả Trên GitHub.com

### Bước 1: Tạo Repository Trống
1. Vào https://github.com/new
2. **Repository name**: `auto_LiRPA_CLAUDE`
3. **Public/Private**: Chọn theo ý
4. Click **Create repository**

### Bước 2: Upload Files
1. Trong repository vừa tạo, click **uploading an existing file**
2. Kéo thả toàn bộ folder `auto_LiRPA_CLAUDE` vào
3. **⚠️ Lưu ý**: Phương pháp này có giới hạn file size (100MB/file)

### Bước 3: Commit
1. Nhập commit message: "Initial commit"
2. Click **Commit changes**

---

## Kiểm Tra Sau Khi Upload

Sau khi upload thành công, repository của bạn trên GitHub sẽ có:

```
auto_LiRPA_CLAUDE/
├── README.md                              ✓
├── .gitignore                             ✓
├── auto_LiRPA/                            ✓
├── gtsrb_project/                         ✓
├── systolic_fault_sim/                    ✓
├── integrated_pe_sensitivity_analysis_v2.py  ✓
└── ...
```

**Các file KHÔNG được upload** (theo .gitignore):
- `__pycache__/`
- `*.pth` (model checkpoints - quá lớn)
- `*.csv` (results)
- `*.png` (visualizations)
- `data/` (dataset - quá lớn)

---

## Cập Nhật Code Sau Này

### Sử dụng GitHub Desktop
1. Mở GitHub Desktop
2. Chọn repository `auto_LiRPA_CLAUDE`
3. Thay đổi code
4. Commit changes (ở dưới cùng)
5. Click **Push origin**

### Sử dụng Command Line
```bash
cd /Users/springbaby/Documents/Nguyen/PHD/EXP/DNN_verification/auto_LiRPA_CLAUDE

# Add changed files
git add .

# Commit
git commit -m "Update: describe your changes"

# Push to GitHub
git push
```

---

## Troubleshooting

### Lỗi: "Permission denied"
- Tạo Personal Access Token: https://github.com/settings/tokens
- Dùng token thay vì password

### Lỗi: "Repository not found"
- Kiểm tra URL: `git remote -v`
- Update URL: `git remote set-url origin https://github.com/YOUR_USERNAME/auto_LiRPA_CLAUDE.git`

### Lỗi: "File too large"
- Files > 100MB không thể upload
- Đảm bảo `.gitignore` đã exclude các file lớn (checkpoints, data)
- Dùng Git LFS cho files lớn: https://git-lfs.github.com/

---

## Liên Hệ

Nếu gặp vấn đề, tham khảo:
- GitHub Docs: https://docs.github.com/
- Git Tutorial: https://git-scm.com/docs/gittutorial
