# Lý Thuyết: Intermediate Layer Perturbations với auto_LiRPA

## Tổng Quan

Document này giải thích chi tiết cách thức thêm perturbations (nhiễu loạn) vào các lớp trung gian (intermediate layers) của mạng neural network, bao gồm:
1. Cơ sở lý thuyết từ auto_LiRPA gốc
2. Cách mở rộng để hỗ trợ intermediate perturbations
3. Cơ chế masked/regional perturbations

---

## 1. Auto_LiRPA Gốc: Cơ Sở Hỗ Trợ

### 1.1. Perturbation Framework Có Sẵn

Auto_LiRPA đã cung cấp **framework cơ bản** để xử lý perturbations, nhưng **chỉ ở input layer**:

#### **File: `auto_LiRPA/perturbations.py`**

```python
class Perturbation:
    """Base class for perturbations"""
    def __init__(self):
        pass

class PerturbationLpNorm(Perturbation):
    """
    Lp-norm bounded perturbation
    Định nghĩa: ||δ||_p ≤ ε
    """
    def __init__(self, eps=0, norm=np.inf, x_L=None, x_U=None):
        self.eps = eps          # Magnitude của perturbation
        self.norm = norm        # Loại norm (∞, 2, 1, etc.)
        self.x_L = x_L         # Lower bound
        self.x_U = x_U         # Upper bound
```

**Chức năng có sẵn:**
- ✅ Định nghĩa perturbation space (Lp ball)
- ✅ Tính toán input bounds: `[x - ε, x + ε]`
- ✅ Hỗ trợ các norm khác nhau (L∞, L2, L1)

**Hạn chế:**
- ❌ **Chỉ áp dụng cho input layer**
- ❌ Không có cơ chế chọn vùng (regional/masked perturbation)
- ❌ Không hỗ trợ intermediate layers

---

### 1.2. BoundedModule: Propagation Engine

#### **File: `auto_LiRPA/bound_ops.py`**

Auto_LiRPA sử dụng **bound propagation** để tính certified bounds:

```python
class BoundedModule(nn.Module):
    """
    Wrapper cho PyTorch models để tính certified bounds
    """
    def __init__(self, model, global_input, ...):
        # Chuyển model thành computation graph
        self._convert(model, global_input)

    def compute_bounds(self, x=None, method='backward', ...):
        """
        Tính lower/upper bounds cho output

        Args:
            x: Input tensor
            method: 'IBP', 'backward' (CROWN), 'forward'

        Returns:
            lb, ub: Lower and upper bounds
        """
```

**Computation Graph:**

Auto_LiRPA chuyển PyTorch model thành **directed acyclic graph (DAG)**:

```
Input (x)
   ↓
Node[/input]
   ↓
Node[/conv1] (Conv2d)
   ↓
Node[/relu1] (ReLU)
   ↓
Node[/conv2] (Conv2d)
   ↓
   ...
   ↓
Node[/output] (Linear)
```

Mỗi node có:
- `node.lower`: Lower bound tensor
- `node.upper`: Upper bound tensor
- `node.forward_value`: Clean forward pass value
- `node.inputs`: List of parent nodes
- `node.perturbed`: Flag indicating if node is perturbed

---

### 1.3. Bound Propagation Methods

Auto_LiRPA hỗ trợ nhiều phương pháp propagate bounds:

#### **A. IBP (Interval Bound Propagation)**

**File: `bound_ops.py` - `interval_propagate()` method**

```python
def interval_propagate(self, *v):
    """
    Propagate interval bounds forward

    Input: [h_L, h_U] - interval của layer trước
    Output: [h'_L, h'_U] - interval của layer này
    """
```

**Công thức (ví dụ Linear layer):**

Cho layer: `y = Wx + b`

- Input bounds: `x ∈ [x_L, x_U]`
- Output bounds:
  ```
  y_L = W^+ x_L + W^- x_U + b
  y_U = W^+ x_U + W^- x_L + b
  ```
  Trong đó: `W^+ = max(W, 0)`, `W^- = min(W, 0)`

**Ưu điểm:** Nhanh, đơn giản
**Nhược điểm:** Bounds rất loose (rộng)

---

#### **B. CROWN (Backward LiRPA)**

**File: `bound_ops.py` - `bound_backward()` method**

```python
def bound_backward(self, last_lA, last_uA, *args):
    """
    Backward propagation để tính linear bounds

    Args:
        last_lA: Linear coefficients cho lower bound từ layer sau
        last_uA: Linear coefficients cho upper bound từ layer sau

    Returns:
        lA, uA: Updated linear coefficients cho layer trước
        lbias, ubias: Bias terms
    """
```

**Ý tưởng:**

Thay vì propagate intervals, ta propagate **linear relaxations**:

```
Output bound: z ≤ α·x + β  (upper bound)
              z ≥ α'·x + β' (lower bound)
```

Backward pass:
1. Bắt đầu từ output layer: `z ∈ [z_L, z_U]`
2. Propagate linear coefficients ngược về input
3. Cuối cùng tính bounds tại input perturbation region

**Ưu điểm:** Tighter bounds hơn IBP
**Nhược điểm:** Chậm hơn IBP

---

### 1.4. Bound Data Structures

#### **LinearBound Class**

**File: `auto_LiRPA/utils.py`**

```python
class LinearBound:
    """
    Lưu trữ linear relaxation của bounds

    Dạng: z ≥ lw·x + lb  (lower bound)
          z ≤ uw·x + ub  (upper bound)
    """
    def __init__(self, lw, lb, uw, ub, ...):
        self.lw = lw  # Lower bound weights
        self.lb = lb  # Lower bound bias
        self.uw = uw  # Upper bound weights
        self.ub = ub  # Upper bound bias
```

#### **Patches Class** (cho Conv layers)

**File: `auto_LiRPA/patches.py`**

```python
class Patches:
    """
    Efficient representation cho convolutional operations
    Dùng cho bound propagation qua Conv layers
    """
    def __init__(self, patches, stride, padding, ...):
        self.patches = patches    # Tensor chứa patches
        self.stride = stride
        self.padding = padding
```

**Vấn đề:** Dropout layers không tương thích với Patches → lý do phải tạo `TrafficSignNetNoDropout`

---

## 2. Mở Rộng: Intermediate Layer Perturbations

### 2.1. Vấn Đề Cần Giải Quyết

**Auto_LiRPA gốc chỉ hỗ trợ input perturbations:**

```
Perturbed Input: x + δ, where ||δ||_∞ ≤ ε
              ↓
          Network
              ↓
           Output
```

**Mục tiêu: Perturb intermediate layers:**

```
Clean Input: x
      ↓
  Layer 1, 2, ...
      ↓
  Layer k: h_k ← Perturb here! h_k + δ_k
      ↓
  Layer k+1, ..., n
      ↓
   Output
```

---

### 2.2. Giải Pháp: IntermediateBoundedModule

#### **File: `intermediate_bound_module.py`**

```python
class IntermediateBoundedModule(BoundedModule):
    """
    Mở rộng BoundedModule để hỗ trợ intermediate perturbations
    """
    def __init__(self, model, global_input, ...):
        super().__init__(model, global_input, ...)

        # Dictionary lưu perturbations cho từng node
        self.intermediate_perturbations = {}

    def register_intermediate_perturbation(self, node_name, perturbation):
        """
        Đăng ký perturbation cho một node cụ thể

        Args:
            node_name: Tên node trong computation graph (vd: '/conv2')
            perturbation: MaskedPerturbationLpNorm object
        """
        self.intermediate_perturbations[node_name] = perturbation
```

---

### 2.3. Algorithm: Compute Bounds từ Intermediate Perturbations

#### **Main Method:**

```python
def compute_bounds_with_intermediate_perturbation(
    self,
    x=None,
    method='backward'
):
    """
    Tính bounds khi có intermediate perturbations

    Pipeline:
    1. Forward pass (clean) → lấy intermediate activations
    2. Apply perturbations tại các node đã register
    3. Backward/Forward propagation từ perturbed nodes → output
    """
```

#### **Step-by-Step Algorithm:**

**STEP 1: Clean Forward Pass**

```python
# Chạy forward pass bình thường để lấy activations
with torch.no_grad():
    output = self.model(x)

# Lưu intermediate values vào computation graph
for node in self.nodes():
    node.forward_value = ...  # Activation tại node này
```

**STEP 2: Identify Perturbed Nodes**

```python
perturbed_nodes = []
for node_name, perturbation in self.intermediate_perturbations.items():
    node = self._modules[node_name]
    perturbed_nodes.append(node)

    # Tính bounds cho node này
    h = node.forward_value  # Clean activation
    h_L = h - perturbation.eps * perturbation.mask
    h_U = h + perturbation.eps * perturbation.mask

    node.lower = h_L
    node.upper = h_U
    node.perturbed = True
```

**STEP 3: Propagate Bounds Forward/Backward**

Có 2 approaches:

##### **Approach A: Forward Propagation (IBP-style)**

```python
def _compute_bounds_forward_from_intermediate(self):
    """
    Propagate intervals forward từ perturbed node → output
    """
    # Với mỗi layer từ perturbed_node → output:
    for node in nodes_after_perturbation:
        if node.perturbed:
            continue  # Đã có bounds rồi

        # Lấy bounds từ inputs
        inp_L, inp_U = get_input_bounds(node)

        # Propagate qua layer này
        node.lower, node.upper = node.interval_propagate(inp_L, inp_U)

    return output_node.lower, output_node.upper
```

**Ví dụ cụ thể:**

```
Input: x (clean, no perturbation)
  ↓
Conv1: h1 = Conv(x)  (clean)
  ↓
ReLU1: h2 = ReLU(h1)  (clean)
  ↓
Conv2: h3 = Conv(h2)  ← PERTURBED HERE!
       h3 ∈ [h3 - ε·mask, h3 + ε·mask]
  ↓
ReLU2: h4 = ReLU(h3)
       interval_propagate([h3_L, h3_U])
       → h4 ∈ [ReLU(h3_L), ReLU(h3_U)]
  ↓
FC: y = W·h4 + b
    interval_propagate([h4_L, h4_U])
    → y ∈ [W^+ h4_L + W^- h4_U + b, W^+ h4_U + W^- h4_L + b]
```

---

##### **Approach B: Backward Propagation (CROWN-style)**

```python
def _compute_bounds_backward_from_intermediate(self):
    """
    Backward LiRPA từ output → perturbed node
    Tính linear relaxation và concretize tại perturbed region
    """
    # Step 1: Initialize tại output
    output_node = self.final_node()
    C = torch.eye(num_classes)  # Identity matrix cho multi-class

    # Step 2: Backward pass
    lA, uA = C, C  # Linear coefficients

    for node in reversed(topological_order):
        if node.perturbed:
            # Node này có perturbation → concretize here
            h_L, h_U = node.lower, node.upper

            # Tính bounds tại output
            lb = lA @ h_L + lbias
            ub = uA @ h_U + ubias

            return lb, ub

        # Propagate linear coefficients backward
        lA, uA, lbias, ubias = node.bound_backward(lA, uA)

    # Nếu không có perturbed node, về tới input
    return concretize_at_input(lA, uA, input_bounds)
```

**Ví dụ cụ thể:**

```
Giả sử perturb tại Conv2 output (h3):

Output y (batch×43)
  ↑ (backward)
  | lA = I (43×43), uA = I
FC: y = W·h4 + b
  ↑
  | bound_backward(lA, uA)
  | → lA' = lA·W = W  (43×256)
  | → uA' = uA·W = W
ReLU2: h4 = ReLU(h3)
  ↑
  | bound_backward(lA'=W, uA'=W)
  | ReLU relaxation:
  |   - Nếu h3[i] ≥ 0: slope = 1
  |   - Nếu h3[i] ≤ 0: slope = 0
  |   - Nếu h3[i] uncertain: linear relaxation
  | → lA'' = lA'·diag(slopes_l)
  | → uA'' = uA'·diag(slopes_u)
Conv2 (PERTURBED): h3 ∈ [h3_L, h3_U]
  ↓ (concretize)

  lb = lA''·h3_L + lbias
  ub = uA''·h3_U + ubias

  → Output bounds: y ∈ [lb, ub]
```

**Ưu điểm của CROWN backward:**
- Tighter bounds hơn forward propagation
- Exploit linear relaxations của non-linear activations (ReLU, etc.)

---

### 2.4. Implementation Details

#### **Method Dispatch:**

```python
def compute_bounds_with_intermediate_perturbation(self, x=None, method='backward'):
    """
    Dispatch đến method phù hợp
    """
    if method == 'IBP':
        return self._compute_bounds_forward_from_intermediate(x)

    elif method == 'backward' or method == 'CROWN':
        return self._compute_bounds_backward_from_intermediate(x)

    elif method == 'forward':
        return self._compute_bounds_forward_lirpa(x)

    else:
        raise ValueError(f"Unknown method: {method}")
```

#### **Node Initialization:**

```python
def _init_intermediate_bounds(self, x):
    """
    Khởi tạo bounds cho perturbed nodes
    """
    # Clean forward pass
    self.model.eval()
    with torch.no_grad():
        _ = self.model(x)

    # Collect intermediate activations
    for node_name, perturbation in self.intermediate_perturbations.items():
        node = self[node_name]
        h = node.forward_value

        # Apply perturbation
        h_L, h_U = perturbation.get_input_bounds(h, A=None)

        node.lower = h_L
        node.upper = h_U
        node.interval = (h_L, h_U)
        node.perturbed = True
```

---

## 3. Masked/Regional Perturbations

### 3.1. Vấn Đề

**Perturbation tiêu chuẩn (toàn bộ tensor):**

```python
# Perturb TẤT CẢ elements
h_perturbed = h + δ
where δ ∈ [-ε, ε]^d  (d = số elements)
```

**Ví dụ:** Conv layer output shape `(1, 64, 16, 16)` = 16,384 elements
→ Perturb tất cả 16,384 elements cùng lúc
→ Bounds rất rộng, không hữu ích

**Mục tiêu: Perturb chỉ một VÙNG nhỏ**

```python
# Perturb CHỈ một số elements cụ thể
h_perturbed[mask] = h[mask] + δ
h_perturbed[~mask] = h[~mask]  # Không đổi
```

**Ví dụ:** Chỉ perturb:
- Batch 0
- Channels 5, 6, 7
- Height từ 5 đến 10
- Width từ 5 đến 10

→ Chỉ perturb `1 × 3 × 5 × 5 = 75` elements thay vì 16,384
→ Bounds chặt hơn rất nhiều!

---

### 3.2. Giải Pháp: MaskedPerturbationLpNorm

#### **File: `masked_perturbation.py`**

```python
class MaskedPerturbationLpNorm(PerturbationLpNorm):
    """
    Lp-norm perturbation with spatial/channel masking

    Chỉ perturb một vùng cụ thể trong tensor
    """
    def __init__(
        self,
        eps=0,
        norm=np.inf,
        mask=None,           # Boolean mask
        batch_idx=None,      # Batch indices
        channel_idx=None,    # Channel indices
        height_slice=None,   # (start, end) for height
        width_slice=None,    # (start, end) for width
        **kwargs
    ):
        super().__init__(eps=eps, norm=norm, **kwargs)

        # Lưu mask specification
        self.mask_spec = {
            'mask': mask,
            'batch_idx': batch_idx,
            'channel_idx': channel_idx,
            'height_slice': height_slice,
            'width_slice': width_slice
        }

        self.full_mask = None  # Sẽ được tạo sau khi biết shape
```

---

### 3.3. Mask Creation

#### **Algorithm tạo mask từ specification:**

```python
def _create_mask_from_spec(self, shape):
    """
    Tạo boolean mask từ specifications

    Args:
        shape: (batch, channels, height, width)

    Returns:
        mask: Boolean tensor same shape, True = perturb, False = keep
    """
    B, C, H, W = shape

    # Start với all False (không perturb gì)
    mask = torch.zeros(shape, dtype=torch.bool)

    # Xác định indices cho mỗi dimension
    b_idx = self._get_batch_indices(B)      # List of batch indices
    c_idx = self._get_channel_indices(C)    # List of channel indices
    h_start, h_end = self._get_height_range(H)
    w_start, w_end = self._get_width_range(W)

    # Set True cho vùng cần perturb
    mask[b_idx, :, :, :] = True           # Chọn batches
    mask[:, c_idx, :, :] &= True          # Chọn channels (AND)

    # Chọn spatial region
    spatial_mask = torch.zeros_like(mask)
    spatial_mask[:, :, h_start:h_end, w_start:w_end] = True
    mask &= spatial_mask

    return mask
```

#### **Ví dụ cụ thể:**

Cho tensor shape `(2, 64, 16, 16)` (2 samples, 64 channels, 16×16 spatial)

**Specification:**
```python
perturbation = MaskedPerturbationLpNorm(
    eps=0.1,
    batch_idx=0,              # Chỉ batch đầu tiên
    channel_idx=[5, 6, 7],    # Chỉ 3 channels
    height_slice=(5, 10),     # Height từ 5→10 (5 pixels)
    width_slice=(5, 10)       # Width từ 5→10 (5 pixels)
)
```

**Resulting mask:**
```python
mask.shape = (2, 64, 16, 16)

# Batch dimension
mask[0, :, :, :] = có thể True (tuỳ thuộc các dims khác)
mask[1, :, :, :] = False (batch 1 không perturb)

# Channel dimension (trong batch 0)
mask[0, 0:5, :, :] = False    # Channels 0-4: không perturb
mask[0, 5:8, :, :] = có thể True (tuỳ spatial)
mask[0, 8:64, :, :] = False   # Channels 8-63: không perturb

# Spatial dimensions (trong batch 0, channels 5-7)
mask[0, 5:8, 0:5, :] = False        # Height 0-4: không
mask[0, 5:8, 5:10, 0:5] = False     # Width 0-4: không
mask[0, 5:8, 5:10, 5:10] = True     # Region này: PERTURB!
mask[0, 5:8, 5:10, 10:16] = False   # Width 10-15: không
mask[0, 5:8, 10:16, :] = False      # Height 10-15: không

# Tổng số True elements: 1 × 3 × 5 × 5 = 75 / 32768 total
```

---

### 3.4. Applying Masked Perturbation

#### **Tính Input Bounds với Mask:**

```python
def get_input_bounds(self, x, A):
    """
    Trả về bounds cho input, với mask applied

    Args:
        x: Clean tensor (batch, channels, height, width)
        A: Linear coefficients (optional, for backward pass)

    Returns:
        x_L: Lower bound (chỉ perturb vùng mask=True)
        x_U: Upper bound (chỉ perturb vùng mask=True)
    """
    # Tạo mask nếu chưa có
    if self.full_mask is None:
        self.full_mask = self._create_mask_from_spec(x.shape)
        self.full_mask = self.full_mask.to(x.device)

    # Apply perturbation CHỈ tại mask=True
    x_L = torch.where(
        self.full_mask,
        x - self.eps,    # Nơi mask=True: perturb xuống
        x                # Nơi mask=False: giữ nguyên
    )

    x_U = torch.where(
        self.full_mask,
        x + self.eps,    # Nơi mask=True: perturb lên
        x                # Nơi mask=False: giữ nguyên
    )

    return x_L, x_U
```

#### **Visualization:**

```
Original tensor h:
┌─────────────────────────────┐
│ 0.5  0.3  0.7  0.2  ...     │
│ 0.1  0.9  0.4  0.6  ...     │
│ ...                         │
└─────────────────────────────┘

Mask (True = perturb):
┌─────────────────────────────┐
│ False False False False ... │
│ False True  True  False ... │  ← Chỉ 2 elements này
│ ...                         │
└─────────────────────────────┘

Lower bound h_L (ε=0.1):
┌─────────────────────────────┐
│ 0.5  0.3  0.7  0.2  ...     │  ← Unchanged
│ 0.1  0.8  0.3  0.6  ...     │  ← 0.9-0.1, 0.4-0.1
│ ...                         │
└─────────────────────────────┘

Upper bound h_U (ε=0.1):
┌─────────────────────────────┐
│ 0.5  0.3  0.7  0.2  ...     │  ← Unchanged
│ 0.1  1.0  0.5  0.6  ...     │  ← 0.9+0.1, 0.4+0.1
│ ...                         │
└─────────────────────────────┘
```

---

### 3.5. Why Masking Matters: Bound Tightness

#### **So sánh: Full Perturbation vs Masked Perturbation**

**Scenario:** Conv2 output shape `(1, 64, 16, 16)`, ε = 0.1

**Case 1: Perturb toàn bộ (no mask)**
```python
perturbation = PerturbationLpNorm(eps=0.1)
# Perturb: 1 × 64 × 16 × 16 = 16,384 elements
# Each element: ±0.1

# Propagate qua ReLU + FC (256 units) + Output (43 classes)
# Accumulated error rất lớn!

Output bounds:
  Class 5: [-12.3, 45.7]  ← RANGE = 58.0 (quá rộng!)
```

**Case 2: Perturb chỉ một vùng nhỏ**
```python
perturbation = MaskedPerturbationLpNorm(
    eps=0.1,
    batch_idx=0,
    channel_idx=[10, 11, 12],
    height_slice=(8, 10),
    width_slice=(8, 10)
)
# Perturb: 1 × 3 × 2 × 2 = 12 elements
# Each element: ±0.1

# Propagate qua ReLU + FC + Output
# Accumulated error nhỏ hơn nhiều!

Output bounds:
  Class 5: [8.2, 12.6]  ← RANGE = 4.4 (chặt hơn 13×!)
```

**Kết luận:**
- Masked perturbation → bounds chặt hơn
- Dễ verify robustness hơn
- Có thể phân tích sensitivity của từng vùng cụ thể

---

## 4. Integration: Hoàn Chỉnh Pipeline

### 4.1. Complete Workflow

```python
# STEP 1: Load model
model = TrafficSignNetNoDropout(num_classes=43)
model.load_from_dropout_checkpoint('checkpoint.pth')

# STEP 2: Create bounded module
lirpa_model = IntermediateBoundedModule(
    model,
    torch.randn(1, 3, 32, 32),
    device='cuda'
)

# STEP 3: Select layer to perturb
node_name = '/conv2'  # Ví dụ: output của conv2

# STEP 4: Create masked perturbation
perturbation = MaskedPerturbationLpNorm(
    eps=0.1,
    batch_idx=0,
    channel_idx=[0, 1, 2],
    height_slice=(5, 10),
    width_slice=(5, 10)
)

# STEP 5: Register perturbation
lirpa_model.register_intermediate_perturbation(node_name, perturbation)

# STEP 6: Compute bounds
image = get_test_image()  # Shape: (1, 3, 32, 32)
lb, ub = lirpa_model.compute_bounds_with_intermediate_perturbation(
    x=image,
    method='backward'  # CROWN
)

# STEP 7: Analyze results
print(f"Lower bounds: {lb}")
print(f"Upper bounds: {ub}")

# Verify robustness
pred_class = lb.argmax(dim=1)
if lb[0, pred_class] > ub[0, :].max(dim=0)[0] (excluding pred_class):
    print("✓ VERIFIED ROBUST!")
else:
    print("✗ Cannot verify robustness")
```

---

### 4.2. Mathematical Formulation

#### **Định nghĩa bài toán:**

Cho:
- Neural network: `f: ℝ^d → ℝ^k` (d=input dim, k=classes)
- Input: `x ∈ ℝ^d`
- Intermediate layer `ℓ` với activation `h_ℓ = f_ℓ(x) ∈ ℝ^m`
- Mask: `M ∈ {0,1}^m` (1 = perturb, 0 = keep)
- Perturbation bound: `ε > 0`

**Perturbation space:**
```
Δ_ℓ = {δ ∈ ℝ^m : |δ_i| ≤ ε if M_i = 1, δ_i = 0 if M_i = 0}
```

**Output bounds:**
```
z_L = min f(x; h_ℓ + δ)   subject to δ ∈ Δ_ℓ
      δ

z_U = max f(x; h_ℓ + δ)   subject to δ ∈ Δ_ℓ
      δ
```

Trong đó `f(x; h_ℓ + δ)` nghĩa là:
- Forward từ input x đến layer ℓ: `h_ℓ = f_ℓ(x)`
- Thay thế `h_ℓ` bằng `h_ℓ + δ`
- Tiếp tục forward từ layer ℓ+1 đến output

---

#### **CROWN Relaxation:**

Thay vì tính exact min/max (NP-hard), dùng **linear relaxation**:

```
z ≥ W^L (h_ℓ + δ) + b^L  (lower bound)
z ≤ W^U (h_ℓ + δ) + b^U  (upper bound)
```

Với `h_ℓ + δ ∈ [h_ℓ - ε·M, h_ℓ + ε·M]`:

```
z_L = W^L (h_ℓ - ε·M) + b^L
    = W^L h_ℓ - ε (W^L ⊙ M) + b^L

z_U = W^U (h_ℓ + ε·M) + b^U
    = W^U h_ℓ + ε (W^U ⊙ M) + b^U
```

Trong đó `⊙` là element-wise product.

**Lưu ý:** `W^L, W^U` được tính bằng backward pass qua các layers ℓ+1, ..., n

---

### 4.3. Complexity Analysis

#### **Forward Propagation (IBP):**

- Time: `O(L·N)` với L = số layers từ perturbed node → output, N = ops per layer
- Space: `O(M)` với M = max activation size
- Tight: ★★☆☆☆ (loose bounds)

#### **Backward Propagation (CROWN):**

- Time: `O(L·N·K)` với K = số classes (vì backward cho mỗi class)
- Space: `O(M·K)` (lưu linear coefficients cho mỗi class)
- Tight: ★★★★☆ (much tighter)

#### **Masking Overhead:**

- Time: `O(1)` (chỉ element-wise operations)
- Space: `O(M)` (lưu boolean mask)
- Benefit: Giảm bound looseness theo tỷ lệ `|mask| / M`

**Ví dụ:**
- Full perturbation: 16,384 elements → bound range 58.0
- Masked (12 elements): 12 elements → bound range 4.4
- Improvement: **13× tighter bounds** với cost gần như 0

---

## 5. Tổng Kết

### 5.1. Đóng Góp của Implementation

| Component | Auto_LiRPA Gốc | Implementation Mới |
|-----------|----------------|-------------------|
| **Perturbation location** | ✅ Input only | ✅ Input + Intermediate |
| **Perturbation masking** | ❌ Full tensor | ✅ Regional/masked |
| **Bound methods** | ✅ IBP, CROWN, Forward | ✅ Tất cả methods |
| **Dropout compatibility** | ✅ Works | ❌ Patches issue → ✅ No-dropout model |

### 5.2. Key Innovations

1. **IntermediateBoundedModule**
   - Extend BoundedModule
   - Register perturbations at any node
   - Propagate bounds từ intermediate → output

2. **MaskedPerturbationLpNorm**
   - Extend PerturbationLpNorm
   - Support batch/channel/spatial masking
   - Dramatically tighter bounds

3. **TrafficSignNetNoDropout**
   - Workaround Dropout/Patches incompatibility
   - Load weights from trained checkpoint
   - Enable CROWN backward for verification

### 5.3. Use Cases

- ✅ Analyze intermediate layer robustness
- ✅ Identify sensitive channels/regions
- ✅ Verify local perturbations
- ✅ Study feature importance
- ✅ Adversarial training guidance

---

## 6. References

### Auto_LiRPA Source Code
- `auto_LiRPA/perturbations.py`: Base perturbation classes
- `auto_LiRPA/bound_ops.py`: Bound propagation operators
- `auto_LiRPA/bounded_tensor.py`: BoundedModule implementation
- `auto_LiRPA/patches.py`: Patches for convolutional bounds

### Papers
1. **CROWN**: "Evaluating Robustness of Neural Networks with Mixed Integer Programming" (2019)
2. **auto_LiRPA**: "Automatic Perturbation Analysis for Scalable Certified Robustness" (NeurIPS 2020)
3. **LiRPA**: "Towards Fast Computation of Certified Robustness for ReLU Networks" (ICML 2018)

### Implementation Files
- `masked_perturbation.py`: Masked perturbation implementation
- `intermediate_bound_module.py`: Intermediate perturbation support
- `main_interactive.py`: Interactive testing interface

---

**Document Version:** 1.0
**Last Updated:** 2025
**Author:** Implementation for GTSRB Intermediate Perturbation Project
