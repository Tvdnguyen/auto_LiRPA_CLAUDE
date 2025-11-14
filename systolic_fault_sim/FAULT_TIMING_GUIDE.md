# Fault Timing Guide - Systolic Fault Simulator

## Tổng quan

Document này giải thích chi tiết về **timing của faults** trong systolic array và cách chọn parameters để faults có tác động lên outputs.

---

## Các Phase trong Systolic Array

Mỗi dataflow có các phases khác nhau, và faults chỉ có tác động khi xuất hiện trong **computation phase**.

### 1. Output Stationary (OS)

```
┌──────────────────────────────────────────────────────────┐
│  Phase 1: Accumulation (T cycles)                        │
│  Cycles 0 → T-1                                          │
│  - Each PE performs T MAC operations                     │
│  - Partial sums accumulate in PE registers               │
│  ✅ FAULTS HERE HAVE IMPACT                              │
├──────────────────────────────────────────────────────────┤
│  Phase 2: Output Drain (W-1 cycles)                      │
│  Cycles T → T+W-2                                        │
│  - Outputs flow out column by column                     │
│  ✅ FAULTS HERE MAY HAVE IMPACT (if they affect drain)   │
└──────────────────────────────────────────────────────────┘

Where:
  T = kernel_size² × input_channels
  W = array_width

Example (3×3 kernel, 3 input channels, 8×8 array):
  T = 27 cycles
  W = 8

  Accumulation: cycles 0-26
  Output drain: cycles 27-33

✅ Critical Range: cycles 0-26 (ANY fault here affects output)
```

**Ví dụ cụ thể:**

| Fault Timing | Impact? | Lý do |
|--------------|---------|-------|
| Start: 0, Duration: 27 (permanent) | ✅ YES | Covers entire accumulation |
| Start: 0, Duration: 5 | ✅ YES | Affects 5/27 accumulations |
| Start: 10, Duration: 2 | ✅ YES | Affects 2/27 accumulations |
| Start: 25, Duration: 1 | ✅ YES | Affects last accumulation |
| Start: 30, Duration: 5 | ⚠️ MAYBE | During drain, may affect output write |

**Khuyến nghị:**
- **Minimum start**: 0 (computation starts immediately)
- **Minimum duration**: 1 cycle (even 1 faulty MAC affects output)
- **Maximum effective duration**: T cycles (27 for this example)

---

### 2. Weight Stationary (WS)

```
┌──────────────────────────────────────────────────────────┐
│  Phase 1: Weight Loading (H cycles)                      │
│  Cycles 0 → H-1                                          │
│  - Weights loaded into PEs row by row                    │
│  ❌ FAULTS HERE USUALLY HAVE NO IMPACT                   │
│  (Unless they permanently corrupt weight registers)      │
├──────────────────────────────────────────────────────────┤
│  Phase 2: Input Streaming (Sr cycles)                    │
│  Cycles H → H+Sr-1                                       │
│  - Inputs stream through, MAC operations happen          │
│  ✅ FAULTS HERE HAVE IMPACT                              │
├──────────────────────────────────────────────────────────┤
│  Phase 3: Output Drain (W-1 cycles)                      │
│  Cycles H+Sr → H+Sr+W-2                                  │
│  - Outputs drain horizontally                            │
│  ✅ FAULTS HERE MAY HAVE IMPACT                          │
└──────────────────────────────────────────────────────────┘

Where:
  H = array_height
  Sr = ofmap_pixels (spatial dimension)
  W = array_width

Example (8×8 spatial output, 8×8 array):
  H = 8 cycles
  Sr = 64 cycles
  W = 8

  Weight loading: cycles 0-7
  Input streaming: cycles 8-71
  Output drain: cycles 72-78

✅ Critical Range: cycles 8-71 (faults here affect outputs)
❌ No Impact: cycles 0-7 (weight loading, usually no impact)
```

**Ví dụ cụ thể:**

| Fault Timing | Impact? | Lý do |
|--------------|---------|-------|
| Start: 1, Duration: 4 | ❌ NO | Ends at cycle 5, before computation (cycle 8) |
| Start: 5, Duration: 10 | ✅ YES | Spans cycles 5-15, overlaps computation (8-15) |
| Start: 10, Duration: 20 | ✅ YES | Fully within computation phase |
| Start: 65, Duration: 10 | ✅ YES | Near end of computation + drain |
| Start: 75, Duration: 5 | ⚠️ MAYBE | Only affects drain phase |

**Khuyến nghị:**
- **Minimum start**: H (8 for this example) - start of computation
- **Minimum duration**: 1 cycle
- **Maximum effective duration**: Sr cycles (64 for this example)
- **⚠️ WARNING**: Faults ending before cycle H will have NO impact!

---

### 3. Input Stationary (IS)

```
┌──────────────────────────────────────────────────────────┐
│  Phase 1: Input Loading (H cycles)                       │
│  Cycles 0 → H-1                                          │
│  - Input activations loaded into PEs                     │
│  ❌ FAULTS HERE USUALLY HAVE NO IMPACT                   │
├──────────────────────────────────────────────────────────┤
│  Phase 2: Weight Streaming (T cycles)                    │
│  Cycles H → H+T-1                                        │
│  - Weights stream through, MAC operations happen         │
│  ✅ FAULTS HERE HAVE IMPACT                              │
├──────────────────────────────────────────────────────────┤
│  Phase 3: Output Drain (H-1 cycles)                      │
│  Cycles H+T → H+T+H-2                                    │
│  - Outputs drain vertically                              │
│  ✅ FAULTS HERE MAY HAVE IMPACT                          │
└──────────────────────────────────────────────────────────┘

Where:
  H = array_height
  T = kernel_size² × input_channels

Example (3×3 kernel, 3 input channels, 8×8 array):
  H = 8 cycles
  T = 27 cycles

  Input loading: cycles 0-7
  Weight streaming: cycles 8-34
  Output drain: cycles 35-41

✅ Critical Range: cycles 8-34 (faults here affect outputs)
❌ No Impact: cycles 0-7 (input loading, usually no impact)
```

**Ví dụ cụ thể:**

| Fault Timing | Impact? | Lý do |
|--------------|---------|-------|
| Start: 1, Duration: 5 | ❌ NO | Ends at cycle 6, before computation (cycle 8) |
| Start: 7, Duration: 5 | ✅ YES | Spans cycles 7-12, overlaps computation (8-12) |
| Start: 10, Duration: 10 | ✅ YES | Fully within computation phase |
| Start: 30, Duration: 10 | ✅ YES | End of computation + drain |

**Khuyến nghị:**
- **Minimum start**: H (8 for this example) - start of computation
- **Minimum duration**: 1 cycle
- **Maximum effective duration**: T cycles (27 for this example)

---

## Bảng Tổng hợp Quick Reference

### Conv Layer: 3×3 kernel, 3 input channels, 32×32 output, 8×8 array

| Dataflow | Weight/Input Load | Computation Phase | Output Drain | Critical Cycles |
|----------|-------------------|-------------------|--------------|-----------------|
| **OS** | N/A | Cycles 0-26 (27 cycles) | Cycles 27-33 | **0-26** |
| **WS** | Cycles 0-7 | Cycles 8-71 (64 cycles) | Cycles 72-78 | **8-71** |
| **IS** | Cycles 0-7 | Cycles 8-34 (27 cycles) | Cycles 35-41 | **8-34** |

### Conv Layer: 3×3 kernel, 64 input channels, 8×8 output, 8×8 array

| Dataflow | Weight/Input Load | Computation Phase | Output Drain | Critical Cycles |
|----------|-------------------|-------------------|--------------|-----------------|
| **OS** | N/A | Cycles 0-575 (576 cycles) | Cycles 576-582 | **0-575** |
| **WS** | Cycles 0-7 | Cycles 8-71 (64 cycles) | Cycles 72-78 | **8-71** |
| **IS** | Cycles 0-7 | Cycles 8-583 (576 cycles) | Cycles 584-590 | **8-583** |

---

## Các Tình huống Thường gặp

### Tình huống 1: "Fault của tôi không có impact gì cả!"

**Nguyên nhân thường gặp:**
1. ✅ Fault timing quá sớm (kết thúc trước computation phase)
2. ✅ Fault timing quá muộn (sau khi computation kết thúc)
3. ✅ Fault duration quá ngắn và không trùng với computation cycles

**Cách khắc phục:**
```
WS/IS Dataflow:
- Start cycle >= H (array_height)
- Duration >= 1
- Ensure: start < (H + computation_cycles)

OS Dataflow:
- Start cycle >= 0
- Duration >= 1
- Ensure: start < T (total accumulation cycles)
```

### Tình huống 2: "Tôi muốn test transient fault ngắn"

**Ví dụ: Fault chỉ 1 cycle trong computation**

```python
# OS Dataflow - 1 cycle fault
fault_timing = {
    'start_cycle': 10,  # Middle of accumulation
    'duration': 1        # Just 1 cycle
}
# Expected: Still affects output (corrupts 1/27 MACs)

# WS Dataflow - 1 cycle fault
fault_timing = {
    'start_cycle': 20,  # During input streaming (after cycle 8)
    'duration': 1        # Just 1 cycle
}
# Expected: Affects some outputs (corrupts 1 spatial position)
```

### Tình huống 3: "Tôi muốn test worst-case scenario"

**Ví dụ: Permanent fault suốt quá trình**

```python
fault_timing = {
    'start_cycle': 0,
    'duration': float('inf')  # Permanent
}
# Expected: Maximum impact on outputs
```

### Tình huống 4: "Tôi muốn simulate radiation-induced transient fault"

**Typical radiation fault: Randomly appears, short duration**

```python
import random

# Random timing within computation phase
start_cycle = random.randint(comp_start, comp_end - 10)
duration = random.randint(1, 5)  # 1-5 cycles

fault_timing = {
    'start_cycle': start_cycle,
    'duration': duration
}

# Test multiple times with different timings to get statistics
```

---

## Công thức Tính toán

### Given Layer Parameters:
```python
# Conv layer
input_channels = C_in
output_channels = C_out
kernel_height = K_h
kernel_width = K_w
output_height = H_out
output_width = W_out

# Array
array_height = H
array_width = W
```

### Cycle Calculations:

#### OS Dataflow:
```python
T = K_h * K_w * C_in  # Accumulation cycles
accumulation_phase = (0, T-1)
drain_phase = (T, T + W - 2)

# Critical cycles: 0 to T-1
```

#### WS Dataflow:
```python
Sr = H_out * W_out  # Spatial dimension
weight_load = H
input_stream = Sr
drain = W - 1

loading_phase = (0, H-1)
computation_phase = (H, H + Sr - 1)
drain_phase = (H + Sr, H + Sr + W - 2)

# Critical cycles: H to H+Sr-1
```

#### IS Dataflow:
```python
T = K_h * K_w * C_in
input_load = H
weight_stream = T
drain = H - 1

loading_phase = (0, H-1)
computation_phase = (H, H + T - 1)
drain_phase = (H + T, H + T + H - 2)

# Critical cycles: H to H+T-1
```

---

## Khuyến nghị Chung

### ✅ Best Practices:

1. **Luôn kiểm tra computation phase trước khi chọn timing**
   - Simulator sẽ hiển thị cycle ranges khi bạn chọn transient fault
   - Chú ý warnings nếu fault quá sớm

2. **Test với permanent fault trước**
   - Verify rằng fault propagation logic hoạt động đúng
   - Nếu permanent fault không có impact → có bug trong simulator

3. **Sau đó test transient faults**
   - Bắt đầu với duration dài (10-20 cycles)
   - Giảm dần duration để tìm minimum impact threshold

4. **Test multiple faults**
   - Verify rằng fault impact accumulates correctly
   - Check spatial distribution of affected outputs

### ❌ Common Mistakes:

1. **Chọn start cycle < computation start** (cho WS/IS)
   - Fault ends before computation begins
   - No impact on outputs

2. **Duration quá ngắn ở dataflow yêu cầu nhiều cycles**
   - 1 cycle fault trong 576-cycle accumulation: impact rất nhỏ
   - May not be statistically significant

3. **Không verify với visualization**
   - Always check matplotlib output
   - Verify spatial pattern matches expected dataflow behavior

---

## Debug Checklist

Nếu fault không có impact như mong đợi, check:

- [ ] Fault timing có nằm trong computation phase không?
- [ ] PE coordinates có hợp lệ không? (0 ≤ row < H, 0 ≤ col < W)
- [ ] Component type có phù hợp với fault type không?
- [ ] Dataflow có được chọn đúng không?
- [ ] Layer dimensions có chính xác không?
- [ ] Có chạy test với permanent fault để verify logic chưa?

---

## Ví dụ Interactive Session

```
[Step 3] Define Faults
Enter faulty PE (row,col) or 'done': 2,3
  Select component (1-6) [default: 6]: 6

  Fault Types:
    1. Stuck-at-0
    2. Stuck-at-1
    3. Bit-flip (random)
    4. Permanent (default)
  Select fault type (1-4) [default: 4]: 3

  Fault Duration:
    1. Permanent (active entire simulation)
    2. Transient (time-bounded)
  Select duration (1-2) [default: 1]: 2

    💡 Computation Cycle Info (WS dataflow):
    Weight load: cycles 0-7, Input stream: cycles 8-71, Drain: cycles 72-78
    ⚠️  Faults before cycle 8 may have NO impact!
    ✅ Faults during cycles 8-71 WILL have impact

    Start cycle: 10        ← Good! Within computation
    Duration (cycles): 20  ← Good! Reasonable duration

  → Added bit_flip fault at PE (2, 3), entire_PE
```

---

**Version:** 2.0
**Last Updated:** 2025-01-10
**Author:** Claude (Auto-LiRPA Systolic Fault Simulator)
