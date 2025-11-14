# Fault Report Export Guide

## Tính năng mới: Export Detailed Fault Report

Ngoài visualization bằng matplotlib, simulator hiện tại đã được trang bị thêm khả năng xuất **báo cáo chi tiết** ra file text (.txt), ghi rõ các vùng bị ảnh hưởng trong tensor đầu ra.

---

## Cách sử dụng

### 1. Interactive Mode (Tự động)

Khi chạy simulator ở chế độ interactive, file report sẽ được tự động tạo:

```bash
python fault_simulator.py
```

**Output:**
```
[Step 4] Running Simulation...
[Step 5] Generating Visualization...
[Step 6] Exporting Detailed Report...

================================================================================
Simulation completed!
  Visualization: fault_impact_conv1.png
  Detailed Report: fault_report_conv1.txt
================================================================================
```

### 2. Programmatic API

Nếu dùng API trực tiếp, gọi method `export_fault_report()`:

```python
from fault_simulator import SystolicFaultSimulator, FaultModel

# Create simulator
sim = SystolicFaultSimulator(8, 8, 'OS')

# Define layer and faults
layer_config = {...}
faults = [...]

# Run simulation
results = sim.simulate_layer(layer_config, faults)

# Export report
sim.export_fault_report(results, 'my_fault_report.txt')
```

---

## Nội dung File Report

### Header Section

```
================================================================================
                    SYSTOLIC FAULT SIMULATION REPORT
================================================================================

Layer: conv1 (Conv)
Dataflow: OS
Array Size: 8×8

--------------------------------------------------------------------------------

STATISTICS:
  Total outputs: 32768
  Affected outputs: 28672
  Fault coverage: 87.50%
  Number of faults: 2
  Affected addresses: 145

--------------------------------------------------------------------------------
```

### Per-Channel Impact (Conv Layers)

Với mỗi channel bị ảnh hưởng, báo cáo sẽ hiển thị:

```
Channel 0:
  Affected pixels: 896/1024 (87.50%)
  Bounding box: rows [0, 31], cols [0, 31]
  Box size: 32×32
  Affected coordinates:
    Row 0: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    Row 1: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    Row 2: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    Row 3: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    Row 4: cols [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
    ...
```

**Thông tin gồm:**
- **Affected pixels**: Số pixel và tỷ lệ bị ảnh hưởng
- **Bounding box**: Vùng chữ nhật bao quanh các pixel bị lỗi
- **Box size**: Kích thước bounding box
- **Affected coordinates**: Danh sách chi tiết tọa độ bị ảnh hưởng, nhóm theo hàng

### Spatial Position Impact

Phần này phân tích theo không gian - mỗi vị trí (row, col) ảnh hưởng đến bao nhiêu channels:

```
================================================================================

Spatial Position Impact (across all channels):
--------------------------------------------------------------------------------
  Position (0, 0): 32/32 channels affected (100.00%)
  Position (0, 1): 32/32 channels affected (100.00%)
  Position (0, 2): 32/32 channels affected (100.00%)
  Position (0, 3): 32/32 channels affected (100.00%)
  Position (0, 8): 32/32 channels affected (100.00%)
  Position (0, 9): 32/32 channels affected (100.00%)
  ...
```

### FC Layer Report

Với Fully Connected layers, format sẽ khác:

```
AFFECTED NEURONS (FC Layer):

Total neurons: 512
Affected neurons: 245

Affected Neuron Indices:
--------------------------------------------------------------------------------
  Neurons 0-15 (16 neurons)
  Neuron 24
  Neuron 25
  Neurons 32-47 (16 neurons)
  ...
```

---

## Ví dụ Sử dụng

### Test Script

File `test_export_report.py` đã được tạo sẵn để demo:

```python
"""Test script to demonstrate fault report export"""

from fault_simulator import SystolicFaultSimulator, FaultModel

def test_report_export():
    # Create simulator
    simulator = SystolicFaultSimulator(8, 8, 'OS')

    # Define layer
    layer_config = {
        'name': 'test_conv',
        'type': 'Conv',
        'input_channels': 3,
        'output_channels': 32,
        'input_size': (32, 32),
        'kernel_size': (3, 3),
        'stride': 1,
        'padding': 1
    }

    # Create 2 faults
    faults = [
        FaultModel(
            fault_type=FaultModel.STUCK_AT_0,
            fault_location={
                'pe_row': 2,
                'pe_col': 3,
                'component': 'accumulator_register'
            },
            fault_timing={'start_cycle': 0, 'duration': float('inf')}
        ),
        FaultModel(
            fault_type=FaultModel.BIT_FLIP,
            fault_location={
                'pe_row': 4,
                'pe_col': 5,
                'component': 'MAC_unit'
            },
            fault_timing={'start_cycle': 0, 'duration': float('inf')}
        )
    ]

    # Run simulation
    results = simulator.simulate_layer(layer_config, faults)

    # Export both visualization and report
    simulator.visualize_results(results, 'test_fault_impact.png')
    simulator.export_fault_report(results, 'test_fault_report.txt')

    print("Generated files:")
    print("  - test_fault_impact.png")
    print("  - test_fault_report.txt")

if __name__ == '__main__':
    test_report_export()
```

**Chạy:**
```bash
python test_export_report.py
```

---

## Lợi ích của Text Report

### 1. Machine-Readable
- Dễ parse bằng scripts khác
- Có thể tích hợp vào pipeline tự động
- Format nhất quán, dễ extract thông tin

### 2. Detailed Analysis
- Chi tiết hơn matplotlib visualization
- Liệt kê chính xác tọa độ từng pixel
- Phân tích theo cả channel và spatial position

### 3. Documentation
- Lưu trữ kết quả simulation
- So sánh giữa các lần chạy
- Audit trail cho experiments

### 4. Integration
- Có thể đưa vào verification pipeline
- Phân tích thống kê batch simulations
- Input cho các tools phân tích downstream

---

## Use Cases

### 1. Fault Pattern Analysis
Phân tích pattern của fault propagation:
```bash
# Run multiple simulations with different faults
for pe in range(64):
    # Inject fault at different PEs
    # Compare reports to see pattern
```

### 2. Dataflow Comparison
So sánh fault impact giữa các dataflows:
```python
for dataflow in ['OS', 'WS', 'IS']:
    sim = SystolicFaultSimulator(8, 8, dataflow)
    results = sim.simulate_layer(layer_config, faults)
    sim.export_fault_report(results, f'report_{dataflow}.txt')
```

### 3. Critical Region Identification
Tìm vùng nhạy cảm nhất với faults:
```python
# Parse multiple reports
# Extract bounding boxes
# Identify most frequently affected regions
```

### 4. Verification Input
Sử dụng affected coordinates làm input cho formal verification:
```python
# Read affected coordinates from report
# Generate verification constraints
# Run auto_LiRPA with constraints
```

---

## Format Details

### Conv Layer Format

```
Channel {ch}:
  Affected pixels: {count}/{total} ({percentage}%)
  Bounding box: rows [{min_row}, {max_row}], cols [{min_col}, {max_col}]
  Box size: {height}×{width}
  Affected coordinates:
    Row {r}: cols [{c1}, {c2}, ...]
```

### FC Layer Format

```
Affected Neuron Indices:
  Neuron {idx}          # Single neuron
  Neurons {start}-{end} ({count} neurons)  # Range
```

---

## Tích hợp với Verification Pipeline

### Example Workflow

1. **Run Fault Simulation**
   ```bash
   python fault_simulator.py
   # → fault_report_conv1.txt
   ```

2. **Parse Report**
   ```python
   def parse_affected_regions(report_path):
       # Extract affected coordinates
       # Return structured data
   ```

3. **Generate Verification Constraints**
   ```python
   affected_coords = parse_affected_regions('fault_report_conv1.txt')
   constraints = generate_constraints(affected_coords)
   ```

4. **Run Verification**
   ```python
   # Use auto_LiRPA with constraints
   verify_robustness(model, constraints)
   ```

---

## Troubleshooting

### Q: Report file quá lớn?
**A:** Với layers lớn và nhiều faults, report có thể rất dài. Có thể:
- Filter chỉ channels có nhiều ảnh hưởng
- Summarize affected regions thành ranges
- Tạo separate reports cho từng channel

### Q: Cần format khác (JSON, CSV)?
**A:** Hiện tại chỉ hỗ trợ text. Có thể extend:
```python
def export_fault_report_json(self, results, save_path):
    # Convert to JSON format
    pass
```

### Q: Report không có spatial position impact?
**A:** Kiểm tra:
- Layer type phải là Conv
- Spatial dimension phải square (để reshape)
- Có ít nhất 1 position bị ảnh hưởng

---

**Version:** 1.0
**Last Updated:** 2025-01-10
**Author:** Claude (Systolic Fault Simulator)
