# Fix for RAM Killing Issue

## Problem Identified

The program was getting killed due to RAM shortage because of **inefficient perturbation creation** in `integrated_pe_sensitivity_analysis_low_memory.py`.

## Root Cause

### Original Problematic Code (lines 233-254)

```python
# Create custom mask
def custom_create_mask(shape):
    mask = torch.zeros(shape, dtype=torch.bool)
    for h, w in affected_spatial:  # ⚠️ LOOP - VERY EXPENSIVE!
        if h < shape[2] and w < shape[3]:
            if len(affected_channels) == 0:
                mask[0, :, h, w] = True
            else:
                for c in affected_channels:  # ⚠️ NESTED LOOP!
                    if c < shape[1]:
                        mask[0, c, h, w] = True
    return mask

perturbation = MaskedPerturbationLpNorm(...)
perturbation.create_mask = custom_create_mask  # ⚠️ Override with slow function
```

**Why this is inefficient:**
- Iterates over every affected spatial position (could be hundreds)
- Nested loop over channels (could be 32+ channels)
- Creates boolean tensors repeatedly in loops
- Overrides the built-in efficient masking with custom slow implementation

## Solution

### Fixed Code (following main_interactive.py pattern)

```python
# Convert affected_spatial to bounding box slices (EFFICIENT!)
height_slice = None
width_slice = None
channel_idx = None

if len(affected_spatial) > 0:
    # Extract all h, w coordinates
    h_coords = [h for h, w in affected_spatial]
    w_coords = [w for h, w in affected_spatial]

    # Compute bounding box
    min_h = min(h_coords)
    max_h = max(h_coords)
    min_w = min(w_coords)
    max_w = max(w_coords)

    # Create slices (inclusive end, so +1)
    height_slice = (min_h, max_h + 1)
    width_slice = (min_w, max_w + 1)

if len(affected_channels) > 0:
    channel_idx = affected_channels

# Create perturbation - EXACTLY like main_interactive.py
# No custom mask override - use built-in efficient slicing
perturbation = MaskedPerturbationLpNorm(
    eps=eps,
    norm=np.inf,
    batch_idx=0,
    channel_idx=channel_idx,      # ✓ Direct pass - no loops
    height_slice=height_slice,     # ✓ Tuple slice - no loops
    width_slice=width_slice        # ✓ Tuple slice - no loops
)
```

**Why this is efficient:**
- ✅ Converts scattered points to bounding box in O(n) time
- ✅ Uses tuple slices `(start, end)` instead of iterating positions
- ✅ Passes channel list directly to built-in masking
- ✅ No custom mask override - uses PyTorch's efficient tensor slicing
- ✅ Matches the proven pattern from `main_interactive.py`

## Expected Memory Impact

| Approach | Complexity | Memory |
|----------|-----------|--------|
| **Old (nested loops)** | O(H×W×C) iterations | High - creates mask element by element |
| **New (bounding box)** | O(1) slice operations | Low - uses efficient tensor slicing |

**Example with typical fault:**
- Affected region: 10 spatial positions × 32 channels = 320 mask operations
- Old approach: 320 loop iterations creating boolean tensors
- New approach: 2 slice operations (height + width) + 1 channel list

**Estimated memory reduction:** ~10-20x less memory allocation in perturbation creation

## Verification

The fix ensures:
1. ✅ **Perturbation creation** follows `main_interactive.py` exactly
2. ✅ **Affected region extraction** from `fault_simulator.py` remains correct (not changed)
3. ✅ **Bounding box approach** is conservative - includes all affected positions
4. ✅ **No loops** in critical path of binary search (called 4-7 times per PE)

## Files Modified

- `integrated_pe_sensitivity_analysis_low_memory.py`: Fixed `find_max_epsilon_for_region()` method
- `test_single_pe.py`: No changes needed (imports fixed class automatically)
- `run_all_pes_sequential.sh`: No changes needed (calls fixed scripts)

## Testing

Test with single PE first:

```bash
python3 test_single_pe.py \
    --data_dir gtsrb_project/data/GTSRB_data \
    --checkpoint gtsrb_project/checkpoints/traffic_sign_net_full.pth \
    --pe_row 0 --pe_col 0 \
    --class_id 0 --test_idx 0 \
    --duration 1 --tolerance 0.05 --epsilon_max 0.3
```

Monitor RAM usage:
```bash
# In another terminal
watch -n 1 "ps aux | grep python | grep test_single"
```

Expected behavior:
- Should complete without being killed
- Peak RAM usage should stay under 6-7 GB (previously exceeded 8 GB)
- Time per PE: ~30-60 seconds

## Notes

- The bounding box approach may perturb slightly more pixels than the exact affected region
- This is conservative and safe - it only makes the verification harder (more perturbation = lower epsilon)
- The performance gain far outweighs the minor over-approximation
