#!/usr/bin/env python3
"""
Quick test to verify TrafficSignNetNoDropout works correctly
"""
import torch
from traffic_sign_net import TrafficSignNet, TrafficSignNetNoDropout

def test_nodropout_model():
    print("="*80)
    print("Testing TrafficSignNetNoDropout")
    print("="*80)

    # Create dummy input
    x = torch.randn(2, 3, 32, 32)

    # Test 1: Create both models
    print("\n[Test 1] Creating models...")
    model_with_dropout = TrafficSignNet(num_classes=43)
    model_no_dropout = TrafficSignNetNoDropout(num_classes=43)
    print("✓ Both models created successfully")

    # Test 2: Forward pass
    print("\n[Test 2] Testing forward pass...")
    model_with_dropout.eval()
    model_no_dropout.eval()

    with torch.no_grad():
        out1 = model_with_dropout(x)
        out2 = model_no_dropout(x)

    print(f"  Output shape (with dropout): {out1.shape}")
    print(f"  Output shape (no dropout):   {out2.shape}")
    assert out1.shape == out2.shape == (2, 43), "Output shape mismatch!"
    print("✓ Forward pass successful")

    # Test 3: Check layer structure
    print("\n[Test 3] Comparing layer structures...")
    with_dropout_layers = [name for name, _ in model_with_dropout.named_modules() if 'dropout' in name.lower()]
    no_dropout_layers = [name for name, _ in model_no_dropout.named_modules() if 'dropout' in name.lower()]

    print(f"  Dropout layers in TrafficSignNet: {len(with_dropout_layers)}")
    print(f"  Dropout layers in TrafficSignNetNoDropout: {len(no_dropout_layers)}")
    assert len(with_dropout_layers) > 0, "Original model should have dropout!"
    assert len(no_dropout_layers) == 0, "No-dropout model should have NO dropout!"
    print("✓ Layer structure correct")

    # Test 4: Load from checkpoint (if exists)
    import os
    checkpoint_path = "checkpoints/traffic_sign_net_full.pth"
    if os.path.exists(checkpoint_path):
        print(f"\n[Test 4] Loading from checkpoint: {checkpoint_path}")
        model_no_dropout_loaded = TrafficSignNetNoDropout(num_classes=43)
        model_no_dropout_loaded.load_from_dropout_checkpoint(checkpoint_path)
        model_no_dropout_loaded.eval()

        with torch.no_grad():
            out3 = model_no_dropout_loaded(x)

        print(f"  Output shape: {out3.shape}")
        print(f"  Logits range: [{out3.min():.3f}, {out3.max():.3f}]")
        print("✓ Checkpoint loading successful")
    else:
        print(f"\n[Test 4] Checkpoint not found: {checkpoint_path}")
        print("  (Skipping checkpoint test)")

    # Test 5: get_layer_info
    print("\n[Test 5] Testing get_layer_info()...")
    layer_info = model_no_dropout.get_layer_info()
    print(f"  Number of layers: {len(layer_info)}")
    for i, (name, ltype, _, shape) in enumerate(layer_info[:3]):
        print(f"    {i}: {name:15s} {ltype:10s} {shape}")
    print("✓ get_layer_info() works")

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80)

if __name__ == '__main__':
    test_nodropout_model()
