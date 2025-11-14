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
    print("   ✓ Core libraries imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    exit(1)

try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import auto_LiRPA
    print("   ✓ auto_LiRPA imported successfully")
except ImportError as e:
    print(f"   ✗ auto_LiRPA import failed: {e}")
    print("   Please run: cd .. && pip install -e .")
    exit(1)

# Test 2: PyTorch version
print("\n2. Checking PyTorch...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("   Note: Running on CPU (training will be slower)")

# Test 3: Test model creation
print("\n3. Testing model creation...")
try:
    from traffic_sign_net import TrafficSignNet
    model = TrafficSignNet(num_classes=43)
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 43), f"Wrong output shape: {y.shape}"
    print("   ✓ TrafficSignNet works correctly")
    print(f"   Output shape: {y.shape}")
except Exception as e:
    print(f"   ✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
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

    perturbed_count = (bounds.lower != bounds.upper).sum().item()
    print("   ✓ Masked perturbation works correctly")
    print(f"   Perturbed elements: {perturbed_count}")
except Exception as e:
    print(f"   ✗ Perturbation test failed: {e}")
    import traceback
    traceback.print_exc()
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
            self.pool = nn.MaxPool2d(2)
            self.fc = nn.Linear(16*16*16, 10)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    simple_model = SimpleNet()
    dummy = torch.randn(1, 3, 32, 32)

    lirpa_model = IntermediateBoundedModule(simple_model, dummy)

    # Get layer names
    layers = lirpa_model.get_layer_names(['Conv', 'Linear'])
    print("   ✓ Intermediate bounded module works correctly")
    print(f"   Found {len(layers)} Conv/Linear layers")
except Exception as e:
    print(f"   ✗ Bounded module test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Test dataset loader (if dataset exists)
print("\n6. Testing dataset loader (optional)...")
try:
    import sys
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
        from gtsrb_dataset import GTSRBDataset, get_gtsrb_transforms

        test_dataset = GTSRBDataset(
            root_dir=data_dir,
            train=False,
            transform=get_gtsrb_transforms(train=False)
        )

        print(f"   ✓ Dataset loaded successfully")
        print(f"   Test samples: {len(test_dataset)}")
    else:
        print("   ⊘ Skipped (provide data_dir as argument to test)")
except Exception as e:
    print(f"   ⊘ Dataset test skipped or failed: {e}")

print("\n" + "="*70)
print("✓ All required tests passed! Installation is correct.")
print("="*70)
print("\nYou can now:")
print("  1. Download GTSRB dataset (see SETUP_GUIDE.md)")
print("  2. Train model: python train_gtsrb.py --data_dir /path/to/GTSRB")
print("  3. Run interactive testing")
print("\nFor detailed instructions, see SETUP_GUIDE.md")
