"""
Quick verification script to check if all missing methods are now present
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Force reload
import importlib
if 'intermediate_bound_module_v2' in sys.modules:
    del sys.modules['intermediate_bound_module_v2']

from intermediate_bound_module_v2 import IntermediateBoundedModuleV2

# List of required methods
required_methods = [
    'clear_intermediate_perturbations',
    'compute_bounds_with_intermediate_perturbation',
    'get_layer_names',
    'get_node_by_name',
    'register_intermediate_perturbation',
    'compute_perturbed_bounds',
    'print_model_structure',
]

print("="*80)
print("V2 COMPATIBILITY VERIFICATION")
print("="*80)

all_ok = True
for method in required_methods:
    has_method = hasattr(IntermediateBoundedModuleV2, method)
    status = "✅" if has_method else "❌"
    print(f"{status} {method}")
    if not has_method:
        all_ok = False

print("="*80)
if all_ok:
    print("✅ ALL REQUIRED METHODS PRESENT!")
    print("\nV2 module is now compatible with main_interactive.py and sca_au.py")
else:
    print("❌ SOME METHODS STILL MISSING!")
    print("\nPlease add the missing methods to V2 module")

print("="*80)
