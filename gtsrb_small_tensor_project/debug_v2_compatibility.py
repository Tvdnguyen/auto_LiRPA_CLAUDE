"""
Debug script to check compatibility between V2 module and main_interactive.py

This script will:
1. Find all method calls to lirpa_model in main_interactive.py
2. Check if V2 module has those methods
3. Report missing methods
4. Suggest fixes
"""

import sys
import os
import re
import ast
from typing import Set, List, Tuple

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Force reload V2
import importlib
if 'intermediate_bound_module_v2' in sys.modules:
    del sys.modules['intermediate_bound_module_v2']

from intermediate_bound_module_v2 import IntermediateBoundedModuleV2


def extract_method_calls(file_path: str) -> Set[str]:
    """
    Extract all method calls to self.lirpa_model from a Python file

    Returns:
        Set of method names
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Pattern: self.lirpa_model.method_name(
    pattern = r'self\.lirpa_model\.(\w+)\('
    matches = re.findall(pattern, content)

    return set(matches)


def check_v2_methods() -> Set[str]:
    """
    Get all public methods from V2 module

    Returns:
        Set of method names
    """
    methods = set()
    for attr in dir(IntermediateBoundedModuleV2):
        if not attr.startswith('_') and callable(getattr(IntermediateBoundedModuleV2, attr)):
            methods.add(attr)
    return methods


def main():
    print("="*80)
    print("V2 COMPATIBILITY DEBUG REPORT")
    print("="*80)

    # Files to check
    files_to_check = [
        'main_interactive.py',
        'sca_au.py'
    ]

    # Get V2 methods
    v2_methods = check_v2_methods()
    print(f"\nV2 module has {len(v2_methods)} public methods")

    # Check each file
    all_missing = {}

    for file_name in files_to_check:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        if not os.path.exists(file_path):
            print(f"\n⚠️  File not found: {file_name}")
            continue

        print(f"\n{'='*80}")
        print(f"Checking: {file_name}")
        print(f"{'='*80}")

        # Extract method calls
        called_methods = extract_method_calls(file_path)
        print(f"\nFound {len(called_methods)} method calls to lirpa_model:")
        for method in sorted(called_methods):
            status = "✓" if method in v2_methods else "✗"
            print(f"  {status} {method}")

        # Find missing methods
        missing = called_methods - v2_methods
        if missing:
            all_missing[file_name] = missing
            print(f"\n❌ MISSING METHODS ({len(missing)}):")
            for method in sorted(missing):
                print(f"  - {method}")
        else:
            print(f"\n✅ All methods are available in V2")

    # Summary and fixes
    print(f"\n{'='*80}")
    print("SUMMARY AND FIXES")
    print(f"{'='*80}")

    if not all_missing:
        print("\n✅ No compatibility issues found!")
        return

    # Collect all unique missing methods
    all_missing_methods = set()
    for methods in all_missing.values():
        all_missing_methods.update(methods)

    print(f"\nTotal missing methods: {len(all_missing_methods)}")

    # Check for similar methods (typos, naming differences)
    print(f"\n{'='*80}")
    print("SUGGESTED FIXES")
    print(f"{'='*80}")

    for missing_method in sorted(all_missing_methods):
        print(f"\n❌ Missing: {missing_method}")

        # Look for similar methods
        similar = []
        for v2_method in v2_methods:
            # Check if similar (e.g., register_* vs set_*)
            if missing_method.replace('register_', '') == v2_method.replace('set_', ''):
                similar.append(v2_method)
            elif missing_method.replace('add_', '') == v2_method.replace('set_', ''):
                similar.append(v2_method)

        if similar:
            print(f"  ℹ️  Similar method(s) found in V2:")
            for s in similar:
                print(f"     - {s}")
            print(f"  💡 Suggestion: Replace '{missing_method}' with '{similar[0]}' in V2 module")
            print(f"     OR rename calls in main_interactive.py")
        else:
            print(f"  💡 Suggestion: Add '{missing_method}' method to V2 module")

    # Generate code template for missing methods
    print(f"\n{'='*80}")
    print("CODE TEMPLATE FOR MISSING METHODS")
    print(f"{'='*80}")

    for missing_method in sorted(all_missing_methods):
        print(f"\n# Add this to IntermediateBoundedModuleV2:")
        print(f"def {missing_method}(self, *args, **kwargs):")
        print(f"    \"\"\"")
        print(f"    TODO: Implement {missing_method}")
        print(f"    \"\"\"")

        # Try to suggest implementation based on name
        if 'register' in missing_method and 'set' in missing_method.replace('register', 'set'):
            alt_name = missing_method.replace('register', 'set')
            if alt_name in v2_methods:
                print(f"    # Alias for {alt_name}")
                print(f"    return self.{alt_name}(*args, **kwargs)")
        else:
            print(f"    raise NotImplementedError('{missing_method} not implemented in V2')")
        print()


if __name__ == '__main__':
    main()
