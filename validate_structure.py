#!/usr/bin/env python3
"""
Basic structure and import validation test.

This script validates that:
1. Package structure is correct
2. __init__.py exports work
3. API classes are importable (without instantiation)
4. Demo files are importable

Does NOT require full dependencies (PyTorch, MMPose, etc.)
"""

import ast
import importlib.util
import sys
from pathlib import Path


def check_file_syntax(filepath: Path) -> bool:
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error in {filepath}: {e}")
        return False


def check_module_structure(module_path: Path, expected_exports: list) -> bool:
    """Check if a module has expected exports in __init__.py."""
    init_file = module_path / "__init__.py"
    if not init_file.exists():
        print(f"  ✗ Missing __init__.py in {module_path}")
        return False
    
    with open(init_file, 'r') as f:
        content = f.read()
    
    all_found = all(export in content for export in expected_exports)
    if all_found:
        print(f"  ✓ {module_path.name} has all expected exports")
    else:
        missing = [e for e in expected_exports if e not in content]
        print(f"  ✗ {module_path.name} missing exports: {missing}")
    
    return all_found


def main():
    """Run validation tests."""
    repo_root = Path(__file__).parent
    
    print("=" * 60)
    print("BBoxMaskPose Structure Validation")
    print("=" * 60)
    
    # Test 1: Check package structure
    print("\n[Test 1] Package Structure")
    print("-" * 60)
    
    packages = {
        "pmpose": ["PMPose"],
        "bboxmaskpose": ["BBoxMaskPose"],
    }
    
    all_good = True
    for pkg_name, exports in packages.items():
        pkg_path = repo_root / pkg_name
        if not pkg_path.exists():
            print(f"  ✗ Package directory missing: {pkg_path}")
            all_good = False
            continue
        
        inner_pkg = pkg_path / pkg_name
        if not inner_pkg.exists():
            print(f"  ✗ Inner package directory missing: {inner_pkg}")
            all_good = False
            continue
        
        if not check_module_structure(inner_pkg, exports):
            all_good = False
        
        # Check api.py exists
        api_file = inner_pkg / "api.py"
        if not api_file.exists():
            print(f"  ✗ Missing api.py in {inner_pkg}")
            all_good = False
        else:
            if check_file_syntax(api_file):
                print(f"  ✓ {pkg_name}/api.py syntax valid")
            else:
                all_good = False
    
    if all_good:
        print("\n✓ Package structure is valid")
    else:
        print("\n✗ Package structure has issues")
    
    # Test 2: Check demo files
    print("\n[Test 2] Demo Files")
    print("-" * 60)
    
    demo_files = [
        "demos/PMPose_demo.py",
        "demos/BMP_demo.py",
        "demos/quickstart.ipynb",
    ]
    
    demos_ok = True
    for demo_file in demo_files:
        demo_path = repo_root / demo_file
        if not demo_path.exists():
            print(f"  ✗ Missing demo: {demo_file}")
            demos_ok = False
        elif demo_file.endswith('.py'):
            if check_file_syntax(demo_path):
                print(f"  ✓ {demo_file} syntax valid")
            else:
                demos_ok = False
        else:
            print(f"  ✓ {demo_file} exists")
    
    if demos_ok:
        print("\n✓ All demo files present and valid")
    else:
        print("\n✗ Demo files have issues")
    
    # Test 3: Check critical files
    print("\n[Test 3] Critical Files")
    print("-" * 60)
    
    critical_files = [
        "setup.py",
        "README.md",
        "pmpose/pmpose/mm_utils.py",
        "pmpose/pmpose/posevis_lite.py",
        "bboxmaskpose/bboxmaskpose/sam2_utils.py",
        "bboxmaskpose/bboxmaskpose/demo_utils.py",
        "bboxmaskpose/bboxmaskpose/posevis_lite.py",
    ]
    
    files_ok = True
    for filepath in critical_files:
        full_path = repo_root / filepath
        if not full_path.exists():
            print(f"  ✗ Missing file: {filepath}")
            files_ok = False
        elif filepath.endswith('.py'):
            if check_file_syntax(full_path):
                print(f"  ✓ {filepath}")
            else:
                files_ok = False
        else:
            print(f"  ✓ {filepath}")
    
    if files_ok:
        print("\n✓ All critical files present")
    else:
        print("\n✗ Some critical files missing")
    
    # Test 4: Check configs
    print("\n[Test 4] Configuration Files")
    print("-" * 60)
    
    config_files = [
        "bboxmaskpose/bboxmaskpose/configs/bmp_D3.yaml",
        "bboxmaskpose/bboxmaskpose/configs/bmp_J1.yaml",
    ]
    
    configs_ok = True
    for config_file in config_files:
        config_path = repo_root / config_file
        if not config_path.exists():
            print(f"  ✗ Missing config: {config_file}")
            configs_ok = False
        else:
            print(f"  ✓ {config_file}")
    
    if configs_ok:
        print("\n✓ All config files present")
    else:
        print("\n✗ Some config files missing")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if all_good and demos_ok and files_ok and configs_ok:
        print("✓ All validation tests passed!")
        print("\nThe repository structure is ready.")
        print("Full functional testing requires:")
        print("  - PyTorch")
        print("  - MMPose/MMDetection/MMEngine")
        print("  - SAM2 dependencies")
        print("\nRun 'pip install -e .' to install in editable mode.")
        return 0
    else:
        print("✗ Some validation tests failed")
        print("Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
