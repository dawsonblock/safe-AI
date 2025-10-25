#!/usr/bin/env python3
"""Syntax validation script - runs without dependencies."""

import ast
import sys
from pathlib import Path


def check_syntax(file_path: Path) -> tuple[bool, str]:
    """Check Python file syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def main() -> int:
    """Run syntax checks."""
    print("=" * 60)
    print("SAFE-AI Governor Syntax Check")
    print("=" * 60)
    print()

    root = Path(__file__).parent.parent
    python_files = list(root.glob("app/**/*.py"))
    
    print(f"✓ Checking {len(python_files)} Python files...")
    
    failed = []
    for py_file in sorted(python_files):
        if "__pycache__" in str(py_file):
            continue
        
        success, error = check_syntax(py_file)
        if not success:
            rel_path = py_file.relative_to(root)
            print(f"  ✗ {rel_path}: {error}")
            failed.append(rel_path)
    
    if not failed:
        print("  ✓ All Python files have valid syntax")
    
    print()
    print("✓ Checking file structure...")
    
    required_files = [
        "app/core/models.py",
        "app/core/keys.py",
        "app/core/gate.py",
        "app/core/policy_vm.py",
        "app/core/apply.py",
        "app/core/sandbox.py",
        "app/core/audit.py",
        "app/core/kill.py",
        "app/core/rbac.py",
        "app/core/metrics.py",
        "app/api/routes.py",
        "app/api/deps.py",
        "app/cli/main.py",
        "pyproject.toml",
        "Dockerfile",
        "README.md",
    ]
    
    missing = []
    for req_file in required_files:
        if not (root / req_file).exists():
            print(f"  ✗ Missing: {req_file}")
            missing.append(req_file)
    
    if not missing:
        print("  ✓ All required files present")
    
    print()
    print("=" * 60)
    
    if failed or missing:
        print("✗ Checks failed")
        print("=" * 60)
        return 1
    else:
        print("✓ All syntax checks passed!")
        print()
        print("Note: To run full validation with imports:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run: python3 scripts/validate.py")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
