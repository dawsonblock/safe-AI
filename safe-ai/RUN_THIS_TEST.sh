#!/bin/bash
# Quick verification test - NO DEPENDENCIES REQUIRED

echo "=========================================="
echo "SAFE-AI Governor - Quick Verification"
echo "=========================================="
echo ""

echo "✓ Testing Python compilation..."
python3 -m py_compile app/core/models.py 2>&1 && echo "  ✅ models.py compiles" || echo "  ❌ models.py failed"
python3 -m py_compile app/core/keys.py 2>&1 && echo "  ✅ keys.py compiles" || echo "  ❌ keys.py failed"
python3 -m py_compile app/core/gate.py 2>&1 && echo "  ✅ gate.py compiles" || echo "  ❌ gate.py failed"
python3 -m py_compile app/core/policy_vm.py 2>&1 && echo "  ✅ policy_vm.py compiles" || echo "  ❌ policy_vm.py failed"
python3 -m py_compile app/api/routes.py 2>&1 && echo "  ✅ routes.py compiles" || echo "  ❌ routes.py failed"
python3 -m py_compile app/cli/main.py 2>&1 && echo "  ✅ main.py compiles" || echo "  ❌ main.py failed"

echo ""
echo "✓ Checking file structure..."
[ -f "app/core/models.py" ] && echo "  ✅ Core modules present" || echo "  ❌ Missing core modules"
[ -f "app/api/routes.py" ] && echo "  ✅ API modules present" || echo "  ❌ Missing API modules"
[ -f "app/cli/main.py" ] && echo "  ✅ CLI modules present" || echo "  ❌ Missing CLI modules"
[ -f "README.md" ] && echo "  ✅ Documentation present" || echo "  ❌ Missing documentation"
[ -f "pyproject.toml" ] && echo "  ✅ Config files present" || echo "  ❌ Missing config files"

echo ""
echo "✓ Running syntax check..."
python3 scripts/syntax_check.py

echo ""
echo "=========================================="
echo "RESULT: All static checks completed"
echo "=========================================="
echo ""
echo "To run full validation with imports:"
echo "  1. pip install -r requirements.txt"
echo "  2. python3 scripts/validate.py"
echo ""
