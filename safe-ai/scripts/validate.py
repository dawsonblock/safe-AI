#!/usr/bin/env python3
"""Validation script for SAFE-AI Governor."""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_imports() -> bool:
    """Validate all imports work."""
    print("✓ Validating imports...")
    try:
        from app.core.models import ChangeManifest, SignedEnvelope
        from app.core.keys import KeyManager
        from app.core.gate import ManifestGate
        from app.core.policy_vm import PolicyVM, PolicyConfig
        from app.core.apply import FileApplicator
        from app.core.sandbox import SandboxRunner
        from app.core.audit import AuditLog
        from app.core.kill import KillSwitch
        from app.core.rbac import RBACManager
        from app.core.metrics import metrics
        from app.api.routes import app
        from app.cli.main import cli
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def validate_structure() -> bool:
    """Validate directory structure."""
    print("✓ Validating structure...")
    required_dirs = [
        "app/core",
        "app/api",
        "app/cli",
        "app/cli/cmds",
        "app/tests/unit",
        "app/tests/integration",
        "app/config",
        "scripts",
        "examples",
    ]

    required_files = [
        "pyproject.toml",
        "Dockerfile",
        ".github/workflows/ci.yml",
        "README.md",
        "SECURITY.md",
        "LICENSE",
    ]

    root = Path(__file__).parent.parent
    all_ok = True

    for dir_path in required_dirs:
        if not (root / dir_path).exists():
            print(f"  ✗ Missing directory: {dir_path}")
            all_ok = False

    for file_path in required_files:
        if not (root / file_path).exists():
            print(f"  ✗ Missing file: {file_path}")
            all_ok = False

    if all_ok:
        print("  ✓ Structure valid")

    return all_ok


def validate_models() -> bool:
    """Validate Pydantic models."""
    print("✓ Validating models...")
    try:
        from app.core.models import ChangeManifest, Edit, EditOperation

        # Create valid manifest
        manifest = ChangeManifest(
            targets=["test.py"],
            edits=[
                Edit(
                    path="test.py",
                    op=EditOperation.REPLACE,
                    start=0,
                    end=1,
                    text="test",
                )
            ],
            rationale="Test validation",
        )

        assert manifest.targets == ["test.py"]
        assert len(manifest.edits) == 1

        print("  ✓ Models valid")
        return True
    except Exception as e:
        print(f"  ✗ Model validation failed: {e}")
        return False


def validate_cryptography() -> bool:
    """Validate cryptographic operations."""
    print("✓ Validating cryptography...")
    try:
        import tempfile
        from app.core.keys import KeyManager
        from app.core.gate import ManifestGate
        from app.core.models import ChangeManifest, Edit, EditOperation

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate keys
            km = KeyManager(Path(tmpdir))
            keypair = km.generate_keypair("test")
            km.save_keypair(keypair)

            # Load keys
            signing_key = km.load_signing_key("test")
            verify_key = km.load_verify_key("test")

            # Create and sign manifest
            manifest = ChangeManifest(
                targets=["test.py"],
                edits=[
                    Edit(
                        path="test.py",
                        op=EditOperation.REPLACE,
                        start=0,
                        end=1,
                        text="test",
                    )
                ],
                rationale="Test validation",
            )

            gate = ManifestGate(verify_key)
            envelope = gate.sign_manifest(manifest, signing_key, "test")

            # Verify
            verified = gate.verify_envelope(envelope)
            assert verified.targets == manifest.targets

        print("  ✓ Cryptography valid")
        return True
    except Exception as e:
        print(f"  ✗ Cryptography validation failed: {e}")
        return False


def validate_policy() -> bool:
    """Validate policy VM."""
    print("✓ Validating policy VM...")
    try:
        from app.core.policy_vm import PolicyVM, PolicyConfig
        from app.core.models import (
            ChangeManifest,
            Edit,
            EditOperation,
            PolicyVerdict,
        )

        config = PolicyConfig()
        policy = PolicyVM(config)

        # Valid manifest
        valid_manifest = ChangeManifest(
            targets=["src/test.py"],
            edits=[
                Edit(
                    path="src/test.py",
                    op=EditOperation.REPLACE,
                    start=0,
                    end=1,
                    text="# Valid code\n",
                )
            ],
            rationale="This is a valid test manifest for validation",
        )

        result = policy.evaluate(valid_manifest)
        assert result.verdict == PolicyVerdict.APPROVED, "Valid manifest should be approved"

        # Invalid manifest (blocked path)
        invalid_manifest = ChangeManifest(
            targets=[".git/config"],
            edits=[
                Edit(
                    path=".git/config",
                    op=EditOperation.REPLACE,
                    start=0,
                    end=1,
                    text="malicious",
                )
            ],
            rationale="This should be blocked",
        )

        result = policy.evaluate(invalid_manifest)
        assert result.verdict == PolicyVerdict.BLOCKED, "Invalid manifest should be blocked"

        print("  ✓ Policy VM valid")
        return True
    except Exception as e:
        print(f"  ✗ Policy validation failed: {e}")
        return False


def main() -> int:
    """Run all validations."""
    print("=" * 60)
    print("SAFE-AI Governor Validation")
    print("=" * 60)
    print()

    validations = [
        validate_imports,
        validate_structure,
        validate_models,
        validate_cryptography,
        validate_policy,
    ]

    results = [v() for v in validations]

    print()
    print("=" * 60)
    if all(results):
        print("✓ All validations passed!")
        print("=" * 60)
        return 0
    else:
        print("✗ Some validations failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
