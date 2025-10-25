"""Test end-to-end workflow."""

import json
import tempfile
from pathlib import Path

import pytest

from app.core.apply import FileApplicator
from app.core.audit import AuditLog
from app.core.gate import ManifestGate
from app.core.keys import KeyManager
from app.core.models import AuditRecord, ChangeManifest, Edit, EditOperation, PolicyVerdict
from app.core.policy_vm import PolicyConfig, PolicyVM


def test_full_workflow() -> None:
    """Test complete sign-verify-policy-apply workflow."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Setup
        keys_dir = tmppath / "keys"
        data_dir = tmppath / "data"
        project_root = tmppath / "project"
        project_root.mkdir()

        # Create test file
        test_file = project_root / "src" / "test.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("# Original content\n")

        # Generate keys
        key_manager = KeyManager(keys_dir)
        keypair = key_manager.generate_keypair("default")
        key_manager.save_keypair(keypair)

        # Create manifest
        manifest = ChangeManifest(
            targets=["src/test.py"],
            edits=[
                Edit(
                    path="src/test.py",
                    op=EditOperation.REPLACE,
                    start=0,
                    end=0,
                    text="# Updated content\n",
                )
            ],
            tests=[],
            rationale="Update test file for integration test",
        )

        # Sign
        signing_key = key_manager.load_signing_key("default")
        verify_key = key_manager.load_verify_key("default")
        gate = ManifestGate(verify_key)
        envelope = gate.sign_manifest(manifest, signing_key, "default")

        # Verify
        verified_manifest = gate.verify_envelope(envelope)
        manifest_hash = gate.hash_manifest(verified_manifest)

        # Policy check
        policy_config = PolicyConfig()
        policy_vm = PolicyVM(policy_config)
        policy_result = policy_vm.evaluate(verified_manifest)

        assert policy_result.verdict == PolicyVerdict.APPROVED

        # Apply
        applicator = FileApplicator(project_root)
        result = applicator.apply_manifest(verified_manifest)
        applicator.cleanup()

        assert result["files_modified"] == 1

        # Verify file was updated
        updated_content = test_file.read_text()
        assert "Updated content" in updated_content

        # Audit
        audit_log = AuditLog(data_dir / "audit.log")
        audit_record = AuditRecord(
            actor="cli",
            manifest_sha256=manifest_hash,
            policy_verdict=policy_result.verdict,
            merkle_root="",
            prev_root="",
            applied=True,
        )
        audit_index = audit_log.append(audit_record)

        assert audit_index >= 0

        # Verify audit chain
        is_valid, error = audit_log.verify_chain()
        assert is_valid


def test_workflow_with_policy_block() -> None:
    """Test workflow with policy blocking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        keys_dir = tmppath / "keys"
        project_root = tmppath / "project"
        project_root.mkdir()

        # Generate keys
        key_manager = KeyManager(keys_dir)
        keypair = key_manager.generate_keypair("default")
        key_manager.save_keypair(keypair)

        # Create manifest with blocked path
        manifest = ChangeManifest(
            targets=[".git/config"],
            edits=[
                Edit(
                    path=".git/config",
                    op=EditOperation.REPLACE,
                    start=0,
                    end=0,
                    text="malicious",
                )
            ],
            rationale="This should be blocked by policy",
        )

        # Sign
        signing_key = key_manager.load_signing_key("default")
        verify_key = key_manager.load_verify_key("default")
        gate = ManifestGate(verify_key)
        envelope = gate.sign_manifest(manifest, signing_key, "default")

        # Verify signature (should succeed)
        verified_manifest = gate.verify_envelope(envelope)

        # Policy check (should fail)
        policy_config = PolicyConfig()
        policy_vm = PolicyVM(policy_config)
        policy_result = policy_vm.evaluate(verified_manifest)

        assert policy_result.verdict == PolicyVerdict.BLOCKED
        assert len(policy_result.reasons) > 0
