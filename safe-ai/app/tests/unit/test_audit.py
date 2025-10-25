"""Test audit log."""

import tempfile
from pathlib import Path

import pytest

from app.core.audit import AuditLog
from app.core.models import AuditRecord, PolicyVerdict, SandboxResult


def test_audit_log_genesis() -> None:
    """Test audit log genesis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.log"
        audit_log = AuditLog(log_path)

        # Should have genesis root
        root = audit_log.get_current_root()
        assert len(root) == 64  # SHA-256


def test_audit_log_append() -> None:
    """Test appending records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.log"
        audit_log = AuditLog(log_path)

        record = AuditRecord(
            actor="cli",
            manifest_sha256="abc123",
            policy_verdict=PolicyVerdict.APPROVED,
            merkle_root="",
            prev_root="",
            applied=True,
        )

        index = audit_log.append(record)

        assert index == 0


def test_audit_log_chain() -> None:
    """Test Merkle chain."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.log"
        audit_log = AuditLog(log_path)

        # Append multiple records
        for i in range(5):
            record = AuditRecord(
                actor="cli",
                manifest_sha256=f"hash{i}",
                policy_verdict=PolicyVerdict.APPROVED,
                merkle_root="",
                prev_root="",
                applied=True,
            )
            audit_log.append(record)

        # Verify chain
        is_valid, error = audit_log.verify_chain()

        assert is_valid
        assert error == ""


def test_audit_log_find_by_hash() -> None:
    """Test finding records by hash."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.log"
        audit_log = AuditLog(log_path)

        target_hash = "target123"

        # Append records
        for i in range(5):
            record = AuditRecord(
                actor="cli",
                manifest_sha256=target_hash if i == 3 else f"hash{i}",
                policy_verdict=PolicyVerdict.APPROVED,
                merkle_root="",
                prev_root="",
                applied=True,
            )
            audit_log.append(record)

        # Find
        found = audit_log.find_by_manifest_hash(target_hash)

        assert found is not None
        assert found.manifest_sha256 == target_hash


def test_audit_log_get_records() -> None:
    """Test getting records."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "audit.log"
        audit_log = AuditLog(log_path)

        # Append records
        for i in range(10):
            record = AuditRecord(
                actor="cli",
                manifest_sha256=f"hash{i}",
                policy_verdict=PolicyVerdict.APPROVED,
                merkle_root="",
                prev_root="",
                applied=True,
            )
            audit_log.append(record)

        # Get last 5
        records = audit_log.get_records(limit=5)

        assert len(records) == 5
