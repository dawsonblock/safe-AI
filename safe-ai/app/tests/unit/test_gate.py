"""Test manifest gate."""

import tempfile
from pathlib import Path

import pytest

from app.core.gate import GateError, ManifestGate
from app.core.keys import KeyManager
from app.core.models import ChangeManifest, Edit, EditOperation


@pytest.fixture
def test_manifest() -> ChangeManifest:
    """Create test manifest."""
    return ChangeManifest(
        targets=["src/test.py"],
        edits=[
            Edit(
                path="src/test.py",
                op=EditOperation.REPLACE,
                start=0,
                end=5,
                text="# New code\n",
            )
        ],
        tests=["pytest"],
        rationale="Test change",
    )


@pytest.fixture
def key_manager() -> KeyManager:
    """Create key manager with test keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KeyManager(Path(tmpdir))
        keypair = km.generate_keypair("test")
        km.save_keypair(keypair)
        yield km


def test_hash_manifest(test_manifest: ChangeManifest) -> None:
    """Test manifest hashing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KeyManager(Path(tmpdir))
        keypair = km.generate_keypair("test")
        gate = ManifestGate(keypair.verify_key)

        hash1 = gate.hash_manifest(test_manifest)
        hash2 = gate.hash_manifest(test_manifest)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256


def test_sign_and_verify(test_manifest: ChangeManifest) -> None:
    """Test signing and verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KeyManager(Path(tmpdir))
        keypair = km.generate_keypair("test")
        signing_key = km.load_signing_key("test")
        verify_key = km.load_verify_key("test")

        gate = ManifestGate(verify_key)

        # Sign
        envelope = gate.sign_manifest(test_manifest, signing_key, "test")

        # Verify
        verified_manifest = gate.verify_envelope(envelope)

        assert verified_manifest.targets == test_manifest.targets
        assert len(verified_manifest.edits) == len(test_manifest.edits)


def test_verify_invalid_signature() -> None:
    """Test verification with invalid signature."""
    with tempfile.TemporaryDirectory() as tmpdir:
        km = KeyManager(Path(tmpdir))

        # Create two different keys
        keypair1 = km.generate_keypair("key1")
        km.save_keypair(keypair1)

        keypair2 = km.generate_keypair("key2")
        km.save_keypair(keypair2)

        signing_key1 = km.load_signing_key("key1")
        verify_key2 = km.load_verify_key("key2")

        manifest = ChangeManifest(
            targets=["test.py"],
            edits=[
                Edit(path="test.py", op=EditOperation.REPLACE, start=0, end=1, text="test")
            ],
            rationale="Test",
        )

        # Sign with key1
        gate1 = ManifestGate(keypair1.verify_key)
        envelope = gate1.sign_manifest(manifest, signing_key1, "key1")

        # Try to verify with key2
        gate2 = ManifestGate(verify_key2)

        with pytest.raises(GateError, match="Invalid signature"):
            gate2.verify_envelope(envelope)
