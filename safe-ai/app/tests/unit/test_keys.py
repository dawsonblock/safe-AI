"""Test key management."""

import tempfile
from pathlib import Path

import pytest

from app.core.keys import KeyManager


def test_generate_keypair() -> None:
    """Test keypair generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_manager = KeyManager(Path(tmpdir))
        keypair = key_manager.generate_keypair("test")

        assert keypair.key_id == "test"
        assert len(keypair.signing_key) == 32
        assert len(keypair.verify_key) == 32


def test_save_and_load_keypair() -> None:
    """Test saving and loading keypairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        keys_dir = Path(tmpdir)
        key_manager = KeyManager(keys_dir)

        # Generate and save
        keypair = key_manager.generate_keypair("test")
        key_manager.save_keypair(keypair)

        # Load
        signing_key = key_manager.load_signing_key("test")
        verify_key = key_manager.load_verify_key("test")

        assert bytes(signing_key) == keypair.signing_key
        assert bytes(verify_key) == keypair.verify_key


def test_key_rotation() -> None:
    """Test key rotation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_manager = KeyManager(Path(tmpdir))

        # Generate initial keypair
        keypair1 = key_manager.generate_keypair("test")
        key_manager.save_keypair(keypair1)

        # Rotate
        keypair2 = key_manager.rotate_keys("test")

        assert keypair2.key_id != keypair1.key_id
        assert keypair2.signing_key != keypair1.signing_key


def test_list_keys() -> None:
    """Test listing keys."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_manager = KeyManager(Path(tmpdir))

        # Generate keys
        for key_id in ["key1", "key2", "key3"]:
            keypair = key_manager.generate_keypair(key_id)
            key_manager.save_keypair(keypair)

        # List
        keys = key_manager.list_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
