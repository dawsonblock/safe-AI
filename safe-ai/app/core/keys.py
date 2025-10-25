"""Cryptographic key management with Ed25519."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import nacl.encoding
import nacl.signing
from pydantic import BaseModel

from app.core.models import KeyInfo


class KeyPair(BaseModel):
    """Ed25519 key pair."""

    signing_key: bytes
    verify_key: bytes
    key_id: str
    created_at: datetime

    class Config:
        arbitrary_types_allowed = True


class KeyManager:
    """Manages Ed25519 keys for signing and verification."""

    def __init__(self, keys_dir: Path) -> None:
        """Initialize key manager.

        Args:
            keys_dir: Directory containing key files
        """
        self.keys_dir = keys_dir
        self.keys_dir.mkdir(parents=True, exist_ok=True)

    def generate_keypair(self, key_id: str = "default") -> KeyPair:
        """Generate new Ed25519 keypair.

        Args:
            key_id: Identifier for the key pair

        Returns:
            KeyPair object
        """
        signing_key = nacl.signing.SigningKey.generate()
        verify_key = signing_key.verify_key

        return KeyPair(
            signing_key=bytes(signing_key),
            verify_key=bytes(verify_key),
            key_id=key_id,
            created_at=datetime.utcnow(),
        )

    def save_keypair(self, keypair: KeyPair) -> None:
        """Save keypair to disk.

        Args:
            keypair: KeyPair to save
        """
        sk_path = self.keys_dir / f"ed25519.{keypair.key_id}.sk"
        vk_path = self.keys_dir / f"ed25519.{keypair.key_id}.vk"

        # Save signing key (private)
        sk_b64 = base64.b64encode(keypair.signing_key).decode("ascii")
        sk_path.write_text(sk_b64)
        sk_path.chmod(0o600)  # Read/write for owner only

        # Save verify key (public)
        vk_b64 = base64.b64encode(keypair.verify_key).decode("ascii")
        vk_path.write_text(vk_b64)
        vk_path.chmod(0o644)

    def load_signing_key(self, key_id: str = "default") -> nacl.signing.SigningKey:
        """Load signing key from disk.

        Args:
            key_id: Key identifier

        Returns:
            SigningKey object

        Raises:
            FileNotFoundError: If key file not found
        """
        sk_path = self.keys_dir / f"ed25519.{key_id}.sk"
        if not sk_path.exists():
            raise FileNotFoundError(f"Signing key not found: {sk_path}")

        sk_b64 = sk_path.read_text().strip()
        sk_bytes = base64.b64decode(sk_b64)
        return nacl.signing.SigningKey(sk_bytes)

    def load_verify_key(self, key_id: str = "default") -> nacl.signing.VerifyKey:
        """Load verify key from disk.

        Args:
            key_id: Key identifier

        Returns:
            VerifyKey object

        Raises:
            FileNotFoundError: If key file not found
        """
        vk_path = self.keys_dir / f"ed25519.{key_id}.vk"
        if not vk_path.exists():
            raise FileNotFoundError(f"Verify key not found: {vk_path}")

        vk_b64 = vk_path.read_text().strip()
        vk_bytes = base64.b64decode(vk_b64)
        return nacl.signing.VerifyKey(vk_bytes)

    def get_key_info(self, key_id: str = "default") -> KeyInfo:
        """Get public key information.

        Args:
            key_id: Key identifier

        Returns:
            KeyInfo object
        """
        vk_path = self.keys_dir / f"ed25519.{key_id}.vk"
        if not vk_path.exists():
            raise FileNotFoundError(f"Key not found: {key_id}")

        vk_b64 = vk_path.read_text().strip()
        stat = vk_path.stat()

        return KeyInfo(
            key_id=f"ed25519:{key_id}",
            algorithm="ed25519",
            public_key_b64=vk_b64,
            created_at=datetime.fromtimestamp(stat.st_ctime),
        )

    def rotate_keys(self, old_key_id: str = "default") -> KeyPair:
        """Rotate keys by generating new keypair and archiving old.

        Args:
            old_key_id: Old key identifier

        Returns:
            New KeyPair
        """
        # Generate new keypair
        new_key_id = f"{old_key_id}.{int(datetime.utcnow().timestamp())}"
        new_keypair = self.generate_keypair(new_key_id)

        # Archive old keys if they exist
        for suffix in ["sk", "vk"]:
            old_path = self.keys_dir / f"ed25519.{old_key_id}.{suffix}"
            if old_path.exists():
                archive_path = self.keys_dir / f"ed25519.{old_key_id}.{suffix}.archived"
                old_path.rename(archive_path)

        # Save new keys
        self.save_keypair(new_keypair)

        # Create symlink for default
        if old_key_id == "default":
            for suffix in ["sk", "vk"]:
                link_path = self.keys_dir / f"ed25519.default.{suffix}"
                if link_path.is_symlink():
                    link_path.unlink()
                target = f"ed25519.{new_key_id}.{suffix}"
                link_path.symlink_to(target)

        return new_keypair

    def list_keys(self) -> list[str]:
        """List available key IDs.

        Returns:
            List of key identifiers
        """
        keys = set()
        for vk_file in self.keys_dir.glob("ed25519.*.vk"):
            if not vk_file.name.endswith(".archived"):
                key_id = vk_file.stem.replace("ed25519.", "")
                keys.add(key_id)
        return sorted(keys)
