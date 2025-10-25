"""Manifest gating: parsing, hashing, and signature verification."""

import base64
import hashlib
import json
from typing import Any

import nacl.encoding
import nacl.exceptions
import nacl.signing

from app.core.models import ChangeManifest, SignedEnvelope


class GateError(Exception):
    """Gate validation error."""

    pass


class ManifestGate:
    """Validates and processes change manifests."""

    def __init__(self, verify_key: nacl.signing.VerifyKey) -> None:
        """Initialize gate with verification key.

        Args:
            verify_key: Ed25519 verification key
        """
        self.verify_key = verify_key

    def hash_manifest(self, manifest: ChangeManifest) -> str:
        """Compute SHA-256 hash of manifest.

        Args:
            manifest: ChangeManifest to hash

        Returns:
            Hex-encoded SHA-256 hash
        """
        manifest_json = manifest.model_dump_json(sort_keys=True, exclude_none=True)
        manifest_bytes = manifest_json.encode("utf-8")
        return hashlib.sha256(manifest_bytes).hexdigest()

    def sign_manifest(
        self, manifest: ChangeManifest, signing_key: nacl.signing.SigningKey, key_id: str
    ) -> SignedEnvelope:
        """Sign a manifest.

        Args:
            manifest: ChangeManifest to sign
            signing_key: Ed25519 signing key
            key_id: Key identifier

        Returns:
            SignedEnvelope with signature
        """
        # Serialize manifest
        manifest_json = manifest.model_dump_json(sort_keys=True, exclude_none=True)
        payload_bytes = manifest_json.encode("utf-8")

        # Sign
        signed = signing_key.sign(payload_bytes)
        signature = signed.signature

        # Encode
        payload_b64 = base64.b64encode(payload_bytes).decode("ascii")
        sig_b64 = base64.b64encode(signature).decode("ascii")

        return SignedEnvelope(
            payload_b64=payload_b64, sig_b64=sig_b64, pubkey_id=f"ed25519:{key_id}"
        )

    def verify_envelope(self, envelope: SignedEnvelope) -> ChangeManifest:
        """Verify signed envelope and extract manifest.

        Args:
            envelope: SignedEnvelope to verify

        Returns:
            Verified ChangeManifest

        Raises:
            GateError: If verification fails
        """
        try:
            # Decode
            payload_bytes = base64.b64decode(envelope.payload_b64)
            signature = base64.b64decode(envelope.sig_b64)

            # Verify signature
            verified_bytes = self.verify_key.verify(payload_bytes, signature)

            # Parse manifest
            manifest_data = json.loads(verified_bytes.decode("utf-8"))
            manifest = ChangeManifest.model_validate(manifest_data)

            return manifest

        except nacl.exceptions.BadSignatureError as e:
            raise GateError("Invalid signature") from e
        except (ValueError, json.JSONDecodeError) as e:
            raise GateError(f"Invalid envelope format: {e}") from e

    def verify_manifest_hash(self, manifest: ChangeManifest, expected_hash: str) -> bool:
        """Verify manifest hash matches expected value.

        Args:
            manifest: ChangeManifest to verify
            expected_hash: Expected SHA-256 hash

        Returns:
            True if hash matches
        """
        actual_hash = self.hash_manifest(manifest)
        return actual_hash == expected_hash

    def parse_manifest(self, manifest_json: str) -> ChangeManifest:
        """Parse and validate manifest JSON.

        Args:
            manifest_json: JSON string

        Returns:
            Validated ChangeManifest

        Raises:
            GateError: If parsing fails
        """
        try:
            manifest_data = json.loads(manifest_json)
            return ChangeManifest.model_validate(manifest_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise GateError(f"Invalid manifest: {e}") from e
