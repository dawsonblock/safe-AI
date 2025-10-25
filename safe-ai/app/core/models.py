"""Pydantic models for SAFE-AI Governor."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EditOperation(str, Enum):
    """Allowed edit operations."""

    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"


class Edit(BaseModel):
    """Single edit operation."""

    path: str = Field(..., description="File path relative to project root")
    op: EditOperation = Field(..., description="Operation type")
    start: int = Field(..., ge=0, description="Start line number (0-indexed)")
    end: int = Field(..., ge=0, description="End line number (0-indexed)")
    text: str = Field(default="", description="New text content")

    @field_validator("text")
    @classmethod
    def validate_text_size(cls, v: str) -> str:
        """Validate edit size limit."""
        if len(v.encode("utf-8")) > 64 * 1024:  # 64KB
            raise ValueError("Edit text exceeds 64KB limit")
        return v

    @field_validator("end")
    @classmethod
    def validate_line_order(cls, v: int, info: Any) -> int:
        """Validate end >= start."""
        if "start" in info.data and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class ChangeManifest(BaseModel):
    """Change manifest describing edits to apply."""

    targets: list[str] = Field(..., min_length=1, description="Target file paths")
    edits: list[Edit] = Field(..., min_length=1, description="Edit operations")
    tests: list[str] = Field(default_factory=list, description="Test commands to run")
    rationale: str = Field(..., min_length=1, description="Change rationale")
    version: str = Field(default="1.0.0", description="Manifest version")

    @field_validator("edits")
    @classmethod
    def validate_total_size(cls, v: list[Edit]) -> list[Edit]:
        """Validate total change size."""
        total_bytes = sum(len(e.text.encode("utf-8")) for e in v)
        if total_bytes > 512 * 1024:  # 512KB
            raise ValueError(f"Total change size {total_bytes} exceeds 512KB limit")
        return v


class SignedEnvelope(BaseModel):
    """Cryptographically signed envelope."""

    payload_b64: str = Field(..., description="Base64-encoded JSON payload")
    sig_b64: str = Field(..., description="Base64-encoded Ed25519 signature")
    pubkey_id: str = Field(..., description="Public key identifier")


class PolicyVerdict(str, Enum):
    """Policy evaluation result."""

    APPROVED = "approved"
    BLOCKED = "blocked"


class PolicyResult(BaseModel):
    """Policy evaluation result."""

    verdict: PolicyVerdict
    reasons: list[str] = Field(default_factory=list)
    blocked_paths: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)


class SandboxResult(BaseModel):
    """Sandbox execution result."""

    rc: int = Field(..., description="Exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    duration_ms: float = Field(..., ge=0, description="Execution duration in ms")
    timed_out: bool = Field(default=False, description="Whether execution timed out")


class AuditRecord(BaseModel):
    """Immutable audit log record."""

    ts: datetime = Field(default_factory=datetime.utcnow, description="Timestamp")
    actor: Literal["api", "cli"] = Field(..., description="Actor type")
    manifest_sha256: str = Field(..., description="Manifest hash")
    policy_verdict: PolicyVerdict = Field(..., description="Policy result")
    sandbox_result: SandboxResult | None = Field(None, description="Sandbox execution result")
    merkle_root: str = Field(..., description="Current Merkle root")
    prev_root: str = Field(..., description="Previous Merkle root")
    version: str = Field(default="1.0.0", description="Record version")
    applied: bool = Field(default=False, description="Whether change was applied")
    error: str | None = Field(None, description="Error message if failed")


class KeyInfo(BaseModel):
    """Public key information."""

    key_id: str
    algorithm: Literal["ed25519"] = "ed25519"
    public_key_b64: str
    created_at: datetime
    rotated_at: datetime | None = None


class JWTPayload(BaseModel):
    """JWT token payload."""

    sub: str = Field(..., description="Subject (user ID)")
    role: Literal["admin", "operator", "auditor"] = Field(..., description="User role")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")


class ApplyRequest(BaseModel):
    """Request to apply a change."""

    signed_envelope: SignedEnvelope
    manifest: ChangeManifest
    idempotency_key: str | None = Field(None, description="Optional idempotency key")


class ApplyResponse(BaseModel):
    """Response from applying a change."""

    success: bool
    manifest_hash: str
    merkle_root: str
    audit_index: int
    message: str
    sandbox_result: SandboxResult | None = None
    error: str | None = None


class RollbackRequest(BaseModel):
    """Request to rollback a change."""

    manifest_hash: str | None = Field(None, description="Manifest hash to rollback")
    audit_index: int | None = Field(None, description="Audit index to rollback to")


class RollbackResponse(BaseModel):
    """Response from rollback operation."""

    success: bool
    reverted_hash: str
    new_merkle_root: str
    message: str
