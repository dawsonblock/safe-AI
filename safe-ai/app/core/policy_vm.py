"""Policy VM for evaluating change requests against security rules."""

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from app.core.models import ChangeManifest, EditOperation, PolicyResult, PolicyVerdict


class PolicyConfig(BaseSettings):
    """Policy configuration."""

    safe_mode: bool = Field(default=True, description="Enable safe mode")
    autonomy_tier: str = Field(default="gamma", description="Autonomy tier: alpha/beta/gamma")

    # Path rules
    allow_paths: list[str] = Field(
        default_factory=lambda: ["src/", "tests/", "app/"], description="Allowed path prefixes"
    )
    deny_paths: list[str] = Field(
        default_factory=lambda: [".git/", "config/keys/", "/etc/", "/usr/", "secrets/"],
        description="Denied path prefixes",
    )

    # Content rules
    deny_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(?i)subprocess\.Popen\(",
            r"rm\s+-rf",
            r"curl\s+http",
            r"wget\s+",
            r"(?i)ssh\s+",
            r"pip\s+install",
            r"(?i)exec\(",
            r"(?i)eval\(",
            r"(?i)__import__",
            r"os\.system\(",
        ],
        description="Denied regex patterns",
    )

    # Operation limits
    allowed_operations: list[str] = Field(
        default_factory=lambda: ["replace", "insert", "delete"], description="Allowed operations"
    )
    max_edit_size_bytes: int = Field(default=64 * 1024, description="Max edit size (64KB)")
    max_total_size_bytes: int = Field(default=512 * 1024, description="Max total size (512KB)")

    class Config:
        env_prefix = "SAFEAI_POLICY_"


class PolicyVM:
    """Policy virtual machine for evaluating changes."""

    def __init__(self, config: PolicyConfig) -> None:
        """Initialize policy VM.

        Args:
            config: Policy configuration
        """
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self.deny_regexes = [re.compile(pattern) for pattern in self.config.deny_patterns]

    def evaluate(self, manifest: ChangeManifest) -> PolicyResult:
        """Evaluate manifest against policy rules.

        Args:
            manifest: ChangeManifest to evaluate

        Returns:
            PolicyResult with verdict and reasons
        """
        reasons: list[str] = []
        blocked_paths: list[str] = []
        blocked_patterns: list[str] = []

        # Check paths
        for target in manifest.targets:
            if not self._is_path_allowed(target):
                reasons.append(f"Path not in allowed list: {target}")
                blocked_paths.append(target)

            if self._is_path_denied(target):
                reasons.append(f"Path explicitly denied: {target}")
                blocked_paths.append(target)

        # Check edit paths
        for edit in manifest.edits:
            if not self._is_path_allowed(edit.path):
                reasons.append(f"Edit path not allowed: {edit.path}")
                blocked_paths.append(edit.path)

            if self._is_path_denied(edit.path):
                reasons.append(f"Edit path denied: {edit.path}")
                blocked_paths.append(edit.path)

        # Check operations
        for edit in manifest.edits:
            if edit.op.value not in self.config.allowed_operations:
                reasons.append(f"Operation not allowed: {edit.op.value}")

        # Check edit sizes
        for edit in manifest.edits:
            edit_size = len(edit.text.encode("utf-8"))
            if edit_size > self.config.max_edit_size_bytes:
                reasons.append(
                    f"Edit size {edit_size} exceeds limit {self.config.max_edit_size_bytes}"
                )

        # Check total size
        total_size = sum(len(e.text.encode("utf-8")) for e in manifest.edits)
        if total_size > self.config.max_total_size_bytes:
            reasons.append(
                f"Total size {total_size} exceeds limit {self.config.max_total_size_bytes}"
            )

        # Check content patterns
        for edit in manifest.edits:
            for pattern_idx, regex in enumerate(self.deny_regexes):
                if regex.search(edit.text):
                    pattern = self.config.deny_patterns[pattern_idx]
                    reasons.append(f"Denied pattern found in {edit.path}: {pattern}")
                    blocked_patterns.append(pattern)

        # Check rationale
        if not manifest.rationale or len(manifest.rationale) < 10:
            reasons.append("Rationale too short or missing")

        # Determine verdict
        verdict = PolicyVerdict.BLOCKED if reasons else PolicyVerdict.APPROVED

        return PolicyResult(
            verdict=verdict,
            reasons=reasons,
            blocked_paths=list(set(blocked_paths)),
            blocked_patterns=list(set(blocked_patterns)),
        )

    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is in allow list.

        Args:
            path: Path to check

        Returns:
            True if allowed
        """
        if not self.config.allow_paths:
            return True  # No restrictions

        normalized = self._normalize_path(path)
        return any(normalized.startswith(allowed) for allowed in self.config.allow_paths)

    def _is_path_denied(self, path: str) -> bool:
        """Check if path is in deny list.

        Args:
            path: Path to check

        Returns:
            True if denied
        """
        normalized = self._normalize_path(path)
        return any(normalized.startswith(denied) for denied in self.config.deny_paths)

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Normalize path for comparison.

        Args:
            path: Path to normalize

        Returns:
            Normalized path
        """
        # Remove leading ./
        if path.startswith("./"):
            path = path[2:]
        # Ensure trailing / for directories
        return path

    def check_kill_switch(self, kill_switch_path: Path) -> bool:
        """Check if kill switch is activated.

        Args:
            kill_switch_path: Path to kill switch file

        Returns:
            True if kill switch is active
        """
        return kill_switch_path.exists()
