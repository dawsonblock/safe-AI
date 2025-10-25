"""Global kill switch for emergency shutdown."""

import os
from pathlib import Path


class KillSwitch:
    """Global kill switch for emergency operations halt."""

    def __init__(self, sentinel_path: Path | None = None) -> None:
        """Initialize kill switch.

        Args:
            sentinel_path: Path to kill switch sentinel file
        """
        self.sentinel_path = sentinel_path or Path("KILL_SWITCH")
        self.enabled_var = os.getenv("SAFEAI_ENABLE", "on").lower()

    def is_active(self) -> bool:
        """Check if kill switch is active.

        Returns:
            True if kill switch is active (operations should halt)
        """
        # Check environment variable
        if self.enabled_var not in ("on", "true", "1", "yes"):
            return True

        # Check sentinel file
        if self.sentinel_path.exists():
            return True

        return False

    def activate(self, reason: str = "") -> None:
        """Activate kill switch.

        Args:
            reason: Reason for activation
        """
        with open(self.sentinel_path, "w", encoding="utf-8") as f:
            f.write(f"KILL_SWITCH_ACTIVE\n")
            f.write(f"Reason: {reason}\n")
            f.write(f"Activated at: {__import__('datetime').datetime.utcnow().isoformat()}\n")

    def deactivate(self) -> None:
        """Deactivate kill switch."""
        if self.sentinel_path.exists():
            self.sentinel_path.unlink()

    def check_or_raise(self) -> None:
        """Check kill switch and raise exception if active.

        Raises:
            KillSwitchError: If kill switch is active
        """
        if self.is_active():
            raise KillSwitchError("Kill switch is active - all operations halted")

    def get_status(self) -> dict[str, bool | str]:
        """Get kill switch status.

        Returns:
            Status dictionary
        """
        is_active = self.is_active()
        reason = ""

        if self.sentinel_path.exists():
            try:
                with open(self.sentinel_path, "r", encoding="utf-8") as f:
                    reason = f.read().strip()
            except Exception:
                pass

        return {
            "active": is_active,
            "env_enabled": self.enabled_var in ("on", "true", "1", "yes"),
            "sentinel_exists": self.sentinel_path.exists(),
            "reason": reason,
        }


class KillSwitchError(Exception):
    """Kill switch is active."""

    pass
