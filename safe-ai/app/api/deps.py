"""FastAPI dependencies."""

import os
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from app.core.apply import FileApplicator
from app.core.audit import AuditLog
from app.core.gate import ManifestGate
from app.core.keys import KeyManager
from app.core.kill import KillSwitch
from app.core.metrics import metrics
from app.core.policy_vm import PolicyConfig, PolicyVM
from app.core.rbac import RBACConfig, RBACError, RBACManager
from app.core.sandbox import SandboxConfig, SandboxRunner


# Paths
DATA_DIR = Path(os.getenv("SAFEAI_DATA_DIR", "./data"))
KEYS_DIR = DATA_DIR / "keys"
AUDIT_LOG_PATH = DATA_DIR / "audit.log"
KILL_SWITCH_PATH = Path(os.getenv("SAFEAI_KILL_SWITCH", "./KILL_SWITCH"))
PROJECT_ROOT = Path(os.getenv("SAFEAI_PROJECT_ROOT", "."))

# Initialize components
_key_manager = KeyManager(KEYS_DIR)
_audit_log = AuditLog(AUDIT_LOG_PATH)
_kill_switch = KillSwitch(KILL_SWITCH_PATH)
_policy_config = PolicyConfig()
_policy_vm = PolicyVM(_policy_config)
_sandbox_config = SandboxConfig()
_rbac_config = RBACConfig(
    jwt_secret=os.getenv("SAFEAI_JWT_SECRET", "development_secret_key")
)
_rbac_manager = RBACManager(_rbac_config)


def get_key_manager() -> KeyManager:
    """Get KeyManager instance."""
    return _key_manager


def get_audit_log() -> AuditLog:
    """Get AuditLog instance."""
    return _audit_log


def get_kill_switch() -> KillSwitch:
    """Get KillSwitch instance."""
    return _kill_switch


def get_policy_vm() -> PolicyVM:
    """Get PolicyVM instance."""
    return _policy_vm


def get_sandbox_runner() -> SandboxRunner:
    """Get SandboxRunner instance."""
    return SandboxRunner(_sandbox_config, PROJECT_ROOT)


def get_file_applicator() -> FileApplicator:
    """Get FileApplicator instance."""
    return FileApplicator(PROJECT_ROOT)


def get_manifest_gate() -> ManifestGate:
    """Get ManifestGate instance."""
    try:
        verify_key = _key_manager.load_verify_key("default")
        return ManifestGate(verify_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification key not found. Initialize keys first.",
        )


def get_rbac_manager() -> RBACManager:
    """Get RBACManager instance."""
    return _rbac_manager


def get_metrics_collector() -> type[metrics]:  # type: ignore
    """Get MetricsCollector instance."""
    return metrics


async def verify_token(
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    """Verify JWT token from Authorization header.

    Args:
        authorization: Authorization header

    Returns:
        JWT token

    Raises:
        HTTPException: If token is missing or invalid
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return authorization[7:]  # Remove "Bearer " prefix


async def require_admin(
    token: Annotated[str, Depends(verify_token)],
    rbac: Annotated[RBACManager, Depends(get_rbac_manager)],
) -> None:
    """Require admin role.

    Args:
        token: JWT token
        rbac: RBAC manager

    Raises:
        HTTPException: If not authorized
    """
    try:
        payload = rbac.verify_token(token)
        if payload.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin role required",
            )
    except RBACError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_operator(
    token: Annotated[str, Depends(verify_token)],
    rbac: Annotated[RBACManager, Depends(get_rbac_manager)],
) -> None:
    """Require operator or admin role.

    Args:
        token: JWT token
        rbac: RBAC manager

    Raises:
        HTTPException: If not authorized
    """
    try:
        payload = rbac.verify_token(token)
        if payload.role not in ("admin", "operator"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator or admin role required",
            )
    except RBACError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


async def check_kill_switch(
    kill_switch: Annotated[KillSwitch, Depends(get_kill_switch)],
) -> None:
    """Check kill switch status.

    Args:
        kill_switch: KillSwitch instance

    Raises:
        HTTPException: If kill switch is active
    """
    if kill_switch.is_active():
        metrics.set_kill_switch_status(True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kill switch active - operations halted",
        )
    metrics.set_kill_switch_status(False)
