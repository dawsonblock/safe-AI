"""Role-based access control with JWT."""

import time
from typing import Literal

import jwt
from pydantic import BaseModel, Field

from app.core.models import JWTPayload

# Role permissions
ROLE_PERMISSIONS = {
    "admin": ["verify", "plan", "apply", "rollback", "audit", "keys"],
    "operator": ["verify", "plan", "apply", "audit"],
    "auditor": ["audit"],
}


class RBACConfig(BaseModel):
    """RBAC configuration."""

    jwt_algorithm: str = Field(default="RS256", description="JWT algorithm")
    jwt_secret: str | None = Field(default=None, description="JWT secret (for HS256)")
    jwt_public_key: str | None = Field(default=None, description="JWT public key (for RS256)")
    jwt_private_key: str | None = Field(default=None, description="JWT private key (for RS256)")
    token_expiry_seconds: int = Field(default=3600, description="Token expiry in seconds")


class RBACManager:
    """Manages role-based access control."""

    def __init__(self, config: RBACConfig) -> None:
        """Initialize RBAC manager.

        Args:
            config: RBAC configuration
        """
        self.config = config

    def create_token(
        self, user_id: str, role: Literal["admin", "operator", "auditor"]
    ) -> str:
        """Create JWT token.

        Args:
            user_id: User identifier
            role: User role

        Returns:
            JWT token string
        """
        now = int(time.time())
        payload = JWTPayload(
            sub=user_id,
            role=role,
            exp=now + self.config.token_expiry_seconds,
            iat=now,
        )

        if self.config.jwt_algorithm.startswith("HS"):
            key = self.config.jwt_secret or "development_secret_key"
        else:
            key = self.config.jwt_private_key or ""

        return jwt.encode(
            payload.model_dump(),
            key,
            algorithm=self.config.jwt_algorithm,
        )

    def verify_token(self, token: str) -> JWTPayload:
        """Verify JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded JWTPayload

        Raises:
            RBACError: If verification fails
        """
        try:
            if self.config.jwt_algorithm.startswith("HS"):
                key = self.config.jwt_secret or "development_secret_key"
            else:
                key = self.config.jwt_public_key or ""

            payload_dict = jwt.decode(
                token,
                key,
                algorithms=[self.config.jwt_algorithm],
            )

            return JWTPayload.model_validate(payload_dict)

        except jwt.ExpiredSignatureError as e:
            raise RBACError("Token expired") from e
        except jwt.InvalidTokenError as e:
            raise RBACError(f"Invalid token: {e}") from e

    def check_permission(
        self, payload: JWTPayload, required_permission: str
    ) -> bool:
        """Check if user has required permission.

        Args:
            payload: JWT payload
            required_permission: Required permission

        Returns:
            True if user has permission
        """
        role_permissions = ROLE_PERMISSIONS.get(payload.role, [])
        return required_permission in role_permissions

    def require_permission(
        self, token: str, required_permission: str
    ) -> JWTPayload:
        """Verify token and check permission.

        Args:
            token: JWT token
            required_permission: Required permission

        Returns:
            JWTPayload if authorized

        Raises:
            RBACError: If not authorized
        """
        payload = self.verify_token(token)

        if not self.check_permission(payload, required_permission):
            raise RBACError(
                f"Insufficient permissions: role '{payload.role}' cannot '{required_permission}'"
            )

        return payload

    def get_role_permissions(self, role: str) -> list[str]:
        """Get permissions for role.

        Args:
            role: Role name

        Returns:
            List of permissions
        """
        return ROLE_PERMISSIONS.get(role, [])


class RBACError(Exception):
    """RBAC error."""

    pass
