"""OCI sandbox execution using rootless podman."""

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.core.models import SandboxResult


class SandboxConfig(BaseModel):
    """Sandbox configuration."""

    runtime: str = Field(default="podman", description="Container runtime")
    image: str = Field(default="python:3.12-slim", description="Container image")
    cpu_limit: str = Field(default="1.0", description="CPU limit")
    memory_limit: str = Field(default="512m", description="Memory limit")
    timeout_seconds: int = Field(default=300, description="Execution timeout")
    network_mode: str = Field(default="none", description="Network mode")
    read_only: bool = Field(default=True, description="Mount project read-only")


class SandboxRunner:
    """Runs commands in isolated OCI containers."""

    def __init__(self, config: SandboxConfig, project_root: Path) -> None:
        """Initialize sandbox runner.

        Args:
            config: Sandbox configuration
            project_root: Project root directory
        """
        self.config = config
        self.project_root = project_root

    def run_tests(self, test_commands: list[str]) -> SandboxResult:
        """Run test commands in sandbox.

        Args:
            test_commands: List of commands to run

        Returns:
            SandboxResult with execution results
        """
        if not test_commands:
            return SandboxResult(
                rc=0, stdout="No tests specified", stderr="", duration_ms=0.0, timed_out=False
            )

        # Combine commands
        combined_command = " && ".join(test_commands)

        return self.run_command(combined_command)

    def run_command(self, command: str, env: dict[str, str] | None = None) -> SandboxResult:
        """Run command in sandbox.

        Args:
            command: Command to execute
            env: Environment variables

        Returns:
            SandboxResult with execution results
        """
        start_time = time.time()

        # Build podman command
        podman_cmd = [
            self.config.runtime,
            "run",
            "--rm",
            "--network",
            self.config.network_mode,
            "--cpus",
            self.config.cpu_limit,
            "--memory",
            self.config.memory_limit,
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
        ]

        # Mount project
        mount_mode = "ro" if self.config.read_only else "rw"
        podman_cmd.extend(
            ["-v", f"{self.project_root.absolute()}:/workspace:{mount_mode}", "-w", "/workspace"]
        )

        # Add tmpfs for temporary files
        podman_cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=128m"])

        # Add environment variables
        if env:
            for key, value in env.items():
                podman_cmd.extend(["-e", f"{key}={value}"])

        # Add image and command
        podman_cmd.append(self.config.image)
        podman_cmd.extend(["/bin/bash", "-c", command])

        # Execute
        try:
            result = subprocess.run(
                podman_cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )

            duration_ms = (time.time() - start_time) * 1000

            return SandboxResult(
                rc=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=duration_ms,
                timed_out=False,
            )

        except subprocess.TimeoutExpired as e:
            duration_ms = (time.time() - start_time) * 1000

            return SandboxResult(
                rc=-1,
                stdout=e.stdout.decode("utf-8") if e.stdout else "",
                stderr=e.stderr.decode("utf-8") if e.stderr else "Timeout",
                duration_ms=duration_ms,
                timed_out=True,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return SandboxResult(
                rc=-1, stdout="", stderr=str(e), duration_ms=duration_ms, timed_out=False
            )

    def validate_runtime(self) -> bool:
        """Validate that container runtime is available.

        Returns:
            True if runtime is available
        """
        try:
            result = subprocess.run(
                [self.config.runtime, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def pull_image(self) -> bool:
        """Pull container image.

        Returns:
            True if successful
        """
        try:
            subprocess.run(
                [self.config.runtime, "pull", self.config.image],
                capture_output=True,
                timeout=300,
                check=True,
            )
            return True
        except Exception:
            return False
