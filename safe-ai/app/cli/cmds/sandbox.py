"""Sandbox command."""

import sys
from pathlib import Path

import click

from app.core.sandbox import SandboxConfig, SandboxRunner


@click.command(name="sandbox")
@click.argument("command", nargs=-1, required=True)
@click.option(
    "--project-root",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Project root directory",
)
def sandbox_cmd(command: tuple[str, ...], project_root: Path) -> None:
    """Run command in sandbox."""
    try:
        config = SandboxConfig()
        sandbox = SandboxRunner(config, project_root)

        # Validate runtime
        if not sandbox.validate_runtime():
            raise click.ClickException(f"Container runtime '{config.runtime}' not available")

        # Join command
        cmd_str = " ".join(command)
        click.echo(f"Running in sandbox: {cmd_str}")

        # Execute
        result = sandbox.run_command(cmd_str)

        # Output
        click.echo(result.stdout, nl=False)
        if result.stderr:
            click.echo(result.stderr, err=True, nl=False)

        if result.timed_out:
            click.echo(f"✗ Timeout after {result.duration_ms / 1000:.1f}s", err=True)
            sys.exit(124)

        if result.rc != 0:
            click.echo(f"✗ Exit code: {result.rc}", err=True)

        sys.exit(result.rc)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
