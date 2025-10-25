"""Policy command."""

import json
import sys
from pathlib import Path

import click

from app.core.models import ChangeManifest
from app.core.policy_vm import PolicyConfig, PolicyVM


@click.command(name="plan")
@click.option(
    "--manifest",
    "-m",
    type=click.File("r"),
    required=True,
    help="Manifest JSON file",
)
def policy_cmd(manifest: click.File) -> None:
    """Evaluate manifest against policy."""
    try:
        # Load manifest
        manifest_data = json.load(manifest)  # type: ignore
        change_manifest = ChangeManifest.model_validate(manifest_data)

        # Evaluate policy
        config = PolicyConfig()
        policy_vm = PolicyVM(config)
        result = policy_vm.evaluate(change_manifest)

        # Output
        if result.verdict.value == "approved":
            click.echo("✓ APPROVED", err=True)
            sys.exit(0)
        else:
            click.echo("✗ BLOCKED", err=True)
            for reason in result.reasons:
                click.echo(f"  - {reason}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
