"""Audit command."""

import json
import sys
from pathlib import Path

import click

from app.core.audit import AuditLog


@click.command(name="audit")
@click.option("--tail", "-n", default=50, help="Number of records to show")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    help="Data directory",
)
@click.option("--verify", is_flag=True, help="Verify Merkle chain")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def audit_cmd(
    tail: int,
    data_dir: Path,
    verify: bool,
    json_output: bool,
) -> None:
    """View audit log."""
    try:
        audit_log = AuditLog(data_dir / "audit.log")

        if verify:
            is_valid, error = audit_log.verify_chain()
            if is_valid:
                click.echo("✓ Audit chain valid")
                click.echo(f"  Current root: {audit_log.get_current_root()}")
            else:
                click.echo(f"✗ Audit chain invalid: {error}", err=True)
                sys.exit(1)
            return

        # Get records
        records = audit_log.get_records(limit=tail)

        if json_output:
            output = [r.model_dump(mode="json") for r in records]
            click.echo(json.dumps(output, indent=2))
        else:
            click.echo(f"Last {len(records)} audit record(s):\n")
            for idx, record in enumerate(records):
                click.echo(f"[{idx}] {record.ts}")
                click.echo(f"    Actor: {record.actor}")
                click.echo(f"    Manifest: {record.manifest_sha256[:16]}...")
                click.echo(f"    Verdict: {record.policy_verdict.value}")
                click.echo(f"    Applied: {record.applied}")
                if record.error:
                    click.echo(f"    Error: {record.error}")
                click.echo(f"    Merkle: {record.merkle_root[:16]}...")
                click.echo()

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
