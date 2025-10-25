"""Rollback command."""

import sys
from pathlib import Path

import click

from app.core.audit import AuditLog
from app.core.kill import KillSwitch


@click.command(name="rollback")
@click.option("--hash", "manifest_hash", help="Manifest hash to rollback")
@click.option("--index", "audit_index", type=int, help="Audit index to rollback")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    help="Data directory",
)
def rollback_cmd(
    manifest_hash: str | None,
    audit_index: int | None,
    data_dir: Path,
) -> None:
    """Rollback a change."""
    try:
        # Check kill switch
        kill_switch = KillSwitch()
        kill_switch.check_or_raise()

        if not manifest_hash and audit_index is None:
            raise click.ClickException("Must specify --hash or --index")

        # Find record
        audit_log = AuditLog(data_dir / "audit.log")

        if manifest_hash:
            record = audit_log.find_by_manifest_hash(manifest_hash)
        else:
            record = audit_log.get_record_by_index(audit_index)  # type: ignore

        if not record:
            raise click.ClickException("Record not found")

        if not record.applied:
            raise click.ClickException("Record was not applied, cannot rollback")

        click.echo(f"Rollback not yet implemented for: {record.manifest_sha256}")
        click.echo("This would revert the changes in the manifest.")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
