"""Verify command."""

import json
import sys
from pathlib import Path

import click

from app.core.gate import GateError, ManifestGate
from app.core.keys import KeyManager
from app.core.models import SignedEnvelope


@click.command(name="verify")
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.File("r"),
    default=sys.stdin,
    help="Signed envelope JSON (default: stdin)",
)
@click.option(
    "--keys-dir",
    type=click.Path(path_type=Path),
    default=Path("./data/keys"),
    help="Keys directory",
)
@click.option("--key-id", default="default", help="Key ID to use")
def verify_cmd(
    input_file: click.File,
    keys_dir: Path,
    key_id: str,
) -> None:
    """Verify a signed envelope."""
    try:
        # Load envelope
        envelope_data = json.load(input_file)  # type: ignore
        envelope = SignedEnvelope.model_validate(envelope_data)

        # Load verify key
        key_manager = KeyManager(keys_dir)
        verify_key = key_manager.load_verify_key(key_id)

        # Verify
        gate = ManifestGate(verify_key)
        manifest = gate.verify_envelope(envelope)
        manifest_hash = gate.hash_manifest(manifest)

        click.echo(f"✓ Signature valid")
        click.echo(f"  Manifest hash: {manifest_hash}")
        click.echo(f"  Targets: {', '.join(manifest.targets)}")
        click.echo(f"  Edits: {len(manifest.edits)}")
        click.echo(f"  Tests: {len(manifest.tests)}")

    except GateError as e:
        click.echo(f"✗ Verification failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
