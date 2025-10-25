"""Sign command."""

import json
import sys
from pathlib import Path

import click

from app.core.gate import ManifestGate
from app.core.keys import KeyManager
from app.core.models import ChangeManifest


@click.command(name="sign")
@click.option(
    "--manifest",
    "-m",
    type=click.File("r"),
    default=sys.stdin,
    help="Manifest JSON file (default: stdin)",
)
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default=sys.stdout,
    help="Output file (default: stdout)",
)
@click.option(
    "--keys-dir",
    type=click.Path(path_type=Path),
    default=Path("./data/keys"),
    help="Keys directory",
)
@click.option("--key-id", default="default", help="Key ID to use")
def sign_cmd(
    manifest: click.File,
    output: click.File,
    keys_dir: Path,
    key_id: str,
) -> None:
    """Sign a change manifest."""
    try:
        # Load manifest
        manifest_data = json.load(manifest)  # type: ignore
        change_manifest = ChangeManifest.model_validate(manifest_data)

        # Load signing key
        key_manager = KeyManager(keys_dir)
        signing_key = key_manager.load_signing_key(key_id)

        # Sign
        gate = ManifestGate(signing_key.verify_key)
        envelope = gate.sign_manifest(change_manifest, signing_key, key_id)

        # Output
        output.write(envelope.model_dump_json(indent=2))  # type: ignore
        output.write("\n")  # type: ignore

        click.echo(f"✓ Signed with key: {key_id}", err=True)

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
