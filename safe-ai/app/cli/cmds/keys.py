"""Keys command."""

import sys
from pathlib import Path

import click

from app.core.keys import KeyManager


@click.group(name="keys")
def keys_cmd() -> None:
    """Manage cryptographic keys."""
    pass


@keys_cmd.command(name="generate")
@click.option(
    "--keys-dir",
    type=click.Path(path_type=Path),
    default=Path("./data/keys"),
    help="Keys directory",
)
@click.option("--key-id", default="default", help="Key ID")
def generate_keys(keys_dir: Path, key_id: str) -> None:
    """Generate new Ed25519 keypair."""
    try:
        key_manager = KeyManager(keys_dir)
        keypair = key_manager.generate_keypair(key_id)
        key_manager.save_keypair(keypair)

        click.echo(f"✓ Generated keypair: {key_id}")
        click.echo(f"  Keys directory: {keys_dir}")
        click.echo(f"  Public key: ed25519.{key_id}.vk")
        click.echo(f"  Private key: ed25519.{key_id}.sk")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@keys_cmd.command(name="list")
@click.option(
    "--keys-dir",
    type=click.Path(path_type=Path),
    default=Path("./data/keys"),
    help="Keys directory",
)
def list_keys(keys_dir: Path) -> None:
    """List available keys."""
    try:
        key_manager = KeyManager(keys_dir)
        keys = key_manager.list_keys()

        if not keys:
            click.echo("No keys found")
            return

        click.echo("Available keys:")
        for key_id in keys:
            try:
                info = key_manager.get_key_info(key_id)
                click.echo(f"  - {key_id} ({info.algorithm})")
                click.echo(f"    Created: {info.created_at}")
            except Exception:
                click.echo(f"  - {key_id} (error reading info)")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@keys_cmd.command(name="rotate")
@click.option(
    "--keys-dir",
    type=click.Path(path_type=Path),
    default=Path("./data/keys"),
    help="Keys directory",
)
@click.option("--key-id", default="default", help="Key ID to rotate")
def rotate_keys(keys_dir: Path, key_id: str) -> None:
    """Rotate keys."""
    try:
        key_manager = KeyManager(keys_dir)
        new_keypair = key_manager.rotate_keys(key_id)

        click.echo(f"✓ Rotated keys: {key_id}")
        click.echo(f"  New key ID: {new_keypair.key_id}")
        click.echo(f"  Old keys archived")

    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
