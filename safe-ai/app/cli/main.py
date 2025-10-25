"""Main CLI entry point."""

import sys
from pathlib import Path

import click

from app.cli.cmds import apply, audit, keys, policy, rollback, sandbox, sign, verify


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """SAFE-AI Governor - Cryptographically-gated AI change management."""
    pass


# Register commands
cli.add_command(sign.sign_cmd)
cli.add_command(verify.verify_cmd)
cli.add_command(policy.policy_cmd)
cli.add_command(apply.apply_cmd)
cli.add_command(rollback.rollback_cmd)
cli.add_command(audit.audit_cmd)
cli.add_command(keys.keys_cmd)
cli.add_command(sandbox.sandbox_cmd)


if __name__ == "__main__":
    cli()
