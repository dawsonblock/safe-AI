"""Apply command."""

import json
import sys
from pathlib import Path

import click

from app.core.apply import ApplyError, FileApplicator
from app.core.audit import AuditLog
from app.core.gate import GateError, ManifestGate
from app.core.keys import KeyManager
from app.core.kill import KillSwitch
from app.core.models import AuditRecord, ChangeManifest, PolicyVerdict, SignedEnvelope
from app.core.policy_vm import PolicyConfig, PolicyVM
from app.core.sandbox import SandboxConfig, SandboxRunner


@click.command(name="apply")
@click.option(
    "--signed",
    "-s",
    type=click.File("r"),
    required=True,
    help="Signed envelope JSON",
)
@click.option(
    "--manifest",
    "-m",
    type=click.File("r"),
    required=True,
    help="Manifest JSON file",
)
@click.option(
    "--keys-dir",
    type=click.Path(path_type=Path),
    default=Path("./data/keys"),
    help="Keys directory",
)
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=Path("./data"),
    help="Data directory",
)
@click.option(
    "--project-root",
    type=click.Path(path_type=Path),
    default=Path("."),
    help="Project root directory",
)
def apply_cmd(
    signed: click.File,
    manifest: click.File,
    keys_dir: Path,
    data_dir: Path,
    project_root: Path,
) -> None:
    """Apply a signed change."""
    try:
        # Check kill switch
        kill_switch = KillSwitch()
        kill_switch.check_or_raise()

        # Load envelope and manifest
        envelope_data = json.load(signed)  # type: ignore
        envelope = SignedEnvelope.model_validate(envelope_data)

        manifest_data = json.load(manifest)  # type: ignore
        change_manifest = ChangeManifest.model_validate(manifest_data)

        # Verify signature
        key_manager = KeyManager(keys_dir)
        verify_key = key_manager.load_verify_key("default")
        gate = ManifestGate(verify_key)

        verified_manifest = gate.verify_envelope(envelope)
        manifest_hash = gate.hash_manifest(verified_manifest)

        click.echo(f"✓ Signature verified: {manifest_hash}")

        # Verify manifest matches
        if not gate.verify_manifest_hash(change_manifest, manifest_hash):
            raise click.ClickException("Manifest hash mismatch")

        # Evaluate policy
        policy_config = PolicyConfig()
        policy_vm = PolicyVM(policy_config)
        policy_result = policy_vm.evaluate(change_manifest)

        if policy_result.verdict == PolicyVerdict.BLOCKED:
            click.echo("✗ Blocked by policy:", err=True)
            for reason in policy_result.reasons:
                click.echo(f"  - {reason}", err=True)
            sys.exit(1)

        click.echo("✓ Policy check passed")

        # Run tests in sandbox
        sandbox_result = None
        if change_manifest.tests:
            click.echo(f"Running {len(change_manifest.tests)} test(s)...")
            sandbox_config = SandboxConfig()
            sandbox = SandboxRunner(sandbox_config, project_root)

            sandbox_result = sandbox.run_tests(change_manifest.tests)

            if sandbox_result.rc != 0:
                click.echo(f"✗ Tests failed (exit code: {sandbox_result.rc})", err=True)
                click.echo(sandbox_result.stdout)
                click.echo(sandbox_result.stderr, err=True)
                sys.exit(1)

            click.echo("✓ Tests passed")

        # Apply changes
        click.echo("Applying changes...")
        applicator = FileApplicator(project_root)
        result = applicator.apply_manifest(change_manifest)
        applicator.cleanup()

        click.echo(f"✓ Applied: {result['files_modified']} files modified")

        # Record in audit log
        audit_log = AuditLog(data_dir / "audit.log")
        audit_record = AuditRecord(
            actor="cli",
            manifest_sha256=manifest_hash,
            policy_verdict=policy_result.verdict,
            sandbox_result=sandbox_result,
            merkle_root="",
            prev_root="",
            applied=True,
        )
        audit_index = audit_log.append(audit_record)

        click.echo(f"✓ Audit record: #{audit_index}")
        click.echo(f"  Merkle root: {audit_log.get_current_root()}")

    except GateError as e:
        click.echo(f"✗ Gate error: {e}", err=True)
        sys.exit(1)
    except ApplyError as e:
        click.echo(f"✗ Apply error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)
