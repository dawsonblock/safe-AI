"""Test policy VM."""

import pytest

from app.core.models import ChangeManifest, Edit, EditOperation, PolicyVerdict
from app.core.policy_vm import PolicyConfig, PolicyVM


def test_policy_allows_valid_manifest() -> None:
    """Test policy allows valid manifest."""
    config = PolicyConfig()
    policy_vm = PolicyVM(config)

    manifest = ChangeManifest(
        targets=["src/module.py"],
        edits=[
            Edit(
                path="src/module.py",
                op=EditOperation.REPLACE,
                start=0,
                end=5,
                text="def new_function():\n    pass\n",
            )
        ],
        tests=["pytest"],
        rationale="Add new function for feature X",
    )

    result = policy_vm.evaluate(manifest)

    assert result.verdict == PolicyVerdict.APPROVED
    assert len(result.reasons) == 0


def test_policy_blocks_denied_path() -> None:
    """Test policy blocks denied paths."""
    config = PolicyConfig()
    policy_vm = PolicyVM(config)

    manifest = ChangeManifest(
        targets=[".git/config"],
        edits=[
            Edit(
                path=".git/config", op=EditOperation.REPLACE, start=0, end=1, text="malicious"
            )
        ],
        rationale="Test",
    )

    result = policy_vm.evaluate(manifest)

    assert result.verdict == PolicyVerdict.BLOCKED
    assert ".git/config" in result.blocked_paths


def test_policy_blocks_dangerous_pattern() -> None:
    """Test policy blocks dangerous patterns."""
    config = PolicyConfig()
    policy_vm = PolicyVM(config)

    manifest = ChangeManifest(
        targets=["src/evil.py"],
        edits=[
            Edit(
                path="src/evil.py",
                op=EditOperation.REPLACE,
                start=0,
                end=1,
                text="import subprocess\nsubprocess.Popen(['rm', '-rf', '/'])\n",
            )
        ],
        rationale="Dangerous code",
    )

    result = policy_vm.evaluate(manifest)

    assert result.verdict == PolicyVerdict.BLOCKED
    assert len(result.blocked_patterns) > 0


def test_policy_blocks_short_rationale() -> None:
    """Test policy blocks short rationale."""
    config = PolicyConfig()
    policy_vm = PolicyVM(config)

    manifest = ChangeManifest(
        targets=["src/test.py"],
        edits=[Edit(path="src/test.py", op=EditOperation.REPLACE, start=0, end=1, text="x")],
        rationale="short",
    )

    result = policy_vm.evaluate(manifest)

    assert result.verdict == PolicyVerdict.BLOCKED
    assert any("rationale" in r.lower() for r in result.reasons)


def test_policy_blocks_oversized_edit() -> None:
    """Test policy blocks oversized edits."""
    config = PolicyConfig(max_edit_size_bytes=100)
    policy_vm = PolicyVM(config)

    large_text = "x" * 200

    manifest = ChangeManifest(
        targets=["src/test.py"],
        edits=[
            Edit(path="src/test.py", op=EditOperation.REPLACE, start=0, end=1, text=large_text)
        ],
        rationale="Test large edit",
    )

    result = policy_vm.evaluate(manifest)

    assert result.verdict == PolicyVerdict.BLOCKED
    assert any("size" in r.lower() for r in result.reasons)
