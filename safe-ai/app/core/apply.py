"""Atomic file operations with rollback support."""

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.core.models import ChangeManifest, Edit, EditOperation


class ApplyError(Exception):
    """Error applying changes."""

    pass


class FileApplicator:
    """Applies edits to files atomically with rollback."""

    def __init__(self, project_root: Path) -> None:
        """Initialize applicator.

        Args:
            project_root: Root directory of project
        """
        self.project_root = project_root
        self.backup_dir: Path | None = None
        self.applied_files: list[Path] = []

    def apply_manifest(self, manifest: ChangeManifest) -> dict[str, Any]:
        """Apply manifest changes atomically.

        Args:
            manifest: ChangeManifest to apply

        Returns:
            Dict with results

        Raises:
            ApplyError: If application fails
        """
        # Create backup directory
        self.backup_dir = Path(tempfile.mkdtemp(prefix="safeai_backup_"))

        try:
            # Group edits by file
            edits_by_file: dict[str, list[Edit]] = {}
            for edit in manifest.edits:
                if edit.path not in edits_by_file:
                    edits_by_file[edit.path] = []
                edits_by_file[edit.path].append(edit)

            # Apply edits to each file
            for file_path, edits in edits_by_file.items():
                self._apply_edits_to_file(file_path, edits)

            return {"files_modified": len(self.applied_files), "success": True}

        except Exception as e:
            # Rollback on error
            self.rollback()
            raise ApplyError(f"Failed to apply manifest: {e}") from e

    def _apply_edits_to_file(self, file_path: str, edits: list[Edit]) -> None:
        """Apply edits to a single file atomically.

        Args:
            file_path: Path to file
            edits: List of edits to apply

        Raises:
            ApplyError: If edits fail
        """
        full_path = self.project_root / file_path

        # Backup original file if it exists
        if full_path.exists():
            backup_path = self.backup_dir / file_path  # type: ignore
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full_path, backup_path)

            # Read original content
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []
            full_path.parent.mkdir(parents=True, exist_ok=True)

        # Sort edits by line number (reverse for operations that change line count)
        sorted_edits = sorted(edits, key=lambda e: e.start, reverse=True)

        # Apply each edit
        for edit in sorted_edits:
            lines = self._apply_single_edit(lines, edit)

        # Write atomically using temporary file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=full_path.parent, prefix=f".{full_path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.writelines(lines)

            # Atomic rename
            os.replace(temp_path, full_path)
            self.applied_files.append(full_path)

        except Exception as e:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise ApplyError(f"Failed to write {file_path}: {e}") from e

    def _apply_single_edit(self, lines: list[str], edit: Edit) -> list[str]:
        """Apply single edit to lines.

        Args:
            lines: Current file lines
            edit: Edit to apply

        Returns:
            Modified lines

        Raises:
            ApplyError: If edit is invalid
        """
        # Ensure lines is long enough
        while len(lines) < edit.end + 1:
            lines.append("\n")

        if edit.op == EditOperation.REPLACE:
            # Replace lines from start to end
            new_text = edit.text if edit.text.endswith("\n") else edit.text + "\n"
            new_lines = new_text.splitlines(keepends=True)
            lines[edit.start : edit.end + 1] = new_lines

        elif edit.op == EditOperation.INSERT:
            # Insert at start position
            new_text = edit.text if edit.text.endswith("\n") else edit.text + "\n"
            new_lines = new_text.splitlines(keepends=True)
            lines[edit.start : edit.start] = new_lines

        elif edit.op == EditOperation.DELETE:
            # Delete lines from start to end
            del lines[edit.start : edit.end + 1]

        return lines

    def rollback(self) -> None:
        """Rollback applied changes."""
        if not self.backup_dir or not self.backup_dir.exists():
            return

        # Restore backed up files
        for applied_file in self.applied_files:
            rel_path = applied_file.relative_to(self.project_root)
            backup_path = self.backup_dir / rel_path

            if backup_path.exists():
                shutil.copy2(backup_path, applied_file)

        # Clean up backup directory
        shutil.rmtree(self.backup_dir, ignore_errors=True)
        self.backup_dir = None
        self.applied_files.clear()

    def cleanup(self) -> None:
        """Clean up backup directory after successful apply."""
        if self.backup_dir and self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)
            self.backup_dir = None

    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hex-encoded SHA-256 hash
        """
        full_path = self.project_root / file_path
        if not full_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(full_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()
