"""Append-only audit log with Merkle chain."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.models import AuditRecord


class AuditLog:
    """Append-only audit log with Merkle chaining."""

    def __init__(self, log_path: Path) -> None:
        """Initialize audit log.

        Args:
            log_path: Path to audit log file
        """
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with genesis root if empty
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            self._write_genesis()

    def _write_genesis(self) -> None:
        """Write genesis record."""
        genesis = {
            "ts": datetime.utcnow().isoformat(),
            "event": "genesis",
            "merkle_root": self._compute_genesis_root(),
            "version": "1.0.0",
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(genesis) + "\n")

    def _compute_genesis_root(self) -> str:
        """Compute genesis Merkle root.

        Returns:
            Hex-encoded SHA-256 hash
        """
        return hashlib.sha256(b"SAFEAI_GENESIS_2025").hexdigest()

    def append(self, record: AuditRecord) -> int:
        """Append record to audit log.

        Args:
            record: AuditRecord to append

        Returns:
            Record index (0-based)
        """
        # Get previous root
        prev_root = self.get_current_root()

        # Compute new root
        record_dict = record.model_dump(mode="json")
        record_dict["prev_root"] = prev_root
        record_json = json.dumps(record_dict, sort_keys=True)

        # Compute Merkle root: hash(prev_root || record_json)
        merkle_data = f"{prev_root}{record_json}".encode("utf-8")
        new_root = hashlib.sha256(merkle_data).hexdigest()

        # Update record with new root
        record.merkle_root = new_root
        record.prev_root = prev_root

        # Append to log
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

        # Return index
        return self._count_records() - 1

    def get_current_root(self) -> str:
        """Get current Merkle root.

        Returns:
            Current Merkle root hash
        """
        if not self.log_path.exists():
            return self._compute_genesis_root()

        # Read last line
        with open(self.log_path, "rb") as f:
            try:
                # Seek to end
                f.seek(-2, 2)

                # Find last newline
                while f.read(1) != b"\n":
                    f.seek(-2, 1)

                # Read last line
                last_line = f.readline().decode("utf-8")
                record = json.loads(last_line)

                return record.get("merkle_root", self._compute_genesis_root())

            except (OSError, json.JSONDecodeError):
                return self._compute_genesis_root()

    def get_records(self, limit: int = 100, offset: int = 0) -> list[AuditRecord]:
        """Get audit records.

        Args:
            limit: Maximum number of records
            offset: Offset from start

        Returns:
            List of AuditRecords
        """
        if not self.log_path.exists():
            return []

        records: list[AuditRecord] = []

        with open(self.log_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx < offset:
                    continue

                if len(records) >= limit:
                    break

                try:
                    record_data = json.loads(line)

                    # Skip genesis
                    if record_data.get("event") == "genesis":
                        continue

                    record = AuditRecord.model_validate(record_data)
                    records.append(record)

                except (json.JSONDecodeError, ValueError):
                    continue

        return records

    def get_record_by_index(self, index: int) -> AuditRecord | None:
        """Get record by index.

        Args:
            index: Record index (0-based, excluding genesis)

        Returns:
            AuditRecord or None
        """
        if not self.log_path.exists():
            return None

        current_idx = -1  # Start at -1 to account for genesis

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record_data = json.loads(line)

                    # Skip genesis
                    if record_data.get("event") == "genesis":
                        continue

                    current_idx += 1

                    if current_idx == index:
                        return AuditRecord.model_validate(record_data)

                except (json.JSONDecodeError, ValueError):
                    continue

        return None

    def find_by_manifest_hash(self, manifest_hash: str) -> AuditRecord | None:
        """Find record by manifest hash.

        Args:
            manifest_hash: Manifest SHA-256 hash

        Returns:
            AuditRecord or None
        """
        if not self.log_path.exists():
            return None

        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record_data = json.loads(line)

                    # Skip genesis
                    if record_data.get("event") == "genesis":
                        continue

                    record = AuditRecord.model_validate(record_data)

                    if record.manifest_sha256 == manifest_hash:
                        return record

                except (json.JSONDecodeError, ValueError):
                    continue

        return None

    def verify_chain(self) -> tuple[bool, str]:
        """Verify Merkle chain integrity.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.log_path.exists():
            return True, ""

        prev_root = self._compute_genesis_root()

        with open(self.log_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                try:
                    record_data = json.loads(line)

                    # Skip genesis
                    if record_data.get("event") == "genesis":
                        prev_root = record_data["merkle_root"]
                        continue

                    # Check prev_root matches
                    if record_data.get("prev_root") != prev_root:
                        return False, f"Chain broken at record {idx}: prev_root mismatch"

                    # Recompute root
                    record_copy = record_data.copy()
                    merkle_root = record_copy.pop("merkle_root")

                    record_json = json.dumps(record_copy, sort_keys=True)
                    merkle_data = f"{prev_root}{record_json}".encode("utf-8")
                    computed_root = hashlib.sha256(merkle_data).hexdigest()

                    if computed_root != merkle_root:
                        return False, f"Chain broken at record {idx}: root mismatch"

                    prev_root = merkle_root

                except (json.JSONDecodeError, ValueError) as e:
                    return False, f"Invalid record at line {idx}: {e}"

        return True, ""

    def _count_records(self) -> int:
        """Count total records (excluding genesis).

        Returns:
            Number of records
        """
        if not self.log_path.exists():
            return 0

        count = 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record_data = json.loads(line)
                    if record_data.get("event") != "genesis":
                        count += 1
                except json.JSONDecodeError:
                    continue

        return count
