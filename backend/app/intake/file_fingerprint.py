"""File fingerprinting for change detection.

Purpose
-------
Decides whether an input file is new, unchanged, or changed without running it
through the expensive ingestion pipeline first.

What it does
------------
Builds a stable fingerprint per file (path, size, modified time, and a content
hash) and normalizes paths so comparisons are reliable across runs and platforms.

Flow
----
A file is fingerprinted and compared against the stored fingerprint; a difference
in content hash means it changed and must be reprocessed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class FileFingerprint:
    """
    Stable fingerprint for one file.
    """

    file_path: str
    normalized_path: str
    file_name: str
    extension: str
    size_bytes: int
    modified_time_ns: int
    modified_time_iso: str
    sha256: str

    @property
    def signature(self) -> str:
        """
        Compact signature used for quick equality checks.
        """

        return f"{self.size_bytes}:{self.modified_time_ns}:{self.sha256}"

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "normalized_path": self.normalized_path,
            "file_name": self.file_name,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "modified_time_ns": self.modified_time_ns,
            "modified_time_iso": self.modified_time_iso,
            "sha256": self.sha256,
            "signature": self.signature,
        }


def normalize_relative_path(path: str | Path, root: str | Path) -> str:
    """
    Return a stable path key relative to the configured input root.

    Windows paths are case-insensitive in practice, so this normalizes to lower
    case and forward slashes. That prevents the same file from being tracked as
    different records because of slash style or drive-letter casing.
    """

    file_path = Path(path).resolve()
    root_path = Path(root).resolve()

    try:
        rel = file_path.relative_to(root_path)
    except ValueError:
        # Fallback for older rows or externally supplied paths. This should not
        # happen for normal ingestion, but it keeps cleanup code defensive.
        rel = Path(path)

    return rel.as_posix().replace("\\", "/").lower()


def normalize_stored_path(path: str | Path, root: str | Path) -> str:
    """
    Normalize a path read from SQLite metadata.

    Existing rows may store relative paths such as data/input/foo.pdf or absolute
    paths. This helper maps both forms to the same normalized relative key used
    by the file registry.
    """

    raw = Path(path)
    root_path = Path(root).resolve()

    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((root_path.parent.parent / raw).resolve())
        candidates.append((root_path / raw).resolve())

    for candidate in candidates:
        try:
            return normalize_relative_path(candidate, root_path)
        except Exception:
            continue

    # Last-resort string normalization. Useful for comparison, not for opening.
    text = str(path).replace("\\", "/")
    marker = str(Path(root).as_posix()).replace("\\", "/").rstrip("/").lower() + "/"
    lower = text.lower()
    if marker in lower:
        return lower.split(marker, 1)[1]
    return lower


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute a streaming SHA-256 hash.

    This reads the full file. For this PoC-scale dataset, correctness is more
    important than shaving a few seconds from the scan. The expensive operations
    are OCR/vision, not hashing.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: str | Path, input_root: str | Path) -> FileFingerprint:
    """
    Build a fingerprint for one input file.
    """

    file_path = Path(path)
    stat = file_path.stat()
    modified_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)

    return FileFingerprint(
        file_path=str(file_path),
        normalized_path=normalize_relative_path(file_path, input_root),
        file_name=file_path.name,
        extension=file_path.suffix.lower(),
        size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
        modified_time_iso=modified_dt.isoformat(),
        sha256=sha256_file(file_path),
    )
