"""Archive extraction.

Purpose
-------
Expands archives (ZIP and RAR) before ingestion so their contents can be processed
as normal files.

What it does
------------
Finds archives in the input directory and extracts them into a dedicated folder,
using built-in ZIP support and an available local backend for RAR, with safe
extraction that avoids path-escape issues.

Flow
----
``extract_archives()`` locates archives, creates a deterministic output folder for
each, extracts safely, counts the extracted files, and returns a report.
"""

import shutil
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from backend.app.ingestion.metadata import normalize_identifier, short_hash


ARCHIVE_EXTENSIONS = {".zip", ".rar"}


@dataclass(frozen=True)
class ArchiveExtractionReport:
    """
    Summary of one archive extraction run.
    """

    input_dir: str
    extract_dir: str
    archives_seen: int = 0
    archives_extracted: int = 0
    archives_skipped: int = 0
    files_created: int = 0
    errors: list[str] = field(default_factory=list)


def iter_archive_files(input_dir: str | Path, extract_dir: str | Path) -> list[Path]:
    """
    Return archive files from input_dir recursively.

    Files inside extract_dir are ignored to avoid re-extracting already
    extracted archives.
    """

    input_root = Path(input_dir)
    extract_root = Path(extract_dir)

    archives: list[Path] = []

    if not input_root.exists():
        return archives

    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if path.suffix.lower() not in ARCHIVE_EXTENSIONS:
            continue

        try:
            path.relative_to(extract_root)
            continue
        except ValueError:
            pass

        archives.append(path)

    return archives


def archive_output_dir(archive_path: Path, input_root: Path, extract_root: Path) -> Path:
    """
    Create a deterministic output folder for an archive.

    The hash prevents collisions when archives have the same filename.
    """

    try:
        relative = archive_path.relative_to(input_root)
    except ValueError:
        relative = archive_path.name

    relative_text = str(relative).replace("\\", "/")
    safe_name = normalize_identifier(archive_path.stem)
    path_hash = short_hash(relative_text)

    return extract_root / f"{safe_name}-{path_hash}"


def extract_archives(
    input_dir: str | Path,
    extract_dir: str | Path,
    clear_extract_dir: bool = False,
) -> ArchiveExtractionReport:
    """
    Extract ZIP/RAR archives into extract_dir.

    Args:
        input_dir:
            Root folder containing source files and archives.

        extract_dir:
            Folder where archives should be extracted.

        clear_extract_dir:
            If true, delete extract_dir before extraction.

    Returns:
        ArchiveExtractionReport:
            Extraction summary.
    """

    input_root = Path(input_dir)
    extract_root = Path(extract_dir)

    if clear_extract_dir and extract_root.exists():
        shutil.rmtree(extract_root)

    extract_root.mkdir(parents=True, exist_ok=True)

    archives = iter_archive_files(input_root, extract_root)

    archives_extracted = 0
    archives_skipped = 0
    files_created_total = 0
    errors: list[str] = []

    for archive_path in archives:
        destination = archive_output_dir(archive_path, input_root, extract_root)
        destination.mkdir(parents=True, exist_ok=True)

        before_files = _count_files(destination)

        try:
            suffix = archive_path.suffix.lower()

            if suffix == ".zip":
                _extract_zip(archive_path, destination)

            elif suffix == ".rar":
                _extract_rar(archive_path, destination)

            else:
                raise ValueError(f"Unsupported archive type: {suffix}")

            after_files = _count_files(destination)
            files_created_total += max(0, after_files - before_files)
            archives_extracted += 1

        except Exception as exc:
            archives_skipped += 1
            errors.append(f"{archive_path}: {exc}")

    return ArchiveExtractionReport(
        input_dir=str(input_root),
        extract_dir=str(extract_root),
        archives_seen=len(archives),
        archives_extracted=archives_extracted,
        archives_skipped=archives_skipped,
        files_created=files_created_total,
        errors=errors,
    )


def _extract_zip(archive_path: Path, destination: Path) -> None:
    """
    Extract a ZIP archive safely.

    Prevents path traversal by ensuring each output path stays inside the
    destination folder.
    """

    with zipfile.ZipFile(archive_path, "r") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            target_path = destination / member.filename
            resolved_target = target_path.resolve()
            resolved_destination = destination.resolve()

            if not str(resolved_target).startswith(str(resolved_destination)):
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")

            target_path.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(member, "r") as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def _extract_rar(archive_path: Path, destination: Path) -> None:
    """
    Extract a RAR archive using an available local backend.

    Supported backends checked in order:
        - 7z from PATH
        - 7zz from PATH
        - 7za from PATH
        - common Windows 7-Zip install paths
        - unrar from PATH
        - bsdtar from PATH

    This avoids requiring users to manually add 7-Zip to PATH.
    """

    seven_zip = _find_7zip()
    unrar = shutil.which("unrar")
    bsdtar = shutil.which("bsdtar")

    if seven_zip:
        command = [
            seven_zip,
            "x",
            "-y",
            f"-o{str(destination)}",
            str(archive_path),
        ]
        _run_command(command)
        return

    if unrar:
        command = [
            unrar,
            "x",
            "-y",
            str(archive_path),
            str(destination),
        ]
        _run_command(command)
        return

    if bsdtar:
        command = [
            bsdtar,
            "-xf",
            str(archive_path),
            "-C",
            str(destination),
        ]
        _run_command(command)
        return

    raise RuntimeError(
        "No RAR extraction backend found. Install 7-Zip, unrar, or bsdtar, "
        "or manually extract the RAR files into data/input/_extracted/. "
        "If 7-Zip is installed, confirm that C:\\Program Files\\7-Zip\\7z.exe exists."
    )


def _find_7zip() -> str | None:
    """
    Find a 7-Zip executable.

    Checks PATH first, then common Windows install locations.
    """

    path_candidates = [
        shutil.which("7z"),
        shutil.which("7zz"),
        shutil.which("7za"),
    ]

    windows_candidates = [
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
        Path(r"C:\Program Files\7-Zip\7zz.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7zz.exe"),
    ]

    for candidate in path_candidates:
        if candidate:
            return candidate

    for candidate in windows_candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _run_command(command: list[str]) -> None:
    """
    Run an external extraction command.
    """

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Archive extraction command failed.\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT: {completed.stdout}\n"
            f"STDERR: {completed.stderr}"
        )


def _count_files(path: Path) -> int:
    """
    Count files under a folder recursively.
    """

    if not path.exists():
        return 0

    return sum(1 for item in path.rglob("*") if item.is_file())