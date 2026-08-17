#!/usr/bin/env python3
"""Generate a non-overwriting SHA-256 manifest for selected paths."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDES = (
    ".git/**",
    "**/.git/**",
    "archive/**",
    "scripts/ros2_ws/build/**",
    "scripts/ros2_ws/install/**",
    "scripts/ros2_ws/log/**",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    "**/__pycache__/**",
    "**/*.pyc",
)


def file_fingerprint(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    before_signature = (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    )
    after_signature = (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    )
    if before_signature != after_signature:
        raise RuntimeError(f"File changed while it was being hashed: {path}")
    return after.st_size, digest.hexdigest()


def sha256_file(path: Path) -> str:
    return file_fingerprint(path)[1]


def is_excluded(relative: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def selected_files(root: Path, selected: list[str], excludes: list[str]) -> list[Path]:
    files: set[Path] = set()
    for item in selected:
        unresolved = (
            root / item if not Path(item).is_absolute() else Path(item)
        )
        if unresolved.is_symlink():
            raise ValueError(
                f"Selected path is an unsupported symlink: {unresolved}"
            )
        path = unresolved.resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Included path is outside manifest root: {path}") from exc
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else path.rglob("*")
        for candidate in candidates:
            relative = str(candidate.relative_to(root))
            if is_excluded(relative, excludes):
                continue
            if candidate.is_symlink():
                raise ValueError(
                    f"Selected path contains an unsupported symlink: "
                    f"{relative}"
                )
            if candidate.is_file():
                files.add(candidate)
    return sorted(files, key=lambda path: str(path.relative_to(root)))


def tracked_files(root: Path) -> set[str]:
    git_root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    git_root = Path(git_root_result.stdout.strip()).resolve()
    scope = root.resolve().relative_to(git_root)
    result = subprocess.run(
        ["git", "ls-files", "--full-name", "--", str(scope)],
        cwd=git_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    prefix = f"{scope}/"
    return {
        line[len(prefix):] if line.startswith(prefix) else line
        for line in result.stdout.splitlines()
    }


def _output_path(value: str | Path) -> Path:
    """Make an output absolute without following its final path component."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.parent.resolve() / candidate.name


def _checksum_line(row: dict[str, Any]) -> str:
    relative = str(row["relative_path"])
    escaped = relative.replace("\\", "\\\\").replace("\n", "\\n")
    prefix = "\\" if escaped != relative else ""
    return f"{prefix}{row['sha256']}  {escaped}\n"


def _render_outputs(rows: list[dict[str, Any]]) -> tuple[str, str]:
    manifest_stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        manifest_stream,
        fieldnames=("relative_path", "size_bytes", "sha256", "git_tracked"),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    checksum_text = "".join(_checksum_line(row) for row in rows)
    return manifest_stream.getvalue(), checksum_text


def _temporary_output(destination: Path, content: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _remove_if_ours(destination: Path, temporary: Path) -> None:
    try:
        if destination.exists() and destination.samefile(temporary):
            destination.unlink()
    except FileNotFoundError:
        pass


def publish_outputs(
    manifest: Path,
    checksums: Path,
    manifest_text: str,
    checksum_text: str,
) -> None:
    """Publish both files without overwriting or leaving a half-pair."""
    if manifest == checksums:
        raise ValueError("Manifest and checksum outputs must be different paths")
    if os.path.lexists(manifest) or os.path.lexists(checksums):
        raise FileExistsError(
            "Refusing to overwrite an existing manifest or checksum file"
        )
    manifest_temp = _temporary_output(manifest, manifest_text)
    try:
        checksum_temp = _temporary_output(checksums, checksum_text)
    except BaseException:
        manifest_temp.unlink(missing_ok=True)
        raise
    manifest_published = False
    checksums_published = False
    try:
        # Hard-linking gives O_EXCL-like semantics: link(2) fails if
        # another process creates a destination after the preflight check.
        os.link(manifest_temp, manifest)
        manifest_published = True
        os.link(checksum_temp, checksums)
        checksums_published = True
    except BaseException:
        if checksums_published:
            _remove_if_ours(checksums, checksum_temp)
        if manifest_published:
            _remove_if_ours(manifest, manifest_temp)
        raise
    finally:
        manifest_temp.unlink(missing_ok=True)
        checksum_temp.unlink(missing_ok=True)


def generate_manifest(
    root: Path,
    manifest: Path,
    checksums: Path,
    includes: list[str],
    extra_excludes: list[str],
) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest = _output_path(manifest)
    checksums = _output_path(checksums)
    if manifest == checksums:
        raise ValueError("Manifest and checksum outputs must be different paths")
    if os.path.lexists(manifest) or os.path.lexists(checksums):
        raise FileExistsError(
            "Refusing to overwrite an existing manifest or checksum file"
        )

    excludes = list(DEFAULT_EXCLUDES) + list(extra_excludes)
    output_relatives = set()
    for output in (manifest, checksums):
        try:
            output_relatives.add(str(output.relative_to(root)))
        except ValueError:
            pass
    excludes.extend(output_relatives)

    files = selected_files(root, includes, excludes)
    if not files:
        raise ValueError("The selected paths contain no regular files")
    tracked = tracked_files(root)
    rows: list[dict[str, Any]] = []
    for path in files:
        relative = str(path.relative_to(root))
        size_bytes, digest = file_fingerprint(path)
        rows.append({
            "relative_path": relative,
            "size_bytes": size_bytes,
            "sha256": digest,
            "git_tracked": relative in tracked,
        })
    manifest_text, checksum_text = _render_outputs(rows)
    publish_outputs(manifest, checksums, manifest_text, checksum_text)
    print(f"Wrote {len(rows)} entries to {manifest} and {checksums}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checksums", required=True)
    parser.add_argument("--include", action="append", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    generate_manifest(
        Path(args.root),
        Path(args.manifest),
        Path(args.checksums),
        args.include,
        args.exclude,
    )


if __name__ == "__main__":
    main()
