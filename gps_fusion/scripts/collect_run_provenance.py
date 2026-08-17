#!/usr/bin/env python3
"""Capture start/end metadata for an offline fusion run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_QUERIES = (
    "python3-cairo",
    "ros-humble-launch",
    "ros-humble-launch-ros",
    "ros-humble-rclpy",
    "ros-humble-fastrtps",
    "ros-humble-robot-localization",
    "ros-humble-rmw-fastrtps-cpp",
    "ros-humble-ros2bag",
    "ros-humble-rosbag2",
    "ros-humble-rosbag2-storage-default-plugins",
    "ros-humble-tf2-ros",
)
PYTHON_DISTRIBUTIONS = ("numpy", "PyYAML", "pycairo")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def absolute_path_no_follow(path: str | Path) -> Path:
    """Return an absolute normalized path without dereferencing its leaf."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def run_text(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "output": result.stdout.rstrip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def sha256_file(path: Path) -> str:
    before = _stat_signature(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = _stat_signature(path)
    if before != after:
        raise RuntimeError(f"File changed while it was being hashed: {path}")
    return digest.hexdigest()


def _stat_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _inventory_file(path: Path, include_text: bool = False) -> dict[str, Any]:
    before = _stat_signature(path)
    digest = hashlib.sha256()
    captured = bytearray() if include_text else None
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            if captured is not None:
                captured.extend(block)
    after = _stat_signature(path)
    if before != after:
        raise RuntimeError(f"File changed while it was being hashed: {path}")
    item: dict[str, Any] = {
        "size_bytes": after[2],
        "sha256": digest.hexdigest(),
    }
    if captured is not None:
        item["content"] = captured.decode("utf-8")
    return item


def inventory_path(path: Path, include_text: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    if resolved.is_file():
        item: dict[str, Any] = {
            "path": str(resolved),
            "exists": True,
            "type": "file",
        }
        item.update(_inventory_file(resolved, include_text=include_text))
        return item

    if not resolved.is_dir():
        raise ValueError(f"Unsupported input path type: {resolved}")
    children_before = sorted(
        str(child.relative_to(resolved)) for child in resolved.rglob("*")
    )
    files: list[dict[str, Any]] = []
    symlinks: list[dict[str, str]] = []
    for relative in children_before:
        child = resolved / relative
        if child.is_symlink():
            symlinks.append({
                "relative_path": relative,
                "target": os.readlink(child),
            })
        elif child.is_file():
            file_item = {"relative_path": relative}
            file_item.update(_inventory_file(child))
            files.append(file_item)
    children_after = sorted(
        str(child.relative_to(resolved)) for child in resolved.rglob("*")
    )
    if children_before != children_after:
        raise RuntimeError(f"Directory changed while it was inventoried: {resolved}")
    return {
        "path": str(resolved),
        "exists": True,
        "type": "directory",
        "files": files,
        "symlinks": symlinks,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def rosbag_info(path: Path) -> dict[str, Any] | None:
    """Return ROS bag topic/count output when *path* looks like a bag."""
    resolved = path.resolve()
    if not resolved.is_dir() or not (resolved / "metadata.yaml").is_file():
        return None
    # Humble's ros2 bag info has no --yaml option. Its plain output contains
    # duration, message count, storage files, and per-topic message counts.
    return run_text(["ros2", "bag", "info", str(resolved)])


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    git_root_result = run_text(["git", "rev-parse", "--show-toplevel"], repo_root)
    if git_root_result.get("returncode") != 0:
        return {"available": False, "diagnostic": git_root_result}
    git_root = Path(git_root_result["output"])
    try:
        scope = str(repo_root.resolve().relative_to(git_root.resolve()))
    except ValueError:
        scope = str(repo_root.resolve())
    head = run_text(["git", "rev-parse", "HEAD"], git_root)
    scoped_status = run_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", scope],
        git_root,
    )
    branch = run_text(["git", "branch", "--show-current"], git_root)
    remote = run_text(["git", "remote", "get-url", "origin"], git_root)
    commit_time = run_text(["git", "show", "-s", "--format=%cI", "HEAD"], git_root)
    return {
        "available": True,
        "git_root": str(git_root),
        "scope": scope,
        "commit": head.get("output") if head.get("returncode") == 0 else None,
        "commit_time": (
            commit_time.get("output")
            if commit_time.get("returncode") == 0 else None
        ),
        "branch": branch.get("output") if branch.get("returncode") == 0 else None,
        "origin": remote.get("output") if remote.get("returncode") == 0 else None,
        "scope_clean": (
            scoped_status.get("returncode") == 0
            and scoped_status.get("output", "") == ""
        ),
        "scope_status": scoped_status.get("output", "").splitlines(),
        "status_command": scoped_status,
    }


def dependency_snapshot() -> dict[str, Any]:
    packages = {}
    for package in PACKAGE_QUERIES:
        result = run_text([
            "dpkg-query", "-W", "-f=${Version}", package,
        ])
        packages[package] = (
            result.get("output") if result.get("returncode") == 0 else None
        )
    python_packages = {}
    for distribution in PYTHON_DISTRIBUTIONS:
        try:
            python_packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            python_packages[distribution] = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "python_packages": python_packages,
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY"),
        "executables": {
            name: shutil.which(name) for name in ("python3", "ros2", "git")
        },
        "packages": packages,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish JSON without replacing any existing path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        # link(2), unlike rename(2), fails atomically when path already exists.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def begin(args: argparse.Namespace) -> int:
    capture_started = utc_now()
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        raise NotADirectoryError(repo_root)
    output_bag = absolute_path_no_follow(args.output_bag)
    if os.path.lexists(output_bag):
        raise FileExistsError(f"Refusing to use an existing output bag: {output_bag}")
    if not args.command:
        raise ValueError("The recorded command must not be empty")

    git_before = git_snapshot(repo_root)
    input_records = []
    for item in args.input:
        record = inventory_path(Path(item))
        if not record["exists"]:
            raise FileNotFoundError(record["path"])
        bag_details = rosbag_info(Path(item))
        if bag_details is not None:
            record["rosbag_info"] = bag_details
            if (
                args.require_clean_git
                and bag_details.get("returncode") != 0
            ):
                raise RuntimeError(
                    f"Could not inspect input ROS bag topic counts: {item}"
                )
        input_records.append(record)
    config_records = []
    for item in args.config:
        record = inventory_path(Path(item), include_text=True)
        if not record["exists"]:
            raise FileNotFoundError(record["path"])
        config_records.append(record)
    git_after = git_snapshot(repo_root)
    git_stable = (
        git_before.get("commit") == git_after.get("commit")
        and git_before.get("scope_status") == git_after.get("scope_status")
        and git_before.get("scope_clean") == git_after.get("scope_clean")
    )
    if args.require_clean_git and (
        not git_after.get("available")
        or not git_after.get("scope_clean")
        or not git_stable
    ):
        raise RuntimeError(
            "Clean-run capture requires a clean, stable Git scope"
        )
    payload = {
        "schema_version": 2,
        "status": "started",
        "provenance_capture_started_utc": capture_started,
        "started_utc": utc_now(),
        "repo_root": str(repo_root),
        "output_bag": str(output_bag),
        "command": args.command,
        "formal_run_requested": args.require_clean_git,
        "git": git_after,
        "git_stable_during_capture": git_stable,
        "environment": dependency_snapshot(),
        "inputs": input_records,
        "effective_configs": config_records,
    }
    write_exclusive(Path(args.out), payload)
    return 0


def finish(args: argparse.Namespace) -> int:
    start_path = Path(args.start_json)
    with start_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict) or payload.get("status") != "started":
        raise ValueError("The start provenance record is not in started state")
    output_bag = absolute_path_no_follow(args.output_bag)
    if payload.get("output_bag") != str(output_bag):
        raise ValueError("Start and finish output-bag paths do not match")
    output_record = inventory_path(output_bag)
    log_record = inventory_path(Path(args.run_log))
    bag_details = rosbag_info(output_bag)
    integrity_errors = []
    if args.exit_code == 0:
        if not output_record.get("exists") or output_record.get("type") != "directory":
            integrity_errors.append("output_bag_directory_missing")
        elif not (output_bag / "metadata.yaml").is_file():
            integrity_errors.append("output_bag_metadata_missing")
        if not log_record.get("exists") or log_record.get("type") != "file":
            integrity_errors.append("run_log_missing")
        if bag_details is None:
            integrity_errors.append("rosbag_info_unavailable")
        elif bag_details.get("returncode") != 0:
            integrity_errors.append("rosbag_info_failed")
    completed = args.exit_code == 0 and not integrity_errors
    payload.update({
        "status": "completed" if completed else "failed",
        "ended_utc": utc_now(),
        "exit_code": args.exit_code,
        "command_exit_code": args.command_exit_code,
        "log_capture_exit_code": args.log_exit_code,
        "start_record_sha256": sha256_file(start_path),
        "output": output_record,
        "run_log": log_record,
        "integrity_errors": integrity_errors,
    })
    if bag_details is not None:
        payload["rosbag_info"] = bag_details
    write_exclusive(Path(args.out), payload)
    return 0 if args.exit_code != 0 or completed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    start = subparsers.add_parser("begin")
    start.add_argument("--repo-root", required=True)
    start.add_argument("--output-bag", required=True)
    start.add_argument("--out", required=True)
    start.add_argument("--input", action="append", default=[])
    start.add_argument("--config", action="append", default=[])
    start.add_argument("--require-clean-git", action="store_true")
    start.add_argument("command", nargs=argparse.REMAINDER)
    start.set_defaults(handler=begin)

    end = subparsers.add_parser("finish")
    end.add_argument("--start-json", required=True)
    end.add_argument("--output-bag", required=True)
    end.add_argument("--run-log", required=True)
    end.add_argument("--exit-code", required=True, type=int)
    end.add_argument("--command-exit-code", type=int)
    end.add_argument("--log-exit-code", type=int)
    end.add_argument("--out", required=True)
    end.set_defaults(handler=finish)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "begin" and args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
