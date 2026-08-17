#!/usr/bin/env python3
"""Regression tests for manifest and run-metadata tools."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


provenance = load_script("collect_run_provenance")
manifest_tool = load_script("generate_artifact_manifest")


class ProvenanceTests(unittest.TestCase):

    def test_formal_begin_records_clean_commit_command_config_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_file = root / "input.csv"
            input_file.write_text("time,value\n0,1\n")
            config = root / "config.yaml"
            config.write_text("frequency: 20.0\n")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "add", "input.csv", "config.yaml"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    "git", "-c", "user.name=Test",
                    "-c", "user.email=test@example.invalid",
                    "commit", "-q", "-m", "fixture",
                ],
                cwd=root,
                check=True,
            )
            start = root / "start.json"
            command = ["ros2", "launch", "package", "experiment.launch.py"]
            args = argparse.Namespace(
                repo_root=str(root),
                output_bag=str(root / "future_bag"),
                out=str(start),
                input=[str(input_file)],
                config=[str(config)],
                require_clean_git=True,
                command=command,
            )
            with mock.patch.object(
                provenance, "dependency_snapshot", return_value={"test": True}
            ):
                self.assertEqual(provenance.begin(args), 0)
            record = json.loads(start.read_text())
            self.assertTrue(record["formal_run_requested"])
            self.assertTrue(record["git"]["scope_clean"])
            self.assertTrue(record["git_stable_during_capture"])
            self.assertEqual(record["command"], command)
            self.assertEqual(
                record["effective_configs"][0]["content"],
                "frequency: 20.0\n",
            )
            self.assertEqual(
                record["inputs"][0]["sha256"],
                provenance.sha256_file(input_file),
            )

    def test_write_exclusive_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "record.json"
            provenance.write_exclusive(destination, {"first": True})
            with self.assertRaises(FileExistsError):
                provenance.write_exclusive(destination, {"second": True})
            self.assertEqual(json.loads(destination.read_text()), {"first": True})

    def test_finish_uses_humble_rosbag_info_and_writes_complete_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "bag"
            output.mkdir()
            (output / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
            (output / "data.db3").write_bytes(b"bag")
            run_log = root / "run.log"
            run_log.write_text("finished\n")
            start = root / "start.json"
            start.write_text(json.dumps({
                "schema_version": 2,
                "status": "started",
                "output_bag": str(output.resolve()),
            }))
            final = root / "final.json"
            calls = []

            def fake_run(command, cwd=None):
                calls.append(command)
                return {"command": command, "returncode": 0, "output": "Files: data.db3"}

            args = argparse.Namespace(
                start_json=str(start),
                output_bag=str(output),
                run_log=str(run_log),
                exit_code=0,
                command_exit_code=0,
                log_exit_code=0,
                out=str(final),
            )
            with mock.patch.object(provenance, "run_text", side_effect=fake_run):
                self.assertEqual(provenance.finish(args), 0)
            record = json.loads(final.read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["integrity_errors"], [])
            self.assertEqual(
                calls,
                [["ros2", "bag", "info", str(output.resolve())]],
            )
            self.assertNotIn("--yaml", record["rosbag_info"]["command"])

    def test_zero_exit_with_missing_bag_is_failed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "missing_bag"
            run_log = root / "run.log"
            run_log.write_text("unexpectedly no bag\n")
            start = root / "start.json"
            start.write_text(json.dumps({
                "schema_version": 2,
                "status": "started",
                "output_bag": str(output.resolve()),
            }))
            final = root / "final.json"
            args = argparse.Namespace(
                start_json=str(start),
                output_bag=str(output),
                run_log=str(run_log),
                exit_code=0,
                command_exit_code=0,
                log_exit_code=0,
                out=str(final),
            )
            self.assertEqual(provenance.finish(args), 3)
            record = json.loads(final.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("output_bag_directory_missing", record["integrity_errors"])


class ManifestTests(unittest.TestCase):

    @staticmethod
    def initialize_repository(root: Path) -> None:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", "artifact.txt"], cwd=root, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "commit", "-q", "-m", "fixture",
            ],
            cwd=root,
            check=True,
        )

    def test_generates_verifiable_pair_and_marks_tracked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.txt").write_text("immutable\n")
            self.initialize_repository(root)
            manifest = root / "release" / "manifest.csv"
            checksums = root / "release" / "checksums.sha256"
            rows = manifest_tool.generate_manifest(
                root, manifest, checksums, ["artifact.txt"], []
            )
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["git_tracked"])
            subprocess.run(
                ["sha256sum", "--check", str(checksums)],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )

    def test_default_excludes_root_and_nested_pytest_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.txt").write_text("immutable\n")
            (root / ".pytest_cache").mkdir()
            (root / ".pytest_cache" / "root-cache.txt").write_text("cache\n")
            nested_cache = root / "package" / ".pytest_cache"
            nested_cache.mkdir(parents=True)
            (nested_cache / "nested-cache.txt").write_text("cache\n")
            self.initialize_repository(root)

            rows = manifest_tool.generate_manifest(
                root,
                root / "release" / "manifest.csv",
                root / "release" / "checksums.sha256",
                ["."],
                [],
            )

            self.assertEqual(
                [row["relative_path"] for row in rows], ["artifact.txt"]
            )

    def test_existing_checksum_prevents_any_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifact.txt").write_text("immutable\n")
            self.initialize_repository(root)
            manifest = root / "manifest.csv"
            checksums = root / "checksums.sha256"
            checksums.write_text("keep me\n")
            with self.assertRaises(FileExistsError):
                manifest_tool.generate_manifest(
                    root, manifest, checksums, ["artifact.txt"], []
                )
            self.assertFalse(manifest.exists())
            self.assertEqual(checksums.read_text(), "keep me\n")

    def test_second_publication_failure_rolls_back_first_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            checksums = root / "checksums.sha256"
            real_link = manifest_tool.os.link

            def fail_second_link(source, destination):
                if Path(destination) == checksums:
                    raise FileExistsError("simulated output race")
                return real_link(source, destination)

            with mock.patch.object(
                manifest_tool.os, "link", side_effect=fail_second_link
            ):
                with self.assertRaises(FileExistsError):
                    manifest_tool.publish_outputs(
                        manifest, checksums, "manifest\n", "checksums\n"
                    )
            self.assertFalse(manifest.exists())
            self.assertFalse(checksums.exists())

    def test_rejects_include_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "root"
            root.mkdir()
            (root / "artifact.txt").write_text("inside\n")
            self.initialize_repository(root)
            outside = base / "outside.txt"
            outside.write_text("outside\n")
            with self.assertRaises(ValueError):
                manifest_tool.generate_manifest(
                    root,
                    root / "manifest.csv",
                    root / "checksums.sha256",
                    [str(outside)],
                    [],
                )


if __name__ == "__main__":
    unittest.main()
