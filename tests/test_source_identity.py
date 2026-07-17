import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sunfish.source_tree import source_tree_digest
from sunfish_tpu.source_identity import (
    require_launcher_run_id,
    source_identity_from_environment,
)


ROOT = Path(__file__).resolve().parents[1]


class SourceIdentityTests(unittest.TestCase):
    def test_workspace_digest_is_deterministic_and_has_a_file_count(self):
        command = [
            "python3",
            "scripts/source_tree_digest.py",
            "--root",
            ".",
            "--with-count",
        ]
        first = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        second = subprocess.run(
            command, cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
        self.assertEqual(first, second)
        digest, count = first.split()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertGreater(int(count), 75)

    def test_digest_covers_content_and_executable_bit_but_ignores_other_modes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "src").mkdir()
            tracked = root / "src" / "run.sh"
            tracked.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            tracked.chmod(0o644)
            (root / ".gitignore").write_text("scratch/\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(root), "add", "src/run.sh", ".gitignore"],
                check=True,
            )

            initial, initial_count = source_tree_digest(root)
            (root / "scratch").mkdir()
            (root / "scratch" / "ignored.txt").write_text("ignored", encoding="utf-8")
            self.assertEqual(source_tree_digest(root), (initial, initial_count))

            (root / "coordination").mkdir()
            (root / "coordination" / "CHANNEL.wire").write_text(
                "mutable coordination", encoding="utf-8"
            )
            self.assertEqual(source_tree_digest(root), (initial, initial_count))

            tracked.chmod(0o664)
            self.assertEqual(source_tree_digest(root), (initial, initial_count))

            tracked.chmod(0o755)
            mode_changed, _ = source_tree_digest(root)
            self.assertNotEqual(mode_changed, initial)

            tracked.chmod(0o775)
            self.assertEqual(source_tree_digest(root), (mode_changed, initial_count))

            tracked.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            content_changed, _ = source_tree_digest(root)
            self.assertNotEqual(content_changed, mode_changed)

    def test_environment_contract_fails_closed_on_partial_identity(self):
        with mock.patch.dict(
            os.environ,
            {"SUNFISH_GIT_COMMIT": "a" * 40},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SOURCE_TREE"):
                source_identity_from_environment(required=True)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                source_identity_from_environment(required=False)["git_commit"],
                "unrecorded",
            )

    def test_hardware_program_run_id_must_match_launcher(self):
        with mock.patch.dict(
            os.environ, {"SUNFISH_RUN_ID": "expected-run"}, clear=True
        ):
            require_launcher_run_id("expected-run")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                require_launcher_run_id("different-run")
        with mock.patch.dict(os.environ, {}, clear=True):
            require_launcher_run_id("local-development", required=False)


if __name__ == "__main__":
    unittest.main()
