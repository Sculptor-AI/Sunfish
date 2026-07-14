import base64
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sunfish_tpu.upload_transaction import (
    MARKER_NAME,
    cleanup_transaction,
    prepare_transaction,
    publish_files,
)


IDENTITY = "a" * 64


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class UploadTransactionTests(unittest.TestCase):
    def test_dependency_free_source_executes_as_the_embedded_remote_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "release"
            source_path = (
                Path(__file__).resolve().parents[1]
                / "src/sunfish_tpu/upload_transaction.py"
            )
            encoded = base64.b64encode(source_path.read_bytes()).decode("ascii")
            bootstrap = (
                "import base64,sys; "
                "source=base64.b64decode(sys.argv.pop(1),validate=True); "
                "exec(compile(source,'sunfish-upload-transaction.py','exec'))"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    bootstrap,
                    encoded,
                    "prepare",
                    "--final",
                    str(final),
                    "--identity",
                    IDENTITY,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                Path(result.stdout.strip()),
                Path(f"{final}.upload-{IDENTITY[:16]}"),
            )

    def test_prepare_reconciles_a_marked_partial_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "release"
            staging = prepare_transaction(final, IDENTITY)
            (staging / "partial.tar").write_bytes(b"partial")
            nested = staging / "partial-tree"
            nested.mkdir()
            (nested / "file").write_text("partial", encoding="utf-8")

            self.assertEqual(prepare_transaction(final, IDENTITY), staging)
            self.assertEqual({path.name for path in staging.iterdir()}, {MARKER_NAME})

    def test_publish_is_idempotent_when_another_worker_already_finished(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "configs"
            files = {"config.toml": digest(b"config")}
            staging = prepare_transaction(final, IDENTITY)
            (staging / "config.toml").write_bytes(b"config")
            publish_files(final, IDENTITY, files)
            self.assertEqual((final / "config.toml").read_bytes(), b"config")

            staging = prepare_transaction(final, IDENTITY)
            (staging / "config.toml").write_bytes(b"config")
            publish_files(final, IDENTITY, files)
            self.assertFalse(Path(f"{final}.upload-{IDENTITY[:16]}").exists())

    def test_publish_refuses_an_existing_divergent_final_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "configs"
            final.mkdir()
            (final / "config.toml").write_bytes(b"wrong")
            staging = prepare_transaction(final, IDENTITY)
            (staging / "config.toml").write_bytes(b"config")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                publish_files(
                    final,
                    IDENTITY,
                    {"config.toml": digest(b"config")},
                )
            self.assertTrue(staging.exists())

    def test_cleanup_refuses_an_unmarked_or_wrongly_marked_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "release"
            staging = Path(f"{final}.upload-{IDENTITY[:16]}")
            staging.mkdir()
            with self.assertRaisesRegex(ValueError, "trusted marker"):
                cleanup_transaction(final, IDENTITY)

            staging.rmdir()
            prepare_transaction(final, IDENTITY)
            marker = staging / MARKER_NAME
            marker.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "another release"):
                cleanup_transaction(final, IDENTITY)

    def test_unsafe_final_paths_are_rejected_before_any_cleanup(self):
        for final in (Path("/"), Path("/tmp/../unsafe")):
            with self.subTest(final=final):
                with self.assertRaisesRegex(ValueError, "safe non-root"):
                    prepare_transaction(final, IDENTITY)


if __name__ == "__main__":
    unittest.main()
