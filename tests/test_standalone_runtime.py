import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from sunfish_tpu import standalone_runtime


_PROBE_PAYLOAD = {
    "ensurepip": "25.0",
    "implementation": "CPython",
    "libc": ["glibc", "2.31"],
    "machine": "x86_64",
    "python_version": "3.12.13",
    "system": "Linux",
}


def _add_directory(archive: tarfile.TarFile, name: str) -> None:
    member = tarfile.TarInfo(name)
    member.type = tarfile.DIRTYPE
    member.mode = 0o755
    archive.addfile(member)


def _add_file(archive: tarfile.TarFile, name: str, content: bytes, mode: int) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    member.mode = mode
    archive.addfile(member, io.BytesIO(content))


def _runtime_archive(path: Path) -> None:
    probe = (
        "#!/bin/sh\n"
        "printf '%s\\n' "
        + __import__("shlex").quote(json.dumps(_PROBE_PAYLOAD, sort_keys=True))
        + "\n"
    ).encode("utf-8")
    with tarfile.open(path, "w:gz") as archive:
        _add_directory(archive, "python")
        _add_directory(archive, "python/bin")
        _add_file(archive, "python/bin/python3.12", probe, 0o755)
        _add_directory(archive, "python/lib")
        _add_file(archive, "python/lib/runtime.txt", b"runtime\n", 0o644)
        symlink = tarfile.TarInfo("python/bin/python3")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "python3.12"
        symlink.mode = 0o777
        archive.addfile(symlink)


class StandaloneRuntimeTests(unittest.TestCase):
    def _bundle(self, root: Path):
        bundle = root / "sunfish-tpu-offline"
        runtime_directory = bundle / standalone_runtime.RUNTIME_ARCHIVE_DIRECTORY
        runtime_directory.mkdir(parents=True)
        archive = runtime_directory / standalone_runtime.RUNTIME_ARCHIVE_NAME
        _runtime_archive(archive)
        patcher = mock.patch.multiple(
            standalone_runtime,
            RUNTIME_ARCHIVE_SHA256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            RUNTIME_ARCHIVE_SIZE=archive.stat().st_size,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        metadata = standalone_runtime.write_runtime_metadata(
            archive, bundle / standalone_runtime.RUNTIME_METADATA_NAME
        )
        return bundle, metadata

    def test_safe_install_is_idempotent_and_tree_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, _ = self._bundle(root)
            destination = root / "installed"
            first = standalone_runtime.install_runtime(
                bundle, destination=destination
            )
            second = standalone_runtime.install_runtime(
                bundle, destination=destination
            )
            self.assertEqual(first, second)
            self.assertEqual(first["probe"]["python_version"], "3.12.13")
            self.assertTrue((destination / "python/bin/python3").is_symlink())

            (destination / "python/lib/runtime.txt").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "installed tree differs"):
                standalone_runtime.verify_installed_runtime(
                    bundle, destination=destination
                )

    def test_runtime_artifacts_are_bound_to_the_outer_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, metadata = self._bundle(Path(temporary))
            metadata_path = bundle / standalone_runtime.RUNTIME_METADATA_NAME
            archive_path = bundle / metadata["archive"]
            manifest = {
                "python_runtime": metadata,
                "files": [
                    {
                        "path": standalone_runtime.RUNTIME_METADATA_NAME,
                        "size": metadata_path.stat().st_size,
                        "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    },
                    {
                        "path": metadata["archive"],
                        "size": archive_path.stat().st_size,
                        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                    },
                ],
            }
            (bundle / standalone_runtime.BUNDLE_MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(
                standalone_runtime.verify_runtime_artifacts(
                    bundle, require_bundle_manifest=True
                ),
                metadata,
            )
            manifest["python_runtime"]["release"] = "forged"
            (bundle / standalone_runtime.BUNDLE_MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not bind"):
                standalone_runtime.verify_runtime_artifacts(
                    bundle, require_bundle_manifest=True
                )

    def test_archive_traversal_and_escaping_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "destination"
            destination.mkdir()
            traversal = root / "traversal.tar"
            with tarfile.open(traversal, "w") as archive:
                _add_file(archive, "python/../../escaped", b"bad", 0o644)
            with self.assertRaisesRegex(ValueError, "escapes"):
                standalone_runtime.safe_extract_archive(
                    traversal, destination, required_prefix="python"
                )
            self.assertFalse((root / "escaped").exists())

            link_archive = root / "link.tar"
            with tarfile.open(link_archive, "w") as archive:
                _add_directory(archive, "python")
                link = tarfile.TarInfo("python/escape")
                link.type = tarfile.SYMTYPE
                link.linkname = "../../victim"
                archive.addfile(link)
            with self.assertRaisesRegex(ValueError, "escapes"):
                standalone_runtime.safe_extract_archive(
                    link_archive, destination, required_prefix="python"
                )

    def test_outer_bundle_extraction_rejects_other_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "destination"
            destination.mkdir()
            archive_path = root / "bundle.tar"
            with tarfile.open(archive_path, "w") as archive:
                _add_file(archive, "unexpected/file", b"bad", 0o644)
            with self.assertRaisesRegex(ValueError, "unexpected root"):
                standalone_runtime.extract_bundle_archive(
                    archive_path, destination
                )

    def test_outer_bundle_without_an_explicit_root_member_extracts_safely(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "destination"
            destination.mkdir()
            archive_path = root / "bundle.tar"
            with tarfile.open(archive_path, "w") as archive:
                _add_file(
                    archive,
                    "sunfish-tpu-offline/source/scripts/run.sh",
                    b"#!/bin/sh\nexit 0\n",
                    0o755,
                )
            extracted = standalone_runtime.extract_bundle_archive(
                archive_path, destination
            )
            script = extracted / "source/scripts/run.sh"
            self.assertEqual(script.read_bytes(), b"#!/bin/sh\nexit 0\n")
            self.assertEqual(script.stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
