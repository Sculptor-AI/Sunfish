import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from sunfish.source_tree import _digest_source_names, workspace_source_identity
from sunfish_tpu.offline_bundle import (
    create_bundle_manifest,
    pack_bundle,
    parse_resolved_lock,
    read_archive_sidecar,
    verify_bundle,
)
from sunfish_tpu.training.dependencies import (
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)


ROOT = Path(__file__).resolve().parents[1]


def make_exported_source(root: Path) -> dict[str, object]:
    files = {
        "src/example.py": "VALUE = 1\n",
        "src/sunfish/__init__.py": "NAME = 'sunfish'\n",
        "src/sunfish_tpu/__init__.py": "NAME = 'sunfish_tpu'\n",
        "scripts/run.sh": "#!/usr/bin/env bash\nexit 0\n",
        "configs/test.toml": "value = 1\n",
        "pyproject.toml": "[project]\nname = 'example'\n",
        "requirements-tpu.lock": "example==1\n",
        "reference/upstream/audit.json": "{}\n",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        if name.endswith(".sh"):
            path.chmod(0o755)
    names = sorted(files)
    digest, count = _digest_source_names(
        root, [name.encode("utf-8") for name in names]
    )
    release = {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "source_tree_sha256": digest,
        "source_files": count,
        "deployment_files": names,
    }
    (root / ".sunfish-release.json").write_text(
        json.dumps(release), encoding="utf-8"
    )
    return release


class ExportedSourceIdentityTests(unittest.TestCase):
    def test_exported_tree_has_the_same_strict_identity_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = make_exported_source(root)
            self.assertEqual(
                workspace_source_identity(root),
                {
                    "git_commit": expected["git_commit"],
                    "source_tree_sha256": expected["source_tree_sha256"],
                    "source_files": expected["source_files"],
                },
            )
            (root / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "digest differs"):
                workspace_source_identity(root)

    def test_exported_tree_rejects_uninventoried_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_exported_source(root)
            (root / "src/injected.py").write_text("BAD = True\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "file set changed"):
                workspace_source_identity(root)


class OfflineBundleTests(unittest.TestCase):
    @staticmethod
    def _bundle(root: Path) -> Path:
        bundle = root / "sunfish-tpu-offline"
        make_exported_source(bundle / "source")
        wheelhouse = bundle / "wheelhouse"
        wheelhouse.mkdir(parents=True)
        pins = {**RUNTIME_VERSIONS, **TPU_ONLY_RUNTIME_VERSIONS}
        pins["sunfish-diffusion"] = "0.1.0"
        for name, version in sorted(pins.items()):
            stem = name.replace("-", "_")
            wheel = wheelhouse / f"{stem}-{version}-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr(
                    f"{stem}-{version}.dist-info/METADATA",
                    f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                )
                if name == "sunfish-diffusion":
                    archive.writestr("sunfish/__init__.py", "NAME = 'sunfish'\n")
                    archive.writestr(
                        "sunfish_tpu/__init__.py", "NAME = 'sunfish_tpu'\n"
                    )
        (bundle / "offline-requirements.lock").write_text(
            "".join(f"{name}=={version}\n" for name, version in sorted(pins.items())),
            encoding="utf-8",
        )
        create_bundle_manifest(bundle)
        manifest_path = bundle / "offline-bundle.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["builder"] = {
            "operating_system": "linux",
            "machine": "x86_64",
            "python": "3.12",
            "libc": ["glibc", "2.31"],
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return bundle

    def test_bundle_inventory_is_source_and_wheel_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(Path(temporary))
            report = verify_bundle(
                bundle,
                expected_commit="a" * 40,
                expected_tree=workspace_source_identity(bundle / "source")[
                    "source_tree_sha256"
                ],
            )
            self.assertEqual(report["network_policy"], "worker-no-egress")
            self.assertIn("sunfish-diffusion", report["resolved_distributions"])
            wheel = bundle / "wheelhouse/sunfish_diffusion-0.1.0-py3-none-any.whl"
            wheel.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                ValueError, "size/type mismatch|hash mismatch|invalid wheel metadata"
            ):
                verify_bundle(bundle)

    def test_pack_writes_a_strict_sha256_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._bundle(root)
            archive = root / "release.tar"
            packed = pack_bundle(bundle, archive)
            self.assertEqual(read_archive_sidecar(archive), packed["sha256"])
            self.assertGreater(archive.stat().st_size, 0)

    def test_resolved_lock_rejects_urls_and_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            path.write_text("gemma @ git+https://example.invalid/repo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "name==version"):
                parse_resolved_lock(path)
            path.write_text("example==1#not-a-version\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid offline lock"):
                parse_resolved_lock(path)


class IapTransportTests(unittest.TestCase):
    @staticmethod
    def _environment(root: Path, capture: Path) -> dict[str, str]:
        fake = root / "gcloud"
        fake.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >> \"$CAPTURE\"\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        return {
            **os.environ,
            "CAPTURE": str(capture),
            "SUNFISH_GCLOUD_BIN": str(fake),
            "TPU_NAME": "sunfish-v4",
            "PROJECT_ID": "sunfish-project",
            "ZONE": "us-central2-b",
        }

    def test_local_cli_check_needs_no_tpu_identity_and_performs_no_remote_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            environment = self._environment(root, capture)
            for name in ("TPU_NAME", "PROJECT_ID", "ZONE"):
                environment.pop(name)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/tpu_iap.sh"), "check-cli"],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments.count("alpha"), 2)
            self.assertIn("ssh", arguments)
            self.assertIn("scp", arguments)
            self.assertNotIn("--worker=all", arguments)
            self.assertIn("command surface is installed", result.stdout)

    def test_ssh_is_all_worker_alpha_iap_and_has_no_lifecycle_surface(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            environment = self._environment(root, capture)
            subprocess.run(
                ["bash", str(ROOT / "scripts/tpu_iap.sh"), "ssh-all", "--command", "true"],
                env=environment,
                check=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:5], ["alpha", "compute", "tpus", "tpu-vm", "ssh"])
            for required in ("--worker=all", "--batch-size=all", "--tunnel-through-iap"):
                self.assertIn(required, arguments)

            forbidden = {
                "sudo shutdown now": "allocation lifecycle",
                "python3 -m pip install jax": "air-gapped worker",
                "curl https://example.invalid": "air-gapped worker",
                "scripts/bootstrap_parity.sh --connected-compute-host": "air-gapped worker",
            }
            for command, message in forbidden.items():
                with self.subTest(command=command):
                    capture.unlink(missing_ok=True)
                    result = subprocess.run(
                        [
                            "bash",
                            str(ROOT / "scripts/tpu_iap.sh"),
                            "ssh-all",
                            "--command",
                            command,
                        ],
                        env=environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(capture.exists())
                    self.assertIn(message, result.stderr)

    def test_scp_is_all_worker_iap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            local = root / "bundle.tar"
            local.write_bytes(b"bundle")
            environment = self._environment(root, capture)
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/tpu_iap.sh"),
                    "scp-all",
                    str(local),
                    "/home/sunfish/bundle.tar",
                ],
                env=environment,
                check=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:5], ["alpha", "compute", "tpus", "tpu-vm", "scp"])
            self.assertIn("--worker=all", arguments)
            self.assertIn("--tunnel-through-iap", arguments)
            self.assertNotIn("--batch-size=all", arguments)

    def test_base_image_probe_is_backend_free_and_all_worker_iap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            environment = self._environment(root, capture)
            environment["SUNFISH_CONTROLLER_LOG_DIR"] = str(root / "logs")
            subprocess.run(
                ["bash", str(ROOT / "scripts/probe_tpu_worker_base.sh")],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            arguments = capture.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:5], ["alpha", "compute", "tpus", "tpu-vm", "ssh"])
            self.assertIn("--worker=all", arguments)
            self.assertIn("--batch-size=all", arguments)
            self.assertIn("--tunnel-through-iap", arguments)
            command = arguments[arguments.index("--command") + 1]
            self.assertIn("python3.12", command)
            self.assertNotIn("jax", command.lower())
            self.assertNotIn("http", command.lower())

    def test_offline_archive_deployer_uses_only_all_worker_iap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            fake = root / "gcloud"
            fake.write_text(
                "#!/usr/bin/env bash\nprintf 'CALL\\n' >> \"$CAPTURE\"\nprintf '%s\\n' \"$@\" >> \"$CAPTURE\"\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            archive = root / "release.tar"
            archive.write_bytes(b"fixture")
            (root / "release.tar.sha256").write_text(
                f"{'a' * 64}  release.tar\n", encoding="ascii"
            )
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/deploy_tpu_offline_bundle.sh"),
                    "--bundle",
                    str(archive),
                    "--remote-dir",
                    "/home/sunfish/releases/a",
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "CAPTURE": str(capture),
                    "SUNFISH_GCLOUD_BIN": str(fake),
                    "TPU_NAME": "sunfish-v4",
                    "PROJECT_ID": "sunfish-project",
                    "ZONE": "us-central2-b",
                },
                check=True,
                capture_output=True,
                text=True,
            )
            calls = capture.read_text(encoding="utf-8")
            self.assertEqual(calls.count("CALL\n"), 3)
            self.assertEqual(calls.count("--worker=all\n"), 3)
            self.assertEqual(calls.count("--tunnel-through-iap\n"), 3)
            self.assertEqual(calls.count("--batch-size=all\n"), 2)
            self.assertIn("tpu-vm\nscp\n", calls)


class ReleaseSafetyTests(unittest.TestCase):
    def test_static_worker_egress_iap_and_lifecycle_policy(self):
        subprocess.run(
            ["python3", "scripts/check_tpu_release_safety.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
