import json
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

from sunfish.source_tree import _digest_source_names, workspace_source_identity
from sunfish_tpu.offline_bundle import (
    create_bundle_manifest,
    minimum_glibc_for_wheels,
    pack_bundle,
    parse_resolved_lock,
    read_archive_sidecar,
    verify_bundle,
    verify_worker_runtime_compatibility,
    write_resolved_lock,
)
from sunfish_tpu.runtime_provenance import resolve_gemma_source_commit
from sunfish_tpu import standalone_runtime
from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_umask(mask: int):
    previous = os.umask(mask)
    try:
        yield
    finally:
        os.umask(previous)


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


class StandaloneRuntimePinTests(unittest.TestCase):
    def test_exact_python_build_standalone_asset_is_pinned(self):
        self.assertEqual(standalone_runtime.RUNTIME_RELEASE, "20260623")
        self.assertEqual(standalone_runtime.RUNTIME_PYTHON_VERSION, "3.12.13")
        self.assertEqual(
            standalone_runtime.RUNTIME_ARCHIVE_NAME,
            "cpython-3.12.13+20260623-"
            "x86_64-unknown-linux-gnu-install_only.tar.gz",
        )
        self.assertEqual(
            standalone_runtime.RUNTIME_ARCHIVE_SHA256,
            "9fa869d69be54f6b8eeae64272fbd9bb0646e0e1a8da9d80e51ba5a3bee48930",
        )
        self.assertEqual(standalone_runtime.RUNTIME_ARCHIVE_SIZE, 111_146_559)


class OfflineBundleTests(unittest.TestCase):
    def setUp(self):
        self.runtime_bytes = b"sunfish-standalone-runtime-fixture"
        patcher = mock.patch.multiple(
            standalone_runtime,
            RUNTIME_ARCHIVE_SHA256=hashlib.sha256(self.runtime_bytes).hexdigest(),
            RUNTIME_ARCHIVE_SIZE=len(self.runtime_bytes),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _bundle(root: Path) -> Path:
        bundle = root / "sunfish-tpu-offline"
        make_exported_source(bundle / "source")
        runtime_directory = bundle / standalone_runtime.RUNTIME_ARCHIVE_DIRECTORY
        runtime_directory.mkdir(parents=True)
        runtime_archive = runtime_directory / standalone_runtime.RUNTIME_ARCHIVE_NAME
        runtime_archive.write_bytes(b"sunfish-standalone-runtime-fixture")
        standalone_runtime.write_runtime_metadata(
            runtime_archive, bundle / standalone_runtime.RUNTIME_METADATA_NAME
        )
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
            "python": "3.12.13",
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
            self.assertEqual(report["target"]["minimum_glibc"], "2.17")
            self.assertEqual(report["target"]["python"], "3.12.13")
            self.assertEqual(
                report["python_runtime"]["archive_sha256"],
                standalone_runtime.RUNTIME_ARCHIVE_SHA256,
            )
            inventory = {record["path"] for record in report["files"]}
            self.assertIn(standalone_runtime.RUNTIME_METADATA_NAME, inventory)
            self.assertIn(report["python_runtime"]["archive"], inventory)
            self.assertIn("sunfish-diffusion", report["resolved_distributions"])
            wheel = bundle / "wheelhouse/sunfish_diffusion-0.1.0-py3-none-any.whl"
            wheel.write_bytes(b"tampered")
            with self.assertRaisesRegex(
                ValueError, "size/type mismatch|hash mismatch|invalid wheel metadata"
            ):
                verify_bundle(bundle)

    def test_manylinux_wheel_tags_define_the_worker_glibc_floor(self):
        wheels = [
            Path("pure-1.0-py3-none-any.whl"),
            Path("native-1.0-cp312-cp312-manylinux_2_27_x86_64.whl"),
            Path(
                "portable-1.0-cp312-cp312-"
                "manylinux_2_28_x86_64.manylinux2014_x86_64.whl"
            ),
        ]
        self.assertEqual(minimum_glibc_for_wheels(wheels), "2.27")
        with self.assertRaisesRegex(ValueError, "no versioned manylinux"):
            minimum_glibc_for_wheels(
                [Path("native-1.0-cp312-cp312-linux_x86_64.whl")]
            )

    def test_bundle_rejects_a_forged_glibc_floor(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(Path(temporary))
            manifest_path = bundle / "offline-bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["target"]["minimum_glibc"] = "2.99"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from wheel tags"):
                verify_bundle(bundle)

    def test_bundle_rejects_runtime_archive_or_metadata_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(Path(temporary))
            archive = bundle / standalone_runtime.expected_runtime_metadata()["archive"]
            original = archive.read_bytes()
            archive.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(ValueError, "hash mismatch|hash differs"):
                verify_bundle(bundle)

    def test_worker_runtime_rejects_old_glibc_and_proxy_settings(self):
        manifest = {"target": {"minimum_glibc": "2.28"}}
        compatible = {
            "system": "Linux",
            "machine": "x86_64",
            "python_version": "3.12.13",
            "libc": ("glibc", "2.31"),
            "environment": {},
            "discovered_proxies": {},
        }
        report = verify_worker_runtime_compatibility(manifest, **compatible)
        self.assertTrue(report["proxy_environment_clear"])
        with self.assertRaisesRegex(ValueError, "older than bundle floor"):
            verify_worker_runtime_compatibility(
                manifest,
                **{**compatible, "libc": ("glibc", "2.27")},
            )
        with self.assertRaisesRegex(ValueError, "proxy settings are forbidden"):
            verify_worker_runtime_compatibility(
                manifest,
                **{
                    **compatible,
                    "environment": {"HTTPS_PROXY": "http://proxy.invalid"},
                },
            )

    def test_pack_writes_a_strict_sha256_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._bundle(root)
            archive = root / "release.tar"
            packed = pack_bundle(bundle, archive)
            self.assertEqual(read_archive_sidecar(archive), packed["sha256"])
            self.assertGreater(archive.stat().st_size, 0)

    def test_pack_extract_verify_is_portable_across_builder_and_controller_umasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller_source = root / "controller" / "source"
            with temporary_umask(0o022):
                make_exported_source(controller_source)
            controller_identity = workspace_source_identity(controller_source)

            with temporary_umask(0o002):
                bundle = self._bundle(root / "builder")
            builder_source = bundle / "source"
            self.assertEqual(
                (controller_source / "src/example.py").stat().st_mode & 0o777,
                0o644,
            )
            self.assertEqual(
                (builder_source / "src/example.py").stat().st_mode & 0o777,
                0o664,
            )

            archive = root / "release.tar"
            pack_bundle(bundle, archive)
            destination = root / "worker"
            destination.mkdir()
            extracted = standalone_runtime.extract_bundle_archive(
                archive, destination
            )
            self.assertEqual(
                extracted, destination / "sunfish-tpu-offline"
            )
            self.assertEqual(
                (extracted / "source/src/example.py").stat().st_mode & 0o777,
                0o664,
            )
            report = verify_bundle(
                extracted,
                expected_commit=controller_identity["git_commit"],
                expected_tree=controller_identity["source_tree_sha256"],
            )
            self.assertEqual(report["sunfish_source"], controller_identity)

    def test_resolved_lock_rejects_urls_and_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lock"
            path.write_text("gemma @ git+https://example.invalid/repo\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "name==version"):
                parse_resolved_lock(path)
            path.write_text("example==1#not-a-version\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid offline lock"):
                parse_resolved_lock(path)

    def test_resolved_tpu_lock_requires_exact_requests_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "offline-requirements.lock"
            with mock.patch(
                "sunfish_tpu.offline_bundle._installed_distributions",
                return_value={"jax": "0.10.2", "requests": "2.32.5"},
            ):
                result = write_resolved_lock(Path("python"), output)
            self.assertEqual(result["requests"], "2.32.5")
            self.assertEqual(
                [line for line in output.read_text().splitlines() if line.startswith("requests==")],
                ["requests==2.32.5"],
            )

            with mock.patch(
                "sunfish_tpu.offline_bundle._installed_distributions",
                return_value={"jax": "0.10.2"},
            ):
                with self.assertRaisesRegex(RuntimeError, "missing requests"):
                    write_resolved_lock(Path("python"), output)

    def test_worker_gemma_provenance_comes_from_verified_offline_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = self._bundle(Path(temporary))
            archive_direct_url = mock.Mock()
            archive_direct_url.read_text.return_value = json.dumps(
                {
                    "url": "file:///wheelhouse/gemma.whl",
                    "archive_info": {"hash": "sha256=" + "a" * 64},
                }
            )
            with mock.patch(
                "sunfish_tpu.runtime_provenance.importlib.metadata.distribution",
                return_value=archive_direct_url,
            ) as distribution:
                commit, provenance = resolve_gemma_source_commit(
                    environment={"SUNFISH_OFFLINE_BUNDLE_ROOT": str(bundle)}
                )
            self.assertEqual(commit, GEMMA_SOURCE_COMMIT)
            self.assertEqual(provenance, "verified-offline-bundle")
            distribution.assert_not_called()

    def test_archive_direct_url_is_not_mistaken_for_vcs_provenance(self):
        distribution = mock.Mock()
        distribution.read_text.return_value = json.dumps(
            {
                "url": "file:///wheelhouse/gemma.whl",
                "archive_info": {"hash": "sha256=" + "a" * 64},
            }
        )
        with mock.patch(
            "sunfish_tpu.runtime_provenance.importlib.metadata.distribution",
            return_value=distribution,
        ):
            self.assertEqual(
                resolve_gemma_source_commit(environment={}),
                (None, "direct-url"),
            )

    def test_offline_root_and_manifest_must_identify_the_same_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = self._bundle(root / "one")
            other = self._bundle(root / "two")
            with self.assertRaisesRegex(ValueError, "disagree"):
                resolve_gemma_source_commit(
                    environment={
                        "SUNFISH_OFFLINE_BUNDLE_ROOT": str(bundle),
                        "SUNFISH_OFFLINE_BUNDLE_MANIFEST": str(
                            other / "offline-bundle.json"
                        ),
                    }
                )


class IapTransportTests(unittest.TestCase):
    @staticmethod
    def _environment(root: Path, capture: Path) -> dict[str, str]:
        fake = root / "gcloud"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s\\n' \"$@\" >> \"$CAPTURE\"\n"
            "case \"${1:-}\" in\n"
            "  version) printf '{\"Google Cloud SDK\":\"%s\"}\\n' \"${GCLOUD_SDK_VERSION:-400.0.0}\" ;;\n"
            "  auth) printf '%s\\n' \"${GCLOUD_ACTIVE_ACCOUNT:-operator@example.com}\" ;;\n"
            "  config) printf '%s\\n' \"${GCLOUD_CONFIG_PROJECT:-sunfish-project}\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        fake.chmod(0o755)
        python = root / "python3.12"
        python.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$*\" == *\"sys.version_info[:2]\"* ]]; then exit 0; fi\n"
            f"exec {shlex.quote(sys.executable)} \"$@\"\n",
            encoding="utf-8",
        )
        python.chmod(0o755)
        key = root / "google_compute_engine"
        key.write_text("test-private-key\n", encoding="utf-8")
        key.with_suffix(".pub").write_text(
            "ssh-ed25519 AAAATEST sunfish-test\n", encoding="utf-8"
        )
        ssh_add = root / "ssh-add"
        ssh_add.write_text(
            "#!/usr/bin/env bash\nprintf 'ssh-ed25519 AAAATEST sunfish-test\\n'\n",
            encoding="utf-8",
        )
        ssh_add.chmod(0o755)
        return {
            **os.environ,
            "CAPTURE": str(capture),
            "SUNFISH_GCLOUD_BIN": str(fake),
            "SUNFISH_CONTROLLER_PYTHON": str(python),
            "SUNFISH_SSH_ADD_BIN": str(ssh_add),
            "SUNFISH_COMPUTE_SSH_KEY": str(key),
            "SSH_AUTH_SOCK": str(root / "agent.sock"),
            "TPU_NAME": "sunfish-v4",
            "PROJECT_ID": "sunfish-project",
            "ZONE": "us-central2-b",
            "SUNFISH_IAP_TUNNEL_ROLE_CONFIRMED": "1",
            "SUNFISH_IAP_SSH_FIREWALL_CONFIRMED": "1",
            "SUNFISH_PRIVATE_GOOGLE_ACCESS_CONFIRMED": "1",
            "SUNFISH_GCS_IAM_CONFIRMED": "1",
            "SUNFISH_OFFLINE_BUNDLE_ROOT": "/home/sunfish/releases/a",
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
            for keepalive in (
                "--ssh-flag=-oServerAliveInterval=30",
                "--ssh-flag=-oServerAliveCountMax=6",
                "--ssh-flag=-oTCPKeepAlive=yes",
            ):
                self.assertIn(keepalive, arguments)

            forbidden = {
                "sudo shutdown now": "allocation lifecycle",
                "python3 -m pip install jax": "air-gapped worker",
                "curl https://example.invalid": "air-gapped worker",
                "scripts/bootstrap_parity.sh --connected-compute-host": "air-gapped worker",
                "gcloud alpha compute tpus tpu-vm attach-disk sunfish-v4 --disk=d": "control-plane",
                "gcloud alpha compute tpus tpu-vm detach-disk sunfish-v4 --disk=d": "control-plane",
                "gcloud alpha compute tpus tpu-vm perform-maintenance sunfish-v4": "control-plane",
                "gcloud alpha compute tpus tpu-vm simulate-maintenance-event sunfish-v4": "control-plane",
                "gcloud alpha compute tpus tpu-vm describe sunfish-v4": "control-plane",
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
            self.assertEqual(arguments[0], "version")
            self.assertIn(
                "alpha\ncompute\ntpus\ntpu-vm\nssh\n",
                capture.read_text(encoding="utf-8"),
            )
            self.assertIn("--worker=all", arguments)
            self.assertIn("--batch-size=all", arguments)
            self.assertIn("--tunnel-through-iap", arguments)
            command = arguments[arguments.index("--command") + 1]
            self.assertIn("python3", command)
            self.assertNotIn("python3.12", command)
            self.assertNotIn("jax", command.lower())
            self.assertIn("HTTP_PROXY", command)
            self.assertIn("proxy_environment_clear", command)
            self.assertIn("bundled_python_required", command)

    def test_controller_preflight_checks_version_account_project_key_and_confirmations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture"
            environment = self._environment(root, capture)
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/preflight_tpu_controller.sh")],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("without contacting TPU workers or GCS", result.stdout)

            environment["GCLOUD_SDK_VERSION"] = "343.0.0"
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/preflight_tpu_controller.sh")],
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("too old", result.stderr)

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
            self.assertEqual(calls.count("verify_tpu_bundled_runtime.sh"), 2)
            self.assertIn("extract-bundle", calls)
            self.assertIn("python3", calls)
            self.assertNotIn("https://", calls)


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
