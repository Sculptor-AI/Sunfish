import ast
from pathlib import Path
import subprocess
import sys
import tomllib
import unittest

from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)


class TpuDependencyLockTests(unittest.TestCase):
    def test_training_runtime_imports_json_used_for_run_identity(self):
        root = Path(__file__).resolve().parents[1]
        source = (
            root / "src/sunfish_tpu/training/runtime.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names = {
            alias.asname or alias.name.split(".", 1)[0]
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        loaded_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertIn("json", loaded_names)
        self.assertIn("json", imported_names)

    def test_training_checkpoint_recovers_abandoned_orbax_saves(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint = (
            root / "src/sunfish_tpu/training/checkpoint.py"
        ).read_text(encoding="utf-8")
        config = (
            root / "src/sunfish_tpu/training/kauldron_config.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class ResumableCheckpointer(kd.ckpts.Checkpointer)", checkpoint)
        self.assertIn("cleanup_tmp_directories=True", checkpoint)
        self.assertIn(
            "cfg.checkpointer = sunfish_checkpoint.ResumableCheckpointer(",
            config,
        )
        self.assertNotIn("cfg.checkpointer = kd.ckpts.Checkpointer(", config)

    def test_every_review_dependency_is_exactly_pinned(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        requirements = project["project"]["optional-dependencies"]["tpu"]
        expected_names = {
            "dialog",
            "etils",
            "flax",
            "gemma",
            "google-cloud-storage",
            "grain",
            "hackable-diffusion",
            "jax",
            "jaxlib",
            "kauldron",
            "libtpu",
            "numpy",
            "optax",
            "orbax-checkpoint",
            "sentencepiece",
        }
        names = {
            requirement.split(" @ ", 1)[0].split("==", 1)[0].split("[", 1)[0]
            for requirement in requirements
        }
        self.assertEqual(names, expected_names)
        for requirement in requirements:
            if " @ " in requirement:
                self.assertRegex(requirement, r"@[0-9a-f]{40}$")
            else:
                self.assertIn("==", requirement)

    def test_lock_matches_project_direct_pins(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        project_pins = set(project["project"]["optional-dependencies"]["tpu"])
        lock_pins = {
            line
            for line in (root / "requirements-tpu.lock").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(lock_pins, project_pins)

    def test_worker_bootstrap_uses_only_the_verified_offline_bundle(self):
        root = Path(__file__).resolve().parents[1]
        base = (root / "requirements-tpu-base.lock").read_text(encoding="utf-8")
        source = (root / "requirements-gemma-source.lock").read_text(encoding="utf-8")
        bootstrap = (root / "scripts/bootstrap_tpu.sh").read_text(encoding="utf-8")
        host_entrypoint = (root / "scripts/tpu_host_entrypoint.sh").read_text(
            encoding="utf-8"
        )
        builder = (root / "scripts/build_tpu_offline_bundle.sh").read_text(
            encoding="utf-8"
        )
        deployer = (root / "scripts/deploy_tpu_offline_bundle.sh").read_text(
            encoding="utf-8"
        )
        probe = (root / "scripts/probe_tpu_worker_base.sh").read_text(
            encoding="utf-8"
        )
        config_upload = (root / "scripts/upload_tpu_configs.sh").read_text(
            encoding="utf-8"
        )
        standalone = (root / "src/sunfish_tpu/standalone_runtime.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("gemma @", base)
        self.assertRegex(source, r"gemma @ .*@[0-9a-f]{40}")
        self.assertNotIn("requirements-gemma-source.lock", bootstrap)
        self.assertNotIn("pip install --upgrade", bootstrap)
        self.assertIn("PIP_NO_INDEX=1", bootstrap)
        self.assertIn("--no-index", bootstrap)
        self.assertIn("--no-deps", bootstrap)
        self.assertIn("--only-binary=:all:", bootstrap)
        self.assertIn("offline-requirements.lock", bootstrap)
        self.assertIn(".sunfish-offline-building", bootstrap)
        self.assertIn(".sunfish-offline-complete", bootstrap)
        self.assertIn("refusing to remove an unmarked existing environment", bootstrap)
        self.assertNotIn("staging_venv", bootstrap)
        self.assertIn("-m pip check", bootstrap)
        self.assertIn("requirements-gemma-source.lock", builder)
        self.assertIn("--connected-build-host", builder)
        self.assertIn("-m pip download", builder)
        self.assertIn("--only-binary=:all:", builder)
        self.assertIn('"${wheel_files[@]}"', builder)
        self.assertIn("-m pip check", builder)
        self.assertIn('"${second_venv}/bin/python" -I -c', builder)
        self.assertIn("_packaged_kauldron_config_path", builder)
        self.assertIn("sunfish-runtime-api-audit", bootstrap)
        self.assertIn("SUNFISH_OFFLINE_BUNDLE_MANIFEST", host_entrypoint)
        self.assertIn("--require-worker-runtime", bootstrap)
        self.assertIn("${SUNFISH_OFFLINE_BUNDLE_ROOT}/python/bin/python3", bootstrap)
        self.assertIn("sunfish_tpu.standalone_runtime verify-installed", bootstrap)
        self.assertIn("${derived_bundle_root}/python/bin/python3", host_entrypoint)
        self.assertIn(
            "${SUNFISH_OFFLINE_BUNDLE_ROOT}/python/bin/python3", config_upload
        )
        self.assertIn("SUNFISH_STOCK_PYTHON_BIN:-python3", probe)
        self.assertNotIn("ensurepip", probe)
        self.assertNotIn("python3.12", probe)
        self.assertIn("SUNFISH_STOCK_PYTHON_BIN:-python3", deployer)
        self.assertIn("extract-bundle", deployer)
        self.assertIn("verify_tpu_bundled_runtime.sh", deployer)
        self.assertNotIn("https://", deployer)
        self.assertIn("cpython-3.12.13+20260623", builder)
        self.assertIn("builder_runtime_root", builder)
        self.assertLess(
            builder.index("builder_runtime_root"), builder.index("-m pip download")
        )
        for exact_pin in (
            'RUNTIME_RELEASE = "20260623"',
            'RUNTIME_PYTHON_VERSION = "3.12.13"',
            "9fa869d69be54f6b8eeae64272fbd9bb0646e0e1a8da9d80e51ba5a3bee48930",
            "RUNTIME_ARCHIVE_SIZE = 111_146_559",
        ):
            self.assertIn(exact_pin, standalone)
        self.assertIn("HTTP_PROXY HTTPS_PROXY ALL_PROXY", host_entrypoint)
        launcher = (root / "scripts/launch_tpu_pod.sh").read_text(encoding="utf-8")
        for source_text in (launcher, host_entrypoint):
            self.assertIn("XLA_PYTHON_CLIENT_PREALLOCATE", source_text)
        self.assertIn("--xla-python-client-preallocate", launcher)
        self.assertIn("publish_pid_file", host_entrypoint)
        self.assertIn("set -o noclobber", host_entrypoint)
        self.assertIn("stop_unpublished_child", host_entrypoint)
        self.assertIn("pid_publish_gate.py", host_entrypoint)
        self.assertIn("exit 126", host_entrypoint)
        self.assertIn("sunfish_tpu.worker_hygiene", host_entrypoint)
        exact_interrupt = (
            root / "src/sunfish_tpu/exact_process_interrupt.py"
        ).read_text(encoding="utf-8")
        self.assertIn("pidfd_send_signal", exact_interrupt)
        self.assertIn("exact_recorded_descendants", exact_interrupt)
        self.assertNotIn("killpg", exact_interrupt)
        worker_hygiene = (root / "src/sunfish_tpu/worker_hygiene.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("/dev/accel", worker_hygiene)
        self.assertIn("/tmp/libtpu_lockfile", worker_hygiene)
        self.assertNotIn("os.kill(", worker_hygiene)
        self.assertLess(
            bootstrap.index("sunfish-runtime-api-audit"),
            bootstrap.index("sunfish-tpu-preflight"),
        )
        training_data = (root / "src/sunfish_tpu/training/data.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def __kd_ocp_restore_post__", training_data)
        self.assertIn("super().__kd_ocp_restore_post__(value)", training_data)
        self.assertIn("type(self)(source=restored.source, iter=restored.iter)", training_data)

    def test_runtime_contract_matches_every_direct_distribution(self):
        root = Path(__file__).resolve().parents[1]
        pins = {}
        for line in (root / "requirements-tpu.lock").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            if " @ " in line:
                name, source = line.split(" @ ", 1)
                pins[name] = "4.1.0"
                self.assertTrue(source.endswith(GEMMA_SOURCE_COMMIT))
            else:
                requirement_name, version = line.split("==", 1)
                name = requirement_name.split("[", 1)[0]
                pins[name] = version
        self.assertEqual(pins, {**RUNTIME_VERSIONS, **TPU_ONLY_RUNTIME_VERSIONS})

    def test_controller_environment_stays_accelerator_free_and_exact(self):
        root = Path(__file__).resolve().parents[1]
        lines = {
            line
            for line in (root / "requirements-controller.lock")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(
            lines,
            {
                "etils[epath-gcs]==1.14.0",
                "google-cloud-storage==3.12.1",
            },
        )
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        self.assertEqual(
            set(project["project"]["optional-dependencies"]["controller"]),
            lines,
        )
        bootstrap = (root / "scripts/bootstrap_tpu_controller.sh").read_text()
        self.assertIn("--no-deps --editable .", bootstrap)
        self.assertIn("parity_evidence", bootstrap)
        self.assertIn("offline_bundle", bootstrap)
        self.assertIn("validate_readiness_unlock", bootstrap)
        for forbidden in ("jax", "libtpu", "flax", "kauldron", "orbax"):
            self.assertNotIn(forbidden + "==", "\n".join(lines))

    def test_controller_gate_imports_do_not_import_accelerator_stack(self):
        root = Path(__file__).resolve().parents[1]
        program = """
import sys
from sunfish_tpu import deployment_config, parity_evidence, readiness_ledger
for name in ('jax', 'flax', 'kauldron', 'orbax'):
    assert name not in sys.modules, (name, sorted(k for k in sys.modules if k.startswith(name)))
assert callable(deployment_config.main)
assert callable(parity_evidence.validate_stage0_parity_report)
assert callable(readiness_ledger.validate_readiness_unlock)
"""
        subprocess.run(
            [sys.executable, "-c", program],
            cwd=root,
            env={"PYTHONPATH": "src"},
            check=True,
            capture_output=True,
            text=True,
        )


if __name__ == "__main__":
    unittest.main()
