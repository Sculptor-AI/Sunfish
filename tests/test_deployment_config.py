import json
import tempfile
import unittest
from pathlib import Path

from sunfish.source_tree import workspace_source_identity
from sunfish_tpu.deployment_config import (
    render_stage05_configs,
    validate_rendered_config_file,
)
from sunfish_tpu.training.spec import HarnessConfig
from tests.test_parity_evidence import valid_parity_payload


ROOT = Path(__file__).resolve().parents[1]


class DeploymentConfigTests(unittest.TestCase):
    def _parity_report(self, directory: Path) -> Path:
        payload = valid_parity_payload()
        source = workspace_source_identity(ROOT)
        payload["sunfish_source"] = source
        payload["environment"]["float32"]["sunfish_source"] = source
        payload["environment"]["bfloat16"]["sunfish_source"] = source
        path = directory / "parity-report.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def test_renders_three_isolated_valid_configs_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            output = temporary_path / "bundle"
            payload = render_stage05_configs(
                template_directory=ROOT / "configs/training",
                output_directory=output,
                storage_root="gs://bucket/sunfish",
                run_tag="grant-001",
                dataset_manifest_sha256="a" * 64,
                seed_manifest_sha256="b" * 64,
                parity_report_path=self._parity_report(temporary_path),
                expected_devices=32,
                expected_processes=8,
                expected_local_devices=4,
                source_root=ROOT,
            )
            configs = {
                name: HarnessConfig.load(output / name)
                for name in (
                    "sunfish-smoke.toml",
                    "sunfish-resume-smoke.toml",
                    "sunfish-preemption-smoke.toml",
                )
            }
            self.assertEqual(len({config.run.run_id for config in configs.values()}), 3)
            self.assertEqual(len({config.run.workdir for config in configs.values()}), 3)
            for config in configs.values():
                self.assertEqual(config.data.manifest_sha256, "a" * 64)
                self.assertEqual(
                    config.data.directory,
                    "gs://bucket/sunfish/data/tiny-overfit-grant-001",
                )
                self.assertEqual(config.checkpoint.init_manifest_sha256, "b" * 64)
                self.assertEqual(config.topology.expected_devices, 32)
            on_disk = json.loads((output / "rendered-configs.json").read_text())
            self.assertEqual(on_disk, payload)
            self.assertEqual(payload["stage0_parity"]["p1_tensors_compared"], 691)
            self.assertEqual(
                payload["stage0_parity"]["filename"], "stage0-parity-report.json"
            )
            self.assertRegex(payload["sunfish_source"]["source_tree_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                validate_rendered_config_file(
                    output / "sunfish-smoke.toml", source_root=ROOT
                ),
                payload,
            )
            with (output / "sunfish-smoke.toml").open("a", encoding="utf-8") as file:
                file.write("# changed after rendering\n")
            with self.assertRaisesRegex(ValueError, "differs from bundle"):
                validate_rendered_config_file(
                    output / "sunfish-smoke.toml", source_root=ROOT
                )

    def test_rejects_missing_or_changed_parity_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            with self.assertRaisesRegex(ValueError, "requires a rendered config"):
                validate_rendered_config_file(
                    ROOT / "configs/training/sunfish-smoke.toml",
                    source_root=ROOT,
                    require_bundle=True,
                )
            parity = self._parity_report(temporary_path)
            output = temporary_path / "bundle"
            render_stage05_configs(
                template_directory=ROOT / "configs/training",
                output_directory=output,
                storage_root="gs://bucket/sunfish",
                run_tag="grant-001",
                dataset_manifest_sha256="a" * 64,
                seed_manifest_sha256="b" * 64,
                parity_report_path=parity,
                expected_devices=32,
                expected_processes=8,
                expected_local_devices=4,
                source_root=ROOT,
            )
            copied = output / "stage0-parity-report.json"
            copied.write_bytes(copied.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "parity report differs"):
                validate_rendered_config_file(
                    output / "sunfish-smoke.toml", source_root=ROOT
                )

    def test_selected_launch_validates_every_config_in_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            output = temporary_path / "bundle"
            render_stage05_configs(
                template_directory=ROOT / "configs/training",
                output_directory=output,
                storage_root="gs://bucket/sunfish",
                run_tag="grant-001",
                dataset_manifest_sha256="a" * 64,
                seed_manifest_sha256="b" * 64,
                parity_report_path=self._parity_report(temporary_path),
                expected_devices=32,
                expected_processes=8,
                expected_local_devices=4,
                source_root=ROOT,
            )
            with (output / "sunfish-resume-smoke.toml").open(
                "a", encoding="utf-8"
            ) as file:
                file.write("# changed after rendering\n")
            with self.assertRaisesRegex(ValueError, "differs from bundle"):
                validate_rendered_config_file(
                    output / "sunfish-smoke.toml", source_root=ROOT
                )

    def test_rejects_placeholders_bad_topology_and_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            base = dict(
                template_directory=ROOT / "configs/training",
                output_directory=temporary_path / "bundle",
                storage_root="gs://bucket/sunfish",
                run_tag="grant-001",
                dataset_manifest_sha256="a" * 64,
                seed_manifest_sha256="b" * 64,
                parity_report_path=self._parity_report(temporary_path),
                expected_devices=32,
                expected_processes=8,
                expected_local_devices=4,
                source_root=ROOT,
            )
            with self.assertRaisesRegex(ValueError, "nonzero"):
                render_stage05_configs(
                    **{**base, "dataset_manifest_sha256": "0" * 64}
                )
            with self.assertRaisesRegex(ValueError, "must equal"):
                render_stage05_configs(**{**base, "expected_devices": 64})
            base["output_directory"].mkdir()
            with self.assertRaises(FileExistsError):
                render_stage05_configs(**base)


if __name__ == "__main__":
    unittest.main()
