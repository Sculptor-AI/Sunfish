import contextlib
import io
import os
import unittest
from pathlib import Path
from unittest import mock

from sunfish_tpu.training import train


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs/training/sunfish-smoke.toml"


class TrainingLauncherTests(unittest.TestCase):
    def test_harness_does_not_import_upstream_checkpoint_evaluator_side_effect(self):
        training_root = ROOT / "src/sunfish_tpu/training"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in training_root.glob("*.py")
        )
        self.assertNotIn("import sft_model", source)
        self.assertNotIn("checkpointed_evaluator", source)

    def test_validate_only_never_launches_jax(self):
        with (
            mock.patch.object(train.kauldron_launch, "main") as launch,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            train.main(["--config", str(SMOKE), "--validate-only"])
        launch.assert_not_called()
        self.assertIn("config_sha256", output.getvalue())

    def test_local_override_zeros_hardware_expectations(self):
        with (
            mock.patch.object(train.kauldron_launch, "main") as launch,
            contextlib.redirect_stdout(io.StringIO()),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            train.main(["--config", str(SMOKE), "--allow-non-tpu"])
            configured_path = os.environ["SUNFISH_TRAIN_CONFIG"]
            allow_non_tpu = os.environ["SUNFISH_ALLOW_NON_TPU"]
        forwarded = launch.call_args.args[0]
        self.assertIn("--allow-non-tpu", forwarded)
        self.assertEqual(forwarded[forwarded.index("--expected-devices") + 1], "0")
        self.assertEqual(configured_path, str(SMOKE.resolve()))
        self.assertEqual(allow_non_tpu, "1")
        self.assertTrue(any(arg.endswith("configs/training/sunfish.py") for arg in forwarded))

    def test_konfig_overrides_are_refused(self):
        with self.assertRaisesRegex(SystemExit, "run-identity"):
            train.main(
                [
                    "--config",
                    str(SMOKE),
                    "--allow-non-tpu",
                    "--",
                    "--cfg.workdir=/tmp/other",
                ]
            )


if __name__ == "__main__":
    unittest.main()
