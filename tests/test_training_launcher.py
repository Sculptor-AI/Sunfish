import contextlib
import io
import os
import tempfile
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
        self.assertIn(
            f"--cfg={Path(train.__file__).resolve().with_name('kauldron_config.py')}",
            forwarded,
        )

    def test_noneditable_wheel_resolves_config_inside_installed_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = (
                root
                / "venv/lib/python3.12/site-packages/sunfish_tpu/training"
            )
            package.mkdir(parents=True)
            installed_train = package / "train.py"
            installed_train.touch()
            installed_config = package / "kauldron_config.py"
            installed_config.touch()
            expected_config = installed_config.resolve()
            outside_checkout = root / "outside-checkout"
            outside_checkout.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(outside_checkout)
                with (
                    mock.patch.object(train, "__file__", str(installed_train)),
                    mock.patch.object(train.kauldron_launch, "main") as launch,
                    contextlib.redirect_stdout(io.StringIO()),
                    mock.patch.dict(os.environ, {}, clear=False),
                ):
                    train.main(["--config", str(SMOKE), "--allow-non-tpu"])
            finally:
                os.chdir(previous_cwd)

        forwarded = launch.call_args.args[0]
        self.assertIn(f"--cfg={expected_config}", forwarded)
        self.assertNotIn(str(ROOT / "configs/training/sunfish.py"), forwarded)

    def test_packaged_config_must_be_a_regular_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "site-packages/sunfish_tpu/training"
            package.mkdir(parents=True)
            installed_train = package / "train.py"
            installed_train.touch()
            with mock.patch.object(train, "__file__", str(installed_train)):
                with (
                    self.assertRaisesRegex(FileNotFoundError, "config is missing"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    train.main(["--config", str(SMOKE), "--validate-only"])
                (package / "kauldron_config.py").mkdir()
                with self.assertRaisesRegex(RuntimeError, "not a regular file"):
                    train._packaged_kauldron_config_path()

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
