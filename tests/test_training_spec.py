import copy
import dataclasses
import unittest
from pathlib import Path

from sunfish_tpu.training.spec import CheckpointFormat, HarnessConfig, Phase


ROOT = Path(__file__).resolve().parents[1]


class TrainingSpecTests(unittest.TestCase):
    def test_checked_in_configs_are_strict_and_digest_stable(self):
        smoke = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        router = HarnessConfig.load(ROOT / "configs/training/sunfish-router.toml")
        recovery = HarnessConfig.load(ROOT / "configs/training/sunfish-recovery.toml")
        self.assertEqual(smoke.run.phase, Phase.SMOKE)
        self.assertEqual(router.run.phase, Phase.ROUTER)
        self.assertEqual(recovery.run.phase, Phase.LORA)
        self.assertEqual(
            recovery.checkpoint.format, CheckpointFormat.KAULDRON_PARAMS
        )
        self.assertEqual(recovery.checkpoint.init_step, 10_000)
        self.assertEqual(recovery.objective.noise_draws, 4)
        self.assertEqual(len(recovery.digest), 64)
        self.assertEqual(recovery.digest, HarnessConfig.from_mapping(recovery.canonical_dict()).digest)

    def test_unknown_key_is_rejected_instead_of_silently_ignored(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        payload = copy.deepcopy(config.canonical_dict())
        payload["optimizer"]["peak_learing_rate"] = payload["optimizer"]["peak_learning_rate"]
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            HarnessConfig.from_mapping(payload)

    def test_smoke_update_range_is_binding(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        invalid = dataclasses.replace(
            config, training=dataclasses.replace(config.training, steps=99)
        )
        with self.assertRaisesRegex(ValueError, "100-500"):
            invalid.validate()

    def test_global_batch_must_divide_across_topology(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        invalid = dataclasses.replace(
            config, data=dataclasses.replace(config.data, global_batch_size=33)
        )
        with self.assertRaisesRegex(ValueError, "processes"):
            invalid.validate()

    def test_full_architecture_is_audited_not_free_form(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        invalid = dataclasses.replace(
            config, model=dataclasses.replace(config.model, hidden_size=2048)
        )
        with self.assertRaisesRegex(ValueError, "audited"):
            invalid.validate()

    def test_phase_promotion_checkpoint_must_pin_a_step(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-recovery.toml")
        invalid = dataclasses.replace(
            config,
            checkpoint=dataclasses.replace(config.checkpoint, init_step=-1),
        )
        with self.assertRaisesRegex(ValueError, "explicit Kauldron step"):
            invalid.validate()


if __name__ == "__main__":
    unittest.main()
