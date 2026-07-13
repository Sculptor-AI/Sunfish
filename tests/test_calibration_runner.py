import dataclasses
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from sunfish_tpu.calibration_runner import (
    _existing_reconstruction_counts,
    calibration_bucket_names,
    mass_candidate_payload,
    pack_calibration_tokens,
    run_calibration,
    usable_record_count,
)

SOURCE = {"git_commit": "c" * 40, "source_tree_sha256": "d" * 64}


@dataclasses.dataclass
class Result:
    selected: tuple[int, ...]
    coverage: dict[str, float]
    weighted_retained: float
    satisfied: bool


class CalibrationRunnerTests(unittest.TestCase):
    def test_stage1_requires_pinned_stage05_ledger(self):
        parameters = inspect.signature(run_calibration).parameters
        self.assertIn("readiness_ledger_path", parameters)
        self.assertIn("readiness_ledger_sha256", parameters)
        source = inspect.getsource(run_calibration)
        self.assertLess(
            source.index("validate_readiness_unlock"),
            source.index("initialize_distributed_jax"),
        )

    def test_plan_taxonomy_has_prefill_and_position_tertiles(self):
        names = calibration_bucket_names()
        self.assertEqual(len(names), 60)
        self.assertEqual(len(set(names)), 60)
        self.assertIn("prefill/repo_edit", names)
        self.assertIn("denoise_low/tool_calls/pos2", names)

    def test_fixed_shape_packing_splits_short_and_long_documents(self):
        short = pack_calibration_tokens(
            list(range(10)),
            prompt_length=8,
            canvas_size=6,
            pad_token=0,
            vocab_size=100,
        )
        self.assertEqual(short["prompt_tokens"], 5)
        self.assertEqual(short["canvas_tokens"], 5)
        self.assertEqual(len(short["prompt"]), 8)
        self.assertEqual(len(short["canvas"]), 6)
        long = pack_calibration_tokens(
            list(range(20)),
            prompt_length=8,
            canvas_size=6,
            pad_token=0,
            vocab_size=100,
        )
        self.assertEqual(long["prompt_tokens"], 8)
        self.assertEqual(long["canvas_tokens"], 6)

    def test_usable_records_are_exactly_process_divisible(self):
        self.assertEqual(usable_record_count(101, process_count=8), 96)
        self.assertEqual(
            usable_record_count(101, process_count=8, max_records=65), 64
        )
        with self.assertRaises(ValueError):
            usable_record_count(3, process_count=8)

    def test_mass_candidate_is_never_promotable_before_reconstruction(self):
        results = [
            Result(tuple(range(32)), {"prefill/code": 0.3}, 1.0, True)
            for _ in range(30)
        ]
        payload = mass_candidate_payload(
            results,
            min_coverage=0.225,
            source_revision="generation:123",
            dataset_manifest_sha256="a" * 64,
            sunfish_source=SOURCE,
        )
        self.assertTrue(payload["mass_gate_satisfied"])
        self.assertFalse(payload["reconstruction_gate_satisfied"])
        self.assertFalse(payload["promotion_allowed"])
        self.assertEqual(len(payload["layers"]), 30)
        fallback = mass_candidate_payload(
            [
                Result(tuple(range(48)), {"prefill/code": 0.4}, 1.0, True)
                for _ in range(30)
            ],
            min_coverage=0.3375,
            source_revision="generation:123",
            dataset_manifest_sha256="a" * 64,
            retained_experts=48,
            sunfish_source=SOURCE,
        )
        self.assertEqual(fallback["retained_experts"], 48)
        self.assertFalse(fallback["promotion_allowed"])

    def test_reconstruction_counts_resume_by_host_and_bucket(self):
        with tempfile.TemporaryDirectory() as temporary:
            host = Path(temporary) / "host-00003"
            host.mkdir()
            for index, tokens in enumerate((4, 7)):
                (host / f"step-{index:09d}-denoise_high.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "process_index": 3,
                            "bucket": "denoise_high/code_completion",
                            "tokens": tokens,
                        }
                    )
                )
            counts, total = _existing_reconstruction_counts(
                Path(temporary), process_index=3
            )
            self.assertEqual(counts["denoise_high/code_completion"], 11)
            self.assertEqual(total, 11)


if __name__ == "__main__":
    unittest.main()
