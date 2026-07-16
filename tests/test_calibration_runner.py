import dataclasses
import inspect
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from sunfish_tpu.calibration_runner import (
    CalibrationSource,
    _existing_reconstruction_counts,
    calibration_bucket_names,
    calibration_completion_status,
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
    def test_calibration_source_rejects_negative_and_past_end_indices(self):
        source = CalibrationSource.__new__(CalibrationSource)
        source._source = ["first", "second"]
        source._ends = [1, 2]
        source._workloads = ["code_completion", "repo_edit"]
        self.assertEqual(source[0], ("first", 0))
        self.assertEqual(source[1], ("second", 1))
        for index in (-1, 2):
            with self.subTest(index=index):
                with self.assertRaises(IndexError):
                    source[index]

    def test_stage1_requires_pinned_stage05_ledger(self):
        parameters = inspect.signature(run_calibration).parameters
        self.assertIn("readiness_ledger_path", parameters)
        self.assertIn("readiness_ledger_sha256", parameters)
        self.assertIn("data_source_receipt_path", parameters)
        self.assertIn("data_source_receipt_sha256", parameters)
        source = inspect.getsource(run_calibration)
        self.assertLess(
            source.index("validate_readiness_unlock"),
            source.index("initialize_distributed_jax"),
        )
        self.assertLess(
            source.index("validate_calibration_source_contract"),
            source.index("initialize_distributed_jax"),
        )
        self.assertLess(
            source.index("verify_live_calibration_data_inventory"),
            source.index("initialize_distributed_jax"),
        )
        self.assertGreaterEqual(
            source.count("verify_live_calibration_data_inventory"), 2
        )
        source_init = inspect.getsource(CalibrationSource.__init__)
        self.assertIn("verify_shard_hashes=True", source_init)
        self.assertIn("corpus_gcs_inventory_sha256", source)
        self.assertIn("_broadcast_process0_error", source)

    def test_teacher_inventory_spans_actual_checkpoint_load(self):
        source = inspect.getsource(run_calibration)
        load_index = source.index("gm.ckpts.load_params")
        post_load_inventory_index = source.index(
            "verify_live_gcs_inventory(", load_index
        )
        self.assertGreater(post_load_inventory_index, load_index)

    def test_stage1_thresholds_cannot_be_weakened(self):
        arguments = {
            "source_checkpoint": "gs://bucket/teacher",
            "source_revision": "gcs-inventory-sha256:" + "a" * 64,
            "source_anonymous": True,
            "readiness_ledger_path": "gs://bucket/ledger.json",
            "readiness_ledger_sha256": "b" * 64,
            "data_directory": "gs://bucket/data",
            "data_manifest_sha256": "c" * 64,
            "output_dir": "gs://bucket/calibration",
            "raw_output_dir": "gs://bucket/raw",
            "run_id": "calibration-v1",
            "expected_devices": 64,
            "expected_processes": 8,
            "expected_local_devices": 8,
            "prompt_length": 512,
            "canvas_size": 256,
            "flush_every_records": 256,
            "max_records": 0,
            "reconstruction_tokens": 100_000,
            "seed": 1,
            "min_coverage": 0.0,
            "fallback_min_coverage": 0.3375,
            "min_source_tokens": 75_000_000,
        }
        with mock.patch.dict("os.environ", {"SUNFISH_RUN_ID": "calibration-v1"}):
            with self.assertRaisesRegex(ValueError, "canonical and non-overridable"):
                run_calibration(**arguments)
        arguments["source_revision"] = "generation:123"
        with self.assertRaisesRegex(ValueError, "GCS inventory SHA-256"):
            run_calibration(**arguments)

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
        completion = calibration_completion_status(
            total_records=96,
            process_count=8,
            processed_records=96,
            max_records=0,
            observed_input_tokens=75_000_000,
            minimum_input_tokens=75_000_000,
        )
        results = [
            Result(tuple(range(32)), {"prefill/code": 0.3}, 1.0, True)
            for _ in range(30)
        ]
        payload = mass_candidate_payload(
            results,
            min_coverage=0.225,
            source_revision="generation:123",
            dataset_manifest_sha256="a" * 64,
            corpus_gcs_inventory_sha256="f" * 64,
            calibration_run_sha256="b" * 64,
            completion=completion,
            sunfish_source=SOURCE,
        )
        self.assertTrue(payload["mass_gate_satisfied"])
        self.assertFalse(payload["reconstruction_gate_satisfied"])
        self.assertFalse(payload["promotion_allowed"])
        self.assertTrue(payload["promotion_eligible"])
        self.assertEqual(payload["calibration_mode"], "full")
        self.assertEqual(len(payload["layers"]), 30)
        fallback = mass_candidate_payload(
            [
                Result(tuple(range(48)), {"prefill/code": 0.4}, 1.0, True)
                for _ in range(30)
            ],
            min_coverage=0.3375,
            source_revision="generation:123",
            dataset_manifest_sha256="a" * 64,
            corpus_gcs_inventory_sha256="f" * 64,
            calibration_run_sha256="b" * 64,
            completion=completion,
            retained_experts=48,
            sunfish_source=SOURCE,
        )
        self.assertEqual(fallback["retained_experts"], 48)
        self.assertFalse(fallback["promotion_allowed"])

    def test_capped_calibration_is_always_debug_and_non_promotable(self):
        status = calibration_completion_status(
            total_records=96,
            process_count=8,
            processed_records=96,
            max_records=96,
            observed_input_tokens=75_000_000,
            minimum_input_tokens=75_000_000,
        )
        self.assertEqual(status["run_mode"], "debug-capped")
        self.assertTrue(status["debug_run"])
        self.assertFalse(status["full_corpus_consumed"])
        self.assertFalse(status["promotion_eligible"])

    def test_full_calibration_still_requires_observed_token_minimum(self):
        status = calibration_completion_status(
            total_records=96,
            process_count=8,
            processed_records=96,
            max_records=0,
            observed_input_tokens=74_999_999,
            minimum_input_tokens=75_000_000,
        )
        self.assertTrue(status["full_corpus_consumed"])
        self.assertFalse(status["minimum_tokens_observed"])
        self.assertFalse(status["promotion_eligible"])

    def test_reconstruction_counts_resume_by_host_and_bucket(self):
        run_id = "calibration-v1"
        calibration_run_sha256 = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            host = Path(temporary) / "host-00003"
            host.mkdir()
            for index, tokens in enumerate((4, 7)):
                (host / f"step-{index:09d}-denoise_high.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "run_id": run_id,
                            "calibration_run_sha256": calibration_run_sha256,
                            "artifact_id": f"step-{index:09d}-denoise_high",
                            "process_index": 3,
                            "bucket": "denoise_high/code_completion",
                            "tokens": tokens,
                        }
                    )
                )
            counts, total = _existing_reconstruction_counts(
                Path(temporary),
                process_index=3,
                run_id=run_id,
                calibration_run_sha256=calibration_run_sha256,
            )
            self.assertEqual(counts["denoise_high/code_completion"], 11)
            self.assertEqual(total, 11)


if __name__ == "__main__":
    unittest.main()
