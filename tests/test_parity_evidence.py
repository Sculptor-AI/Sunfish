import json
import tempfile
import unittest
from pathlib import Path

from sunfish_tpu.parity_dependencies import (
    MODEL_CLASS,
    PARITY_RUNTIME_VERSIONS,
    TRANSFORMERS_RELEASE,
)
from sunfish_tpu.parity_evidence import (
    validate_stage0_parity_payload,
    validate_stage0_parity_report,
)


SOURCE = {"git_commit": "c" * 40, "source_tree_sha256": "d" * 64}


def _runtime():
    return {
        "versions": dict(PARITY_RUNTIME_VERSIONS),
        "python": "3.12.10",
        "platform": "Linux",
        "machine": "x86_64",
        "processor": "x86_64",
        "device": "cpu",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "deterministic_algorithms": True,
        "mkldnn_enabled": True,
        "torch_config_sha256": "1" * 64,
        "model_class": MODEL_CLASS,
        "transformers_release": TRANSFORMERS_RELEASE,
        "sunfish_source": SOURCE,
    }


def valid_parity_payload():
    prompts = []
    categories = ("code", "prose", "multilingual", "structured")
    for category in categories:
        for index in range(8):
            prompts.append(
                {
                    "id": f"{category}-{index}",
                    "category": category,
                    "fixture_text_sha256": "2" * 64,
                    "raw_token_count": 1200 if index < 2 else 512,
                    "token_count": 1200 if index < 2 else 512,
                    "truncated_to_contract_max": False,
                    "crosses_sliding_window": index < 2,
                    "token_ids_sha256": "3" * 64,
                }
            )
    exact = {
        "passed": True,
        "contract_errors": [],
        "differences": 0,
        "mismatches": [],
        "signature_count": 10,
        "argmax_agreement": 1.0,
        "max_abs_diff": 0.0,
    }
    return {
        "schema_version": 1,
        "stage": "stage-0-parity",
        "upstream_revision": "a" * 40,
        "sunfish_source": SOURCE,
        "passed": True,
        "checks": {
            "p1": {
                "passed": True,
                "contract_errors": [],
                "static_report": {
                    "passed": True,
                    "tensors_compared": 691,
                    "checks": {
                        "p1.1_tokenizer_identical": True,
                        "p1.2_config_diff_exact": True,
                        "p1.2_vision_config_null": True,
                        "p1.3_tensor_set_exact": True,
                        "p1.3_all_tensor_hashes_equal": True,
                    },
                },
            },
            "p2": dict(exact),
            "p3": dict(exact),
            "p4": {
                **exact,
                "signature_count": 0,
                "argmax_agreement": None,
                "max_abs_diff": None,
            },
            "p5": dict(exact),
        },
        "environment": {
            "float32": _runtime(),
            "bfloat16": _runtime(),
            "requirements": dict(PARITY_RUNTIME_VERSIONS),
        },
        "prompt_fixture": {
            "path": "/mnt/parity-prompts.json",
            "sha256": "4" * 64,
            "tokenizer_class": "GemmaTokenizerFast",
            "tokenizer_vocab_size": 262144,
            "prompts": prompts,
        },
        "conversion_manifest": {
            "source_experts": 128,
            "retained_experts": 128,
            "top_k": 8,
            "text_only": True,
        },
        "checkpoint_metadata": {
            "upstream": {
                "fingerprint": "5" * 64,
                "metadata_sha256": {
                    "config.json": "6" * 64,
                    "model.safetensors.index.json": "7" * 64,
                },
            },
            "control": {
                "fingerprint": "8" * 64,
                "metadata_sha256": {
                    "config.json": "9" * 64,
                    "model.safetensors.index.json": "a" * 64,
                    "sunfish_conversion.json": "b" * 64,
                },
            },
        },
        "artifacts": {
            "p1_report": {"path": "/p1.json", "sha256": "c" * 64},
            "conversion_manifest": {
                "path": "/conversion.json",
                "sha256": "d" * 64,
            },
            "traces": {
                "upstream_float32": "e" * 64,
                "control_float32": "f" * 64,
                "upstream_bfloat16": "1" * 64,
                "control_bfloat16": "2" * 64,
            },
        },
    }


class ParityEvidenceTests(unittest.TestCase):
    def test_complete_report_passes_and_returns_compact_pin(self):
        summary = validate_stage0_parity_payload(
            valid_parity_payload(), expected_source=SOURCE
        )
        self.assertEqual(summary["upstream_revision"], "a" * 40)
        self.assertEqual(summary["p1_tensors_compared"], 691)
        self.assertRegex(summary["checks_sha256"], r"^[0-9a-f]{64}$")

    def test_any_failed_gate_is_rejected(self):
        payload = valid_parity_payload()
        payload["checks"]["p4"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "P4 did not pass"):
            validate_stage0_parity_payload(payload, expected_source=SOURCE)

    def test_zero_signature_exactness_claim_is_rejected(self):
        payload = valid_parity_payload()
        payload["checks"]["p3"]["signature_count"] = 0
        with self.assertRaisesRegex(ValueError, "P3 compared no"):
            validate_stage0_parity_payload(payload, expected_source=SOURCE)

    def test_source_or_runtime_drift_is_rejected(self):
        payload = valid_parity_payload()
        payload["environment"]["bfloat16"]["versions"]["torch"] = "drifted"
        with self.assertRaisesRegex(ValueError, "runtime versions"):
            validate_stage0_parity_payload(payload, expected_source=SOURCE)
        payload = valid_parity_payload()
        with self.assertRaisesRegex(ValueError, "top-level source"):
            validate_stage0_parity_payload(
                payload,
                expected_source={
                    "git_commit": "e" * 40,
                    "source_tree_sha256": "f" * 64,
                },
            )

    def test_file_validator_pins_exact_report_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            raw = json.dumps(valid_parity_payload(), indent=2).encode() + b"\n"
            path.write_bytes(raw)
            summary, observed = validate_stage0_parity_report(
                path, expected_source=SOURCE
            )
            self.assertEqual(observed, raw)
            self.assertRegex(summary["report_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
