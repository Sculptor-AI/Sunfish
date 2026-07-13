import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from sunfish_tpu.parity import (
    SCHEMA_VERSION,
    _require_trace_pair,
    build_parser,
    checkpoint_metadata,
    compare_check,
    compare_traces,
)
from sunfish_tpu.parity_dependencies import PARITY_RUNTIME_VERSIONS


def signature(digest="a" * 64, argmax=7):
    return {
        "sha256": digest,
        "argmax": argmax,
        "dtype": "torch.float32",
        "shape": [262_144],
    }


class ParityComparisonTests(unittest.TestCase):
    def test_exact_streaming_signatures_pass_with_zero_diff(self):
        payload = {
            "contract_errors": [],
            "prompts": {"p": {"positions": [signature(), signature(argmax=8)]}},
        }
        result = compare_check("p2", payload, json.loads(json.dumps(payload)))
        self.assertTrue(result["passed"])
        self.assertEqual(result["max_abs_diff"], 0.0)
        self.assertEqual(result["argmax_agreement"], 1.0)
        self.assertEqual(result["signature_count"], 2)

    def test_any_digest_mismatch_fails_closed(self):
        upstream = {"contract_errors": [], "positions": [signature()]}
        control = {
            "contract_errors": [],
            "positions": [signature(digest="b" * 64)],
        }
        result = compare_check("p3", upstream, control)
        self.assertFalse(result["passed"])
        self.assertIsNone(result["max_abs_diff"])
        self.assertEqual(result["argmax_agreement"], 1.0)
        self.assertIn("streaming hash mismatch", result["max_abs_diff_note"])

    def test_identical_contract_error_is_still_a_failure(self):
        payload = {"contract_errors": ["only one canvas"], "value": 1}
        result = compare_check("p4", payload, dict(payload))
        self.assertFalse(result["passed"])
        self.assertEqual(result["differences"], 0)

    def test_trace_pair_requires_identical_runtime_and_tokens(self):
        base = {
            "schema_version": SCHEMA_VERSION,
            "role": "upstream",
            "dtype": "float32",
            "runtime": {"torch": "pinned"},
            "prompt_fixture": {"sha256": "fixture", "prompts": []},
            "checks": {"p2": {}, "p3": {}, "p4": {}},
            "upstream_revision": "revision",
        }
        control = json.loads(json.dumps(base))
        control["role"] = "control"
        _require_trace_pair(
            base, control, dtype_name="float32", checks={"p2", "p3", "p4"}
        )
        control["runtime"]["torch"] = "drifted"
        with self.assertRaises(ValueError):
            _require_trace_pair(
                base, control, dtype_name="float32", checks={"p2", "p3", "p4"}
            )


class ParityContractTests(unittest.TestCase):
    def test_four_trace_merge_emits_all_pass_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_fixture = {"sha256": "f" * 64, "prompts": []}
            runtime = {
                "versions": PARITY_RUNTIME_VERSIONS,
                "device": "cpu",
                "sunfish_source": {
                    "git_commit": "c" * 40,
                    "source_tree_sha256": "d" * 64,
                },
            }

            def trace(role, dtype, checks):
                return {
                    "schema_version": SCHEMA_VERSION,
                    "role": role,
                    "dtype": dtype,
                    "runtime": runtime,
                    "prompt_fixture": prompt_fixture,
                    "checks": checks,
                    "upstream_revision": "a" * 40,
                    "checkpoint": {"fingerprint": role},
                }

            fp_checks = {
                "p2": {"contract_errors": [], "positions": [signature()]},
                "p3": {"contract_errors": [], "positions": [signature(argmax=9)]},
                "p4": {"contract_errors": [], "tokens": [1, 2], "steps": [4, 3]},
            }
            bf_checks = {
                "p5": {"contract_errors": [], "positions": [signature(argmax=11)]}
            }
            paths = {}
            for name, payload in {
                "upstream_fp32": trace("upstream", "float32", fp_checks),
                "control_fp32": trace("control", "float32", fp_checks),
                "upstream_bf16": trace("upstream", "bfloat16", bf_checks),
                "control_bf16": trace("control", "bfloat16", bf_checks),
            }.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            p1_path = root / "p1.json"
            p1_path.write_text(
                json.dumps({"passed": True, "checks": {"all": True}}),
                encoding="utf-8",
            )
            conversion_path = root / "conversion.json"
            conversion_path.write_text(
                json.dumps(
                    {
                        "source_experts": 128,
                        "retained_experts": 128,
                        "top_k": 8,
                        "text_only": True,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "report.json"
            report = compare_traces(
                upstream_fp32_path=paths["upstream_fp32"],
                control_fp32_path=paths["control_fp32"],
                upstream_bf16_path=paths["upstream_bf16"],
                control_bf16_path=paths["control_bf16"],
                p1_report_path=p1_path,
                conversion_manifest_path=conversion_path,
                upstream_revision="a" * 40,
                output_path=output,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(output.is_file())
            self.assertEqual(report["checks"]["p2"]["max_abs_diff"], 0.0)

    def test_checkpoint_metadata_hashes_only_small_contract_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}', encoding="utf-8"
            )
            metadata = checkpoint_metadata(root)
            self.assertEqual(
                sorted(metadata["metadata_sha256"]),
                ["config.json", "model.safetensors.index.json"],
            )
            self.assertEqual(len(metadata["fingerprint"]), 64)

    def test_direct_parity_lock_matches_project_and_runtime_contract(self):
        root = Path(__file__).resolve().parents[1]
        with (root / "pyproject.toml").open("rb") as source:
            project = tomllib.load(source)
        project_pins = set(project["project"]["optional-dependencies"]["parity"])
        lock_pins = {
            line
            for line in (root / "requirements-parity.lock")
            .read_text(encoding="utf-8")
            .splitlines()
            if line and not line.startswith("#")
        }
        self.assertEqual(project_pins, lock_pins)
        parsed = dict(line.split("==", 1) for line in lock_pins)
        self.assertEqual(parsed, PARITY_RUNTIME_VERSIONS)

    def test_trace_cli_requires_explicit_revision_and_output(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "trace",
                "--role",
                "upstream",
                "--model",
                "/model",
                "--tokenizer",
                "/model",
                "--prompts",
                "/prompts.json",
                "--dtype",
                "float32",
                "--checks",
                "p2",
                "p3",
                "p4",
                "--upstream-revision",
                "a" * 40,
                "--output",
                "/trace.json",
            ]
        )
        self.assertEqual(args.upstream_revision, "a" * 40)
        self.assertEqual(args.checks, ["p2", "p3", "p4"])


if __name__ == "__main__":
    unittest.main()
