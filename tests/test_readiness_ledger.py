import copy
import hashlib
import json
import unittest

from sunfish_tpu.parity_evidence import validate_stage0_parity_payload
from sunfish_tpu.checkpoint_smoke import verify_checkpoint_evidence
from sunfish_tpu.input_smoke import verify_evidence as verify_input_evidence
from sunfish_tpu.real_resume_smoke import (
    verify_real_resume_evidence,
    verify_real_resume_prepare_evidence,
)
from sunfish_tpu.readiness_ledger import (
    validate_readiness_unlock,
    verify_readiness_ledger,
)
from sunfish_tpu.seed_load_smoke import verify_seed_load_evidence
from sunfish_tpu.topology_smoke import verify_topology_evidence
from sunfish_tpu.training.dependencies import (
    GEMMA_SOURCE_COMMIT,
    RUNTIME_VERSIONS,
    TPU_ONLY_RUNTIME_VERSIONS,
)
from tests.test_input_smoke import host as input_host
from tests.test_parity_evidence import valid_parity_payload
from tests.test_real_resume_smoke import (
    host as real_resume_host,
    prepare_host as real_resume_prepare_host,
)
from tests.test_seed_load_smoke import host as seed_load_host
from tests.test_topology_smoke import host as topology_host


def _payload_sha256(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixtures():
    source = {"git_commit": "c" * 40, "source_tree_sha256": "f" * 64}
    identity = {
        "schema_version": 1,
        "phase": "smoke",
        "run_id": "smoke",
        "config_sha256": "a" * 64,
        "config_file_sha256": "9" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "init_checkpoint_manifest_sha256": "c" * 64,
        "init_checkpoint": "gs://bucket/seed",
        "init_checkpoint_format": "orbax-exact-tree",
        "init_checkpoint_step": -1,
        "init_checkpoint_manifest": "gs://bucket/seed.json",
        "model": {
            "num_experts": 32,
            "top_k_experts": 4,
            "vocab_size": 262144,
            "hidden_size": 2816,
            "num_layers": 30,
            "expert_hidden_size": 704,
            "dtype": "bfloat16",
        },
        "runtime_versions": {**RUNTIME_VERSIONS, **TPU_ONLY_RUNTIME_VERSIONS},
        "gemma_source_commit": GEMMA_SOURCE_COMMIT,
        "sunfish_source": source,
        "topology": {
            "device_count": 8,
            "process_count": 2,
            "local_device_count": 4,
        },
    }
    topology_hosts = [topology_host(0), topology_host(1)]
    for item in topology_hosts:
        item["sunfish_source"] = source
    topology_summary = verify_topology_evidence(
        topology_hosts,
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
    )
    input_hosts = [
        input_host(0, [0, 2, 4], manifest="b" * 64),
        input_host(1, [1, 3, 5], manifest="b" * 64),
    ]
    for item in input_hosts:
        item["sunfish_source"] = source
    input_summary = verify_input_evidence(
        input_hosts, total_records=6, expected_processes=2
    )
    seed_hosts = [seed_load_host(0), seed_load_host(1)]
    for item in seed_hosts:
        item["sunfish_source"] = source
    seed_summary = verify_seed_load_evidence(
        seed_hosts,
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
    )
    checkpoint_hosts = []
    for process in range(2):
        checkpoint_hosts.append(
            {
                "schema_version": 1,
                "ready": True,
                "run_id": "checkpoint",
                "destination": "gs://bucket/sunfish/readiness/tag/checkpoint",
                "process_index": process,
                "process_count": 2,
                "global_device_count": 8,
                "local_device_count": 4,
                "topology": {"ready": True},
                "restored_addressable_shards_exact": True,
                "next_loss_exact": True,
                "next_gradients_exact": True,
                "next_update_exact": True,
                "sunfish_source": source,
            }
        )
    checkpoint_summary = verify_checkpoint_evidence(
        checkpoint_hosts,
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
    )
    real_prepare_hosts = [real_resume_prepare_host(0), real_resume_prepare_host(1)]
    for item in real_prepare_hosts:
        item.update(
            {
                "run_id": "real-resume",
                "config_sha256": "e" * 64,
                "config_file_sha256": "7" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "seed_manifest_sha256": "c" * 64,
                "sunfish_source": source,
            }
        )
    real_prepare_summary = verify_real_resume_prepare_evidence(
        real_prepare_hosts,
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
    )
    real_hosts = [
        real_resume_host(0, real_prepare_summary),
        real_resume_host(1, real_prepare_summary),
    ]
    for item in real_hosts:
        item.update(
            {
                "run_id": "real-resume",
                "config_sha256": "e" * 64,
                "config_file_sha256": "7" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "seed_manifest_sha256": "c" * 64,
                "sunfish_source": source,
            }
        )
    real_resume_summary = verify_real_resume_evidence(
        real_hosts,
        prepare_summary=real_prepare_summary,
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
    )
    preemption_identity = copy.deepcopy(identity)
    preemption_identity["run_id"] = "preemption"
    preemption_identity["config_sha256"] = "d" * 64
    preemption_identity["config_file_sha256"] = "8" * 64
    parity = valid_parity_payload()
    parity["sunfish_source"] = source
    parity["environment"]["float32"]["sunfish_source"] = source
    parity["environment"]["bfloat16"]["sunfish_source"] = source
    parity_summary = validate_stage0_parity_payload(
        parity, expected_source=source
    )
    parity_sha256 = _payload_sha256(parity)
    bundle = {
        "schema_version": 1,
        "purpose": "stage-0.5-rendered-config-bundle",
        "storage_root": "gs://bucket/sunfish",
        "dataset_manifest_sha256": "b" * 64,
        "seed_manifest_sha256": "c" * 64,
        "sunfish_source": source,
        "topology": {
            "global_devices": 8,
            "processes": 2,
            "local_devices": 4,
        },
        "readiness_global_batch_size": 8,
        "stage0_parity": {
            "filename": "stage0-parity-report.json",
            **parity_summary,
            "report_sha256": parity_sha256,
        },
        "configs": {
            "sunfish-smoke.toml": {
                "run_id": "smoke",
                "workdir": "gs://bucket/runs/smoke",
                "config_sha256": "a" * 64,
                "config_file_sha256": "9" * 64,
            },
            "sunfish-resume-smoke.toml": {
                "run_id": "real-resume",
                "workdir": "gs://bucket/runs/real-resume",
                "config_sha256": "e" * 64,
                "config_file_sha256": "7" * 64,
            },
            "sunfish-preemption-smoke.toml": {
                "run_id": "preemption",
                "workdir": "gs://bucket/runs/preemption",
                "config_sha256": "d" * 64,
                "config_file_sha256": "8" * 64,
            },
        },
    }
    return {
        "topology": topology_summary,
        "input": input_summary,
        "seed_load": seed_summary,
        "smoke": {
            "schema_version": 1,
            "attempt_id": "resume",
            "expected_processes": 2,
            "run_id": "smoke",
            "config_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
            "seed_manifest_sha256": "c" * 64,
            "sunfish_source": source,
            "passed": True,
            "gates": {
                "4": {
                    "passed": True,
                    "errors": [],
                    "metric_steps": 100,
                    "first_loss_median": 2.0,
                    "final_loss_median": 1.5,
                    "relative_loss_reduction": 0.25,
                    "max_gradient_norm": 1.0,
                    "max_update_norm": 0.1,
                    "required_relative_loss_reduction": 0.10,
                    "max_peak_hbm_fraction": 0.80,
                    "max_allowed_peak_hbm_fraction": 0.90,
                    "local_device_count": 4,
                },
                "8": {
                    "passed": True,
                    "errors": [],
                    "steady_state_steps": list(range(20, 100)),
                    "p95_input_wait_ratio": 0.05,
                    "max_p95_input_wait_ratio": 0.10,
                    "local_cache_policy": "none-direct-gcs-range-reads",
                    "memory_prefetch_policy": "grain-mp-prefetch-bounded",
                    "per_worker_prefetch_batches": 2,
                },
            },
            "errors": [],
        },
        "checkpoint": checkpoint_summary,
        "real_resume": real_resume_summary,
        "preemption": {
            "schema_version": 1,
            "gate": 7,
            "passed": True,
            "finalized_checkpoint_survived": True,
            "automatic_same_workdir_restore": True,
            "resume_continued_from_checkpoint": True,
            "fresh_start_metric_absent": True,
            "exact_recorded_processes_interrupted": True,
            "interrupt_process_policy": (
                "pre-signal-exact-root-and-descendant-snapshot-with-pidfd"
            ),
            "preempted_launch_exit_policy": "signal-status-only-137-or-143",
            "same_attempt_descendants_absent": True,
            "owner_intervention_required": False,
            "interrupt_timeout_seconds": 120,
            "manual_gcs_cleanup_performed": False,
            "train_complete_found": True,
            "final_checkpoint_found": True,
            "preempted_launch_returncode": 137,
            "resumed_launch_returncode": 0,
            "preempted_output_sha256": "4" * 64,
            "resumed_output_sha256": "5" * 64,
            "interrupt_output_sha256": "6" * 64,
            "plan": {
                "run_id": "preemption",
                "workdir": "gs://bucket/runs/preemption",
                "config_sha256": "d" * 64,
                "preempt_attempt": "kill",
                "resume_attempt": "resume",
                "preempt_after_step": 25,
                "preempt_marker": (
                    "gs://bucket/runs/preemption/checkpoints/ckpt_25/"
                    "commit_success.txt"
                ),
                "final_marker": (
                    "gs://bucket/runs/preemption/checkpoints/ckpt_100/"
                    "commit_success.txt"
                ),
                "train_complete": "gs://bucket/runs/preemption/train_complete.txt",
                "resume_first_metric": (
                    "gs://bucket/runs/preemption/readiness/resume/metrics/"
                    "step-000000025.json"
                ),
                "fresh_start_metric": (
                    "gs://bucket/runs/preemption/readiness/resume/metrics/"
                    "step-000000000.json"
                ),
            },
            "resume_proof": {
                "schema_version": 1,
                "attempt_id": "resume",
                "run_id": "preemption",
                "config_sha256": "d" * 64,
                "dataset_manifest_sha256": "b" * 64,
                "seed_manifest_sha256": "c" * 64,
                "step": 25,
                "sunfish_source": source,
                "metric_sha256": "6" * 64,
            },
        },
        "run_identity": identity,
        "preemption_run_identity": preemption_identity,
        "stage0_parity": parity,
        "config_bundle": bundle,
    }


def verify(evidence):
    return verify_readiness_ledger(
        evidence,
        expected_devices=8,
        expected_processes=2,
        expected_local_devices=4,
        evidence_sha256={
            name: _payload_sha256(payload) for name, payload in evidence.items()
        },
    )


def persisted_ledger():
    evidence = fixtures()
    ledger = verify(evidence)
    ledger["evidence"] = {
        name: {"path": f"gs://bucket/evidence/{name}.json", "sha256": _payload_sha256(payload)}
        for name, payload in evidence.items()
    }
    return ledger


class ReadinessLedgerTests(unittest.TestCase):
    def test_persisted_all_pass_ledger_unlocks_later_stages(self):
        ledger = persisted_ledger()
        validated = validate_readiness_unlock(
            ledger,
            expected_source={
                "git_commit": "c" * 40,
                "source_tree_sha256": "f" * 64,
            },
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(validated["passed"])

    def test_later_stage_rejects_edited_ledger(self):
        ledger = persisted_ledger()
        ledger["ordered_gates"]["6"]["passed"] = False
        with self.assertRaisesRegex(ValueError, "gate 6"):
            validate_readiness_unlock(
                ledger,
                expected_source={
                    "git_commit": "c" * 40,
                    "source_tree_sha256": "f" * 64,
                },
                expected_devices=8,
                expected_processes=2,
                expected_local_devices=4,
            )

    def test_all_eight_scoped_gates_pass(self):
        ledger = verify(fixtures())
        self.assertTrue(ledger["passed"])
        self.assertEqual(list(ledger["ordered_gates"]), [str(i) for i in range(1, 9)])

    def test_synthetic_resume_cannot_replace_real_gate_six(self):
        evidence = copy.deepcopy(fixtures())
        evidence["real_resume"]["scope"] = "synthetic-sharded-state"
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertFalse(ledger["ordered_gates"]["6"]["passed"])

    def test_gate_six_recomputes_embedded_prepare_evidence(self):
        evidence = copy.deepcopy(fixtures())
        evidence["real_resume"]["prepare_summary"]["passed"] = False
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertFalse(ledger["ordered_gates"]["6"]["passed"])
        self.assertTrue(
            any("real_resume summary" in error for error in ledger["errors"])
        )

    def test_wrong_preemption_identity_or_seed_fails_lineage(self):
        evidence = copy.deepcopy(fixtures())
        evidence["preemption_run_identity"]["dataset_manifest_sha256"] = "e" * 64
        evidence["seed_load"]["hosts"][0]["seed_manifest_sha256"] = "d" * 64
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertTrue(
            any("gate-7 identity" in error for error in ledger["errors"])
        )
        self.assertTrue(any("gate-3 seed" in error for error in ledger["errors"]))

    def test_gate_seven_rejects_surviving_same_attempt_descendants(self):
        evidence = copy.deepcopy(fixtures())
        evidence["preemption"]["same_attempt_descendants_absent"] = False
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertFalse(ledger["ordered_gates"]["7"]["passed"])

    def test_gate_seven_rejects_cleanup_hard_stop_as_interruption_evidence(self):
        evidence = copy.deepcopy(fixtures())
        evidence["preemption"]["preempted_launch_returncode"] = 126
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertFalse(ledger["ordered_gates"]["7"]["passed"])

    def test_smoke_summary_from_another_run_cannot_pass(self):
        evidence = copy.deepcopy(fixtures())
        evidence["smoke"]["config_sha256"] = "f" * 64
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertFalse(ledger["ordered_gates"]["4"]["passed"])
        self.assertFalse(ledger["ordered_gates"]["8"]["passed"])

    def test_failed_or_replaced_stage0_parity_cannot_unlock_tpus(self):
        evidence = copy.deepcopy(fixtures())
        evidence["stage0_parity"]["checks"]["p5"]["passed"] = False
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertTrue(any("P5 did not pass" in error for error in ledger["errors"]))

    def test_unbound_rendered_config_cannot_pass(self):
        evidence = copy.deepcopy(fixtures())
        evidence["config_bundle"]["configs"]["sunfish-resume-smoke.toml"][
            "config_sha256"
        ] = "0" * 64
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertTrue(any("resume-smoke.toml" in error for error in ledger["errors"]))

    def test_readiness_batch_must_stay_one_example_per_device(self):
        evidence = copy.deepcopy(fixtures())
        evidence["config_bundle"]["readiness_global_batch_size"] = 16
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertTrue(
            any("one example per global device" in error for error in ledger["errors"])
        )

    def test_top_level_pass_cannot_hide_failed_host_evidence(self):
        evidence = copy.deepcopy(fixtures())
        evidence["topology"]["hosts"][1]["preflight"]["checks"][-1][
            "status"
        ] = "fail"
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertTrue(any("embedded host" in error for error in ledger["errors"]))

    def test_quantitative_smoke_thresholds_are_rechecked(self):
        evidence = copy.deepcopy(fixtures())
        evidence["smoke"]["gates"]["4"]["relative_loss_reduction"] = 0.01
        evidence["smoke"]["gates"]["8"]["p95_input_wait_ratio"] = 0.20
        ledger = verify(evidence)
        self.assertFalse(ledger["ordered_gates"]["4"]["passed"])
        self.assertFalse(ledger["ordered_gates"]["8"]["passed"])

    def test_runtime_or_checkpoint_path_drift_fails(self):
        evidence = copy.deepcopy(fixtures())
        evidence["run_identity"]["runtime_versions"]["jax"] = "drifted"
        evidence["seed_load"]["hosts"][0]["seed_path"] = "gs://bucket/other"
        ledger = verify(evidence)
        self.assertFalse(ledger["passed"])
        self.assertTrue(any("runtime versions" in error for error in ledger["errors"]))
        self.assertTrue(any("seed path" in error for error in ledger["errors"]))


if __name__ == "__main__":
    unittest.main()
