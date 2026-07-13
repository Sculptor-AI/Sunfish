import unittest

from sunfish_tpu.smoke_evidence import analyze_smoke_evidence


LINEAGE = {
    "run_id": "smoke-run",
    "config_sha256": "a" * 64,
    "dataset_manifest_sha256": "b" * 64,
    "seed_manifest_sha256": "c" * 64,
    "sunfish_source": {
        "git_commit": "d" * 40,
        "source_tree_sha256": "e" * 64,
    },
}


def metric(step, loss, *, attempt="a1", sps=2.0, grad=1.0, update=0.1):
    return {
        "schema_version": 1,
        "attempt_id": attempt,
        **LINEAGE,
        "process_index": 0,
        "process_count": 2,
        "step": step,
        "scalars": {
            "losses/total": loss,
            "metrics/gradient_norm": grad,
            "metrics/update_norm": update,
            "perf_stats/steps_per_sec": sps,
        },
    }


def wait(step, process, seconds=0.01, *, attempt="a1"):
    return {
        "schema_version": 1,
        "attempt_id": attempt,
        **LINEAGE,
        "process_index": process,
        "process_count": 2,
        "step": step,
        "input_wait": {
            "samples": 1,
            "total_seconds": seconds,
            "mean_seconds": seconds,
            "max_seconds": seconds,
        },
        "local_cache_policy": "none-direct-gcs-range-reads",
        "local_cache_bytes": 0,
    }


class SmokeEvidenceTests(unittest.TestCase):
    def test_overfit_and_nonstarved_attempt_passes(self):
        metrics = [metric(step, 2.0 - 0.02 * step) for step in range(20)]
        waits = [wait(step, process) for step in range(20) for process in range(2)]
        result = analyze_smoke_evidence(
            metrics,
            waits,
            expected_processes=2,
            min_steps=20,
            steady_state_start=5,
        )
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["gates"]["4"]["passed"])
        self.assertTrue(result["gates"]["8"]["passed"])
        self.assertEqual(result["sunfish_source"], LINEAGE["sunfish_source"])

    def test_flat_loss_fails_gate4(self):
        metrics = [metric(step, 2.0) for step in range(20)]
        waits = [wait(step, process) for step in range(20) for process in range(2)]
        result = analyze_smoke_evidence(
            metrics,
            waits,
            expected_processes=2,
            min_steps=20,
            steady_state_start=5,
        )
        self.assertFalse(result["gates"]["4"]["passed"])
        self.assertTrue(result["gates"]["8"]["passed"])

    def test_missing_host_and_high_wait_fail_gate8(self):
        metrics = [metric(step, 2.0 - 0.02 * step) for step in range(20)]
        waits = [wait(step, 0, seconds=0.2) for step in range(20)]
        result = analyze_smoke_evidence(
            metrics,
            waits,
            expected_processes=2,
            min_steps=20,
            steady_state_start=5,
        )
        self.assertFalse(result["gates"]["8"]["passed"])
        self.assertTrue(any("hosts" in error for error in result["errors"]))

    def test_noncontiguous_or_nonfinite_metrics_fail(self):
        metrics = [metric(step, 1.0) for step in range(5)]
        for payload in metrics:
            payload["process_count"] = 1
        metrics.pop(2)
        metrics[-1]["scalars"]["losses/total"] = float("nan")
        result = analyze_smoke_evidence(
            metrics,
            [],
            expected_processes=1,
            min_steps=4,
            steady_state_start=0,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("contiguous" in error for error in result["errors"]))
        self.assertTrue(any("non-finite" in error for error in result["errors"]))

    def test_cross_run_lineage_cannot_be_merged(self):
        metrics = [metric(step, 2.0 - 0.02 * step) for step in range(20)]
        waits = [wait(step, process) for step in range(20) for process in range(2)]
        waits[-1]["sunfish_source"] = {
            **waits[-1]["sunfish_source"],
            "source_tree_sha256": "0" * 64,
        }
        result = analyze_smoke_evidence(
            metrics,
            waits,
            expected_processes=2,
            min_steps=20,
            steady_state_start=5,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["8"]["passed"])
        self.assertTrue(any("lineages differ" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
