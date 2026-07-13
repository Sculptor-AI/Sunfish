import copy
import unittest

from sunfish_tpu.topology_smoke import verify_topology_evidence


CHECKS = (
    "jax-distributed-initialize",
    "jax-tpu-topology",
    "jax-global-device-count",
    "jax-process-count",
    "jax-process-index",
    "jax-unique-process-indices",
    "jax-local-device-count",
    "jax-global-psum",
)
SOURCE = {"git_commit": "c" * 40, "source_tree_sha256": "d" * 64}


def host(process):
    return {
        "schema_version": 1,
        "gate": 1,
        "run_id": "topology-1",
        "process_index": process,
        "process_count": 2,
        "global_device_count": 8,
        "local_device_count": 4,
        "preflight": {
            "ready": True,
            "checks": [
                {"name": name, "status": "pass", "detail": "ok"}
                for name in CHECKS
            ],
        },
        "sunfish_source": SOURCE,
    }


class TopologySmokeTests(unittest.TestCase):
    def test_exact_all_host_evidence_passes(self):
        result = verify_topology_evidence(
            [host(0), host(1)],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(result["passed"], result["errors"])

    def test_missing_host_or_collective_failure_fails(self):
        broken = copy.deepcopy(host(1))
        broken["preflight"]["checks"][-1]["status"] = "fail"
        result = verify_topology_evidence(
            [host(0), broken],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("jax-global-psum" in error for error in result["errors"]))
        result = verify_topology_evidence(
            [host(0)],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
