import unittest

from sunfish_tpu.seed_load_smoke import verify_seed_load_evidence


def host(process):
    return {
        "schema_version": 1,
        "run_id": "seed-load",
        "process_index": process,
        "process_count": 2,
        "global_device_count": 8,
        "local_device_count": 4,
        "gate": 3,
        "scope": "real-8b-orbax-seed-target-sharded-restore",
        "seed_path": "gs://bucket/seed",
        "seed_manifest_path": "gs://bucket/seed.json",
        "tree_sha256": "a" * 64,
        "sharding_sha256": "b" * 64,
        "seed_manifest_sha256": "c" * 64,
        "seed_gcs_inventory_sha256": "e" * 64,
        "restored_exact_target_tree": True,
        "restored_exact_target_shardings": True,
        "host_does_not_hold_full_model": True,
        "global_parameter_bytes": 1000,
        "host_parameter_bytes": 300,
        "device_parameter_bytes": {f"TPU:{process}": 300},
        "topology": {"ready": True},
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
    }


class SeedLoadSmokeTests(unittest.TestCase):
    def test_every_host_proves_target_sharded_real_seed(self):
        summary = verify_seed_load_evidence(
            [host(0), host(1)],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(summary["gate"], 3)

    def test_one_full_host_replica_fails(self):
        hosts = [host(0), host(1)]
        hosts[1]["host_parameter_bytes"] = 1000
        hosts[1]["host_does_not_hold_full_model"] = False
        summary = verify_seed_load_evidence(
            hosts,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(summary["passed"])
        self.assertTrue(any("process 1" in error for error in summary["errors"]))


if __name__ == "__main__":
    unittest.main()
