import inspect
import unittest
from unittest import mock

from sunfish_tpu import checkpoint_smoke


class MeshShapeTests(unittest.TestCase):
    def test_default_data_axis_spans_all_global_devices(self):
        self.assertEqual(
            checkpoint_smoke._resolve_mesh_shape(
                global_device_count=64,
                process_count=8,
                data_axis_size=0,
            ),
            (64, 1),
        )

    def test_data_axis_must_equal_global_devices(self):
        with self.assertRaises(ValueError):
            checkpoint_smoke._resolve_mesh_shape(
                global_device_count=64,
                process_count=8,
                data_axis_size=7,
            )


class CheckpointInitOrderTests(unittest.TestCase):
    def test_distributed_init_is_before_backend_adjacent_imports(self):
        source = inspect.getsource(checkpoint_smoke.run_smoke)
        initialize = source.index("initialize_distributed_jax")
        self.assertLess(initialize, source.index("import jax.numpy"))
        self.assertLess(initialize, source.index("import orbax.checkpoint"))

    def test_invalid_run_id_fails_before_jax_import(self):
        with (
            mock.patch.object(checkpoint_smoke, "initialize_distributed_jax") as initialize,
            self.assertRaises(ValueError),
        ):
            checkpoint_smoke.run_smoke("/tmp/sunfish", "bad/run")
        initialize.assert_not_called()


class CheckpointEvidenceTests(unittest.TestCase):
    def test_all_host_evidence_requires_every_exact_comparison(self):
        hosts = []
        for process in range(2):
            hosts.append(
                {
                    "schema_version": 1,
                    "ready": True,
                    "run_id": "resume-1",
                    "destination": "gs://bucket/resume-1",
                    "process_index": process,
                    "process_count": 2,
                    "global_device_count": 8,
                    "local_device_count": 4,
                    "topology": {"ready": True},
                    "restored_addressable_shards_exact": True,
                    "next_loss_exact": True,
                    "next_gradients_exact": True,
                    "next_update_exact": True,
                    "sunfish_source": {
                        "git_commit": "c" * 40,
                        "source_tree_sha256": "d" * 64,
                    },
                }
            )
        result = checkpoint_smoke.verify_checkpoint_evidence(
            hosts,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(result["passed"], result["errors"])
        hosts[1]["next_gradients_exact"] = False
        result = checkpoint_smoke.verify_checkpoint_evidence(
            hosts,
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
