import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    HAVE_NUMPY = True
except ModuleNotFoundError:
    HAVE_NUMPY = False

from sunfish_tpu.reconstruction_drain import ReconstructionDrain


@unittest.skipUnless(HAVE_NUMPY, "numpy not installed")
class ReconstructionDrainTests(unittest.TestCase):
    def test_double_buffered_drain_caps_tokens_and_writes_exact_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = {
                "shared_pre_router_residual": np.ones((4, 2, 3), np.float16),
                "topk_indices": np.zeros((4, 2, 2), np.uint8),
                "final_scaled_topk_weights": np.ones((4, 2, 2), np.float16),
            }
            drain = ReconstructionDrain(
                output_dir=temporary,
                process_index=0,
                run_id="calibration-v1",
                calibration_run_sha256="a" * 64,
                max_tokens=5,
            )
            self.assertTrue(
                drain.submit(
                    artifacts,
                    bucket="denoise_high/code_completion",
                    artifact_id="batch-000",
                    valid_tokens=4,
                )
            )
            self.assertTrue(
                drain.submit(
                    artifacts,
                    bucket="denoise_mid/code_completion",
                    artifact_id="batch-001",
                    valid_tokens=4,
                )
            )
            self.assertEqual(len(drain.flush()), 2)
            self.assertEqual(drain.pending, 0)
            self.assertLessEqual(drain.pending, 2)
            self.assertFalse(
                drain.submit(
                    artifacts,
                    bucket="denoise_low/code_completion",
                    artifact_id="batch-002",
                    valid_tokens=4,
                )
            )
            manifests = drain.close()
            self.assertEqual(drain.tokens_submitted, 5)
            self.assertEqual(len(manifests), 2)
            payloads = [json.loads(Path(path).read_text()) for path in manifests]
            self.assertEqual(sum(payload["tokens"] for payload in payloads), 5)
            for payload in payloads:
                self.assertEqual(payload["run_id"], "calibration-v1")
                self.assertEqual(payload["calibration_run_sha256"], "a" * 64)
                self.assertEqual(
                    set(payload["fields"]),
                    {
                        "shared_pre_router_residual",
                        "topk_indices",
                        "final_scaled_topk_weights",
                    },
                )

    def test_restart_quota_and_completed_manifest_tracking(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = {
                "shared_pre_router_residual": np.ones((1, 1, 1), np.float16),
                "topk_indices": np.zeros((1, 1, 1), np.uint8),
                "final_scaled_topk_weights": np.ones((1, 1, 1), np.float16),
            }
            drain = ReconstructionDrain(
                output_dir=temporary,
                process_index=0,
                run_id="calibration-v1",
                calibration_run_sha256="a" * 64,
                max_tokens=4,
                initial_tokens=2,
            )
            for index in range(3):
                drain.submit(
                    artifacts,
                    bucket="denoise_high/code_completion",
                    artifact_id=f"batch-{index:03d}",
                    valid_tokens=1,
                )
            manifests = drain.close()
            self.assertEqual(drain.tokens_submitted, 4)
            self.assertEqual(len(manifests), 2)

    def test_contract_rejects_unbounded_or_incomplete_buffers(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "double buffering"):
                ReconstructionDrain(
                    output_dir=temporary,
                    process_index=0,
                    run_id="calibration-v1",
                    calibration_run_sha256="a" * 64,
                    max_tokens=1,
                    max_pending=3,
                )


if __name__ == "__main__":
    unittest.main()
