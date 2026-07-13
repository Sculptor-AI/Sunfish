import copy
import unittest

from sunfish_tpu.real_resume_smoke import verify_real_resume_evidence


def host(process):
    payload = {
        "schema_version": 1,
        "gate": 6,
        "scope": "production-model-optimizer-grain-orbax",
        "run_id": "resume-real-1",
        "config_sha256": "a" * 64,
        "config_file_sha256": "9" * 64,
        "checkpoint_step": 1,
        "process_index": process,
        "process_count": 2,
        "global_device_count": 8,
        "local_device_count": 4,
        "attempt_id": "resume-proof",
        "dataset_manifest_sha256": "b" * 64,
        "seed_manifest_sha256": "c" * 64,
        "topology": {"ready": True},
        "next_batch_exact": True,
        "next_loss_exact": True,
        "next_trainable_gradients_exact": True,
        "next_trainable_updates_exact": True,
        "next_trainable_params_exact": True,
        "next_optimizer_state_exact": True,
        "next_collections_exact": True,
        "next_step_exact": True,
        "control_frozen_params_unchanged": True,
        "resumed_frozen_params_unchanged": True,
        "digests": {},
        "sunfish_source": {
            "git_commit": "c" * 40,
            "source_tree_sha256": "d" * 64,
        },
    }
    for name in (
        "batch",
        "loss",
        "gradients",
        "updates",
        "params",
        "opt_state",
        "collections",
        "step",
    ):
        payload["digests"][name] = {"control": name, "resumed": name}
    return payload


class RealResumeEvidenceTests(unittest.TestCase):
    def test_all_exact_host_comparisons_pass(self):
        result = verify_real_resume_evidence(
            [host(0), host(1)],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertTrue(result["passed"], result["errors"])

    def test_any_boolean_or_digest_mismatch_fails(self):
        broken = copy.deepcopy(host(1))
        broken["next_loss_exact"] = False
        broken["digests"]["gradients"]["resumed"] = "different"
        result = verify_real_resume_evidence(
            [host(0), broken],
            expected_devices=8,
            expected_processes=2,
            expected_local_devices=4,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("next_loss_exact" in error for error in result["errors"]))
        self.assertTrue(any("gradients" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
