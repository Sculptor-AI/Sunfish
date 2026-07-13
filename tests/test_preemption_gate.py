import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from sunfish.source_tree import workspace_source_identity
from sunfish_tpu.deployment_config import render_stage05_configs
from sunfish_tpu.preemption_gate import (
    build_preemption_plan,
    checkpoint_commit_marker,
    run_preemption_gate,
)
from sunfish_tpu.training.spec import HarnessConfig, Phase
from tests.test_parity_evidence import valid_parity_payload


ROOT = Path(__file__).resolve().parents[1]


class PreemptionGateTests(unittest.TestCase):
    def test_controller_launch_kill_resume_and_upload_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            state.mkdir()
            source = workspace_source_identity(ROOT)
            parity_payload = valid_parity_payload()
            parity_payload["sunfish_source"] = source
            parity_payload["environment"]["float32"]["sunfish_source"] = source
            parity_payload["environment"]["bfloat16"]["sunfish_source"] = source
            parity_path = root / "parity.json"
            parity_path.write_text(json.dumps(parity_payload), encoding="utf-8")
            bundle = root / "bundle"
            render_stage05_configs(
                template_directory=ROOT / "configs/training",
                output_directory=bundle,
                storage_root="gs://bucket/sunfish",
                run_tag="preemption-integration",
                dataset_manifest_sha256="a" * 64,
                seed_manifest_sha256="b" * 64,
                parity_report_path=parity_path,
                expected_devices=32,
                expected_processes=8,
                expected_local_devices=4,
                source_root=ROOT,
            )
            config_path = bundle / "sunfish-preemption-smoke.toml"
            config = HarnessConfig.load(config_path)
            resume_metric_template = state / "resume-metric-template.json"
            resume_metric_template.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "attempt_id": "stage05-resume-integration",
                        "run_id": config.run.run_id,
                        "config_sha256": config.digest,
                        "dataset_manifest_sha256": config.data.manifest_sha256,
                        "seed_manifest_sha256": config.checkpoint.init_manifest_sha256,
                        "step": 25,
                        "sunfish_source": workspace_source_identity(ROOT),
                    }
                ),
                encoding="utf-8",
            )
            fake_gcloud = root / "gcloud"
            fake_gcloud.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
state="${SUNFISH_FAKE_GCLOUD_STATE:?}"
if [[ "$1 $2" == "storage ls" ]]; then
  uri="$3"
  case "$uri" in
    *ckpt_25/commit_success.txt) test -f "$state/preempt-marker" ;;
    *ckpt_100/commit_success.txt) test -f "$state/final-marker" ;;
    *train_complete.txt) test -f "$state/train-complete" ;;
    *stage05-resume-integration/metrics/step-000000025.json) test -f "$state/resume-metric" ;;
    *stage05-resume-integration/metrics/step-000000000.json) test -f "$state/fresh-metric" ;;
    *preemption-summary.json) test -f "$state/evidence.json" ;;
    *) exit 1 ;;
  esac
  exit $?
fi
if [[ "$1 $2" == "storage cat" ]]; then
  cat "$state/resume-metric"
  exit 0
fi
if [[ "$1 $2" == "storage cp" ]]; then
  cp "$3" "$state/evidence.json"
  exit 0
fi
if [[ "$1 $2 $3" == "compute tpus tpu-vm" ]]; then
  command=""
  previous=""
  for argument in "$@"; do
    if [[ "$previous" == "--command" ]]; then command="$argument"; break; fi
    previous="$argument"
  done
  if [[ "$command" == *"kill -KILL"* ]]; then
    touch "$state/killed"
    exit 0
  fi
  if [[ "$command" == *"stage05-kill-integration"* ]]; then
    touch "$state/preempt-marker"
    while [[ ! -f "$state/killed" ]]; do sleep 0.01; done
    exit 137
  fi
  if [[ "$command" == *"stage05-resume-integration"* ]]; then
    cp "$state/resume-metric-template.json" "$state/resume-metric"
    touch "$state/final-marker" "$state/train-complete"
    exit 0
  fi
fi
exit 2
""",
                encoding="utf-8",
            )
            fake_gcloud.chmod(0o755)
            environment = {
                "SUNFISH_GCLOUD_BIN": str(fake_gcloud),
                "SUNFISH_FAKE_GCLOUD_STATE": str(state),
                "SUNFISH_TRAIN_BIN": "fake-sunfish-train",
                "SUNFISH_CONTROLLER_LOG_DIR": str(root / "logs"),
                "TPU_NAME": "fake-tpu",
                "PROJECT_ID": "fake-project",
                "ZONE": "fake-zone",
                "REMOTE_REPO_DIR": str(ROOT),
                "EXPECTED_TPU_DEVICES": "32",
                "EXPECTED_TPU_PROCESSES": "8",
                "EXPECTED_LOCAL_TPU_DEVICES": "4",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                payload = run_preemption_gate(
                    config_path=config_path,
                    preempt_attempt="stage05-kill-integration",
                    resume_attempt="stage05-resume-integration",
                    preempt_after_step=25,
                    evidence_uri=(
                        "gs://fake/sunfish/preemption/preemption-summary.json"
                    ),
                    timeout_seconds=5,
                    poll_seconds=0,
                )
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["preempted_launch_returncode"], 137)
            self.assertEqual(payload["resumed_launch_returncode"], 0)
            self.assertTrue(payload["resume_continued_from_checkpoint"])
            self.assertEqual(payload["resume_proof"]["step"], 25)
            uploaded = json.loads((state / "evidence.json").read_text())
            self.assertTrue(uploaded["automatic_same_workdir_restore"])

    def test_preemption_template_only_changes_run_identity_and_workdir(self):
        smoke = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        for name in ("sunfish-preemption-smoke.toml", "sunfish-resume-smoke.toml"):
            diagnostic = HarnessConfig.load(ROOT / "configs/training" / name)
            smoke_payload = smoke.canonical_dict()
            diagnostic_payload = diagnostic.canonical_dict()
            for payload in (smoke_payload, diagnostic_payload):
                payload["run"].pop("run_id")
                payload["run"].pop("workdir")
            self.assertEqual(smoke_payload, diagnostic_payload)

    def test_plan_uses_pinned_orbax_gcs_finalization_marker(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        plan = build_preemption_plan(
            config,
            preempt_attempt="kill-001",
            resume_attempt="resume-001",
            preempt_after_step=25,
        )
        self.assertEqual(
            plan["preempt_marker"],
            checkpoint_commit_marker(config.run.workdir, 25),
        )
        self.assertTrue(plan["preempt_marker"].endswith("/commit_success.txt"))
        self.assertTrue(plan["train_complete"].endswith("/train_complete.txt"))
        self.assertTrue(
            plan["resume_first_metric"].endswith(
                "/readiness/resume-001/metrics/step-000000025.json"
            )
        )
        self.assertTrue(
            plan["fresh_start_metric"].endswith(
                "/readiness/resume-001/metrics/step-000000000.json"
            )
        )

    def test_preemption_must_happen_at_an_interior_checkpoint(self):
        config = HarnessConfig.load(ROOT / "configs/training/sunfish-smoke.toml")
        with self.assertRaises(ValueError):
            build_preemption_plan(
                config,
                preempt_attempt="same",
                resume_attempt="same",
                preempt_after_step=25,
            )
        with self.assertRaisesRegex(ValueError, "checkpoint step"):
            build_preemption_plan(
                config,
                preempt_attempt="kill",
                resume_attempt="resume",
                preempt_after_step=24,
            )
        non_smoke = dataclasses.replace(
            config,
            run=dataclasses.replace(config.run, phase=Phase.ROUTER),
        )
        with self.assertRaisesRegex(ValueError, "phase=smoke"):
            build_preemption_plan(
                non_smoke,
                preempt_attempt="kill",
                resume_attempt="resume",
                preempt_after_step=25,
            )


if __name__ == "__main__":
    unittest.main()
