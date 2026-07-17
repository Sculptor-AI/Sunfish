# Verify-lane confirmed findings (C, 2026-07-16, against bd85a9c)

Nine findings from the 8-reviewer fan-out were each adversarially verified by
three independent skeptics against the committed tree. Seven were refuted 3-0
(already fixed by the repair lane or wrong). Two were confirmed 3-0.

## MAJOR — source-tree digest is not umask-portable (deploy-day hard block)

`src/sunfish/source_tree.py:150-154` folds `st_mode & 0o777` (all nine
permission bits) into the source-tree SHA-256. Git tracks only the executable
bit; the group/other bits come from each host's umask at checkout time.

Concrete failure: the bundle MUST be built on a separate Linux host, and stock
Ubuntu/GCP images default to umask 002 (user-private groups) → builder
checkout is 0o664/0o775. The macOS controller (umask 022) computes
`--expected-tree` from its own 0o644/0o755 checkout
(`deploy_tpu_offline_bundle.sh:41`, `launch_tpu_pod.sh:105`). The digests
differ, `offline_bundle.py:555` ("offline source tree differs from
controller") fails on EVERY worker, and the deploy hard-blocks — discovered
only after the full bundle build and upload cycle. Fail-closed, so no safety
issue, but it can burn the deployment window.

Empirically reproduced by a verifier: identical one-file tree digests
differently at 0o644 vs 0o664.

Fix options (either closes it):
1. Mask the digested mode to the git-tracked bit only
   (`mode & 0o100` presence → canonical 0o755/0o644 encoding), or drop mode
   from the digest and keep executable-bit verification as a separate check.
2. Keep full-bit binding but enforce `umask 022` in
   `build_tpu_offline_bundle.sh` AND document the controller requirement —
   weaker, still breaks on pre-existing checkouts.

Also add the missing round-trip test: pack → extract → verify_bundle
(tests/test_offline_bundle.py never exercises extraction).

Note: one verifier found the tar `filter="data"` mode-rewrite leg of the
original claim stale in the current tree; the umask leg is what all three
confirmed.

## MINOR — host-log-relay ready-file race (spurious launch failure)

`scripts/host_log_relay.py:66-71` `_create_ready()` creates the ready file
with `O_CREAT|O_EXCL` and writes the PID in a SEPARATE `os.write()` — the
file is transiently visible empty. `scripts/tpu_host_entrypoint.sh:314-323`
polls `-f`, and the FIRST time the file exists does one non-retried `read`;
an empty read leaves the PID unset and the script hits the unconditional
`echo "host log relay ready file is invalid"; exit 2` — no retry path once
the file exists. A healthy relay startup can be misclassified as tampering,
hard-failing the worker launch and consuming one attempt from the bounded
durable-controller budget (automatic retry is forbidden by design).

Fix (either): relay writes PID to a temp path then `os.rename()`s onto the
ready path (atomic, content visible when the entry appears), or the
entrypoint `continue`s the poll loop on an empty/short read instead of
exiting on first observation. No test covers this path today.

## Refuted 3-0 (for the record)

- real_resume_smoke back-to-back initialize settle concern
- preemption resume-phase missing cleanup (covered in final tree)
- interrupt script missing Grain child kill (covered/detect-and-fail is by design)
- reconstruction_gate undefined names (mid-refactor artifact, fixed)
- offline_bundle glibc gate reading wrong glibc (wrong)
- training_harness.md stale config path (fixed in bd85a9c)
- source-tree ignore-list divergence from git --exclude-standard (wrong)
