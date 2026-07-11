import unittest

from sunfish_tpu.tpu_preflight import Check, report, validate_gcs_uri


class GcsUriTests(unittest.TestCase):
    def test_valid_project_prefix(self):
        self.assertEqual(
            validate_gcs_uri("gs://sunfish-checkpoints/training/stage-0"),
            ("sunfish-checkpoints", "training/stage-0"),
        )

    def test_bucket_root_and_non_gcs_are_rejected(self):
        for value in ("gs://bucket", "https://bucket/path", "gs:///path"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_gcs_uri(value)


class ReportTests(unittest.TestCase):
    def test_failure_controls_readiness(self):
        payload = report(
            [
                Check("one", "pass", "ok"),
                Check("two", "warn", "check this"),
                Check("three", "fail", "broken"),
            ]
        )
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["summary"], {"pass": 1, "warn": 1, "fail": 1})

    def test_warnings_do_not_block_local_inspection(self):
        payload = report([Check("cpu", "warn", "not a TPU")])
        self.assertTrue(payload["ready"])


if __name__ == "__main__":
    unittest.main()
