import unittest

from sunfish.calibration_records import (
    calibration_bucket_token_caps,
    calibration_windows,
)


class CalibrationRecordTests(unittest.TestCase):
    def test_full_budget_uses_canonical_workload_shares_exactly(self):
        caps = calibration_bucket_token_caps(75_000_000)
        self.assertEqual(sum(caps.values()), 75_000_000)
        self.assertEqual(caps["code_completion"], 26_250_000)
        self.assertEqual(caps["reasoning_control"], 3_750_000)

    def test_long_documents_are_split_and_cap_is_never_exceeded(self):
        records = list(
            calibration_windows(
                list(range(20)),
                record_tokens=6,
                min_record_tokens=2,
                remaining_tokens=14,
            )
        )
        self.assertEqual([len(record) for record in records], [6, 6, 2])
        self.assertEqual(sum(records, []), list(range(14)))

    def test_short_tail_and_short_remaining_budget_are_skipped(self):
        self.assertEqual(
            list(
                calibration_windows(
                    [1, 2, 3],
                    record_tokens=6,
                    min_record_tokens=4,
                    remaining_tokens=10,
                )
            ),
            [],
        )
        self.assertEqual(
            list(
                calibration_windows(
                    list(range(10)),
                    record_tokens=6,
                    min_record_tokens=4,
                    remaining_tokens=3,
                )
            ),
            [[0, 1, 2]],
        )

    def test_windowing_never_strands_a_one_token_cap_tail(self):
        records = list(
            calibration_windows(
                list(range(20)),
                record_tokens=6,
                min_record_tokens=4,
                remaining_tokens=7,
            )
        )
        self.assertEqual([len(record) for record in records], [5, 2])
        self.assertEqual(sum(len(record) for record in records), 7)

    def test_invalid_contract_fails_closed(self):
        with self.assertRaises(ValueError):
            list(
                calibration_windows(
                    [1, 2],
                    record_tokens=1,
                    min_record_tokens=1,
                    remaining_tokens=2,
                )
            )


if __name__ == "__main__":
    unittest.main()
