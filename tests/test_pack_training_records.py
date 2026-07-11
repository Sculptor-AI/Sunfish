import json
import tempfile
import unittest
from pathlib import Path

from sunfish.datashards import ShardedRecordSource
from sunfish_tpu.training.pack_records import pack_jsonl
from sunfish_tpu.training.record_format import decode_record


class PackTrainingRecordsTests(unittest.TestCase):
    def test_jsonl_to_immutable_shards_and_masks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "records.jsonl"
            records = [
                {"prompt": [2, 3], "response": [4, 5], "bucket_id": 9},
                {
                    "prompt": [6],
                    "response": [7, 8, 9],
                    "prompt_loss_mask": [False],
                    "response_loss_mask": [True, False, True],
                },
                {"prompt": [10], "response": [11]},
            ]
            input_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            output = root / "packed"
            report = pack_jsonl(
                input_path, output, records_per_shard=2, source="unit-test"
            )
            self.assertEqual(report["records"], 3)
            self.assertEqual(report["shards"], 2)
            source = ShardedRecordSource(output)
            decoded = [decode_record(source[index]) for index in range(3)]
            self.assertEqual(decoded[0].bucket_id, 9)
            self.assertEqual(decoded[1].prompt_loss_mask, (False,))
            self.assertEqual(decoded[1].response_loss_mask, (True, False, True))

    def test_unknown_json_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "records.jsonl"
            input_path.write_text(
                json.dumps({"prompt": [1], "response": [2], "typo": True}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                pack_jsonl(input_path, root / "packed", records_per_shard=2, source="test")


if __name__ == "__main__":
    unittest.main()
