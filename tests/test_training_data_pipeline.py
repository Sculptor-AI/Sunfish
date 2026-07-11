import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    from sunfish_tpu.training.data import TrainingRecordSource
except ImportError as error:
    DATA_IMPORT_ERROR = error
else:
    DATA_IMPORT_ERROR = None

from sunfish.datashards import ShardWriter, write_manifest
from sunfish_tpu.training.record_format import TrainingRecord, encode_record


@unittest.skipIf(DATA_IMPORT_ERROR is not None, f"training data stack unavailable: {DATA_IMPORT_ERROR}")
class TrainingDataPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        writer = ShardWriter(self.directory / "records-00000")
        writer.add(
            encode_record(
                TrainingRecord(
                    prompt=(2, 3, 4, 5),
                    response=(6, 7),
                    prompt_loss_mask=(False,) * 4,
                )
            )
        )
        first_shard = writer.close()
        second_writer = ShardWriter(self.directory / "records-00001")
        second_writer.add(
            encode_record(TrainingRecord(prompt=(8,), response=(9, 10)))
        )
        self.manifest_hash = write_manifest(
            self.directory,
            [first_shard, second_writer.close()],
            source="training-pipeline-test",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_source(self, manifest_hash=None):
        return TrainingRecordSource(
            directory=str(self.directory),
            expected_manifest_sha256=manifest_hash or self.manifest_hash,
            verify_shard_hashes=True,
            prompt_length=4,
            canvas_size=4,
            num_canvases=1,
            vocab_size=100,
            pad_token=0,
            eos_token=1,
        )

    def test_source_emits_exact_model_contract(self):
        example = self.make_source()[0]
        self.assertEqual(
            set(example),
            {
                "prompt",
                "canvas",
                "canvas_id",
                "canvas_mask",
                "canvas_loss_mask",
                "encoder_target",
                "encoder_target_mask",
                "bucket_id",
                "record_id",
            },
        )
        self.assertEqual(example["canvas"].shape, (4, 1))
        self.assertEqual(example["prompt"].dtype, np.int32)
        self.assertTrue(example["encoder_target_mask"][-1] == np.bool_(False))

    def test_manifest_identity_is_checked_before_first_read(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.make_source("f" * 64)

    def test_compact_index_crosses_shard_boundary(self):
        source = self.make_source()
        self.assertEqual(len(source), 2)
        self.assertEqual(int(source[1]["record_id"]), 1)
        self.assertEqual(source[1]["canvas"][:2, 0].tolist(), [9, 10])


if __name__ == "__main__":
    unittest.main()
