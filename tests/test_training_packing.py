import unittest

from sunfish_tpu.training.packing import pack_training_record
from sunfish_tpu.training.record_format import TrainingRecord


def pack(record):
    return pack_training_record(
        record,
        record_id=7,
        prompt_length=4,
        canvas_size=4,
        num_canvases=2,
        vocab_size=100,
        pad_token=0,
        eos_token=1,
    )


class TrainingPackingTests(unittest.TestCase):
    def test_shapes_canvas_ids_and_eos_fill(self):
        example = pack(TrainingRecord(prompt=(2, 3), response=(4, 5, 6, 7, 8)))
        self.assertEqual(example.prompt, (2, 3, 0, 0))
        self.assertEqual(example.canvas, (4, 5, 6, 7, 8, 1, 1, 1))
        self.assertEqual(example.canvas_id, (0, 0, 0, 0, 1, 1, 1, 1))
        self.assertEqual(example.canvas_mask, (True,) * 8)
        self.assertEqual(example.canvas_loss_mask, (True,) * 8)

    def test_prefix_only_observation_has_no_encoder_supervision(self):
        example = pack(
            TrainingRecord(
                prompt=(2, 3, 4, 5),
                response=(6, 7),
                prompt_loss_mask=(False, False, False, False),
            )
        )
        # Prompt-to-prompt targets are all conditioning-only. The last prompt
        # token still predicts the first supervised canvas token.
        self.assertEqual(example.encoder_target_mask[:3], (False, False, False))
        self.assertTrue(example.encoder_target_mask[3])

    def test_fully_masked_bad_action_does_not_leak_through_eos(self):
        example = pack(
            TrainingRecord(
                prompt=(2, 3, 4, 5),
                response=(6, 7),
                prompt_loss_mask=(False,) * 4,
                response_loss_mask=(False, False),
            )
        )
        self.assertEqual(example.canvas_loss_mask, (False,) * 8)
        self.assertFalse(any(example.encoder_target_mask))

    def test_oversize_and_out_of_vocab_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "maximum"):
            pack(TrainingRecord(prompt=(1, 2, 3, 4, 5), response=(6,)))
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            pack(TrainingRecord(prompt=(2,), response=(101,)))


if __name__ == "__main__":
    unittest.main()
