import unittest
from array import array

from sunfish_tpu.training.record_format import (
    TrainingRecord,
    decode_record,
    encode_record,
)


class TrainingRecordFormatTests(unittest.TestCase):
    def test_roundtrip_with_default_supervision(self):
        record = TrainingRecord(prompt=(2, 3, 4), response=(5, 6), bucket_id=7)
        decoded = decode_record(encode_record(record))
        self.assertEqual(decoded, record)
        self.assertEqual(decoded.normalized_prompt_mask(), (True, True, True))
        self.assertEqual(decoded.normalized_response_mask(), (True, True))

    def test_bitpacked_masks_cross_word_boundary(self):
        prompt = tuple(range(70))
        prompt_mask = tuple(index % 3 == 0 for index in range(70))
        response_mask = (False, True, False)
        record = TrainingRecord(
            prompt=prompt,
            response=(80, 81, 82),
            prompt_loss_mask=prompt_mask,
            response_loss_mask=response_mask,
        )
        self.assertEqual(decode_record(encode_record(record)), record)

    def test_truncation_and_trailing_words_are_rejected(self):
        payload = encode_record(TrainingRecord(prompt=(1,), response=(2,)))
        with self.assertRaisesRegex(ValueError, "length"):
            decode_record(payload[:-1])
        corrupted = array("I", payload)
        corrupted.append(123)
        with self.assertRaisesRegex(ValueError, "length"):
            decode_record(corrupted)

    def test_nonzero_mask_padding_bits_are_rejected(self):
        payload = encode_record(
            TrainingRecord(
                prompt=(1,),
                response=(2,),
                response_loss_mask=(True,),
            )
        )
        payload[-1] |= 1 << 31
        with self.assertRaisesRegex(ValueError, "padding bits"):
            decode_record(payload)

    def test_mask_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "length"):
            encode_record(
                TrainingRecord(
                    prompt=(1, 2),
                    response=(3,),
                    prompt_loss_mask=(True,),
                )
            )


if __name__ == "__main__":
    unittest.main()
