import json
import struct
import tempfile
import unittest
from pathlib import Path

from sunfish.checkpoint_audit import (
    audit,
    classify,
    read_checkpoint,
    read_safetensors_header,
)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> None:
    """Write a valid safetensors file with zero-filled data."""
    header: dict[str, dict[str, object]] = {}
    offset = 0
    bits = {"BF16": 16, "F32": 32, "F16": 16, "U8": 8}
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * bits[dtype] // 8
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    payload = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        handle.write(b"\x00" * offset)


class HeaderTests(unittest.TestCase):
    def test_roundtrip_single_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "model-00001-of-00002.safetensors"
            write_safetensors(
                shard,
                {
                    "model.layers.0.experts.gate_proj": ("BF16", [128, 704, 2816]),
                    "model.layers.0.router.weight": ("F32", [128, 2816]),
                },
            )
            tensors = read_safetensors_header(shard)
            self.assertEqual(len(tensors), 2)
            info = tensors["model.layers.0.experts.gate_proj"]
            self.assertEqual(info.shape, (128, 704, 2816))
            self.assertEqual(info.numel, 128 * 704 * 2816)
            self.assertEqual(info.bits, 16 * info.numel)

    def test_multiple_shards_merge_and_duplicates_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_safetensors(directory / "a.safetensors", {"x": ("BF16", [2, 2])})
            write_safetensors(directory / "b.safetensors", {"y": ("BF16", [3])})
            self.assertEqual(set(read_checkpoint(directory)), {"x", "y"})

            write_safetensors(directory / "c.safetensors", {"x": ("BF16", [2, 2])})
            with self.assertRaises(ValueError):
                read_checkpoint(directory)

    def test_rejects_non_safetensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "bogus.safetensors"
            bogus.write_bytes(b"\x00" * 4)
            with self.assertRaises(ValueError):
                read_safetensors_header(bogus)

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_checkpoint(Path(tmp))


class AuditTests(unittest.TestCase):
    def test_grouping_and_parameter_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            write_safetensors(
                directory / "model.safetensors",
                {
                    "model.layers.0.experts.down_proj": ("BF16", [4, 8, 16]),
                    "model.layers.0.router.weight": ("F32", [4, 16]),
                    "vision_tower.patch_embed": ("BF16", [10, 10]),
                    "model.embed_tokens.weight": ("BF16", [100, 16]),
                    "model.layers.0.self_attn.q_proj": ("BF16", [16, 16]),
                },
            )
            report = audit(read_checkpoint(directory))
            groups = report["groups"]
            self.assertEqual(groups["routed_experts"]["parameters"], 4 * 8 * 16)
            self.assertEqual(groups["router"]["parameters"], 4 * 16)
            self.assertEqual(groups["vision"]["parameters"], 100)
            self.assertEqual(groups["embedding_or_head"]["parameters"], 1600)
            self.assertEqual(groups["other"]["parameters"], 256)
            self.assertEqual(
                report["total_parameters"],
                4 * 8 * 16 + 4 * 16 + 100 + 1600 + 256,
            )

    def test_custom_rules_override_defaults(self):
        rules = (("everything", r".*"),)
        self.assertEqual(classify("model.layers.0.experts.w1", rules), "everything")

    def test_classify_falls_back_to_other(self):
        self.assertEqual(classify("model.norm.weight", (("v", r"vision"),)), "other")


if __name__ == "__main__":
    unittest.main()
