import json
import struct
import tempfile
import unittest
from pathlib import Path

from sunfish.checkpoint_audit import read_safetensors_header
from sunfish.checkpoint_convert import build_plan, convert, plan_report


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        data.extend(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * ((-len(encoded)) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        handle.write(data)


def tensor_bytes(path: Path, name: str) -> bytes:
    with path.open("rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length))
        start, end = header[name]["data_offsets"]
        handle.seek(8 + header_length + start)
        return handle.read(end - start)


def make_checkpoint(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    config = {
        "model_type": "diffusion_gemma",
        "vision_config": {"hidden_size": 2},
        "text_config": {
            "num_experts": 4,
            "top_k_experts": 2,
            "num_hidden_layers": 2,
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "tokenizer.json").write_text("tokenizer")

    shard_1 = {
        "model.decoder.layers.0.experts.down_proj": ("U8", [4, 1, 2], bytes(range(8))),
        "model.decoder.layers.0.experts.gate_up_proj": (
            "U8",
            [4, 2, 2],
            bytes(range(16)),
        ),
        "model.decoder.layers.0.router.per_expert_scale": (
            "U8",
            [4],
            bytes([10, 11, 12, 13]),
        ),
        "model.decoder.layers.0.router.proj.weight": (
            "U8",
            [4, 2],
            bytes(range(20, 28)),
        ),
        "model.decoder.layers.0.router.scale": ("U8", [2], bytes([30, 31])),
        "model.encoder.vision_tower.patch": ("U8", [3], bytes([40, 41, 42])),
    }
    shard_2 = {
        "model.decoder.layers.1.experts.down_proj": ("U8", [4, 1, 2], bytes(range(50, 58))),
        "model.decoder.layers.1.experts.gate_up_proj": (
            "U8",
            [4, 2, 2],
            bytes(range(60, 76)),
        ),
        "model.decoder.layers.1.router.per_expert_scale": (
            "U8",
            [4],
            bytes([80, 81, 82, 83]),
        ),
        "model.decoder.layers.1.router.proj.weight": (
            "U8",
            [4, 2],
            bytes(range(90, 98)),
        ),
        "model.decoder.layers.1.router.scale": ("U8", [2], bytes([100, 101])),
        "model.encoder.embed_vision.embedding_projection.weight": (
            "U8",
            [2],
            bytes([110, 111]),
        ),
        "model.embed_tokens.weight": ("U8", [2, 2], bytes([120, 121, 122, 123])),
    }
    shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    write_safetensors(source / shard_names[0], shard_1)
    write_safetensors(source / shard_names[1], shard_2)
    weight_map = {
        name: shard_names[index]
        for index, shard in enumerate((shard_1, shard_2))
        for name in shard
    }
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 999, "total_parameters": 999},
                "weight_map": weight_map,
            }
        )
    )
    return source


def add_vision_only_middle_shard(source: Path) -> tuple[str, str, str]:
    """Turn the two-shard fixture into a gapped three-shard text conversion."""
    old_first = "model-00001-of-00002.safetensors"
    old_second = "model-00002-of-00002.safetensors"
    first = "model-00001-of-00003.safetensors"
    middle = "model-00002-of-00003.safetensors"
    last = "model-00003-of-00003.safetensors"
    (source / old_first).rename(source / first)
    (source / old_second).rename(source / last)

    vision_name = "model.encoder.vision_tower.middle_only"
    write_safetensors(
        source / middle,
        {vision_name: ("U8", [2], bytes([200, 201]))},
    )

    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index["weight_map"] = {
        name: first if shard == old_first else last
        for name, shard in index["weight_map"].items()
    }
    index["weight_map"][vision_name] = middle
    index_path.write_text(json.dumps(index))
    return first, middle, last


class ConversionTests(unittest.TestCase):
    def test_text_only_pruning_slices_axis_zero_and_rebuilds_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_checkpoint(root)
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "source_experts": 4,
                        "retained_experts": 2,
                        "layers": {"0": [1, 3], "1": [0, 2]},
                    }
                )
            )
            plan = build_plan(
                source,
                retained_experts=2,
                top_k=1,
                selection_path=selection,
            )
            output = root / "output"
            manifest = convert(plan, output, chunk_mib=1)

            config = json.loads((output / "config.json").read_text())
            self.assertIsNone(config["vision_config"])
            self.assertEqual(config["text_config"]["num_experts"], 2)
            self.assertEqual(config["text_config"]["top_k_experts"], 1)
            self.assertEqual((output / "tokenizer.json").read_text(), "tokenizer")

            first = output / "model-00001-of-00002.safetensors"
            self.assertEqual(
                tensor_bytes(first, "model.decoder.layers.0.experts.down_proj"),
                bytes([2, 3, 6, 7]),
            )
            converted_header = read_safetensors_header(first)
            self.assertEqual(
                converted_header["model.decoder.layers.0.experts.gate_up_proj"].shape,
                (2, 2, 2),
            )
            self.assertEqual(
                tensor_bytes(first, "model.decoder.layers.0.router.per_expert_scale"),
                bytes([11, 13]),
            )
            self.assertEqual(
                tensor_bytes(first, "model.decoder.layers.0.router.scale"),
                bytes([30, 31]),
            )

            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertNotIn("model.encoder.vision_tower.patch", index["weight_map"])
            self.assertNotIn(
                "model.encoder.embed_vision.embedding_projection.weight",
                index["weight_map"],
            )
            self.assertEqual(index["metadata"]["total_size"], plan.output_bytes)
            self.assertEqual(index["metadata"]["total_parameters"], plan.output_parameters)
            self.assertEqual(manifest["dropped_tensors"], 2)

    def test_no_prune_control_copies_language_tensor_bytes_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_checkpoint(root)
            plan = build_plan(source, retained_experts=4, top_k=2)
            output = root / "control"
            convert(plan, output)

            name = "model.decoder.layers.1.experts.gate_up_proj"
            source_shard = source / "model-00002-of-00002.safetensors"
            output_shard = output / "model-00002-of-00002.safetensors"
            self.assertEqual(tensor_bytes(source_shard, name), tensor_bytes(output_shard, name))
            self.assertEqual(plan.selection[0], (0, 1, 2, 3))

    def test_vision_only_source_shard_can_leave_gapped_output_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_checkpoint(root)
            first, empty_middle, last = add_vision_only_middle_shard(source)

            plan = build_plan(source, retained_experts=4, top_k=2)
            output = root / "control"
            convert(plan, output)

            self.assertTrue((output / first).is_file())
            self.assertFalse((output / empty_middle).exists())
            self.assertTrue((output / last).is_file())
            index = json.loads((output / "model.safetensors.index.json").read_text())
            referenced_shards = set(index["weight_map"].values())
            self.assertEqual(referenced_shards, {first, last})
            self.assertTrue(all((output / shard).is_file() for shard in referenced_shards))

    def test_pruning_requires_complete_valid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_checkpoint(root)
            with self.assertRaises(ValueError):
                build_plan(source, retained_experts=2, top_k=1)

            selection = root / "bad.json"
            selection.write_text(json.dumps({"layers": {"0": [1, 1], "1": [0, 2]}}))
            with self.assertRaises(ValueError):
                build_plan(
                    source,
                    retained_experts=2,
                    top_k=1,
                    selection_path=selection,
                )

    def test_dry_plan_has_no_output_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_checkpoint(root)
            output = root / "unused"
            report = plan_report(build_plan(source, retained_experts=4, top_k=2))
            self.assertFalse(output.exists())
            self.assertEqual(report["layers"], 2)
            self.assertEqual(report["dropped_tensors"], 2)

    def test_refuses_to_overwrite_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_checkpoint(root)
            plan = build_plan(source, retained_experts=4, top_k=2)
            output = root / "existing"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                convert(plan, output)


if __name__ == "__main__":
    unittest.main()
