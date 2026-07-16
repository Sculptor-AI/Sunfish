import json
import tempfile
import unittest
from pathlib import Path

from sunfish.checkpoint_convert import load_selection_manifest
from sunfish_tpu.orbax_seed import (
    AUDITED_TARGET_TEXT_PARAMETERS_32E,
    PrunableLeaf,
    classify_prunable_path,
    require_parameter_count,
    audited_target_text_parameters,
    load_selection_snapshot,
    require_unchanged_source_inventory,
    validate_prunable_inventory,
)


class OrbaxSeedContractTests(unittest.TestCase):
    def test_exact_jax_moe_paths_and_axes(self):
        cases = {
            ("layer_7", "mlp", "gating_einsum", "w"): PrunableLeaf(
                7, "gating_einsum", 0
            ),
            ("layer_7", "mlp", "linear", "w"): PrunableLeaf(7, "linear", 0),
            ("layer_7", "mlp", "router_logits", "w"): PrunableLeaf(
                7, "router_logits", -1
            ),
            ("layer_7", "mlp", "per_expert_scale"): PrunableLeaf(
                7, "per_expert_scale", 0
            ),
        }
        for path, expected in cases.items():
            self.assertEqual(classify_prunable_path(path), expected)
        self.assertIsNone(
            classify_prunable_path(("layer_7", "mlp", "router_scale"))
        )
        self.assertIsNone(
            classify_prunable_path(("layer_7", "mlp2", "gating_einsum"))
        )

    def test_inventory_requires_all_four_leaves_in_all_layers(self):
        inventory = {
            layer: {
                "gating_einsum",
                "linear",
                "router_logits",
                "per_expert_scale",
            }
            for layer in range(30)
        }
        validate_prunable_inventory(inventory)
        inventory[3].remove("router_logits")
        with self.assertRaisesRegex(ValueError, "router_logits"):
            validate_prunable_inventory(inventory)

    def test_public_selection_loader_is_shared_with_safetensors_converter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "source_experts": 4,
                        "retained_experts": 2,
                        "layers": {"0": [0, 3], "1": [1, 2]},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_selection_manifest(
                    path, source_experts=4, retained_experts=2
                ),
                {0: (0, 3), 1: (1, 2)},
            )

    def test_audited_parameter_count_is_a_hard_gate(self):
        require_parameter_count(
            {"parameters": AUDITED_TARGET_TEXT_PARAMETERS_32E},
            expected=AUDITED_TARGET_TEXT_PARAMETERS_32E,
            label="target",
        )
        with self.assertRaisesRegex(RuntimeError, "audited contract"):
            require_parameter_count(
                {"parameters": AUDITED_TARGET_TEXT_PARAMETERS_32E - 1},
                expected=AUDITED_TARGET_TEXT_PARAMETERS_32E,
                label="target",
            )
        self.assertGreater(
            audited_target_text_parameters(48),
            AUDITED_TARGET_TEXT_PARAMETERS_32E,
        )

    def test_selection_snapshot_reads_bytes_exactly_once(self):
        first = json.dumps(
            {
                "schema_version": 1,
                "purpose": "stage-0.5-infrastructure-readiness-only",
                "promotion_allowed": False,
                "selection_method": "first-32",
                "source_experts": 128,
                "retained_experts": 32,
                "top_k_experts": 4,
                "layers": {
                    str(layer): list(range(32)) for layer in range(30)
                },
            }
        ).encode()
        changed = first.replace(b'"top_k_experts": 4', b'"top_k_experts": 8')

        class MutatingPath:
            calls = 0

            def read_bytes(self):
                self.calls += 1
                return first if self.calls == 1 else changed

            def __str__(self):
                return "mutating-selection.json"

        path = MutatingPath()
        snapshot = load_selection_snapshot(
            path,  # type: ignore[arg-type]
            source_experts=128,
            retained_experts=32,
            top_k_experts=4,
        )
        self.assertEqual(path.calls, 1)
        self.assertEqual(snapshot.layers[0], tuple(range(32)))

    def test_source_inventory_must_be_unchanged_across_load(self):
        inventory = {"sha256": "a" * 64, "objects": [{"generation": 1}]}
        require_unchanged_source_inventory(inventory, dict(inventory))
        changed = {"sha256": "b" * 64, "objects": [{"generation": 2}]}
        with self.assertRaisesRegex(RuntimeError, "changed while loading"):
            require_unchanged_source_inventory(inventory, changed)


if __name__ == "__main__":
    unittest.main()
