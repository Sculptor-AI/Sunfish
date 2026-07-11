import unittest

from sunfish_tpu.training.sharding_policy import (
    partition_axis_for_path,
    resolve_phase_b_mesh,
)


class TrainingShardingPolicyTests(unittest.TestCase):
    def test_expert_bank_is_forced_to_expert_axis(self):
        self.assertEqual(
            partition_axis_for_path(
                "layer_0.mlp.gating_einsum.w",
                (32, 2, 704, 2816),
                data_axis_size=32,
                itemsize=2,
            ),
            0,
        )
        self.assertEqual(
            partition_axis_for_path(
                "layer_0.mlp.per_expert_scale",
                (32,),
                data_axis_size=32,
                itemsize=4,
            ),
            0,
        )

    def test_shared_mlp_is_not_misclassified_as_expert(self):
        self.assertIsNone(
            partition_axis_for_path(
                "layer_0.mlp2.linear",
                (2112, 2816),
                data_axis_size=32,
                itemsize=2,
                min_size_bytes=1 << 30,
            )
        )

    def test_dense_rows_and_attention_fallback(self):
        self.assertEqual(
            partition_axis_for_path(
                "embedder.input_embedding",
                (262144, 2816),
                data_axis_size=32,
                itemsize=2,
            ),
            0,
        )
        self.assertEqual(
            partition_axis_for_path(
                "layer_0.attn.q",
                (16, 2816, 256),
                data_axis_size=32,
                itemsize=2,
            ),
            1,
        )

    def test_mesh_adapts_to_megacore_device_count(self):
        self.assertEqual(resolve_phase_b_mesh(global_device_count=32, num_experts=32), ((32,), ("data",), 32))
        self.assertEqual(
            resolve_phase_b_mesh(global_device_count=64, num_experts=32),
            ((2, 32), ("replica", "data"), 32),
        )
        self.assertEqual(resolve_phase_b_mesh(global_device_count=16, num_experts=32), ((16,), ("data",), 16))
        with self.assertRaises(ValueError):
            resolve_phase_b_mesh(global_device_count=24, num_experts=32)


if __name__ == "__main__":
    unittest.main()
