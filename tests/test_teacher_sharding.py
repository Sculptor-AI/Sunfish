import unittest

from sunfish_tpu.teacher_sharding import (
    resolve_teacher_mesh,
    teacher_partition_axis,
)


class TeacherShardingTests(unittest.TestCase):
    def test_requested_slice_keeps_expert_collective_within_host(self):
        self.assertEqual(
            resolve_teacher_mesh(
                global_device_count=32,
                process_count=8,
                local_device_count=4,
            ),
            ((8, 4), ("data", "expert")),
        )
        self.assertEqual(
            resolve_teacher_mesh(
                global_device_count=64,
                process_count=16,
                local_device_count=4,
            ),
            ((16, 4), ("data", "expert")),
        )

    def test_only_expert_banks_and_scale_shard(self):
        self.assertEqual(
            teacher_partition_axis(
                "layer_0.mlp.gating_einsum.w", (128, 2, 704, 2816)
            ),
            0,
        )
        self.assertEqual(
            teacher_partition_axis("layer_0.mlp.linear.w", (128, 704, 2816)),
            0,
        )
        self.assertEqual(
            teacher_partition_axis("layer_0.mlp.per_expert_scale", (128,)),
            0,
        )
        self.assertIsNone(
            teacher_partition_axis("layer_0.mlp.router_logits.w", (2816, 128))
        )
        self.assertIsNone(
            teacher_partition_axis("embedder.input_embedding", (262144, 2816))
        )

    def test_invalid_topology_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "must equal"):
            resolve_teacher_mesh(
                global_device_count=32,
                process_count=7,
                local_device_count=4,
            )
        with self.assertRaisesRegex(ValueError, "divide evenly"):
            resolve_teacher_mesh(
                global_device_count=6,
                process_count=2,
                local_device_count=3,
            )


if __name__ == "__main__":
    unittest.main()
