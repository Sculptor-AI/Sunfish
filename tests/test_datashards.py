import json
import tempfile
import unittest
from array import array
from pathlib import Path

from sunfish.datashards import (
    ResumableSampler,
    ShardedRecordSource,
    ShardWriter,
    write_manifest,
)


def build_corpus(directory: Path, shards: int = 3, records_per_shard: int = 20):
    infos = []
    value = 0
    for s in range(shards):
        writer = ShardWriter(directory / f"shard{s}")
        for _ in range(records_per_shard):
            writer.add([value] * (value % 7 + 1))  # varied lengths
            value += 1
        infos.append(writer.close())
    write_manifest(directory, infos, source="unit-test")
    return value  # total records


class ShardFormatTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.total = build_corpus(self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    def test_random_access_roundtrip(self):
        source = ShardedRecordSource(self.dir)
        self.assertEqual(len(source), self.total)
        for i in (0, 7, 19, 20, 45, self.total - 1):
            record = source[i]
            self.assertEqual(list(record), [i] * (i % 7 + 1))

    def test_hash_verification_catches_tampering(self):
        bin_path = self.dir / "shard1.bin"
        data = bytearray(bin_path.read_bytes())
        data[0] ^= 0xFF
        bin_path.write_bytes(bytes(data))
        with self.assertRaises(ValueError):
            ShardedRecordSource(self.dir)
        ShardedRecordSource(self.dir, verify=False)  # opt-out still works

    def test_writer_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            ShardWriter(self.dir / "shard0")


class SamplerTests(unittest.TestCase):
    def make(self, index, count, seed=7, records=60):
        return ResumableSampler(
            records, seed=seed, process_index=index, process_count=count,
            manifest_hash="mh",
        )

    def test_processes_are_disjoint_and_exhaustive_per_epoch(self):
        count = 4
        records = 60
        seen = []
        for p in range(count):
            sampler = self.make(p, count, records=records)
            seen.append([next(sampler) for _ in range(records // count)])
        flat = [r for chunk in seen for r in chunk]
        self.assertEqual(sorted(flat), list(range(records)))  # no dup, no gap

    def test_epochs_reshuffle_deterministically(self):
        a = self.make(0, 1, records=10)
        first = [next(a) for _ in range(10)]
        second = [next(a) for _ in range(10)]
        self.assertEqual(sorted(first), sorted(second))
        self.assertNotEqual(first, second)  # different epoch order
        b = self.make(0, 1, records=10)
        self.assertEqual([next(b) for _ in range(20)], first + second)  # deterministic

    def test_exact_resume_mid_epoch(self):
        a = self.make(1, 3)
        for _ in range(13):
            next(a)
        state = a.get_state()
        tail = [next(a) for _ in range(12)]

        b = self.make(1, 3)
        b.set_state(state)
        self.assertEqual([next(b) for _ in range(12)], tail)

    def test_state_refuses_wrong_dataset_or_topology(self):
        a = self.make(0, 2)
        state = a.get_state()
        wrong_data = dict(state, manifest_hash="other")
        with self.assertRaises(ValueError):
            self.make(0, 2).set_state(wrong_data)
        with self.assertRaises(ValueError):
            self.make(1, 2).set_state(state)

    def test_source_plus_sampler_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            total = build_corpus(directory)
            source = ShardedRecordSource(directory)
            sampler = ResumableSampler(
                len(source), seed=3, process_index=0, process_count=2,
                manifest_hash=source.manifest_hash,
            )
            for _ in range(10):
                record = source[next(sampler)]
                self.assertIsInstance(record, array)
                self.assertGreater(len(record), 0)


if __name__ == "__main__":
    unittest.main()
