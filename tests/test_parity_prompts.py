import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "parity_prompts.json"


class ParityPromptFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.prompts = cls.payload["prompts"]

    def test_counts_match_spec(self):
        self.assertEqual(len(self.prompts), 32)
        by_category = {}
        for prompt in self.prompts:
            by_category.setdefault(prompt["category"], []).append(prompt)
        self.assertEqual(
            sorted(by_category), ["code", "multilingual", "prose", "structured"]
        )
        for category, members in by_category.items():
            self.assertEqual(len(members), 8, category)
        long_count = sum(p["crosses_sliding_window"] for p in self.prompts)
        self.assertGreaterEqual(long_count, 8)  # spec floor; we ship 16

    def test_lengths_have_token_margin(self):
        # Conservative >=8 chars/token margin against the real ~3-4.
        for prompt in self.prompts:
            floor = 8_192 if prompt["crosses_sliding_window"] else 2_000
            self.assertGreaterEqual(len(prompt["text"]), floor, prompt["id"])

    def test_hashes_are_correct(self):
        for prompt in self.prompts:
            digest = hashlib.sha256(prompt["text"].encode("utf-8")).hexdigest()
            self.assertEqual(digest, prompt["sha256"], prompt["id"])

    def test_generator_is_deterministic_and_matches_committed_fixture(self):
        before = FIXTURE.read_bytes()
        try:
            subprocess.run(
                [sys.executable, str(FIXTURES / "generate_parity_prompts.py")],
                check=True,
                capture_output=True,
            )
            self.assertEqual(FIXTURE.read_bytes(), before)
        finally:
            FIXTURE.write_bytes(before)


if __name__ == "__main__":
    unittest.main()
