"""
Simple Text Dataset for Micro SunFish
Uses a curated text corpus for coherent generation
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import urllib.request


SAMPLE_TEXT = """
The quick brown fox jumps over the lazy dog. This is a sample text for training a language model.

Once upon a time, in a small village, there lived a curious young student who loved to read books.
Every day, the student would visit the library and discover new stories about adventure and science.

The weather today is sunny and warm. People are walking in the park, enjoying the beautiful day.
Children are playing games, and birds are singing in the trees.

Science and technology have transformed our world. Computers help us solve complex problems.
The internet connects people from different countries and cultures.

Learning is a lifelong journey. Every day brings new opportunities to discover and grow.
Reading books expands our knowledge and imagination.

In the beginning, there was nothing but darkness. Then light emerged, and the universe was born.
Stars formed in the void, and planets circled around them.

The cat sat on the mat. The dog ran in the yard. The bird flew in the sky.
Animals live in different habitats around the world.

Food is essential for life. We eat breakfast in the morning, lunch at noon, and dinner in the evening.
Healthy eating includes fruits, vegetables, and grains.

Music brings joy to our lives. People sing songs and play instruments.
Different cultures have unique musical traditions.

The ocean is vast and deep. Fish swim in the water, and waves crash on the shore.
Marine life is diverse and fascinating.

Mountains rise high into the clouds. Rivers flow through valleys.
Nature is beautiful and powerful.

Cities are busy places with many people. Cars drive on streets, and buildings reach toward the sky.
Urban life is fast-paced and dynamic.

Friends are important in our lives. They support us and make us happy.
Friendship is a valuable gift.

Dreams can come true with hard work and determination.
Never give up on your goals.

The sun rises in the east and sets in the west. Day turns into night.
Time moves forward, always changing.

Education opens doors to new possibilities. Schools teach students important skills.
Teachers guide and inspire learning.

Art expresses human creativity. Paintings, sculptures, and photographs capture moments and emotions.
Every artwork tells a story.

Sports bring people together. Athletes train hard to compete and excel.
Teamwork and dedication lead to success.

History teaches us about the past. Ancient civilizations built great monuments.
We learn from those who came before us.

The future is full of possibilities. Technology will continue to advance.
Innovation drives progress.

Love is the strongest force in the universe. Families care for each other.
Compassion makes the world better.

Questions lead to answers. Curiosity drives discovery.
We explore to understand our world.

Peace is precious and worth protecting. Cooperation is better than conflict.
Together we can build a better future.

Every ending is a new beginning. Seasons change, and life goes on.
Hope springs eternal.
"""


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for micro training.

    Uses curated sample text repeated many times to provide
    enough training data for learning basic language patterns.
    """

    def __init__(
        self,
        block_size: int = 256,
        tokenizer_name: str = "gpt2",
        num_repeats: int = 1000,  # Repeat text to create more data
        vocab_size: int = 8192,  # Max vocab size for clipping
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize the sample text
        full_text = SAMPLE_TEXT * num_repeats  # Repeat for more training data

        tokens = self.tokenizer(
            full_text,
            truncation=False,
            add_special_tokens=False
        )['input_ids']

        # Clip tokens to vocab size (map high IDs to modulo vocab_size)
        tokens = [min(t, vocab_size - 1) for t in tokens]

        # Create fixed-length blocks
        self.blocks = []
        for i in range(0, len(tokens) - block_size, block_size // 2):  # 50% overlap
            block = tokens[i:i + block_size]
            if len(block) == block_size:
                self.blocks.append(block)

        print(f"📚 Created SimpleTextDataset:")
        print(f"   Total tokens: {len(tokens):,}")
        print(f"   Number of blocks: {len(self.blocks):,}")
        print(f"   Block size: {block_size}")

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return torch.tensor(self.blocks[idx], dtype=torch.long)


def test_simple_dataset():
    """Test the simple text dataset."""
    dataset = SimpleTextDataset(block_size=128, num_repeats=10)

    print(f"\n✅ Dataset created with {len(dataset)} samples")

    # Show first sample
    batch = dataset[0]
    print(f"\nFirst batch shape: {batch.shape}")

    # Decode to show text
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text = tokenizer.decode(batch[:50])
    print(f"\nFirst 50 tokens decoded:\n{text}")


if __name__ == "__main__":
    test_simple_dataset()
