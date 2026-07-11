"""Deterministic generator for the logit-parity prompt fixture.

Produces tests/fixtures/parity_prompts.json per docs/parity_harness.md P2:
32 prompts, 8 per category (code / prose / multilingual / structured), half
of each category long enough to safely cross the 1024-token sliding-window
boundary. Lengths are targeted in characters with a conservative margin
(>=8 chars/token assumed against the real ~3-4), because the gated upstream
tokenizer is not required to build the fixture; the harness records actual
token counts at run time.

No randomness: prompts are composed by cycling fixed seed passages with
numbered section headers until the target size is reached, so regeneration
is byte-identical (verified by tests/test_parity_prompts.py).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SHORT_CHARS = 2_000   # comfortably >256 tokens
LONG_CHARS = 9_000    # comfortably >1024 tokens

CODE_SEEDS = [
    (
        "python",
        "def entropy_bound_mask(entropies, bound):\n"
        "    accepted = [False] * len(entropies)\n"
        "    cumulative = 0.0\n"
        "    for index in sorted(range(len(entropies)), key=entropies.__getitem__):\n"
        "        cumulative += entropies[index]\n"
        "        if cumulative - entropies[index] <= bound:\n"
        "            accepted[index] = True\n"
        "    return tuple(accepted)\n",
    ),
    (
        "typescript",
        "export async function retryWithBackoff<T>(fn: () => Promise<T>, retries = 5): Promise<T> {\n"
        "  let delay = 100;\n"
        "  for (let attempt = 0; ; attempt++) {\n"
        "    try { return await fn(); } catch (error) {\n"
        "      if (attempt >= retries) throw error;\n"
        "      await new Promise(r => setTimeout(r, delay));\n"
        "      delay = Math.min(delay * 2, 5000);\n"
        "    }\n  }\n}\n",
    ),
    (
        "rust",
        "fn softmax(logits: &[f32]) -> Vec<f32> {\n"
        "    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);\n"
        "    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();\n"
        "    let sum: f32 = exps.iter().sum();\n"
        "    exps.iter().map(|x| x / sum).collect()\n}\n",
    ),
    (
        "shell",
        "for shard in model-*.safetensors; do\n"
        "  size=$(stat -f%z \"$shard\" 2>/dev/null || stat -c%s \"$shard\")\n"
        "  echo \"$shard: $((size / 1048576)) MiB\"\n"
        "done | sort -t: -k2 -n\n",
    ),
]

PROSE_SEEDS = [
    "The lighthouse keeper logged the weather twice a day for forty years, and "
    "in all that time the entries never once mentioned the sea itself, only the "
    "sky above it, as if the water were too obvious to record.",
    "Early compilers were written by people who had to hold the whole machine "
    "in their heads at once; the discipline this demanded shaped programming "
    "culture for a generation after the constraint itself had disappeared.",
    "A recipe is a program executed by a cook, and like all programs it embeds "
    "assumptions about its runtime: the strength of the flour, the humidity of "
    "the kitchen, the impatience of the person stirring.",
    "The train timetable is a work of speculative fiction revised four times a "
    "year, describing a country in which everything happens on time.",
]

MULTILINGUAL_SEEDS = [
    ("spanish", "La biblioteca estaba abierta toda la noche, y los estudiantes "
     "llegaban con sus cuadernos como si fueran mapas de territorios aún no descubiertos."),
    ("french", "Le boulanger connaissait ses clients par leurs commandes plutôt "
     "que par leurs noms, et cette taxonomie lui suffisait parfaitement."),
    ("german", "Die Werkstatt roch nach Öl und Metall, und jede Maschine hatte "
     "einen eigenen Rhythmus, den der Meister im Schlaf erkennen konnte."),
    ("chinese", "图书馆的旧书架上摆满了没有人再借阅的书，但管理员每天仍然为它们除尘，仿佛在等待某位迟到多年的读者。"),
    ("japanese", "駅前の小さな喫茶店は五十年間同じ豆を使い続けており、常連客はその味の変化で季節を知ると言われている。"),
    ("russian", "Старый картограф рисовал города, в которых никогда не бывал, "
     "и путешественники уверяли, что его карты точнее всех остальных."),
    ("arabic", "كان صانع الساعات يصلح الوقت نفسه كما كان يقول لزبائنه، وكانت دكانه الصغيرة مليئة بأصوات لا تتفق أبدا."),
    ("hindi", "पुराने पुल के नीचे हर शाम मछुआरे अपनी नावें बांधते थे और नदी की कहानियां एक दूसरे को सुनाते थे।"),
]

STRUCTURED_SEEDS = [
    (
        "json",
        '{"tool": "run_tests", "arguments": {"path": "tests/", "verbose": true, '
        '"markers": ["not slow"], "timeout_seconds": 300}, "call_id": "call_0042"}',
    ),
    (
        "diff",
        "--- a/src/router.py\n+++ b/src/router.py\n@@ -14,7 +14,9 @@\n"
        "-    probs = softmax(logits)\n"
        "+    scaled = hidden * self.scale\n"
        "+    probs = softmax(scaled @ self.proj.T)\n"
        "     topk = probs.topk(self.k)\n",
    ),
    (
        "markdown",
        "## Deployment checklist\n\n- [x] budget alerts created\n- [x] bucket "
        "soft-delete disabled\n- [ ] lifecycle rules verified via describe\n"
        "- [ ] scratch disk attached and mounted\n",
    ),
    (
        "yaml",
        "training:\n  provider: trc-tpu\n  checkpoint:\n    interval_minutes: 30\n"
        "    offsite: true\n  distillation:\n    mode: offline-full\n    layers: [7, 14, 21, 28]\n",
    ),
]


def _compose(seeds: list[tuple[str, str]] | list[str], target_chars: int, label: str) -> str:
    parts: list[str] = []
    total = 0
    index = 0
    while total < target_chars:
        seed = seeds[index % len(seeds)]
        name, body = seed if isinstance(seed, tuple) else (label, seed)
        section = f"### {label} section {index + 1} ({name})\n{body}\n"
        parts.append(section)
        total += len(section)
        index += 1
    return "".join(parts)


def build_prompts() -> list[dict[str, object]]:
    categories = [
        ("code", CODE_SEEDS),
        ("prose", PROSE_SEEDS),
        ("multilingual", MULTILINGUAL_SEEDS),
        ("structured", STRUCTURED_SEEDS),
    ]
    prompts: list[dict[str, object]] = []
    for category, seeds in categories:
        for i in range(8):
            long = i >= 4  # 4 short + 4 long per category => 16 prompts >1024 tokens
            text = _compose(seeds, LONG_CHARS if long else SHORT_CHARS, category)
            prompts.append(
                {
                    "id": f"{category}-{i}",
                    "category": category,
                    "crosses_sliding_window": long,
                    "chars": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "text": text,
                }
            )
    return prompts


def main() -> None:
    output = Path(__file__).parent / "parity_prompts.json"
    payload = {
        "spec": "docs/parity_harness.md P2",
        "generator": "tests/fixtures/generate_parity_prompts.py (deterministic)",
        "prompts": build_prompts(),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    long_count = sum(p["crosses_sliding_window"] for p in payload["prompts"])
    print(f"wrote {output.name}: {len(payload['prompts'])} prompts, {long_count} long")


if __name__ == "__main__":
    main()
