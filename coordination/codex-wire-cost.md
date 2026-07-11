# Codex-side bridge cost findings (2026-07-10)

## 1. Resume works, with an important caveat

Installed `codex-cli 0.144.1` supports:

```text
codex exec resume <SESSION_ID> <PROMPT>
```

Start the lane with `codex exec --json ...`; persist the `thread_id` from the
first `thread.started` event; later calls should use that exact ID, not
`--last` (other Codex activity can race it). Resume preserves conversation
memory, so a narrow prompt can say to use prior repo context and read only
unread wire messages. This should eliminate repeated broad repo orientation.

It does **not** make history token-free. Every model turn still has a growing
context, usually served largely from prompt cache. Rotate the lane after about
5-10 sends, after a major task switch, or when its last-turn input exceeds a
chosen threshold (suggest starting at 120k). A persistent app-server process
may save wall-clock startup but is not the main token lever.

## 2. Measured cost is tool-round-trip dominated

The two recent cold bridge runs recorded by Codex were:

| run | cumulative input | cached input | uncached input | output | tool calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 19:19 | 401,756 | 352,256 | 49,500 | 5,122 | 11 |
| 19:43 | 415,291 | 352,768 | 62,523 | 2,710 | 11 |

The reported 55-90k figures resemble individual late-turn inputs, not the
whole invocation. Each tool call causes another model turn carrying the full
prefix. Cached input was already 85-88%, which reduces monetary cost but does
not reduce TPM/rate-limit accounting. Cutting 11 tool calls to 1-3 turns is
plausibly a 60-85% reduction in cumulative input for narrow bridge asks; this
needs one A/B measurement before treating it as a guarantee.

## 3. Recommended harness changes, in order

1. **Persistent session per repo + agent lane.** Teach `agentwire send` to
   capture `thread_id` from JSONL and use `exec resume` on later sends. Store
   the ID under `.agentwire/`; rotate explicitly rather than using `--last`.
2. **Split `consult` from `work`.** Consult mode should be read-only, low
   reasoning, and instructed to answer from the wire plus supplied refs, with
   no repo scan unless an `ask:`/`do:` requires it. Work mode keeps the full
   coding profile. A/B a minimal Codex profile (`--ignore-user-config`, unused
   plugins/MCP/browser/image tools disabled): tool schemas and instructions are
   input tokens, so this may save far more than WIRE text.
3. **Host-side ref prefetch.** Resolve explicit `ref:` targets once in the
   harness and append a bounded snapshot/diff to stdin. One 5k-token bundle in
   one model turn is usually cheaper than five shell reads that each trigger a
   40k-90k model turn. Reject broad refs or cap attached bytes.
4. **Batch/debounce.** Queue nonurgent asks for 2-5 minutes and invoke once.
   Never direct-send `fyi:`/`ok:` alone; let the peer consume those on its next
   natural run.
5. **Add cost telemetry and budgets.** Parse `turn.completed.usage` from
   `--json`; log cumulative input, cached input, output, and tool-call count per
   send. Warn/stop consult mode after e.g. 2 tool calls. Optimize measured
   cumulative invocation cost, not the final-turn number.

## 4. OpenAI-tokenizer-specific answer

No encoding trick is likely to matter. Keep common lowercase English and
normal spaces: leading-space word merges are efficient, while squeezed words,
CamelCase, UUIDs, hashes, and exotic glyphs often split badly. The `|`, `>`,
`:`, and `;` delimiters may cost several tokens per line, but the whole WIRE
message is a rounding error beside tool schemas and repeated model turns.

For exact measurements, add an optional benchmark using the OpenAI Responses
input-token-count endpoint. OpenAI explicitly notes that local tokenizers miss
message formatting, tools, schemas, files, and model-specific behavior. Test
whole request shapes, not just the WIRE string.

Keep stable material byte-identical and before variable content. OpenAI prompt
caching requires exact prefix matches and begins at 1,024 tokens. GPT-5.6 cache
writes are billed differently from reads, and cached tokens still count toward
TPM, so log both cache reads and total input. The current dynamic turn marker is
small and occurs after the much larger fixed Codex prefix; moving it to the end
is theoretically cleaner but unlikely to move the bill materially.

## Bottom line

Ship persistent explicit-ID resume, consult/work lanes, ref prefetch, and usage
telemetry. Expected win is measured in hundreds of thousands of cumulative
input tokens per multi-tool cold invocation. Further WIRE glyph compression is
unlikely to save enough to measure.
