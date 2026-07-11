# The bridge — direct two-way agent invocation

Symmetric, synchronous communication between **Claude (Fable 5)** and
**Codex (5.6 Sol)**. Either agent can invoke the other and get a reply within
its own working turn. The bridge is for *conversation and coordination*;
`coordination/channel.md` remains the durable record, and the AGENTS.md
decision log remains the contract.

## How to invoke the other agent

Preferred (uniform envelope, turn caps, logging):

```bash
coordination/bridge.sh to-codex  "message"      # Claude -> Codex
coordination/bridge.sh to-claude "message"      # Codex  -> Claude
```

Raw equivalents (what the wrapper runs):

```bash
# Claude -> Codex
codex exec --cd <repo> --sandbox workspace-write "message"

# Codex -> Claude  (run from the repo root)
claude -p "message"                 # fresh session
claude -c -p "message"              # continue Claude's most recent session
```

Native tool registration (one-time, done by Chase — makes the other agent a
first-class tool instead of a shell call):

```bash
claude mcp add codex -- codex mcp-server     # Codex becomes a Claude tool
codex mcp add claude -- claude mcp serve     # Claude becomes a Codex tool
```

## Rules of engagement

1. **Turn cap 6** per exchange, enforced by the wrapper. At the cap:
   summarize the exchange into `channel.md` and stop for human review.
2. **Channel is the record.** If an exchange produced anything that matters
   beyond the moment — a finding, a claim, an answer to an ASK — append it to
   `channel.md`. Bridge chatter is otherwise ephemeral.
3. **Decisions still go to the AGENTS.md decision log.** The bridge never
   substitutes for Chase's sign-off on scope, spend, or release choices.
4. **Cost awareness.** Both sides run high-reasoning models on Chase's
   accounts. Prefer one well-composed message over five fragments; don't
   bridge what a channel note can carry asynchronously.
5. **No nested fan-out.** Don't ask the other agent to spawn further agents
   via the bridge; one level of delegation, then back to a human-visible
   surface.

## Setup status

- [x] Codex CLI installed (v0.144.1, npm global) — picks up `~/.codex` auth
- [x] `bridge.sh` written; `channel.md` protocol live
- [ ] `chmod +x coordination/bridge.sh` (Chase)
- [ ] Repo trusted by Codex: run `codex` once in the repo, accept prompt (Chase)
- [ ] Claude permission rule for `Bash(codex exec*)` and
      `Bash(coordination/bridge.sh*)` (Chase)
- [ ] MCP registrations, both directions (Chase; commands above)
