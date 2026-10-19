# agent-loop

A single tool-using agent for agent traces: ten turns of generate a short
answer, call a fake `search` tool that sleeps 0.5–2 s, feed its result back
through `tools::answer`, repeat. The working set is kept across turns.

```bash
(cd examples && cargo build --release --target wasm32-wasip3 -p agent-loop)
cargo build --release -p pie --features metal

./target/release/pie run --trace agent-1.pie agent-loop
./target/release/pie run --trace agent-3.pie --parallel 3 agent-loop
python3 scripts/check-trace.py agent-1.pie agent-3.pie
```

| Name | Default | Meaning |
| --- | --- | --- |
| `turns` | `10` | Generate / tool / feed-back rounds |
| `max_tokens` | `80` | Most tokens one turn generates |
| `tool_min_ms`, `tool_max_ms` | `500`, `2000` | The fake tool's sleep range |
| `seed` | `0` | Varies sleeps and sampling (mixed with the instance id) |
| `temperature` | `0.7` | Sampling temperature; `0` is greedy |

Traces from this Mac are in [`../traces`](../traces).
