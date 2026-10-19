# .pie trace format, version 1

A `.pie` file is one UTF-8 JSON object. It is also a valid Chrome/Perfetto JSON trace (Perfetto ignores the extra keys).

```json
{
  "format": "pie-trace",
  "version": 1,
  "meta": {
    "pie_version": "0.5.3", "commit": "905e7b531", "started_at": "2026-10-18T12:00:00Z",
    "model": "qwen3.5-0.8b", "scheme": "affine_u4_g64", "device": "metal",
    "hardware": { "name": "Apple M5 Max", "memory_gb": 128, "bandwidth_gbs": 614 },
    "kv_page_bytes": 262144, "kv_pages_total": 4096,
    "weights_bytes": 480000000
  },
  "traceEvents": [ ... ]
}
```

No prompt or output text anywhere: only counts, timings, tool names and ids.

All `ts`/`dur` are microseconds since trace start (`ts` 0 = engine start of tracing).

## Tracks

| pid | process name | tid | thread name | holds |
| --- | --- | --- | --- | --- |
| 1 | Agents | one per agent (1, 2, …) | agent label, e.g. `agent-1 (react)` | the agent's phases |
| 2 | Engine | 1 | Scheduler | one span per step |
| 2 | Engine | 2 | GPU | one span per GPU submission (Metal command buffer GPU start/end) |

Name tracks with metadata events: `{"ph":"M","name":"process_name","pid":1,"args":{"name":"Agents"}}` and `{"ph":"M","name":"thread_name","pid":1,"tid":1,"args":{"name":"agent-1"}}`.

## Agent spans (pid 1, ph "X")

`name` is one of:
- `prefill` — args `{ "tokens": 512, "cached_tokens": 0 }`
- `decode` — args `{ "tokens": 120 }`
- `tool` — args `{ "tool": "search" }` (the agent is waiting on a tool; no GPU work for it)
- `queued` — the agent has work but the scheduler has not put it in a step yet

Gaps between an agent's spans are idle time.

## Scheduler steps (pid 2, tid 1, ph "X", name "step")

args:
```json
{ "step": 17, "prefill_tokens": 0, "decode_tokens": 3,
  "agents": { "1": { "prefill": 0, "decode": 1 }, "3": { "prefill": 0, "decode": 2 } },
  "gpu_us": 1840 }
```
`gpu_us` is the GPU-busy time of the submissions for this step.

## GPU submissions (pid 2, tid 2, ph "X", name "gpu")

args `{ "step": 17 }`. Real GPU start/end from Metal (`MTLCommandBuffer.GPUStartTime/GPUEndTime`), converted to the trace clock.

## Counters (ph "C", pid 2)

- `name: "kv_pages"`, args `{ "agent-1": 120, "agent-2": 64, "free": 3912 }` — per-agent KV cache pages held, emitted when any changes. Per-agent memory = pages × `kv_page_bytes`.
- `name: "gpu_busy"`, args `{ "pct": 73.5 }` — share of the last 10 ms window the GPU was busy.
- `name: "gpu_share"`, args `{ "agent-1": 40.0, "agent-2": 33.5 }` — per-agent share of that window's GPU busy time, splitting each step's `gpu_us` by the agent's tokens in it (prefill and decode tokens weighted equally for v1). Sums to `gpu_busy.pct`.
- `name: "memory"`, args `{ "allocated_gb": 6.2 }` — device memory allocated (Metal `currentAllocatedSize`).

## Instant events (ph "i", s "t")

- pid 1, agent tid: `start`, `end`.
- pid 2, tid 1: `evict` args `{ "agent": 2, "pages": 32 }` when the scheduler evicts an agent's pages.
