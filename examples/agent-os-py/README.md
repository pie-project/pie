# agent-os

An operating system for reasoning agents, written as one Pie inferlet.

Pie is the kernel: it owns KV pages, copy-on-write state, paging and the forward pass.
This program is the user-space half. It runs many agents ("lanes") against one arena of
KV pages and decides, every wave of 32 tokens, which agents run, which die, which are
forked, and what they share. A strategist (a policy) makes those decisions; the runtime
executes them cheaply and records why.

```
 problems ──► Supervisor ──► Policy (the strategist: spawn / kill / judge / finish)
                  │
                  ├─ memory-aware admission (KV pages, recurrent-state seats, GPU budget)
                  ├─ ledger (exact decode / prefill / sharing accounting) + trace (every decision, with its reason)
                  ▼
              wave(): ONE forward pass decodes every running lane (rows of a batch)
                  ▼
              Arena: one WorkingSet of refcounted KV pages
                  Root = a shared, page-aligned prefix (prefilled once)
                  Lane = its own page chain after the root + a copy-on-write recurrent state
```

## What is real, and what is not

Real, and measured on this machine (M5, 24 GB, Qwen3.5-0.8B and 4B, branch
`perf/metal-auto-kv-pool`):

| property | evidence |
|---|---|
| A fork is free and exact. A child forked mid-generation lists the parent's pages and continues token for token like the parent. | `--mode kernel_test`: child equals parent; 128 tokens of KV shared at the fork; fork cost 0.1-0.6 ms for any width |
| Lanes in one pass batch; lanes on separate pipelines do not. | separate pipelines: ~170 tok/s at 1 lane, 82 tok/s at 8 (they timeslice). Batched waves: 417 tok/s at 8, 586 at 16, 697 at 32 lanes (`probe_batching.py`); 700-1000 tok/s with 32 agents in the real runs |
| Shared prefixes are read once. | 120 problems x 4 agents read 15,104 prompt tokens instead of 75,520 (5x); 16 agents: 17x. These prompts are short (~128 tokens); with long shared contexts the gap is far larger |
| Batched rows waste almost nothing. | 99% of decoded row-tokens are useful (waste is only the tail of a finishing lane's last wave) |
| The scheduler never overcommits memory. | admission counts KV pages and recurrent-state seats; a problem that cannot fit waits in CREATED; the benchmark driver freezes the engine at 86% system memory (see below) |

Not real, so not claimed: the strategist does not "understand" problems (it sees only vote
structure and lane progress); savings from kills are labelled estimates in the trace; these
benchmarks use MATH-500, not open research problems. **No agent harness solves Millennium Prize
problems, and this one does not try to.** Whether a controller can allocate inference compute
better than fixed strategies is an empirical question; the numbers below are what this
prototype actually measured, including where it did not help.

## Engine facts this design is built on (each learned the hard way)

1. An inferlet cannot spawn, wait on or kill another inferlet. The orchestrator is therefore
   one inferlet hosting many agents; processes are lanes of one pass, not Pie processes.
2. A working set is scoped to one pipeline at a time, and `run_ahead` closes its pipeline when
   it finishes. Each phase (setup, wave) gets a fresh pipeline.
3. A multi-row pass is only recognized as a device-looped decode when its `pages` and
   `page_indptr` are carried through the epilogue (`pages.put(pages.take())`). Otherwise it is
   treated as a single row and rejected.
4. **Physical KV is backed as a prefix up to the highest page a pass declares writable.**
   Declare exact `writable_pages=(lo, hi)` spans for every pass; the default is the whole
   reservation, which silently backs the entire arena. The arena hands out the lowest free
   page first, so the physical footprint follows the true high-water mark.
5. The engine sizes scratch buffers up front from `max_forward_tokens` and
   `max_forward_requests`. The defaults (10240 / 512) cost ~11 GB of a 24 GB Mac before a token
   is decoded; 2048 / 64 cost ~2 GB and are plenty here. `max_model_len` bounds the highest page
   index of one sequence, so the arena's frontier must fit in it.
6. Recurrent-state seats are a fixed pool (`max_state_slots`); every root and every lane holds
   some. One-shot policies hand their root's seats back right after forking.
7. The engine refuses a deployment that does not fit (`impossible submission: the device does
   not hold this deployment ...`) and says which knob to lower. Trust it.

## Run it

```bash
# build and install (see the repo README); Python inferlets need the language component
cargo build --release -p pie --features metal
pie language install <python.wasm built with python/inferlet/language/build.sh>
pie model import Qwen/Qwen3.5-0.8B

python3 examples/agent-os-py/bench.py --n 24 --levels 1,2,3 --configs fixed1,fixed4,adaptive
pie -c examples/agent-os-py/agentos.toml run examples/agent-os-py/main.py -- --mode kernel_test
python3 examples/agent-os-py/viz.py results/<run>.json > agents.html     # watch the decisions
python3 examples/agent-os-py/report.py results/*.json > report.html      # quality vs compute
python3 examples/agent-os-py/simulate.py                                 # replay recorded lanes, no GPU
```

`bench.py` refuses to start above 78% memory use and freezes the engine (SIGSTOP) at 86%,
resuming at 78%; if pressure does not ease in four minutes it aborts the run instead of letting
the machine swap. Every result file records the peak system memory and engine RSS.

### Scaling up (for a 128 GB machine)

Everything is model-agnostic. For a bigger model, change `model` in `agentos.toml`, then raise
`--arena-pages` (KV pages the arena may use), `--max-rows` (agents per wave), `--state-slots`
(recurrent-state seats; each lane needs 2), and `max_model_len` / `total_pages` to match. If the
engine says the deployment does not fit, it names the knob. With a large machine try
`total_pages = 0` (the auto-sized pool): on this machine it ran the full 32-agent workload with
79% peak system memory against 77% for an explicit pool, with identical accuracy.

## Results

See `RESULTS.md`.
