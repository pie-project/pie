# benches

Benchmark runners for Pie and baseline engines.

Modes:

- `latency`: sequential requests, useful for single-request latency.
- `tput`: launches `--num-requests` requests together, useful for
  saturated throughput.

Pie runs use the `text-completion-bench` inferlet so the runner can record
actual prompt/output token counts.

## Scripts

Engine runners (shared CLI, `latency` / `tput` modes):

- `pie_bench.py` — Pie (`--driver cuda_native`, …).
- `vllm_bench.py` — vLLM.
- `sglang_bench.py` — SGLang.
- `llamacpp_bench.py` — llama.cpp (HTTP server).
- `tensorrt_llm_bench.py` — TensorRT-LLM.

Auxiliary tools:

- `run_microbench.py` — WIT call overhead via the `wit-microbench` inferlet.
- `run_loader_evidence.py` — regenerate loader tests + Pie/vLLM/SGLang evidence.
- `smoke_deterministic.py` — per-model deterministic smoke check (temp 0, sha ledger).
- `sweep_spec_on_off.py` — A/B sweep: speculation on vs off.
- `sweep_pie_cuda_native.py` — sweep Pie cuda_native over local HF models.
- `tune_cuda_memory_profile.py` — profile CUDA memory-planner candidates; install
  the fastest auto-profile cache entry.

## Setup

Run from the repo root.

```bash
rustup target add wasm32-wasip2
cargo build -p text-completion-bench --target wasm32-wasip2 --release
```

For Pie:

```bash
uv --project sdk/python-server sync --reinstall-package pie-server
```

For vLLM or SGLang, use separate virtual environments and install the engine
there. The runners import the engine packages from the Python used to launch
the script.

## Examples

Pie latency:

```bash
uv --project sdk/python-server run python benches/pie_bench.py latency \
  --driver cuda_native \
  --model Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --max-tokens 128 \
  --requests 32 \
  --warmup 4
```

Pie throughput:

```bash
uv --project sdk/python-server run python benches/pie_bench.py tput \
  --driver cuda_native \
  --model Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --max-tokens 128 \
  --num-requests 512 \
  --kv-cache-dtype auto \
  --warmup 16
```

vLLM or SGLang:

```bash
python benches/vllm_bench.py tput --model Qwen/Qwen3-0.6B
python benches/sglang_bench.py tput --model Qwen/Qwen3-0.6B
```

llama.cpp:

```bash
python benches/llamacpp_bench.py tput \
  --url http://127.0.0.1:8080 \
  --model Qwen/Qwen3-0.6B
```

Use `--json-out <path>` to save machine-readable results.

## Reasoning-pattern benchmark

`reasoning_bench.py` compares Direct, Best-of-N, Tree-of-Thought, and
Graph-of-Thought through method-specific benchmark inferlets by default.
Reference answers remain in the Python harness and are never sent to the
model. The older `reasoning-benchmark` inferlet remains available as a
prototype/reference with `--inferlet-mode prototype`.

Build the base and method-specific inferlets:

```bash
cargo build --manifest-path inferlets/reasoning-base/Cargo.toml --target wasm32-wasip2 --release
cargo build --manifest-path inferlets/reasoning-direct/Cargo.toml --target wasm32-wasip2 --release
cargo build --manifest-path inferlets/reasoning-best-of-n/Cargo.toml --target wasm32-wasip2 --release
cargo build --manifest-path inferlets/reasoning-tree-of-thought/Cargo.toml --target wasm32-wasip2 --release
cargo build --manifest-path inferlets/reasoning-graph-of-thought/Cargo.toml --target wasm32-wasip2 --release
```

Run the bundled smoke problems:

```bash
uv --project sdk/python-server run python benches/reasoning_bench.py \
  --driver cuda_native \
  --model Qwen/Qwen3-0.6B \
  --pattern all \
  --json-out .tmp/reasoning-smoke.json
```

For GSM8K, pass a JSONL file containing the official `question` and `answer`
fields. The harness extracts the numeric reference after GSM8K's `####`
delimiter.

`inferlets/reasoning-base` is the minimal prompt-to-completion inferlet. It
does not extract answers, score candidates, or implement a scaling method. The
method-specific benchmark inferlets share one internal implementation crate,
`inferlets/reasoning-core`, so they preserve a common comparison schema:
`pattern`, `final_response`, `final_answer`, `selected_candidate_id`,
`candidates`, and `stats`. Use `--inferlet-dir PATTERN=PATH` to point one
method at an external/user-submitted inferlet with the same comparison schema.

RatioThink's ToT implementation is a useful reference for the next refinement
pass. The pieces worth adopting first are explicit beam/frontier bookkeeping,
separate `ok`/`incomplete`/`error` node states, observable scorer failures, and
final-answer selection from the deepest surviving beam rather than blindly
trusting the last generated level. Its source also records an important Pie
runtime lesson: in the current host path, sibling decode from one inferlet does
not reliably coalesce into larger GPU batches, so a memory-frugal sequential
or coupled expansion strategy can be the right baseline until forward-pass
deferral improves.

## Fairness defaults

- vLLM prefix caching is disabled.
- SGLang radix cache is disabled; CUDA graphs are left **enabled** (SGLang's
  default). Pass `--sglang-disable-cuda-graph` to turn them off for a run.
- llama.cpp requests use `cache_prompt=false`.
- Pie prompts are unique by default and `--ignore-eos` is enabled so every
  request consumes the same output-token budget.
- `tput` runs at **unlimited concurrency** by default (`--concurrency 0`): all
  `--num-requests` are admitted at once. To keep the comparison fair, vLLM's
  `max_num_seqs` is raised to `--num-requests` (lifting its built-in 256-seq
  cap), and the other engines likewise. Pass `--concurrency N` to cap every
  engine at N. (`latency` mode is sequential by definition — concurrency 1.)

## Speculation knobs

The CUDA driver advertises pass-level speculation; the bench can drive
it from the command line for sweeps:

- `--speculation-depth <n>` forwards to `scheduler.speculation_depth`
  (range 0..=64). The bench leaves it unset by default, so the server's
  default applies; pass `0` for the explicit no-spec baseline in any A/B
  comparison, `1` for the piggyback path, or higher for more chain steps
  per real fire to overlap with the inferlet's WASM time.
- `--wasm-delay-us <µs>` makes `text-completion-bench` busy-spin in
  WASM between every `execute()` call. Simulates per-token inferlet
  work (agent reasoning, tool-call deserialization) so the chain
  has time to overlap with non-zero W. At `W = 0` (default) the
  chain has nothing to overlap with — depth-4 runs ~4% slower than
  depth-1 due to extra `poll_events` RPCs.

The bench summary surfaces the relevant counters from the server's
`model_status` query (`spec hits`, `spec misses`, `spec attempted`,
`spec rule skipped`, `spec budget skipped`, `spec chain now/peak`,
`spec longest chain`) plus two derived efficiency lines:

- `spec hit rate` = hits / (hits + misses + attempted)
  — fraction of `execute()` calls that short-circuited via the
  chain.
- `spec chain yield` = hits / (attempted × speculation_depth)
  — how much of the theoretical chain max actually got delivered
  to inferlet calls; lower means truncation by page boundaries or
  orphan-at-ctx-destroy.

### Sample numbers (Qwen3-0.6B, RTX 4090)

**Sequential latency mode, W=10ms** (4 reqs × 64 tokens,
`latency --requests 4 --max-tokens 64 --wasm-delay-us 10000`):

| Mode                    | mean lat | tok/s | hits |
|---                      | ---:     | ---:  | ---: |
| `speculation_depth = 0` | 805 ms   | 79.5  |  0   |
| depth 1 (piggyback)     | 735 ms   | 87.1  | 145  |
| depth 4 (async chain)   | 687 ms   | 93.1  | 235  |

Depth-4 is 14.7% faster than no-spec; chain firing overlaps with
the 10ms WASM gap between calls.

**Many-request throughput mode, W=0** (256 reqs × 256 tokens,
`tput --num-requests 256 --max-tokens 256`):

| Mode                    | wall    | tok/s  | hit rate |
|---                      | ---:    | ---:   | ---:     |
| `speculation_depth = 0` | 6.58 s  |  9964  | —        |
| depth 1 (piggyback)     | 5.44 s  | 12055  | 46.6%    |
| depth 4 (async chain)   | 6.17 s  | 10623  | 74.6%    |

Depth-1 is 17.3% faster than no-spec; depth-4 is +6.2%. At high
request counts the GPU is already saturated by real fires, so the
extra chain kernels at depth-4 compete with real work — depth-1
hits the sweet spot. Depth-4 still beats no-spec because its 74.6%
hit rate skips real kernels often enough to outweigh the extra
chain work.

### Picking `speculation_depth`

| Workload                       | Recommended | Why                                   |
|--                              | ---:        | --                                    |
| Single-ctx, W>0 (agent loops)  | 4 (up to 64)| GPU idle in WASM gap; deep chain fills it |
| Many-ctx, high request count   | 1           | GPU already saturated; extra kernels hurt |
| Single-ctx, W≈0 (chat completion) | 1 or off | No WASM slack to hide chain behind    |

### Correctness verification

Run twice at `--temperature 0` (argmax) with `--dump-first-text` —
once with `--speculation-depth 0`, once with `--speculation-depth 4`.
The sha256 prefixes must match; if they diverge, the chain is
producing different tokens than the underlying kernel would.
