# benches

Benchmark runners for Pie and baseline engines.

Modes:

- `latency`: sequential requests, useful for single-request latency.
- `tput`: launches `--num-requests` requests together, useful for
  saturated throughput.

Pie runs use the `text-completion-bench` inferlet so the runner can record
actual prompt/output token counts.

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
  --model Qwen/Qwen2-0.5B \
  --device cuda:0 \
  --max-tokens 128 \
  --requests 32 \
  --warmup 4
```

Pie throughput:

```bash
uv --project sdk/python-server run python benches/pie_bench.py tput \
  --driver cuda_native \
  --model Qwen/Qwen2-0.5B \
  --device cuda:0 \
  --max-tokens 128 \
  --num-requests 512 \
  --warmup 16
```

vLLM or SGLang:

```bash
python benches/vllm_bench.py tput --model Qwen/Qwen2-0.5B
python benches/sglang_bench.py tput --model Qwen/Qwen2-0.5B
```

llama.cpp:

```bash
python benches/llamacpp_bench.py tput \
  --url http://127.0.0.1:8080 \
  --model Qwen/Qwen2-0.5B
```

Use `--json-out <path>` to save machine-readable results.

## Fairness defaults

- vLLM prefix caching is disabled.
- SGLang radix cache and CUDA graphs are disabled.
- llama.cpp requests use `cache_prompt=false`.
- Pie prompts are unique by default and `--ignore-eos` is enabled so every
  request consumes the same output-token budget.

## Speculation knobs

The CUDA driver advertises pass-level speculation; the bench can drive
it from the command line for sweeps:

- `--speculation-depth <n>` forwards to `scheduler.speculation_depth`
  (range 0..=64, default 1). `0` disables speculation entirely —
  use this as the no-spec baseline in any A/B comparison. `1` is
  the piggyback path; higher values give more chain steps per real
  fire to overlap with the inferlet's WASM time.
- `--max-in-flight-tokens <n>` forwards to
  `scheduler.max_in_flight_tokens` (default 4096; `0` = unlimited).
  Soft per-device cap on aggregate chain entries — when reached, the
  runtime stops sending `predict_flag=true` until the chain drains.
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
