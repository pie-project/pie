# benches

Benchmark runners for Pie and baseline engines.

Modes:

- `latency`: sequential requests, useful for single-request latency.
- `tput`: concurrent requests, useful for saturated throughput.

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
  --concurrency 128 \
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
- SGLang radix cache and CUDA graphs are disabled.
- llama.cpp requests use `cache_prompt=false`.
- Pie prompts are unique by default and `--ignore-eos` is enabled so every
  request consumes the same output-token budget.
