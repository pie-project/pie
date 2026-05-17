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
  --kv-cache-dtype auto \
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
