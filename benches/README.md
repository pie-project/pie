# benches

Canonical benchmark suite for Pie and baseline engines.

`benches` provides two benchmark modes:

- `latency`: one request at a time, intended for single-request latency. Pie uses scheduler policy `greedy`.
- `tput`: high concurrent load, intended for saturated throughput. Pie uses scheduler policy `adaptive`.

The suite records exact output-token counts whenever the engine exposes generated token IDs or usage metadata. Pie uses the `text-completion-bench` inferlet, which returns `num_prompt_tokens` and `num_output_tokens` from the actual generated token stream.

## Fair Defaults

Baseline optimizations that can distort comparison are off by default:

- vLLM: `enable_prefix_caching=False`; CUDA graphs remain enabled by default
- SGLang: `disable_cuda_graph=True`, `disable_radix_cache=True`
- llama.cpp: request payload uses `cache_prompt=false`; concurrent runs require `llama-server`
- Repeated prompt prefix reuse is avoided by default with `--unique-prompts`
- `--ignore-eos` is enabled by default so each request consumes the same output-token budget

## Examples

Pie latency:

```bash
uv --project sdk/python-server run python benches/pie_bench.py latency \
  --driver cuda_native \
  --model Qwen/Qwen2-0.5B --device cuda:0 \
  --max-tokens 128 --requests 16 --warmup 2 \
  --default-token-limit 4096
```

Pie throughput:

```bash
uv --project sdk/python-server run python benches/pie_bench.py tput \
  --driver cuda_native \
  --model Qwen/Qwen2-0.5B --device cuda:0 \
  --max-tokens 128 --num-requests 512 --concurrency 128 \
  --warmup 16 --default-token-limit 200000
```

vLLM throughput, from the vLLM venv:

```bash
/root/Workspace/.venvs/vllm/bin/python benches/vllm_bench.py tput \
  --model Qwen/Qwen2-0.5B \
  --max-tokens 128 --num-requests 512 --concurrency 128
```

SGLang throughput, from the SGLang venv:

```bash
/root/Workspace/.venvs/sglang/bin/python benches/sglang_bench.py tput \
  --model Qwen/Qwen2-0.5B \
  --max-tokens 128 --num-requests 512 --concurrency 128
```

llama.cpp throughput, with a running `llama-server`:

```bash
python benches/llamacpp_bench.py tput \
  --url http://127.0.0.1:8080 \
  --model Qwen/Qwen2-0.5B --max-tokens 128 \
  --num-requests 512 --concurrency 128
```

All runs print a human-readable summary and write JSON if `--json-out` is set.

## Environment Setup

Run all commands from the Pie repo root:

```bash
cd /root/Workspace/pie
```

### 1. Build the Pie benchmark inferlet

Pie measurements use `inferlets/text-completion-bench`, not the regular streaming `text-completion` inferlet. It returns exact token counts in the final `Return` event.

```bash
rustup target add wasm32-wasip2
cd inferlets/text-completion-bench
cargo build --target wasm32-wasip2 --release
cd /root/Workspace/pie
```

Expected artifact:

```bash
inferlets/text-completion-bench/target/wasm32-wasip2/release/text_completion_bench.wasm
```

### 2. Pie environment

Use the Pie Python environment created by `uv`. The reference PyTorch driver is named `dev`; `cuda_native`, `portable`, and `dummy` are embedded in the SDK `pie.server.Server` path used by this benchmark.

```bash
CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
  uv --project sdk/python-server sync --reinstall-package pie-server
```

Add `PIE_PORTABLE_CUDA=1` when you want the embedded portable driver to include ggml CUDA:

```bash
PIE_PORTABLE_CUDA=1 CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
  uv --project sdk/python-server sync --reinstall-package pie-server
```

Run Pie benches with:

```bash
uv --project sdk/python-server run python benches/pie_bench.py latency --help
uv --project sdk/python-server run python benches/pie_bench.py tput --help
```

### 3. vLLM environment

Use an independent venv. The benchmark runner imports `vllm`, `transformers`, and `tokenizers` from this environment.

```bash
cd /root/Workspace
uv venv .venvs/vllm
.venvs/vllm/bin/python -m pip install -U pip
.venvs/vllm/bin/python -m pip install vllm transformers tokenizers
cd /root/Workspace/pie
```

If you already have the vLLM venv from prior setup, use it directly:

```bash
/root/Workspace/.venvs/vllm/bin/python benches/vllm_bench.py tput \
  --model Qwen/Qwen2-0.5B \
  --max-tokens 128 --num-requests 512 --concurrency 128 \
  --json-out /tmp/benches-vllm.json
```

`benches` turns vLLM prefix caching off by default. It does not force eager mode:

- `enable_prefix_caching=False`
- `enforce_eager` is not set

### 4. SGLang environment

Use an independent venv. The benchmark runner imports `sglang`, `transformers`, and `tokenizers` from this environment.

```bash
cd /root/Workspace
uv venv .venvs/sglang
.venvs/sglang/bin/python -m pip install -U pip
.venvs/sglang/bin/python -m pip install sglang transformers tokenizers
cd /root/Workspace/pie
```

If you already have the SGLang venv from prior setup, use it directly:

```bash
/root/Workspace/.venvs/sglang/bin/python benches/sglang_bench.py tput \
  --model Qwen/Qwen2-0.5B \
  --max-tokens 128 --num-requests 512 --concurrency 128 \
  --json-out /tmp/benches-sglang.json
```

`benches` turns SGLang radix cache and CUDA graphs off by default:

- `disable_radix_cache=True`
- `disable_cuda_graph=True`

### 5. llama.cpp environment

Concurrent llama.cpp measurements require `llama-server`, not only `llama-cli` or `llama-completion`. The runner also imports `transformers` to apply the same chat template used by the other engines; install it in the Python environment used to launch `benches`.

```bash
uv --project sdk/python-server pip install transformers tokenizers
```

Build CUDA llama.cpp with the server target:

```bash
cd /root/Workspace/llama.cpp
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build-cuda --target llama-server -j "$(nproc)"
cd /root/Workspace/pie
```

Use a GGUF model. For example, if the Qwen2-0.5B GGUF created during smoke tests exists:

```bash
uv --project sdk/python-server run python benches/llamacpp_bench.py tput \
  --server-bin /root/Workspace/llama.cpp/build-cuda/bin/llama-server \
  --gguf-model /root/Workspace/models/gguf/Qwen2-0.5B-F16.gguf \
  --model Qwen/Qwen2-0.5B \
  --max-tokens 128 --num-requests 512 --concurrency 128 \
  --json-out /tmp/benches-llamacpp.json
```

You can also start `llama-server` yourself and point the runner at it:

```bash
python benches/llamacpp_bench.py tput \
  --url http://127.0.0.1:8080 \
  --model Qwen/Qwen2-0.5B \
  --max-tokens 128 --num-requests 512 --concurrency 128
```

The llama.cpp adapter sends `cache_prompt=false` in each request. When spawned by `benches`, it also starts the server with `--flash-attn off`.

## Canonical Runs

### Latency

Latency mode runs requests sequentially. For Pie it sets the scheduler policy to `greedy`.

```bash
uv --project sdk/python-server run python benches/pie_bench.py latency \
  --driver cuda_native \
  --model Qwen/Qwen2-0.5B --device cuda:0 \
  --requests 32 --warmup 4 --max-tokens 128 \
  --default-token-limit 4096 \
  --json-out /tmp/benches-pie-latency.json
```

### Throughput

Throughput mode drives a high concurrent load. For Pie it sets the scheduler policy to `adaptive`.

```bash
uv --project sdk/python-server run python benches/pie_bench.py tput \
  --driver cuda_native \
  --model Qwen/Qwen2-0.5B --device cuda:0 \
  --num-requests 512 --concurrency 128 --warmup 16 \
  --max-tokens 128 --default-token-limit 200000 \
  --json-out /tmp/benches-pie-tput.json
```

## Output Schema

Each run prints a summary and, with `--json-out`, writes:

```json
{
  "summary": {
    "mode": "tput",
    "engine": "pie",
    "model": "Qwen/Qwen2-0.5B",
    "requests": 512,
    "completed": 512,
    "failed": 0,
    "wall_s": 12.34,
    "output_tokens": 65536,
    "prompt_tokens": 24576,
    "req_per_s": 41.49,
    "output_tok_per_s": 5310.86,
    "latency_mean_ms": 1234.5,
    "latency_p50_ms": 1200.0,
    "latency_p95_ms": 1500.0,
    "latency_p99_ms": 1700.0,
    "config": {}
  },
  "requests": []
}
```

For Pie, `output_tokens` and `prompt_tokens` are exact counts returned by the inferlet. For vLLM and SGLang, output tokens are read from engine token IDs or completion metadata. For llama.cpp, token counts come from the OpenAI-compatible `usage` object when the server returns it.

## Fairness Notes

- Keep `--unique-prompts` enabled unless you are explicitly studying prefix cache behavior.
- Keep `--ignore-eos` enabled for max-token throughput comparisons.
- Use the same model, prompt, output budget, dtype/quantization, tensor parallelism, and GPU set across engines.
- Compare `latency` and `tput` separately. The Pie scheduler policy is intentionally different between them.
- Do not mix numbers from the removed legacy bench scripts with this harness
  without noting the methodology differences.
