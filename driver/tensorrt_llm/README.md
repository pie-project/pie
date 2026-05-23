# pie-driver-tensorrt-llm

Experimental TensorRT-LLM subprocess driver for Pie.

The default driver path is intentionally built on TensorRT-LLM's public
`LLM.generate` API. It supports normal causal, token-producing generation
from Pie contexts and reports a conservative capability surface back to the
Pie runtime. Pie features that need a low-level per-forward/KV-cache API
are rejected explicitly: arbitrary user attention masks, adapter math,
raw logits/distribution/logprob probes, speculative verification, and
context forks that arrive without a replayable token history.

For deterministic generation, the driver uses a small lookahead buffer by
default: TensorRT-LLM generates a short continuation, while Pie still observes
one accepted token per step. Divergent replay tokens or stochastic samplers
fall back to the one-token path.

An experimental `execution_mode = "pyexecutor"` option is available for
TensorRT-LLM 1.2.1 with the PyTorch backend. It stops TensorRT-LLM's background
worker after initialization and drives the private `PyExecutor` forward,
sampling, and resource-manager calls directly so TensorRT-LLM-owned KV remains
resident across Pie decode steps. This mode is version-fragile and requires full
token histories (`max_history_tokens = null`).

For benchmarks, set `max_seq_len` explicitly to the same context limit used by
the comparison backend. Qwen3's model config advertises a much longer context
than the canonical `2048`-token vLLM/SGLang benches, and leaving TensorRT-LLM to
size for the full window increases prefill/setup cost.

Install from a Pie checkout:

```bash
sudo apt-get install -y libopenmpi-dev openmpi-bin
uv venv ~/.pie/venvs/tensorrt_llm --python 3.12
uv pip install --python ~/.pie/venvs/tensorrt_llm/bin/python ./driver/tensorrt_llm
pie driver tensorrt_llm set venv ~/.pie/venvs/tensorrt_llm
```

TensorRT-LLM imports `mpi4py`, so a system MPI runtime must be visible to the
driver process. NVIDIA's TensorRT-LLM container already provides this. The
driver wheel also installs NVIDIA's CUDA 13 CUBLAS wheel and bootstraps the
Python wheel library paths before importing TensorRT-LLM.
