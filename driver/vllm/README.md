# pie-driver-vllm

Pie driver backed by vLLM. Wraps vLLM's model implementations and kernels
under pie's RPC + shmem surface so vLLM-supported models run inside the
pie scheduler.

Shared Python scaffolding (worker loop, `Batch` wire model, shmem IPC,
capabilities handshake, standalone launcher) is imported from
[`pie-driver-bridge`](../bridge_py/) — same bridge used by
`pie-driver-{dev,sglang}`. This wheel ships only `VllmEngine` and the
vllm-specific `worker_main` shim that plugs it into
`._bridge.worker.run_worker`.

## Install

```sh
uv venv ~/envs/pie-vllm
uv pip install --python ~/envs/pie-vllm/bin/python -e driver/vllm
```

Use uv for this driver. The package pins vLLM's CUDA 12.9 release wheel plus
PyTorch's CUDA 12.9 index in `pyproject.toml`; plain pip ignores
`[tool.uv.sources]` and may resolve a CUDA 13 stack that imports but fails on
CUDA 12.x hosts. vLLM + torch + flashinfer + FlashInfer's `ninja` JIT helper
are co-resolved; expect a ~5-10 GB install.

## Run via standalone

```toml
[[model]]
name = "vllm-qwen"

[model.driver]
type = "vllm"
device = ["cuda:0"]

[model.driver.options]
venv = "/home/me/envs/pie-vllm"
```

## Compatibility

- vLLM 0.21.0 CUDA 12.9 wheel + torch 2.11.0 CUDA 12.9 + flashinfer 0.6.8.post1.
- The standalone prepends the configured venv's `bin` directory to `PATH` for
  subprocess drivers so venv-local tools such as `ninja` are visible to
  FlashInfer JIT subprocesses.
- Coexists with `pie-driver-sglang` in the same venv only with the
  `override-dependencies` in `pyproject.toml`. Separate venvs are the
  supported path; `pie driver vllm set venv ...` keeps them isolated.
