# pie-driver-dev

Reference Python implementation of the pie driver — the readable backend
new features iterate against before being ported to C++.

Backed by PyTorch + flashinfer; suitable for prototyping new kernels,
sampling strategies, or model architectures without touching native code.
For production, prefer the C++ drivers under `driver/{portable,cuda}/`.

Shared Python scaffolding (worker loop, `Batch` wire model, shmem IPC,
capabilities handshake, standalone launcher) lives in
[`pie-driver-bridge`](../bridge_py/); this wheel only ships the
flashinfer-based `Engine`, per-arch model code, and the `dev` driver's
`worker_main` shim. `pie-driver-{sglang,vllm}` compose against the same
bridge surface.

## Install

```sh
uv venv ~/envs/pie-dev
uv pip install --python ~/envs/pie-dev/bin/python pie-driver-dev[cu128]
```

Extras:
- `cu126` — CUDA 12.6 wheel (older drivers, torch 2.7+).
- `cu128` — CUDA 12.8 wheel (default; torch 2.7+, flashinfer 0.6.8+).
- `metal` — Apple Silicon (no flashinfer; uses pie_kernels' Metal path).

## Run via standalone

```toml
[[model]]
name = "dev-qwen"

[model.driver]
type = "dev"
device = ["cuda:0"]

[model.driver.options]
venv = "/home/me/envs/pie-dev"
```
