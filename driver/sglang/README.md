# pie-driver-sglang

Pie driver backed by SGLang. Wraps SGLang's model definitions, attention
kernels, and KV cache plumbing under pie's RPC + shmem surface.

Shared Python scaffolding (worker loop, `Batch` wire model, shmem IPC,
capabilities handshake, standalone launcher) is imported from
[`pie-driver-bridge`](../bridge_py/) — same bridge used by
`pie-driver-{dev,vllm}`. This wheel ships only `SGLangEngine` and the
sglang-specific `worker_main` shim that plugs it into
`._bridge.worker.run_worker`.

## Install

```sh
uv venv ~/envs/pie-sglang
uv pip install --python ~/envs/pie-sglang/bin/python pie-driver-sglang
```

sglang + torch + flashinfer pins are co-resolved; expect a ~5-10 GB install.

## Run via standalone

```toml
[[model]]
name = "sglang-llama"

[model.driver]
type = "sglang"
device = ["cuda:0"]

[model.driver.options]
venv = "/home/me/envs/pie-sglang"
```

## Compatibility

- SGLang 0.5.9 + torch 2.9.1 + flashinfer 0.6.3 (matches `pie-driver-vllm`'s
  flashinfer pin so a shared venv is technically possible — see
  `pie-driver-vllm`'s `pyproject.toml`).
