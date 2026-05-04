# pie-driver-sglang

Pie driver backed by SGLang. Wraps SGLang's model definitions, attention
kernels, and KV cache plumbing under pie's RPC + shmem surface.

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
