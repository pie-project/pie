# pie-driver-vllm

Pie driver backed by vLLM. Wraps vLLM's model implementations and kernels
under pie's RPC + shmem surface so vLLM-supported models run inside the
pie scheduler.

## Install

```sh
uv venv ~/envs/pie-vllm
uv pip install --python ~/envs/pie-vllm/bin/python pie-driver-vllm
```

vllm + torch + flashinfer pins are co-resolved; expect a ~5-10 GB install.

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

- vLLM 0.16.0 + torch 2.9.1 + flashinfer 0.6.3.
- Coexists with `pie-driver-sglang` in the same venv only with the
  `override-dependencies` in `pyproject.toml`. Separate venvs are the
  supported path; `pie driver vllm set venv …` keeps them isolated.
