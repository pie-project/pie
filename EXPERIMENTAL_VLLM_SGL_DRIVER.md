# Experimental: vLLM + SGLang drivers in pie's venv

**Status:** wired into `pyproject.toml` as the `vllm` and `sglang` extras (mutually exclusive with `cu126` / `cu128` / `metal`).

Pie ships `pie_driver_vllm` and `pie_driver_sgl` as optional drivers. Their wheels do not co-resolve with pie's default `cu128` extra (torch 2.11.0 + flashinfer 0.6.8.post1) — neither vLLM nor SGLang publishes a release that lines up with that pair. The `vllm` and `sglang` extras encode the most-recent set of wheels that *do* co-resolve with each other, at the cost of downgrading pie's torch.

## Target versions

| Package              | Version       | Source                             |
| -------------------- | ------------- | ---------------------------------- |
| torch                | 2.9.1+cu128   | `download.pytorch.org/whl/cu128`   |
| torchvision          | 0.24.1+cu128  | `download.pytorch.org/whl/cu128`   |
| torchaudio           | 2.9.1+cu128   | `download.pytorch.org/whl/cu128`   |
| flashinfer-python    | 0.6.3         | PyPI                               |
| flashinfer-cubin     | 0.6.3         | PyPI                               |
| flashinfer-jit-cache | 0.6.3+cu128   | `flashinfer.ai/whl/cu128`          |
| vllm                 | 0.16.0        | PyPI (cp38-abi3 manylinux, cu128)  |
| sglang               | 0.5.9         | PyPI                               |
| sgl-kernel           | 0.3.21        | PyPI (cp310-abi3 manylinux)        |
| transformers         | 4.57.1        | PyPI                               |

This is the latest co-resolvable pair. Reasoning:

- vLLM ≤ 0.16.0 is the newest line still pinning `torch==2.9.1`. Anything newer (0.17.x+) requires torch 2.10+, and 0.20.0 requires torch 2.11 + CUDA 13.
- SGLang 0.5.9 is the newest release pinning `flashinfer==0.6.3` (which matches vLLM 0.16.0). SGLang 0.5.10+ moved to flashinfer 0.6.7.x.
- Both pin `torch==2.9.1`. Both publish prebuilt wheels — no source build required.

## Known conflicts (resolved via overrides)

vLLM 0.16.0 and SGLang 0.5.9 disagree on three structured-decoding deps:

| Dep            | vLLM 0.16.0       | SGLang 0.5.9       | Override winner |
| -------------- | ----------------- | ------------------ | --------------- |
| llguidance     | `>=1.3.0,<1.4.0`  | `>=0.7.11,<0.8.0`  | vLLM            |
| xgrammar       | `==0.1.29`        | `==0.1.27`         | vLLM            |
| outlines-core  | `==0.2.11`        | (via outlines)     | vLLM            |
| outlines       | (transitive)      | `==0.1.11`         | vLLM (1.2.x)    |

We force vLLM's versions. SGLang's structured-output paths (guided JSON, regex/EBNF, grammar-constrained decoding) call APIs that do not exist on these newer libraries and **will fail at runtime**. Plain (unconstrained) generation is unaffected.

## Install

```bash
# vLLM driver
cd pie && uv sync --extra vllm

# OR SGLang driver (mutually exclusive)
cd pie && uv sync --extra sglang
```

`pyproject.toml` carries the cohort pins (torch 2.9.1+cu128, vllm 0.16.0 / sglang 0.5.9, flashinfer 0.6.3, etc.) and `[tool.uv].override-dependencies` resolves the four structured-decoding pin conflicts (llguidance, xgrammar, outlines, outlines-core) plus a numba pin (vLLM 0.16 wants 0.61.2 vs pie's base 0.63+).

The two extras are listed in `[tool.uv].conflicts` together with `cu126` / `cu128` / `metal`, so the resolver enforces the choice — you can't have both vLLM and SGLang in the same lock, and you can't combine either with the default `cu128` cohort.

## Verify

```bash
cd /tmp && ./pie/.venv/bin/python - <<'EOF'
import torch, flashinfer, vllm, sglang, transformers, sgl_kernel
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("flashinfer  ", flashinfer.__version__)
print("vllm        ", vllm.__version__)
print("sglang      ", sglang.__version__)
print("transformers", transformers.__version__)
from vllm import _C, _moe_C, LLM                          # vLLM C ext + Python facade
from sglang.srt.entrypoints.engine import Engine          # SGLang engine
print("ok")
EOF
```

Expected:
```
torch        2.9.1+cu128 cuda 12.8
flashinfer   0.6.3
vllm         0.16.0
sglang       0.5.9
transformers 4.57.1
ok
```

> **cwd warning:** do not run the verify script with `cwd` inside `pie/.venv/lib/python3.12/site-packages/vllm/`. vLLM ships a `vllm/tokenizers/` subpackage which shadows the top-level `tokenizers` package when the cwd injects that directory into `sys.path[0]`, producing a spurious circular-import error in `transformers`.

## Caveats

1. **`torchao` is downgraded** from `>=0.14.1` (pie's `[cu128]` pin) to `0.9.0` (SGLang's hard pin) on the `vllm` / `sglang` extras. Pie features that rely on torchao ≥ 0.14 quantization APIs will not work under these extras.
2. **`flashinfer` is downgraded** from `0.6.8.post1` to `0.6.3`. The `flashinfer-jit-cache` pin in `pyproject.toml` for `cu128` is `0.6.8.post1+cu128`; the `vllm` and `sglang` extras override it to `0.6.3+cu128`.
3. **SGLang structured outputs are likely broken** under the override pins (llguidance / xgrammar / outlines all forced to vLLM's versions). Confirm against your actual usage before relying on guided generation through the SGLang driver.
4. **vLLM 0.16.0 wheel is CUDA 12.8-compiled** — works on cu128 hosts. vLLM 0.20.0's PyPI wheel needs `libcudart.so.13` (CUDA 13) and will not load against torch+cu128, which is one reason we are not on 0.20.0.
5. **rand_mv adapter math is disconnected** in this release. `pie_driver/adapter.py` and `pie_driver_sgl/adapter.py` hard-set `RAND_MV_AVAILABLE = False` so the rand_mv kernel is never loaded or JIT-compiled (no ninja / nvcc dependency on server boot). The CMA-ES adapter `update()` method raises `RuntimeError` if invoked. Re-enable by restoring the `from .rand_mv import ...` lines in those files when the adapter feature lands.

## Switching back to the default cohort

```bash
cd pie && uv sync --extra cu128
```

This restores torch 2.11 + flashinfer 0.6.8.post1 and removes vLLM/SGLang.

## When to revisit

- SGLang `main` (post-0.5.10.post1) has bumped flashinfer to 0.6.8.post1, matching pie. Once a release ships with that bump and a torch-2.10/2.11 update, the conflict matrix collapses considerably.
- vLLM publishing cu128 wheels for the 0.20+ line (currently CUDA 13 only on PyPI) would let pie keep its torch 2.11 stack and add vLLM as a normal extra.
