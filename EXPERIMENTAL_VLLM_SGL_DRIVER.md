# Experimental: vLLM + SGLang drivers in pie's venv

**Status:** experimental, not part of `pyproject.toml`. Recreates manually after every `uv sync`.

Pie ships `pie_driver_vllm` and `pie_driver_sgl` as optional drivers. Their wheels do not co-resolve with pie's default `cu128` extra (torch 2.11.0 + flashinfer 0.6.8.post1) — neither vLLM nor SGLang publishes a release that lines up with that pair. This document records the most-recent set of wheels that *do* co-resolve with each other, at the cost of downgrading pie's torch.

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

The pie venv lives at `pie/.venv`. Run from anywhere — paths are absolute.

```bash
cat > /tmp/pie-vllm-sgl-overrides.txt <<'EOF'
llguidance>=1.3.0,<1.4.0
xgrammar==0.1.29
outlines-core==0.2.11
outlines>=1.2.0,<1.3.0
EOF

uv pip install \
  --python ./pie/.venv/bin/python \
  --only-binary=:all: \
  --index-strategy unsafe-best-match \
  --override /tmp/pie-vllm-sgl-overrides.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://flashinfer.ai/whl/cu128 \
  'torch==2.9.1' 'torchvision==0.24.1' 'torchaudio==2.9.1' \
  'vllm==0.16.0' 'sglang==0.5.9' \
  'flashinfer-python==0.6.3' 'flashinfer-cubin==0.6.3' 'flashinfer-jit-cache==0.6.3'
```

Notes on the flags:
- `--only-binary=:all:` — refuses to fall back to source builds.
- `--index-strategy unsafe-best-match` — required because PyPI carries a stale vLLM mirror on the PyTorch index that uv would otherwise pin to.
- `--override` — forces the four structured-decoding pins above; without it the resolver fails on `llguidance` (vLLM 1.3.x vs SGLang 0.7.x) and on `outlines-core` (0.2.11 vs 0.1.26).

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

1. **Not in `pyproject.toml` or `uv.lock`.** Any `uv sync` will undo this and restore the torch 2.11 cohort. Re-run the install block above to recreate.
2. **`torchao` is downgraded** from `>=0.14.1` (pie's `[cu128]` pin) to `0.9.0` (SGLang's hard pin). Pie features that rely on torchao ≥ 0.14 quantization APIs will not work in this venv.
3. **`flashinfer` is downgraded** from `0.6.8.post1` to `0.6.3`. Pie's `flashinfer-jit-cache` pin in `pyproject.toml` is at `0.6.8.post1+cu128`; this install replaces it with `0.6.3+cu128`.
4. **SGLang structured outputs are likely broken** (see override table). Confirm against your actual usage before relying on guided generation through the SGLang driver.
5. **vLLM 0.16.0 wheel is CUDA 12.8-compiled** — works on cu128 hosts. vLLM 0.20.0's PyPI wheel needs `libcudart.so.13` (CUDA 13) and will not load against torch+cu128, which is one reason we are not on 0.20.0.

## Revert

```bash
cd pie && uv sync --extra cu128
```

This restores torch 2.11 + flashinfer 0.6.8.post1 and removes vLLM/SGLang.

## When to revisit

- SGLang `main` (post-0.5.10.post1) has bumped flashinfer to 0.6.8.post1, matching pie. Once a release ships with that bump and a torch-2.10/2.11 update, the conflict matrix collapses considerably.
- vLLM publishing cu128 wheels for the 0.20+ line (currently CUDA 13 only on PyPI) would let pie keep its torch 2.11 stack and add vLLM as a normal extra.
