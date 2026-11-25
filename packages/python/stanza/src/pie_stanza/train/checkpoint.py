"""The checkpoint: the frozen params of a program, read from Hugging Face
safetensors the way `import.json` says.

`import.json` comes from pie's own import contract, the one `pie model
import` and serving use. Each entry names one trace param, the checkpoint
tensors that fill it and an op: `copy` reads one tensor as it is, and `expr`
evaluates the contract's read expression. The trainer evaluates:

- `Src`: a checkpoint tensor;
- `Slice`: `len` entries of an axis from `start` on (one layer's columns of
  Gemma's per-layer embedding table);
- `Transmute` to a raw encoding of the same dtype: a reshape (Qwen3.5's
  convolution, stored `[channels, 1, kernel]`; one expert of a stack);
- `Concat`: parts joined along an axis (a layer's experts);
- `Scale` and `Bias` by a uniform f32.

Anything else (a quantized encoding, a per-block factor) makes it refuse,
naming the param, so a model whose import it cannot follow never trains on
wrong weights.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

import torch
from safetensors import safe_open

from .program import Program

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
"""The raw encodings a `Transmute` may name, by spelling."""


def load(program: Program, weights: str | os.PathLike, *, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    """Every checkpoint param of `program` by trace name, read from the
    safetensors files in `weights`, at `dtype` on `device`; a param the trace
    states f32 stays f32."""
    files = sorted(Path(weights).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{weights}: no .safetensors files")
    where = {}
    for f in files:
        st = safe_open(f, framework="pt")
        where.update((key, st) for key in st.keys())
    shapes = {p["name"]: p["shape"] for p in program.trace["params"] if p["source"] == "Checkpoint"}
    wide = {p["name"] for p in program.trace["params"] if p["dtype"] == "f32"}
    out = {}
    for entry in program.imports:
        name, op = entry["name"], entry["op"]
        want = shapes.get(name)
        if want is None:
            raise ValueError(f"import.json reads `{name}`, which is no checkpoint param of the trace")
        missing = [s for s in entry["checkpoint"] if s not in where]
        if missing:
            raise KeyError(f"param `{name}`: checkpoint tensor `{missing[0]}` is in no file under {weights}")
        if op == "copy" and len(entry["checkpoint"]) == 1:
            t = where[entry["checkpoint"][0]].get_tensor(entry["checkpoint"][0])
        elif op == "expr":
            t = _evaluate(entry["expr"], where, name)
        else:
            raise ValueError(f"param `{name}`: import op {op!r}; the trainer reads `copy` of one tensor or `expr`")
        if list(t.shape) != want:
            raise ValueError(f"param `{name}`: the checkpoint read is {list(t.shape)}, the trace wants {want}")
        out[name] = t.to(device=device, dtype=torch.float32 if name in wide else dtype)
    missing = sorted(set(shapes) - set(out))
    if missing:
        raise ValueError(f"param `{missing[0]}`: import.json names no checkpoint tensor for it")
    return out


def _evaluate(expr: dict, where: dict, name: str) -> torch.Tensor:
    """One read expression of the import contract, over the checkpoint's tensors."""
    (kind, a), = expr.items()
    if kind == "Src":
        return where[a].get_tensor(a)
    if kind == "Slice":
        if "Src" in a["src"]:
            # Read only the slice: a per-layer slice of a 9 GB table must not load the table each time.
            key = a["src"]["Src"]
            at = [slice(None)] * len(where[key].get_slice(key).get_shape())
            at[a["axis"]] = slice(a["start"], a["start"] + a["len"])
            return where[key].get_slice(key)[tuple(at)]
        return _evaluate(a["src"], where, name).narrow(a["axis"], a["start"], a["len"]).clone()
    if kind == "Transmute":
        src = _evaluate(a["src"], where, name)
        raw = a["to"]["encoding"].get("Raw") if isinstance(a["to"]["encoding"], dict) else None
        if DTYPES.get(raw) != src.dtype or src.numel() != _numel(a["to"]["shape"]):
            raise ValueError(f"param `{name}`: the import reads it as {a['to']['encoding']}, which the trainer "
                             "cannot dequantize; train from a bf16 or f32 checkpoint")
        return src.reshape(a["to"]["shape"])
    if kind == "Concat":
        return torch.cat([_evaluate(p, where, name) for p in a["parts"]], dim=a["axis"])
    if kind in ("Scale", "Bias"):
        by = a["factor"] if kind == "Scale" else a["by"]
        if not (isinstance(by, dict) and "Uniform" in by):
            raise ValueError(f"param `{name}`: the import {kind.lower()}s it per block, which the trainer cannot read")
        k = struct.unpack("<f", struct.pack("<I", by["Uniform"]))[0]
        src = _evaluate(a["src"], where, name)
        return (src.float() * k if kind == "Scale" else src.float() + k).to(src.dtype)
    raise ValueError(f"param `{name}`: the import reads it through `{kind}`, which the trainer cannot evaluate")


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n
