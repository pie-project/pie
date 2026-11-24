"""The checkpoint: the frozen params of a program, read from Hugging Face
safetensors the way `import.json` says.

Each import entry names one trace param, the checkpoint tensors that fill it
and an op. The trainer knows the op `copy` (one tensor, same shape). Any other
op makes it refuse, naming the param, so a model whose import it cannot follow
never trains on wrong weights.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors import safe_open

from .program import Program


def load(program: Program, weights: str | os.PathLike, *, dtype: torch.dtype, device: str) -> dict[str, torch.Tensor]:
    """Every checkpoint param of `program` by trace name, read from the
    safetensors files in `weights`, at `dtype` on `device`."""
    files = sorted(Path(weights).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{weights}: no .safetensors files")
    where = {}
    for f in files:
        st = safe_open(f, framework="pt")
        where.update((key, st) for key in st.keys())
    shapes = {p["name"]: p["shape"] for p in program.trace["params"] if p["source"] == "Checkpoint"}
    out = {}
    for entry in program.imports:
        name, sources, op = entry["name"], entry["checkpoint"], entry["op"]
        if op != "copy" or len(sources) != 1:
            raise ValueError(f"param `{name}`: import op {op!r} over {len(sources)} tensors; the trainer reads `copy` of one")
        if sources[0] not in where:
            raise KeyError(f"param `{name}`: checkpoint tensor `{sources[0]}` is in no file under {weights}")
        t = where[sources[0]].get_tensor(sources[0])
        if list(t.shape) != shapes.get(name):
            raise ValueError(f"param `{name}`: checkpoint `{sources[0]}` is {list(t.shape)}, the trace wants {shapes.get(name)}")
        out[name] = t.to(device=device, dtype=dtype)
    missing = sorted(set(shapes) - set(out))
    if missing:
        raise ValueError(f"param `{missing[0]}`: import.json names no checkpoint tensor for it")
    return out
