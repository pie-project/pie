"""The adapter bundle: one trained stanza as a directory pie can serve.

`adapter.safetensors` holds one bf16 tensor per bank of the stanza, keyed by
the Poem bank name and shaped like the bank without its slot axis. Its
metadata names the format and the stanza. `stanza.json` holds the bundle's
identity, its shape, how it was trained and, once `stanza skew` has run, how
far serving drifts from training.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .program import Program

FORMAT = "pie-stanza.adapter/1"


def _sort_header(path: Path) -> None:
    """Rewrite a safetensors file's header with sorted keys, so the same planes
    always give the same bytes (and the same bundle id): safetensors writes its
    metadata map in hash order. The tensor data is left as it is."""
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.dumps(json.loads(raw[8:8 + size]), sort_keys=True, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + raw[8 + size:])


def write_bundle(out_dir: str | os.PathLike, planes: dict[str, torch.Tensor], program: Program, *,
                 stanza: str, name: str, meta: dict | None = None) -> Path:
    """Write `planes` (one slot per bank of `stanza`) as a bundle in `out_dir`;
    `meta` adds to or overrides the fields of `stanza.json`."""
    banks = program.banks_of(stanza)
    want = {b.name: b for b in banks}
    if set(planes) != set(want):
        odd = sorted(set(planes) ^ set(want))
        raise ValueError(f"bank `{odd[0]}`: the planes and stanza `{stanza}`'s banks differ")
    tensors = {}
    for bank in banks:
        plane = planes[bank.name].detach()
        if tuple(plane.shape) != bank.shape[1:]:
            raise ValueError(f"bank `{bank.name}`: plane is {list(plane.shape)}, one slot is {list(bank.shape[1:])}")
        tensors[bank.name] = plane.to("cpu", torch.bfloat16).contiguous()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights = out / "adapter.safetensors"
    save_file(tensors, weights, metadata={"format": FORMAT, "stanza": stanza})
    _sort_header(weights)
    params = sum(t.numel() for t in tensors.values())
    info = {
        "name": name,
        "version": 1,
        "id": "ada_" + hashlib.sha256(weights.read_bytes()).hexdigest()[:16],
        "model": program.model,
        "package": program.package,
        "stanza": stanza,
        "stanza_sha256": program.mount.get("stanza_sha256", {}).get(stanza),
        "package_sha256": program.mount.get("files", {}).get(f"{program.package}.poem"),
        "banks": [{"name": b.name, "shape": list(b.shape[1:]), "dtype": "bf16"} for b in banks],
        "params": params,
        "bytes": 2 * params,
        "data_sha256": None,
        "steps": 0,
        "lr": None,
        "loss_curve": [],
        "train_seconds": 0.0,
        "device": None,
        "skew": None,
    }
    info.update(meta or {})
    (out / "stanza.json").write_text(json.dumps(info, indent=1) + "\n")
    return out


def read_bundle(dir: str | os.PathLike) -> tuple[dict[str, torch.Tensor], dict]:
    """A bundle's bf16 planes by bank name, and its `stanza.json`."""
    path = Path(dir)
    weights = path / "adapter.safetensors"
    with safe_open(weights, framework="pt") as st:
        meta = st.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{weights}: format {meta.get('format')!r}, expected {FORMAT!r}")
        planes = {key: st.get_tensor(key) for key in st.keys()}
    info = json.loads((path / "stanza.json").read_text())
    if meta.get("stanza") != info.get("stanza"):
        raise ValueError(f"{path}: adapter.safetensors is stanza {meta.get('stanza')!r}, stanza.json says {info.get('stanza')!r}")
    return planes, info
