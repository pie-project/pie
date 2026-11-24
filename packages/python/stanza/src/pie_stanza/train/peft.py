"""PEFT interop: Hugging Face PEFT adapters read into the standard-library
stanzas' banks, and bundles of those stanzas written back as PEFT adapters.

Two stanzas have a PEFT form. `lora` holds PEFT LoRA with the scale folded
into B (`alpha/r`, or `alpha/sqrt(r)` for rsLoRA) and the rank zero-padded to
the bank's, as `crates/inferlet/src/eta/peft.rs` reads it. `ia3` holds PEFT
IA3 on output rows as `l = l_peft - 1`, so a zero bank is the identity. The
module each site corrects comes from `import.json`: the checkpoint tensor of
param `layers.{l}.{site}`, without its `.weight`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .bundle import read_bundle
from .program import Bank, Program

PREFIX = "base_model.model."


def from_peft(peft_dir: str | os.PathLike, program: Program, *, stanza: str = "lora") -> dict[str, torch.Tensor]:
    """The f32 planes (one slot per bank) of every bank of `stanza` for the PEFT
    adapter in `peft_dir`; banks the adapter does not touch are zero."""
    path = Path(peft_dir)
    config = json.loads((path / "adapter_config.json").read_text())
    tensors = load_file(path / "adapter_model.safetensors")
    banks = program.banks_of(stanza)
    planes = {b.name: torch.zeros(b.shape[1:]) for b in banks}
    at = {(b.layer, b.site, _leaf(b)): b for b in banks}
    module = _modules(program)
    kind = config.get("peft_type")
    if (stanza, kind) == ("lora", "LORA"):
        rank, alpha = config["r"], config.get("lora_alpha", config["r"])
        scale = alpha / rank**0.5 if config.get("use_rslora") else alpha / rank
        for name, t in tensors.items():
            role = "lora_a" if ".lora_A." in name else "lora_b" if ".lora_B." in name else None
            if role is None:
                if "lora_" in name:
                    raise ValueError(f"adapter tensor `{name}` corrects no projection stanza `lora` adapts")
                continue
            bank = at.get((*module.get(_module(name, ".lora_"), (None, None)), role))
            if bank is None or t.dim() != 2:
                raise ValueError(f"adapter tensor `{name}` corrects no projection stanza `lora` adapts")
            r = t.shape[0] if role == "lora_a" else t.shape[1]
            if r > min(bank.shape[1:]):
                raise ValueError(f"adapter tensor `{name}` is rank {r}; bank `{bank.name}` seats {min(bank.shape[1:])}")
            if role == "lora_a":
                planes[bank.name][:r] = t.float()
            else:
                planes[bank.name][:, :r] = t.float() * scale
    elif (stanza, kind) == ("ia3", "IA3"):
        feedforward = set(config.get("feedforward_modules") or [])
        for name, t in tensors.items():
            if not name.endswith(".ia3_l"):
                continue
            mod = _module(name, ".ia3_l")
            bank = at.get((*module.get(mod, (None, None)), None))
            if bank is None:
                raise ValueError(f"adapter tensor `{name}` scales no projection stanza `ia3` adapts")
            if mod.rsplit(".", 1)[-1] in feedforward or t.shape[-1] != 1:
                raise ValueError(f"adapter tensor `{name}` scales its module's input; stanza `ia3` scales outputs")
            planes[bank.name][:] = t.float()[:, 0] - 1
    else:
        raise ValueError(f"from_peft reads PEFT LORA into stanza `lora` and IA3 into `ia3`, not {kind} into `{stanza}`")
    return planes


def to_peft(bundle_dir: str | os.PathLike, out_dir: str | os.PathLike, program: Program) -> None:
    """Write a `lora` or `ia3` bundle as a PEFT adapter directory (f32 tensors)."""
    from peft import IA3Config, LoraConfig

    planes, info = read_bundle(bundle_dir)
    stanza = info["stanza"]
    banks = program.banks_of(stanza)
    name = {(layer, site): mod for mod, (layer, site) in _modules(program).items()}
    targets = sorted({name[(b.layer, b.site)].rsplit(".", 1)[-1] for b in banks})
    tensors = {}
    if stanza == "lora":
        rank = min(banks[0].shape[1:])
        config = LoraConfig(r=rank, lora_alpha=rank, target_modules=targets, lora_dropout=0.0, task_type="CAUSAL_LM")
        for b in banks:
            role = "lora_A" if _leaf(b) == "lora_a" else "lora_B"
            tensors[f"{PREFIX}{name[(b.layer, b.site)]}.{role}.weight"] = planes[b.name].float()
    elif stanza == "ia3":
        config = IA3Config(target_modules=targets, feedforward_modules=[], task_type="CAUSAL_LM")
        for b in banks:
            tensors[f"{PREFIX}{name[(b.layer, b.site)]}.ia3_l"] = (1 + planes[b.name].float())[:, None]
    else:
        raise ValueError(f"stanza `{stanza}` has no PEFT form; to_peft writes `lora` and `ia3`")
    out = Path(out_dir)
    config.save_pretrained(out)
    save_file({k: v.contiguous() for k, v in tensors.items()}, out / "adapter_model.safetensors")


def _modules(program: Program) -> dict[str, tuple[int, str]]:
    """The checkpoint module of each adapted (layer, site), by module path."""
    sites = {(b.layer, b.site) for b in program.banks}
    out = {}
    for entry in program.imports:
        parts = entry["name"].split(".")
        if len(parts) == 3 and parts[0] == "layers" and (int(parts[1]), parts[2]) in sites:
            out[entry["checkpoint"][0].removesuffix(".weight")] = (int(parts[1]), parts[2])
    return out


def _module(tensor: str, marker: str) -> str:
    return tensor.removeprefix(PREFIX).split(marker)[0]


def _leaf(bank: Bank) -> str | None:
    """`lora_a` or `lora_b` for a LoRA bank; None for a stanza's only bank at a site."""
    leaf = bank.name.rsplit(".", 1)[-1]
    return leaf if leaf in ("lora_a", "lora_b") else None
