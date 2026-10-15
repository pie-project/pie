"""A modelopt NVFP4 export of Qwen 3.8 Flash Next, its quantized planes read out
to float32 under transformers' own names, so `scripts/qwen4_parity_ref.py` reads
the same weights pie serves at the precision of the model rather than of a W4A4
kernel.

    python3 scripts/qwen4_nvfp4_dequant.py <nvfp4-snapshot> <out-dir>

Each routed expert's `gate_proj`/`up_proj`/`down_proj` is `e2m1(code) ·
e4m3(weight_scale) · weight_scale_2`, stacked into the fused `gate_up_proj`
`[experts, 2 · inter, hidden]` and `down_proj` `[experts, hidden, inter]` banks
transformers reads; the e4m3 n-gram shards are each code times the one stored
`ngram_embedding.weight_scale`. The activations' `input_scale`s are not read:
the reference multiplies bf16-or-wider activations, as pie's W4A16 product does.
Every other plane is copied as stored, and the config loses its
`quantization_config`.
"""

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

EXPERT = re.compile(r"^(.*\.mlp\.experts)\.(\d+)\.(gate|up|down)_proj\.(weight|weight_scale|weight_scale_2|input_scale)$")


def nvfp4(codes, scales, tensor):
    """`[rows, cols / 2]` u8 codes under `[rows, cols / 16]` e4m3 scales and
    one f32 tensor scale, as float32 `[rows, cols]`."""
    rows = codes.shape[0]
    pairs = torch.stack([E2M1[(codes & 0xF).long()], E2M1[(codes >> 4).long()]], dim=-1)
    values = pairs.reshape(rows, -1)
    gains = scales.to(torch.float32).repeat_interleave(16, dim=-1) * tensor.to(torch.float32)
    return values * gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    index = json.load(open(os.path.join(args.snapshot, "model.safetensors.index.json")))["weight_map"]
    handles = {}

    def tensor(name):
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(args.snapshot, shard), "pt")
        return handles[shard].get_tensor(name)

    out = {}
    banks = {}
    for name in sorted(index):
        m = EXPERT.match(name)
        if m:
            bank, expert, leg, plane = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            banks.setdefault(bank, {}).setdefault(expert, {}).setdefault(leg, {})[plane] = name
            continue
        if name.endswith("ngram_embedding.weight_scale"):
            continue
        t = tensor(name)
        if ".ngram_embedding.shard_" in name and t.dtype == torch.float8_e4m3fn:
            scale = tensor(name.rsplit(".shard_", 1)[0] + ".weight_scale").to(torch.float32)
            t = t.to(torch.float32) * scale
        out[name] = t

    for bank, experts in banks.items():
        legs = {}
        for e in sorted(experts):
            for leg, planes in experts[e].items():
                w = nvfp4(tensor(planes["weight"]), tensor(planes["weight_scale"]), tensor(planes["weight_scale_2"]))
                legs.setdefault(leg, []).append(w)
        gate, up, down = (torch.stack(legs[leg]) for leg in ("gate", "up", "down"))
        out[bank + ".gate_up_proj"] = torch.cat([gate, up], dim=1).contiguous()
        out[bank + ".down_proj"] = down.contiguous()
        print(f"{bank}: {len(experts)} experts, gate_up {list(out[bank + '.gate_up_proj'].shape)}")

    save_file(out, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})
    config = json.load(open(os.path.join(args.snapshot, "config.json")))
    config.pop("quantization_config", None)
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    for name in os.listdir(args.snapshot):
        if name.endswith(".safetensors") or name in ("config.json", "model.safetensors.index.json", "hf_quant_config.json"):
            continue
        source = os.path.join(args.snapshot, name)
        if os.path.isfile(source):
            shutil.copy(source, os.path.join(args.out, name))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
