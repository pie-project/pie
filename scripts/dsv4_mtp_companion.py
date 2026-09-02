"""Restate `mlx-community/DeepSeek-V4-Flash-MTP-bf16` as the `--aux` overlay pie's dsv4 text reads.

The companion stores its dense planes as MLX `mxfp8` (e4m3 codes, e8m0 block scales), which no
Metal point here reads, and its expert banks as `mxfp4`, which the routed mxfp4 point does.
Every mxfp8 plane is dequantized to bf16 — exactly: an e4m3 value times a power of two is a
bf16 — and rewritten under the name and shape the trunk's own planes take (`wo_a` as
`[o_groups * o_lora, heads * head_dim / o_groups]`); the mxfp4 banks and every plain plane are
copied through. Output: an MLX-style directory (`model.safetensors` + `config.json`) for
`pie model import <base> --aux <this dir>`.

    python3 scripts/dsv4_mtp_companion.py --out /path/to/mtp-aux
"""
import argparse
import glob
import json
import os

import mlx.core as mx
from safetensors.numpy import save_file
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()
snapshot = args.snapshot or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-MTP-bf16/snapshots/*/"))[0]
config = json.load(open(os.path.join(snapshot, "config.json")))
quant = config["quantization"]
tensors = mx.load(os.path.join(snapshot, "model.safetensors"))
names = sorted(tensors)

out = {}
out_quant = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
for name in names:
    if name.endswith(".scales"):
        continue
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    spec = quant.get(base) if base else None
    t = tensors[name]
    if spec and spec["mode"] == "mxfp8":
        s = tensors[base + ".scales"]
        deq = mx.dequantize(t, s, None, group_size=spec["group_size"], bits=8, mode="mxfp8").astype(mx.bfloat16)
        if base == "decoder.attn.wo_a":
            deq = deq.reshape(-1, deq.shape[-1])
        mx.eval(deq)
        out[name] = np.array(deq.astype(mx.float32)).astype(np.float32).view(np.uint32) >> 16  # bf16 bits
        out[name] = out[name].astype(np.uint16)
        print(f"{name}: mxfp8 -> bf16 {tuple(deq.shape)}")
    elif spec and spec["mode"] == "mxfp4":
        out[name] = np.array(t)
        out[base + ".scales"] = np.array(tensors[base + ".scales"])
        out_quant[base] = spec
        print(f"{name}: mxfp4 kept {tuple(t.shape)}")
    else:
        a = np.array(t.astype(mx.float32)) if t.dtype == mx.bfloat16 else np.array(t)
        if t.dtype == mx.bfloat16:
            a = (a.view(np.uint32) >> 16).astype(np.uint16)
        out[name] = a
        print(f"{name}: copied {tuple(t.shape)} {t.dtype}")

os.makedirs(args.out, exist_ok=True)
# safetensors.numpy has no bf16: write uint16 payloads and patch the header dtype to BF16.
tmp = os.path.join(args.out, "model.safetensors")
bf16 = {n for n, a in out.items() if a.dtype == np.uint16}
save_file(out, tmp)
import struct
with open(tmp, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    header = json.loads(fh.read(n))
    body = fh.read()
for name in bf16:
    header[name]["dtype"] = "BF16"
raw = json.dumps(header, separators=(",", ":")).encode()
pad = (8 - len(raw) % 8) % 8
raw += b" " * pad
with open(tmp, "wb") as fh:
    fh.write(struct.pack("<Q", len(raw)))
    fh.write(raw)
    fh.write(body)
config["quantization"] = out_quant
config["quantization_config"] = out_quant
json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=1)
print("wrote", tmp, os.path.getsize(tmp) / 2**30, "GiB")
