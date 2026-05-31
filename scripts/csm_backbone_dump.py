#!/usr/bin/env python3
"""Dump the CSM backbone + depth + mimi weights (bf16) + prompt token ids for the
standalone backbone-forward parity harness (tests/csm_generate_parity.cu).

Loads eustlb/csm-1b, encodes "[0]Hello, this is a test.", and dumps every tensor
the CUDA generation primitive binds, as little-endian bf16 raw .bin files plus a
manifest. Also dumps the prompt token ids. The harness runs csm_generate_audio
and compares emitted codes against /tmp/csm_depth_parity/emitted_codes.npy.
"""
import glob, json, os, struct, sys
import numpy as np
import torch

CKPT = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--eustlb--csm-1b/snapshots/*"))[0]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/csm_bb_dump"
os.makedirs(OUT, exist_ok=True)

from transformers import AutoProcessor, CsmForConditionalGeneration  # noqa

model = CsmForConditionalGeneration.from_pretrained(CKPT).eval()
sd = model.state_dict()
proc = AutoProcessor.from_pretrained(CKPT)
text = "[0]Hello, this is a test."
inputs = proc(text, add_special_tokens=True, return_tensors="pt")
ids = inputs["input_ids"][0].tolist()
print("prompt ids:", ids)


def dump_bf16(name, t):
    a = t.detach().to(torch.bfloat16).cpu().contiguous()
    raw = a.view(torch.uint16).numpy().astype("<u2").tobytes()
    with open(os.path.join(OUT, name + ".bin"), "wb") as f:
        f.write(raw)
    return list(a.shape)


shapes = {}
for k, v in sd.items():
    if (k.startswith("backbone_model.") or k == "embed_text_tokens.weight"
            or k == "lm_head.weight"):
        shapes[k] = dump_bf16(k, v)

manifest = {"prompt": text, "prompt_ids": ids, "shapes": shapes}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print(f"dumped {len(shapes)} backbone tensors -> {OUT}")
_ = struct
