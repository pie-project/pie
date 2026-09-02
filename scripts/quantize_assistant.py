"""Restate a bf16 Gemma 4 assistant (`mlx-community/gemma-4-*-it-assistant-bf16`)
in MLX affine 4-bit (group 64), the encoding pie's quantized gemma4 rows read
their banks in — so `pie model import <trunk> --aux <this>` lands a head the
text can chain cheaply (0.8 GB a step at bf16, 0.2 at four bits).

    python3 scripts/quantize_assistant.py <snapshot-dir> <out-dir> [--bits 4] [--group 64]

Every 2-D bank is quantized; norms, scalars stay bf16. `pre_projection` is
split at its column midpoint into `pre_projection_embed` / `pre_projection_hidden`
first, so each half is quantized on its own and read by name.
"""

import argparse
import json
import os
import shutil

import mlx.core as mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    config = json.load(open(os.path.join(args.snapshot, "config.json")))
    backbone = int(config["backbone_hidden_size"])

    src = os.path.join(args.snapshot, "model.safetensors")
    out = {}
    if True:
        weights = mx.load(src)
        for name, a in weights.items():
            if name == "pre_projection.weight":
                halves = {
                    "pre_projection_embed.weight": a[:, :backbone],
                    "pre_projection_hidden.weight": a[:, backbone:],
                }
            else:
                halves = {name: a}
            for n, w in halves.items():
                if w.ndim == 2 and w.shape[-1] % args.group == 0:
                    wq, scales, biases = mx.quantize(w, group_size=args.group, bits=args.bits)
                    stem = n[: -len(".weight")]
                    out[n] = wq
                    out[stem + ".scales"] = scales.astype(mx.bfloat16)
                    out[stem + ".biases"] = biases.astype(mx.bfloat16)
                else:
                    out[n] = w.astype(mx.bfloat16) if w.dtype in (mx.float32, mx.float16) else w
    mx.save_safetensors(os.path.join(args.out, "model.safetensors"), out, {"format": "mlx"})
    config["quantization"] = {"group_size": args.group, "bits": args.bits, "mode": "affine"}
    json.dump(config, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    for extra in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        p = os.path.join(args.snapshot, extra)
        if os.path.exists(p):
            shutil.copy(p, args.out)
    total = sum(v.nbytes for v in out.values())
    print(f"{len(out)} tensors, {total / 2**20:.0f} MiB -> {args.out}")


if __name__ == "__main__":
    main()
