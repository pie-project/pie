"""Restate a published quantized MTP head (`mlx-community/*-MTP-4bit`: the head
alone, `fc.*` and `layers.0.*` at its root) with `fc.weight` split into its
two column halves, `fc_embed.*` and `fc_hidden.*`, each quantized on its own —
what pie's import reads by name, since a column slice of packed codes is not
a plane it can name.

    python3 scripts/split_mtp_fc.py <head-snapshot-dir> <out-dir>
"""

import json
import os
import shutil
import sys

import mlx.core as mx


def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    config = json.load(open(os.path.join(src, "config.json")))
    q = config.get("quantization") or {}
    group, bits = int(q.get("group_size", 64)), int(q.get("bits", 4))
    weights = mx.load(os.path.join(src, "model.safetensors"))
    fc = mx.dequantize(weights["fc.weight"], weights["fc.scales"], weights["fc.biases"], group_size=group, bits=bits)
    hidden = fc.shape[1] // 2
    out_w = {k: v for k, v in weights.items() if not k.startswith("fc.")}
    for name, half in (("fc_embed", fc[:, :hidden]), ("fc_hidden", fc[:, hidden:])):
        wq, scales, biases = mx.quantize(half, group_size=group, bits=bits)
        out_w[f"{name}.weight"], out_w[f"{name}.scales"], out_w[f"{name}.biases"] = wq, scales, biases
    mx.save_safetensors(os.path.join(out, "model.safetensors"), out_w, {"format": "mlx"})
    for extra in os.listdir(src):
        if extra.endswith(".json") and extra != "model.safetensors.index.json":
            shutil.copy(os.path.join(src, extra), out)
    print(f"{len(out_w)} tensors -> {out} (fc split at column {hidden})")


if __name__ == "__main__":
    main()
