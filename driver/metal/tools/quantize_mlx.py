#!/usr/bin/env python3
"""Arch-agnostic MLX 4-bit (affine group) quantizer.

Operates directly on raw safetensors tensors by name heuristic, so it works on
custom/future architectures (qwen3_5, gemma4) that mlx_lm / llama.cpp can't
parse. Produces an mlx-community-style checkpoint:
  - quantized linear `weight` -> uint32 packed, plus sibling `.scales`/`.biases`
  - everything else (norms, biases, embeddings, conv1d, GDN linear-attn,
    vision/audio/mtp) copied through DENSE
  - config.json gets a top-level `quantization` block {group_size, bits}

Driver consumes via index-as-quant-map (src.has(name + ".scales")). mlx_lm's
loader also applies quant per-module by scales-presence, so a partial-quant
checkpoint stays a valid mlx-lm oracle for supported archs (qwen2).
"""
import argparse, glob, json, os, re, shutil, sys
import mlx.core as mx

# Linear projections that flow through ops::linear in the metal graphs.
QUANT_SUFFIXES = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
    # gemma4 Per-Layer-Embedding projections (compiled PLE region)
    "per_layer_input_gate.weight", "per_layer_projection.weight",
    "per_layer_model_projection.weight",
)
# Never quantize anything under these (unused towers + GDN custom-kernel path +
# embeddings + conv).
SKIP_SUBSTR = ("visual", "vision", "audio", "mtp.", "linear_attn", "conv1d")


def is_embed(name):
    return name.endswith("embed_tokens.weight") or name.endswith(
        "embed_tokens_per_layer.weight")


def should_quant(name, arr, gs):
    if any(s in name for s in SKIP_SUBSTR):
        return False
    if arr.ndim != 2:
        return False
    if not name.endswith(QUANT_SUFFIXES):
        return False
    if arr.shape[-1] % gs != 0:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--lm-head", action="store_true",
                    help="synthesize a quantized lm_head from the (tied) embed")
    ap.add_argument("--lm-head-bits", type=int, default=0,
                    help="bits for the synthesized lm_head (0 = same as --bits)")
    args = ap.parse_args()
    gs, bits = args.group_size, args.bits

    os.makedirs(args.dst, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(args.src, "*.safetensors")))
    if not shards:
        sys.exit(f"no safetensors under {args.src}")

    tensors = {}
    for s in shards:
        tensors.update(mx.load(s))

    out = {}
    nq = nd = 0
    embed_name = None
    for name, arr in tensors.items():
        if is_embed(name) and embed_name is None and name.endswith(
                "embed_tokens.weight"):
            embed_name = name
        if should_quant(name, arr, gs):
            wq, scales, biases = mx.quantize(arr, group_size=gs, bits=bits)
            base = name[: -len(".weight")]
            out[name] = wq
            out[base + ".scales"] = scales
            out[base + ".biases"] = biases
            nq += 1
        else:
            out[name] = arr
            nd += 1

    # Tied lm_head: synthesize a quantized lm_head matmul weight from the embed
    # table, while keeping the embed dense for the gather path.
    has_lm_head = any("lm_head.weight" in n for n in tensors)
    if args.lm_head and not has_lm_head and embed_name is not None:
        emb = tensors[embed_name]
        lmh_bits = args.lm_head_bits if args.lm_head_bits > 0 else bits
        if emb.ndim == 2 and emb.shape[-1] % gs == 0:
            wq, scales, biases = mx.quantize(emb, group_size=gs, bits=lmh_bits)
            out["lm_head.weight"] = wq
            out["lm_head.scales"] = scales
            out["lm_head.biases"] = biases
            nq += 1
            print(f"  synthesized {lmh_bits}-bit lm_head from {embed_name} "
                  f"{tuple(emb.shape)}")

    mx.eval(list(out.values()))
    out_path = os.path.join(args.dst, "model.safetensors")
    mx.save_safetensors(out_path, out, metadata={"format": "pt"})

    # Copy aux files; patch config.json with the quantization block.
    for f in os.listdir(args.src):
        if f.endswith(".safetensors") or f.endswith(".safetensors.index.json"):
            continue
        sp = os.path.join(args.src, f)
        if os.path.isfile(sp):
            shutil.copy2(sp, os.path.join(args.dst, f))
    cfg_path = os.path.join(args.dst, "config.json")
    with open(cfg_path) as fh:
        cfg = json.load(fh)
    cfg["quantization"] = {"group_size": gs, "bits": bits}
    cfg["quantization_config"] = {"group_size": gs, "bits": bits,
                                  "quant_method": "affine"}
    with open(cfg_path, "w") as fh:
        json.dump(cfg, fh, indent=2)

    sz = os.path.getsize(out_path) / 1e6
    print(f"{args.src} -> {args.dst}: quantized {nq} linears, {nd} dense, "
          f"{sz:.0f} MB")


if __name__ == "__main__":
    main()
