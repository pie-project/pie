#!/usr/bin/env python3
"""Arch-agnostic MLX 4-bit (affine group) quantizer.

Operates directly on raw safetensors tensors by name heuristic, so it works on
custom/future architectures (qwen3_5, gemma4) that mlx_lm / llama.cpp can't
parse. Produces an mlx-community-style checkpoint:
  - quantized linear `weight` -> uint32 packed, plus sibling `.scales`/`.biases`
  - everything else (norms, biases, embeddings, conv1d, vision/audio/mtp) copied
    through DENSE
  - with --quant-arch, also quantizes the big arch-specific tensors (qwen3.6 GDN
    projections + gemma4 PLE table) for a genuinely BPW-matched fair bench
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
# Arch-specific big tensors quantized only under --quant-arch (for genuinely
# BPW-matched fair benchmarking vs llama, which quantizes these too):
#   - qwen3.6 GDN linear-attention projections: in_proj_qkv (226MB) + in_proj_z
#     (75MB) + out_proj (75MB) = 377MB / 42% of the text core. These are plain
#     matmuls in the metal graph (the *activations*, not these weights, feed
#     beta's GDN kernel), so they bind as QuantLinear with zero new kernel work.
#     The tiny sensitive gate/state projections (in_proj_a/b, A_log, dt_bias) and
#     conv1d (which DOES feed the GDN kernel) stay dense.
#   - gemma4 PLE table embed_tokens_per_layer (4.7GB / 80% of the file): a gather,
#     dequant-gathered by the graph's apply_embedding (same path as the tied embed).
ARCH_QUANT_SUFFIXES = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.out_proj.weight",
    "embed_tokens_per_layer.weight",
)
# Never quantize anything under these (unused towers + GDN custom-kernel path +
# embeddings + conv).
SKIP_SUBSTR = ("visual", "vision", "audio", "mtp.", "linear_attn", "conv1d")


def is_embed(name):
    return name.endswith("embed_tokens.weight") or name.endswith(
        "embed_tokens_per_layer.weight")


def should_quant(name, arr, gs, quant_arch=False):
    if arr.ndim != 2:
        return False
    if arr.shape[-1] % gs != 0:
        return False
    if quant_arch and name.endswith(ARCH_QUANT_SUFFIXES):
        return True
    if any(s in name for s in SKIP_SUBSTR):
        return False
    if not name.endswith(QUANT_SUFFIXES):
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
    ap.add_argument("--drop-embed", action="store_true",
                    help="omit the dense token embed_tokens.weight (tied-reuse): "
                    "the graph dequant-gathers input embeddings from the tied "
                    "quant lm_head bundle. Requires --lm-head. Zero double-store, "
                    "true-4-bit parity with llama's tied q4_K. Keeps gemma4 PLE "
                    "(embed_tokens_per_layer) intact.")
    ap.add_argument("--quant-arch", action="store_true",
                    help="also quantize arch-specific big tensors for a genuinely "
                    "BPW-matched fair bench vs llama: qwen3.6 GDN projections "
                    "(linear_attn.in_proj_qkv/in_proj_z/out_proj) and the gemma4 "
                    "PLE table (embed_tokens_per_layer). conv1d + tiny gate/state "
                    "projections (in_proj_a/b, A_log, dt_bias) stay dense.")
    args = ap.parse_args()
    if args.drop_embed and not args.lm_head:
        sys.exit("--drop-embed requires --lm-head (the tied bundle to gather from)")
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
        if should_quant(name, arr, gs, args.quant_arch):
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
            if args.drop_embed:
                out.pop(embed_name, None)
                nd -= 1
                print(f"  dropped dense {embed_name} (tied-reuse: graph "
                      f"dequant-gathers from lm_head bundle)")
        elif args.drop_embed:
            sys.exit("--drop-embed: no tied lm_head was synthesized "
                     f"(has_lm_head={has_lm_head}, embed_name={embed_name}); "
                     "refusing to drop the embed with no gather source")

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
