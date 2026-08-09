#!/usr/bin/env python3
"""Dump the CSM backbone's weights, and the HF reference the harness checks against.

`crates/driver-cuda/csrc/tests/csm_backbone_parity.cu` has named this script
since it was written, and the script did not exist -- so the harness has never
been runnable by anyone who did not already have the dumps. That is what makes
the tower kernels untouchable: they can be read but not verified, so a change
to them cannot be shown to be safe. This closes that.

What comes out
--------------
One raw little-endian bf16 `.bin` per weight, named for its HF parameter, in
the layout `load_bin()` expects (`uint16` bit patterns, no header, no shape --
the harness knows the shapes from `CsmBackboneRawWeights`):

    embed_text_tokens.weight.bin
    backbone_model.embed_tokens.embed_audio_tokens.weight.bin
    backbone_model.norm.weight.bin
    lm_head.weight.bin
    backbone_model.layers.{0..15}.{input_layernorm,post_attention_layernorm}.weight.bin
    backbone_model.layers.{0..15}.self_attn.{q,k,v,o}_proj.weight.bin
    backbone_model.layers.{0..15}.mlp.{gate,up,down}_proj.weight.bin

plus two files the harness does not read but a human does:

    prompt_ids.txt      the token ids, one per line
    reference.txt       HF's codebook-0 argmax for frame 0, and the logit

Usage
-----
    pip install torch transformers safetensors
    python scripts/csm_backbone_dump.py                 # -> /tmp/csm_bb_dump
    python scripts/csm_backbone_dump.py --out DIR --model eustlb/csm-1b

then

    nvcc -O2 -arch=sm_89 -std=c++17 -I crates/driver-cuda/csrc/src \\
         crates/driver-cuda/csrc/tests/csm_backbone_parity.cu -o /tmp/cbp
    /tmp/cbp /tmp/csm_bb_dump

Why the reference is recomputed rather than hardcoded
-----------------------------------------------------
The harness prints `[HF reference = 420]` from a comment. A number in a comment
cannot notice that `transformers` changed, that a different checkpoint revision
was pulled, or that the prompt was edited. `reference.txt` is computed from the
same weights being dumped, in the same run, so the two cannot drift apart. If
it disagrees with 420, the harness's constant is what is stale.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# The prompt the harness hardcodes. Keep the two in step: if this changes, the
# `prompt` vector in csm_backbone_parity.cu changes with it.
PROMPT_IDS = [128000, 58, 15, 60, 9906, 11, 420, 374, 264, 1296, 13, 128001]

# HF parameter name -> whether it is per-layer. Everything else about a weight
# (shape, dtype) is the checkpoint's business; the harness reads shapes from
# `CsmBackboneRawWeights`, so a mismatch shows up there as a `WARN numel` line
# rather than silently.
GLOBAL_WEIGHTS = [
    "embed_text_tokens.weight",
    "backbone_model.embed_tokens.embed_audio_tokens.weight",
    "backbone_model.norm.weight",
    "lm_head.weight",
]
LAYER_WEIGHTS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/csm_bb_dump")
    ap.add_argument("--model", default="eustlb/csm-1b")
    ap.add_argument("--revision", default=None,
                    help="pin the checkpoint revision; recorded in reference.txt")
    ap.add_argument("--no-reference", action="store_true",
                    help="dump weights only (skips loading the full model for a forward)")
    args = ap.parse_args()

    try:
        import torch
        # CSM is not an AutoModelForCausalLM -- it is a text-to-waveform model
        # whose backbone happens to be a causal LM. Ask for the concrete class.
        from transformers import AutoConfig, CsmForConditionalGeneration
    except ImportError as e:
        print(f"needs torch + transformers: {e}", file=sys.stderr)
        print("  pip install torch transformers safetensors", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}"
          + (f" @ {args.revision}" if args.revision else ""))
    model = CsmForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16)
    model.eval()
    params = dict(model.named_parameters())

    cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
    n_layers = getattr(getattr(cfg, "backbone_config", cfg), "num_hidden_layers", 16)

    wanted = list(GLOBAL_WEIGHTS)
    for i in range(n_layers):
        wanted += [f"backbone_model.layers.{i}.{w}" for w in LAYER_WEIGHTS]

    missing = [n for n in wanted if n not in params]
    if missing:
        print(f"\n{len(missing)} parameters not found under these names, e.g.:",
              file=sys.stderr)
        for n in missing[:5]:
            print(f"   {n}", file=sys.stderr)
        print("\nnames this checkpoint does have (first 20):", file=sys.stderr)
        for n in list(params)[:20]:
            print(f"   {n}", file=sys.stderr)
        print("\nThe harness loads by HF parameter name, so a rename upstream "
              "breaks it here rather than silently.", file=sys.stderr)
        return 3

    total = 0
    for name in wanted:
        t = params[name].detach().to(torch.bfloat16).contiguous().cpu()
        # bf16 bit patterns as raw uint16 -- what `load_bin` memcpy's straight
        # into a __nv_bfloat16 buffer.
        raw = t.view(torch.uint16).numpy().tobytes()
        (out / f"{name}.bin").write_bytes(raw)
        total += len(raw)
    print(f"wrote {len(wanted)} weights, {total/1e6:.1f} MB -> {out}")

    (out / "prompt_ids.txt").write_text("\n".join(str(i) for i in PROMPT_IDS) + "\n")

    if args.no_reference:
        print("skipping the reference forward (--no-reference)")
        return 0

    print("running the HF forward for the reference argmax")
    with torch.no_grad():
        ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
        outp = model(input_ids=ids)
        logits = outp.logits if hasattr(outp, "logits") else outp[0]
        last = logits[0, -1].float()
        arg = int(last.argmax())
        val = float(last[arg])

    (out / "reference.txt").write_text(
        f"model {args.model}\n"
        f"revision {args.revision or '(default)'}\n"
        f"prompt {' '.join(str(i) for i in PROMPT_IDS)}\n"
        f"frame0_cb0_argmax {arg}\n"
        f"frame0_cb0_logit {val:.6f}\n")
    print(f"frame0 cb0 argmax = {arg} (logit {val:.4f})")
    if arg != 420:
        print(f"\nNOTE: csm_backbone_parity.cu prints `[HF reference = 420]` from a\n"
              f"comment, and this run says {arg}. The harness's constant is the stale\n"
              f"one -- update it, or pin --revision to the checkpoint it was written\n"
              f"against.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
