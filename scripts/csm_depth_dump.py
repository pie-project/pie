#!/usr/bin/env python3
"""Dump the CSM depth decoder's weights and one frame's trace, for csm_depth_decoder_parity.

`csm_depth_decoder_parity.cu` checks the genuinely-new output-modality piece:
the depth decoder plus the RVQ frame sampler. It needs a captured frame --
what the backbone handed down, and what HF produced from it at each of the
thirty-one autoregressive steps -- and no dump or script for it existed.

What comes out
--------------
    weights/<HF name>.npy    the depth decoder's 40 tensors, raw bf16 bits in a
                             uint16 array (`upload_bf` takes u16 directly, so
                             this is lossless; f32 would round twice)
    frame_bb_hidden.npy      [backbone_hidden] f32 -- the seed
    frame_depth_argmax.npy   [31] i64  -- HF's emitted cb1..cb31
    frame_depth_logits.npy   [31, audio_vocab] f32 -- per-step logits
    manifest.json            cb0, and which frame this is

Why cb0 lives in the manifest
-----------------------------
The harness reads it from there rather than from `emitted_codes[0]`, and its
comment says why: the captured frame need not be frame 0. It is here, but the
harness cannot know that, so the dump states it. Same principle as the
backbone dump recomputing its reference instead of trusting a constant in a
comment: what the consumer has to know, the producer writes down.

The names need no mapping -- the harness opens `depth_decoder.model.…`, which
is exactly what the checkpoint calls them.

Usage
-----
    python scripts/csm_depth_dump.py --out /tmp/csm_depth_parity

    export PATH=/usr/local/cuda/bin:$PATH
    nvcc -O2 -arch=sm_89 -std=c++20 --extended-lambda --expt-relaxed-constexpr \\
         -I crates/driver-cuda/csrc/src -I crates/kernels-cuda/csrc/src \\
         crates/driver-cuda/csrc/tests/csm_depth_decoder_parity.cu -o /tmp/cdp
    /tmp/cdp /tmp/csm_depth_parity

That nvcc line no longer runs
-----------------------------
It is kept as the record of what the harness was built against, not as an
instruction. Every path in it is deleted: `crates/driver-cuda/csrc` went at
`b58db6c16`, taking `csm_depth_decoder_parity.cu` with it, and
`crates/kernels-cuda/csrc/src` was the ARCHIVE crate's host header tree --
`.hpp` declaring the `.cu` launchers -- which went with the whole crate at
`85c6c674b`.

The second `-I` is the one worth flagging, because that name has been reused
since. `crates/kernels-cuda/csrc/src` EXISTS again and holds the JIT crate's
device text, `.cuh` that NVRTC compiles at run time and not one host header.
nvcc would take the flag and find nothing this command wanted behind it: an
include root that resolves is not the same as the right one.

The dump below is unaffected. It writes `.npy` and JSON from a checkpoint and
reaches neither tree.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

# The same prompt csm_backbone_dump.py uses, so the two dumps describe the same
# run and csm_generate_parity can read both.
PROMPT_IDS = [128000, 58, 15, 60, 9906, 11, 420, 374, 264, 1296, 13, 128001]
FRAME = 0

GLOBALS = [
    "depth_decoder.model.embed_tokens.weight",
    "depth_decoder.model.inputs_embeds_projector.weight",
    "depth_decoder.model.norm.weight",
    "depth_decoder.codebooks_head.weight",
]
LAYER = [
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight",
    "self_attn.v_proj.weight", "self_attn.o_proj.weight",
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="/tmp/csm_depth_parity")
    ap.add_argument("--model", default="eustlb/csm-1b")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--no-reference", action="store_true",
                    help="weights only; skip the frame capture")
    args = ap.parse_args()

    try:
        import numpy as np
        import torch
        from transformers import AutoConfig, CsmForConditionalGeneration
    except ImportError as e:
        print(f"needs torch + transformers: {e}", file=sys.stderr)
        return 2

    out = pathlib.Path(args.out)
    (out / "weights").mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model}")
    model = CsmForConditionalGeneration.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16)
    model.eval()

    cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
    dcfg = getattr(cfg, "depth_decoder_config", cfg)
    n_layers = getattr(dcfg, "num_hidden_layers", 4)

    params = dict(model.named_parameters())
    wanted = list(GLOBALS)
    for i in range(n_layers):
        wanted += [f"depth_decoder.model.layers.{i}.{w}" for w in LAYER]

    missing = [n for n in wanted if n not in params]
    if missing:
        print(f"\n{len(missing)} of {len(wanted)} not found, e.g.:", file=sys.stderr)
        for n in missing[:5]:
            print(f"   {n}", file=sys.stderr)
        dd = [n for n in params if "depth" in n]
        print(f"\nnames this checkpoint has ({len(dd)} depth-ish, first 20):",
              file=sys.stderr)
        for n in dd[:20]:
            print(f"   {n}", file=sys.stderr)
        return 3

    total = 0
    for name in wanted:
        t = params[name].detach().to(torch.bfloat16).contiguous().cpu()
        # u16 raw bits: `upload_bf` in the harness accepts them directly, so
        # nothing rounds. Writing f32 would round bf16 -> f32 -> bf16.
        a = t.view(torch.uint16).numpy()
        np.save(out / "weights" / f"{name}.npy", a)
        total += a.nbytes
    print(f"wrote {len(wanted)} weights, {total/1e6:.1f} MB -> {out}/weights")

    if args.no_reference:
        print("skipping the frame capture (--no-reference)")
        return 0

    # ── capture one frame ────────────────────────────────────────────────
    print(f"capturing frame {FRAME}")
    with torch.no_grad():
        ids = torch.tensor([PROMPT_IDS], dtype=torch.long)
        bb = model(input_ids=ids, output_hidden_states=True)
        # Post-final-norm, last position: what the driver's `last_hidden` is
        # (csm_backbone_forward.cu rmsnorms resid[R-1] before the depth seed).
        bb_hidden = bb.hidden_states[-1][0, -1].float()
        cb0 = int((bb.logits if hasattr(bb, "logits") else bb[0])[0, -1].float().argmax())
        print(f"  cb0 = {cb0}")

        dd = model.depth_decoder
        n_cb = getattr(cfg, "num_codebooks", 32)
        codes, logit_rows = [], []
        past, cur = None, torch.tensor([[cb0]], dtype=torch.long)
        seed = bb_hidden.unsqueeze(0).to(torch.bfloat16)
        for step in range(n_cb - 1):
            o = dd(input_ids=cur,
                   # Required only for the first step, where `cur` holds the
                   # codebook-0 token the BACKBONE produced.
                   backbone_last_hidden_state=seed if step == 0 else None,
                   past_key_values=past, use_cache=True,
                   # Always the LAST position. With the default (0) the model
                   # takes `slice(1, None)` -- it drops index 0 because on the
                   # first step that slot holds the concatenated backbone
                   # hidden state, not a token. From step 1 the sequence is one
                   # token long and that slice is EMPTY, which crashes inside
                   # codebooks_head. `1` is right for both: step 0's last of
                   # two is the cb0 position, and later steps have only one.
                   logits_to_keep=1)
            past = o.past_key_values
            row = o.logits[0, -1].float()
            logit_rows.append(row.numpy())
            nxt = int(row.argmax())
            codes.append(nxt)
            cur = torch.tensor([[nxt]], dtype=torch.long)

    np.save(out / "frame_bb_hidden.npy", bb_hidden.numpy())
    np.save(out / "frame_depth_argmax.npy", np.asarray(codes, dtype=np.int64))
    logits = np.stack(logit_rows)
    np.save(out / "frame_depth_logits.npy", logits)
    print(f"  emitted {len(codes)} codes, logits {logits.shape}")
    print(f"  cb1..cb5 = {codes[:5]}")

    (out / "manifest.json").write_text(json.dumps({
        "model": args.model,
        "revision": args.revision or "(default)",
        "prompt": PROMPT_IDS,
        "frame": FRAME,
        "cb0": cb0,
        "num_codebooks": n_cb,
        "backbone_hidden": int(bb_hidden.shape[0]),
        "audio_vocab": int(logits.shape[1]),
        "n_layers": n_layers,
    }, indent=2) + "\n")

    if logits.shape[0] != n_cb - 1:
        print(f"\nNOTE: captured {logits.shape[0]} steps; the harness reads "
              f"[{n_cb - 1}, vocab]. Reconcile before trusting a comparison.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
