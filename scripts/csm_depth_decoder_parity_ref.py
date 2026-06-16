#!/usr/bin/env python3
"""CSM depth-decoder standalone parity reference (the genuinely-new engine piece).

The CSM backbone is a stock Llama-3.2-1B (reuses Pie's verified `llama_like`),
and the Mimi decoder (codes -> waveform) is parity-verified separately. The two
new pieces are (1) the depth decoder (4-layer/1024-hidden Llama that turns the
backbone's last hidden + cb0 into codebooks 1..31 of one frame) and (2) the
RVQ frame sampler. This script isolates and dumps a ground-truth trace for both
so the CUDA `csm_depth_decoder_forward` + frame sampler can be checked standalone
without the whole serving loop — exactly mirroring mimi_decoder_parity_ref.py.

It runs the real `CsmForConditionalGeneration` (eustlb/csm-1b) with GREEDY
sampling on a fixed prompt, captures:
  * the ACTUAL emitted codes [32, n_frames]  (NOT a round-trip re-encode, which
    is what csm_generate_ref.py dumps — those are an argmin re-encode of the
    waveform and do NOT match the model's emitted argmax codes),
  * for a chosen frame `f`: the backbone last_hidden_state [2048] (seed) + the
    backbone-sampled cb0 + the depth decoder's full emitted [cb1..cb31] (31),
  * the per-codebook depth logits argmax (the exact target of the CUDA sampler),
  * the depth-decoder + lm_head + codebooks_head + backbone-embed weights in
    bf16 (real safetensors names) so the CUDA harness binds directly,
  * configs/manifest.

Usage: python3 csm_depth_decoder_parity_ref.py [checkpoint_dir] [out_dir] [frame_idx] ["prompt"]
"""
import glob
import json
import os
import sys

import numpy as np
import torch

_DEFAULT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--eustlb--csm-1b/snapshots/*"
)
CKPT = sys.argv[1] if len(sys.argv) > 1 else next(
    (p for p in glob.glob(_DEFAULT_GLOB) if os.path.isdir(p)), None
)
if not CKPT:
    sys.exit("csm-1b checkpoint not found; run huggingface-cli download eustlb/csm-1b")
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/csm_depth_parity"
FRAME = int(sys.argv[3]) if len(sys.argv) > 3 else 0
PROMPT = sys.argv[4] if len(sys.argv) > 4 else "Hello, this is a test."
os.makedirs(OUT, exist_ok=True)
wdir = os.path.join(OUT, "weights")
os.makedirs(wdir, exist_ok=True)

from transformers import AutoProcessor, CsmForConditionalGeneration  # noqa: E402

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
# Load in float32 then cast to bf16 for the dumped weights (the CUDA driver is
# bf16-store/fp32-compute). We RUN the reference in fp32 for a clean oracle.
model = CsmForConditionalGeneration.from_pretrained(CKPT, dtype=torch.float32).to(dev).eval()
cfg = model.config
dcfg = cfg.depth_decoder_config
NCB = cfg.num_codebooks          # 32
VOCAB = cfg.vocab_size           # 2051
processor = AutoProcessor.from_pretrained(CKPT)

text = f"[0]{PROMPT}"
inputs = processor(text, add_special_tokens=True, return_tensors="pt").to(dev)
print(f"prompt {text!r} -> input_ids {tuple(inputs['input_ids'].shape)}")

# ── Hook the depth decoder to capture, for the target frame, its seed +
#    emitted codes + per-step argmax. The backbone hidden seed is captured from
#    the _sample loop via a forward hook on backbone_model.norm output.
captured = {"frame": -1, "bb_hidden": None, "cb0": None, "depth_codes": None,
            "depth_logits_argmax": None}

orig_depth_generate = model.depth_decoder.generate
frame_counter = {"n": 0}


def patched_depth_generate(*args, **kwargs):
    out = orig_depth_generate(*args, **kwargs)
    seq = out if isinstance(out, torch.Tensor) else out.sequences
    fidx = frame_counter["n"]
    if fidx == FRAME:
        bb = kwargs["backbone_last_hidden_state"]
        in_ids = kwargs["input_ids"]
        captured["frame"] = fidx
        captured["bb_hidden"] = bb.detach().float().cpu().numpy()[0]   # [2048]
        captured["cb0"] = int(in_ids[0, 1].item())                     # cb0
        captured["depth_codes"] = seq.detach().cpu().numpy()[0, 1:].tolist()  # 31
    frame_counter["n"] += 1
    return out


model.depth_decoder.generate = patched_depth_generate

with torch.no_grad():
    gen = model.generate(
        **inputs, output_audio=False,
        max_new_tokens=64, do_sample=False, depth_decoder_do_sample=False,
    )
# gen: input_ids stacked frames. After the text prompt row, each row is 32 codes.
# Extract emitted code frames (rows with the codebook dimension).
seqs = gen if isinstance(gen, torch.Tensor) else gen.sequences
# seqs shape [1, n_frames, 32]
codes = seqs[0].detach().cpu().numpy().T  # [32, n_frames]
n_frames = codes.shape[1]
print(f"emitted codes (actual argmax): {codes.shape} (32 x {n_frames} frames)")
print(f"  frame0 cb (col 0): {codes[:, 0].tolist()}")
print(f"  cb0 row: {codes[0].tolist()}")

np.save(os.path.join(OUT, "emitted_codes.npy"), codes.astype(np.int64))

# ── Re-run the depth decoder once on the captured seed to dump per-step logits
#    (so the CUDA harness can compare logits, not just the argmax). ───────────
assert captured["bb_hidden"] is not None, f"frame {FRAME} not captured (only {n_frames} frames)"
bb_hidden = torch.tensor(captured["bb_hidden"], device=dev, dtype=torch.float32)[None]  # [1,2048]
cb0 = captured["cb0"]
print(f"target frame {FRAME}: cb0={cb0}  depth_codes(31)={captured['depth_codes']}")

# Manual 31-step greedy depth-decode reproducing CsmDepthDecoderForCausalLM.generate.
from transformers.cache_utils import DynamicCache
dd = model.depth_decoder
per_step_logits = []
per_step_argmax = []
with torch.no_grad():
    cache = DynamicCache(config=dcfg)
    # step 0: input [placeholder=0, cb0], seeded by bb_hidden at pos 0.
    # logits_to_keep=1 keeps only the last position (slice(-1,None)), avoiding
    # the empty slice(1,None) when generating incrementally.
    cur = torch.tensor([[0, cb0]], device=dev, dtype=torch.long)  # [1,2]
    out = dd(input_ids=cur, backbone_last_hidden_state=bb_hidden,
             past_key_values=cache, use_cache=True, logits_to_keep=1)
    logits = out.logits[:, -1, :].float()
    per_step_logits.append(logits.cpu().numpy()[0])
    nxt = int(torch.argmax(logits, dim=-1).item())
    per_step_argmax.append(nxt)
    cache = out.past_key_values
    for i in range(2, NCB):  # codebooks 2..31
        cur = torch.tensor([[nxt]], device=dev, dtype=torch.long)
        out = dd(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1, :].float()
        per_step_logits.append(logits.cpu().numpy()[0])
        nxt = int(torch.argmax(logits, dim=-1).item())
        per_step_argmax.append(nxt)
        cache = out.past_key_values

print(f"manual depth argmax(31): {per_step_argmax}")
print(f"  matches generate(): {per_step_argmax == captured['depth_codes']}")

np.save(os.path.join(OUT, "frame_bb_hidden.npy"), captured["bb_hidden"].astype(np.float32))
np.save(os.path.join(OUT, "frame_depth_logits.npy"),
        np.stack(per_step_logits).astype(np.float32))   # [31, 2051]
np.save(os.path.join(OUT, "frame_depth_argmax.npy"),
        np.array(per_step_argmax, dtype=np.int64))       # [31]

# ── Dump ALL weights needed by the CUDA depth decoder + frame sampler, bf16,
#    real names (so the harness's name->ptr map mirrors mimi's). ──────────────
sd = model.state_dict()
NAMES = (
    ["lm_head.weight", "embed_text_tokens.weight",
     "backbone_model.embed_tokens.embed_audio_tokens.weight",
     "depth_decoder.model.embed_tokens.weight",
     "depth_decoder.model.inputs_embeds_projector.weight",
     "depth_decoder.model.norm.weight",
     "depth_decoder.codebooks_head.weight"]
)
for L in range(dcfg.num_hidden_layers):
    p = f"depth_decoder.model.layers.{L}."
    NAMES += [p + s for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    )]
dumped = {}
for name in NAMES:
    if name not in sd:
        print(f"  MISSING {name}")
        continue
    t = sd[name]
    # bf16 storage (as u16) — matches the driver and the mimi parity dumps.
    bf = t.to(torch.bfloat16).view(torch.int16).cpu().numpy().astype(np.uint16)
    np.save(os.path.join(wdir, name + ".npy"), bf)
    dumped[name] = list(t.shape)

manifest = {
    "checkpoint": CKPT, "prompt": text, "frame_idx": FRAME, "n_frames": n_frames,
    "num_codebooks": NCB, "vocab_size": VOCAB,
    "depth_decoder_config": {
        "hidden_size": dcfg.hidden_size, "num_hidden_layers": dcfg.num_hidden_layers,
        "num_attention_heads": dcfg.num_attention_heads,
        "num_key_value_heads": dcfg.num_key_value_heads, "head_dim": dcfg.head_dim,
        "intermediate_size": dcfg.intermediate_size, "vocab_size": dcfg.vocab_size,
        "num_codebooks": dcfg.num_codebooks, "backbone_hidden_size": dcfg.backbone_hidden_size,
        "max_position_embeddings": dcfg.max_position_embeddings,
        "rms_norm_eps": dcfg.rms_norm_eps,
        "rope_theta": dcfg.rope_parameters["rope_theta"] if hasattr(dcfg, "rope_parameters") else 500000,
        "rope_scaling": dict(getattr(dcfg, "rope_parameters", {}) or {}),
    },
    "cb0": cb0, "depth_argmax": per_step_argmax,
    "dumped_weights": {k: v for k, v in dumped.items()},
}
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, default=str)
print(f"wrote emitted_codes, frame trace, {len(dumped)} weights -> {OUT}")
