#!/usr/bin/env python3
"""Qwen3-VL vision-tower parity reference.

Runs the *real* transformers `Qwen3VLVisionModel` + the `Qwen3VLImageProcessor`
(loaded from the local `Qwen/Qwen3-VL-2B-Instruct` checkpoint) on a deterministic
synthetic image (and optionally a real one), and dumps the intermediate
activations the CUDA `qwen3_vl_vision` encoder (MULTIMODAL.md Phase 3) must
reproduce.

Unlike gemma, Qwen3-VL is native-resolution: the image processor produces the
`pixel_values` `[n_patch, 3·temporal·patch²]` and `image_grid_thw` `(t,h,w)`,
and the vision tower internally does patch-embed (Conv3d-as-matmul) + learned
abs pos-embed (bilinear-interpolated to the grid) + 24 pre-norm ViT blocks
(LayerNorm, full bidirectional attn, 2D-RoPE, plain gelu-tanh MLP) +
2×2 spatial-merge merger → `[n_token, 2048]`, plus 3 DeepStack mergers at layers
{5,11,17}.

Encoder parity is input-agnostic: identical (pixel_values, grid_thw) feed both
torch and the CUDA driver, so a synthetic-but-valid input is sufficient. We also
dump the text `position_ids` from `get_rope_index` for an image-containing
sequence, which the M-RoPE kernel must reproduce.

Outputs → OUT_DIR as .npy + manifest.json (shape/dtype/mean/std/first values).
Usage: python3 qwen3_vl_vision_parity_ref.py [checkpoint_dir] [out_dir] [image_path]

Mirrors scripts/gemma4_vision_parity_ref.py conventions.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

# ── Locate the checkpoint (guard with a clear message if absent) ─────────────
_DEFAULT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*"
)
if len(sys.argv) > 1:
    CKPT = sys.argv[1]
else:
    _hits = glob.glob(_DEFAULT_GLOB)
    if not _hits:
        sys.exit(
            "Qwen3-VL checkpoint not found. Expected a snapshot under\n"
            f"  {_DEFAULT_GLOB}\n"
            "Download it first:\n"
            "  huggingface-cli download Qwen/Qwen3-VL-2B-Instruct\n"
            "or pass the checkpoint dir as argv[1]."
        )
    CKPT = _hits[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/qwen3_vl_vision_parity"
IMG = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(OUT, exist_ok=True)

from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
)
from safetensors import safe_open

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

cfg = AutoConfig.from_pretrained(CKPT)
vcfg, tcfg = cfg.vision_config, cfg.text_config
SMS = vcfg.spatial_merge_size
DEEP = list(vcfg.deepstack_visual_indexes)
print(f"vision: hidden={vcfg.hidden_size} depth={vcfg.depth} heads={vcfg.num_heads} "
      f"patch={vcfg.patch_size} merge={SMS} out_hidden={vcfg.out_hidden_size} "
      f"num_pos_embed={vcfg.num_position_embeddings} deepstack={DEEP}")

# ── Build the vision tower and load its weights (strip `model.visual.`) ──────
vision = Qwen3VLVisionModel(vcfg).to(dev, dtype).eval()
sf = os.path.join(CKPT, "model.safetensors")
vt_sd = {}
PREFIX = "model.visual."
with safe_open(sf, framework="pt", device="cpu") as f:
    for k in f.keys():
        if k.startswith(PREFIX):
            vt_sd[k[len(PREFIX):]] = f.get_tensor(k).to(dtype)
mv = vision.load_state_dict(vt_sd, strict=False)
print(f"vision load: {len(vt_sd)} tensors, missing={len(mv.missing_keys)} "
      f"unexpected={len(mv.unexpected_keys)}")
real_missing = [k for k in mv.missing_keys if k.endswith(".weight") or k.endswith(".bias")]
assert not real_missing, f"missing vision weights: {real_missing[:8]}"

# ── Input: real image via the processor, else deterministic synthetic ────────
processor = AutoProcessor.from_pretrained(CKPT)
imgproc = processor.image_processor

if IMG is not None:
    from PIL import Image
    pil = Image.open(IMG).convert("RGB")
    proc_out = imgproc(images=[pil], return_tensors="pt")
    pixel_values = proc_out["pixel_values"].to(dev, dtype)
    grid_thw = proc_out["image_grid_thw"].to(dev)
    print(f"real image {IMG}: pixel_values={tuple(pixel_values.shape)} grid_thw={grid_thw.tolist()}")
else:
    # Synthetic: pick a grid divisible by the merge size in both axes so the
    # merger and deepstack reshapes are clean. 32×32 patches → 1024 patches →
    # /(2²) → 256 merged tokens. temporal t=1.
    GH, GW, T = 32, 32, 1
    grid_thw = torch.tensor([[T, GH, GW]], device=dev, dtype=torch.long)
    n_patch = T * GH * GW
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    # Deterministic ramp in [-1, 1]-ish range (processor normally normalizes).
    pixel_values = torch.linspace(
        -1, 1, n_patch * patch_dim, device=dev
    ).reshape(n_patch, patch_dim).to(dtype)
    print(f"synthetic: grid_thw={grid_thw.tolist()} pixel_values={tuple(pixel_values.shape)}")

n_patch = int(pixel_values.shape[0])
t, h, w = [int(x) for x in grid_thw[0].tolist()]
n_token = (t * h * w) // (SMS * SMS)
assert n_token * SMS * SMS == t * h * w, (n_token, t, h, w, SMS)

# ── Hooks to capture per-layer hidden states (post-block) ────────────────────
LAYER_TAPS = sorted(set([0, 5, 11, 17, vcfg.depth - 1]))
caps = {}


def _mk_hook(name):
    def hook(_m, _i, o):
        caps[name] = (o[0] if isinstance(o, tuple) else o).detach()
    return hook


# patch_embed output (after Conv3d-as-matmul, BEFORE pos-embed add).
vision.patch_embed.register_forward_hook(_mk_hook("patch_embed"))
for li in LAYER_TAPS:
    vision.blocks[li].register_forward_hook(_mk_hook(f"layer{li}"))

with torch.no_grad():
    vout = vision(pixel_values, grid_thw=grid_thw)
    merged = vout.pooler_output            # [n_token, 2048]
    deepstack = list(vout.deepstack_features)  # 3 × [n_token, 2048]
    last_hidden = vout.last_hidden_state   # [n_patch, 1024]

caps["last_hidden"] = last_hidden.detach()
caps["merged"] = merged.detach()
for i, d in enumerate(deepstack):
    caps[f"deepstack{i}_layer{DEEP[i]}"] = d.detach()
caps["input_pixel_values"] = pixel_values.detach()
caps["input_grid_thw"] = grid_thw.detach()

# ── Precomputed side-inputs the CUDA encoder consumes (validate independently)
# The interpolated abs pos-embed table `[n_patch, 1024]` (bilinear-interpolated
# from the [2304,1024] table to the grid, in spatial-merge patch order) and the
# 2D-RoPE (row,col) `position_ids` `[n_patch, 2]` — the CUDA header takes the
# interpolated table as a host-precomputed side input (see qwen3_vl_vision_forward.hpp).
try:
    from transformers.vision_utils import (
        get_vision_bilinear_indices_and_weights,
        get_vision_position_ids,
    )
    num_grid_per_side = int(vcfg.num_position_embeddings ** 0.5)
    b_idx, b_wt = get_vision_bilinear_indices_and_weights(
        grid_thw, num_grid_per_side=num_grid_per_side, spatial_merge_size=SMS
    )
    interp = (vision.pos_embed(b_idx.to(dev)) * b_wt.to(dev)[:, :, None]).sum(0)
    caps["pos_embed_interp"] = interp.detach()                 # [n_patch, 1024]
    vpos = get_vision_position_ids(grid_thw, SMS)              # [n_patch, 2] (row,col)
    caps["vision_rope_position_ids"] = vpos.detach()
    print(f"side-inputs: pos_embed_interp={tuple(interp.shape)} "
          f"vision_rope_position_ids={tuple(vpos.shape)}")
except Exception as e:  # noqa: BLE001
    print(f"WARN: side-input dump skipped: {e}")

# ── Text M-RoPE position_ids for an image-containing sequence ────────────────
# Build a tiny sequence: <text><vision_start>[image tokens]<vision_end><text>
# and dump the (3, seq) position_ids from get_rope_index, which the driver's
# M-RoPE kernel must reproduce. mm_token_type_ids: text=0, image=1.
try:
    image_token_id = cfg.image_token_id
    vstart, vend = cfg.vision_start_token_id, cfg.vision_end_token_id
    pre = [10, 11, 12, vstart]
    post = [vend, 13, 14]
    ids = pre + [image_token_id] * n_token + post
    input_ids = torch.tensor([ids], device=dev, dtype=torch.long)
    mm_tt = torch.zeros_like(input_ids)
    mm_tt[0, len(pre):len(pre) + n_token] = 1  # image span
    text_model = Qwen3VLModel(cfg).to(dev).eval()  # weights irrelevant for get_rope_index
    with torch.no_grad():
        pos_ids, deltas = text_model.get_rope_index(
            input_ids, mm_tt, image_grid_thw=grid_thw
        )
    caps["text_position_ids"] = pos_ids.detach()       # (3, 1, seq)
    caps["text_input_ids"] = input_ids.detach()
    caps["text_mm_token_type_ids"] = mm_tt.detach()
    print(f"text position_ids: {tuple(pos_ids.shape)} (seq={len(ids)}, image span "
          f"[{len(pre)}:{len(pre)+n_token}]) deltas={deltas.tolist()}")
except Exception as e:  # noqa: BLE001
    print(f"WARN: get_rope_index dump skipped: {e}")

# ── Dump ─────────────────────────────────────────────────────────────────────
manifest = {
    "checkpoint": CKPT,
    "config": {
        "hidden": vcfg.hidden_size, "depth": vcfg.depth, "heads": vcfg.num_heads,
        "head_dim": vcfg.hidden_size // vcfg.num_heads,
        "intermediate": vcfg.intermediate_size, "patch": vcfg.patch_size,
        "temporal_patch": vcfg.temporal_patch_size, "merge": SMS,
        "out_hidden": vcfg.out_hidden_size,
        "num_pos_embed": vcfg.num_position_embeddings,
        "num_grid_per_side": int(vcfg.num_position_embeddings ** 0.5),
        "deepstack_indexes": DEEP, "rope_theta": 10000.0, "ln_eps": 1e-6,
        "grid_thw": grid_thw[0].tolist(), "n_patch": n_patch, "n_token": n_token,
        "mrope_section": tcfg.rope_parameters.get("mrope_section"),
        "mrope_interleaved": tcfg.rope_parameters.get("mrope_interleaved"),
    },
    "tensors": {},
}
for name, tns in caps.items():
    arr = tns.float().cpu().numpy() if tns.dtype.is_floating_point else tns.cpu().numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    flat = arr.reshape(-1).astype(np.float64)
    manifest["tensors"][name] = {
        "shape": list(arr.shape), "dtype": str(tns.dtype),
        "mean": float(flat.mean()), "std": float(flat.std()),
        "first8": [float(x) for x in flat[:8]],
    }
with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)

# ── fp32 reference run, for a tight standalone CUDA check + all weights ───────
vision.float()
caps32 = {}
vision.patch_embed.register_forward_hook(_mk_hook_f32 := (
    lambda _m, _i, o: caps32.__setitem__("patch_embed", (o[0] if isinstance(o, tuple) else o).detach())
))
for li in LAYER_TAPS:
    vision.blocks[li].register_forward_hook(
        (lambda nm: (lambda _m, _i, o: caps32.__setitem__(
            nm, (o[0] if isinstance(o, tuple) else o).detach())))(f"layer{li}")
    )
with torch.no_grad():
    vout32 = vision(pixel_values.float(), grid_thw=grid_thw)
    caps32["last_hidden"] = vout32.last_hidden_state.detach()
    caps32["merged"] = vout32.pooler_output.detach()
    for i, d in enumerate(vout32.deepstack_features):
        caps32[f"deepstack{i}_layer{DEEP[i]}"] = d.detach()
for name, tns in caps32.items():
    np.save(os.path.join(OUT, name + "_f32.npy"), tns.cpu().numpy())
np.save(os.path.join(OUT, "input_pixel_values_f32.npy"), pixel_values.float().cpu().numpy())

wdir = os.path.join(OUT, "weights")
os.makedirs(wdir, exist_ok=True)
for name, tns in vision.state_dict().items():
    np.save(os.path.join(wdir, "vision." + name + ".npy"), tns.float().cpu().numpy())
print(f"  dumped fp32 refs ({list(caps32)}) + {len(vision.state_dict())} weights → weights/")

print(f"\nwrote {len(caps)} tensors → {OUT}")
for n, m in manifest["tensors"].items():
    print(f"  {n:28s} {str(m['shape']):18s} mean={m['mean']:+.4f} std={m['std']:.4f}")
