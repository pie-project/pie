#!/usr/bin/env python3
"""flux/mk_tiny_flux2_pipe.py: flux2-mini's whole pipeline (models/flux_2.poem's `flux2-mini`: the DiT of
mk_tiny_flux2.py, a two-layer Qwen3 text encoder tapped at its second layer, the VAE of 32-channel blocks) as
diffusers' and transformers' modules, its weights drawn by name as engine-cerebras' tests draw them.

  --ckpt <dir>                      the checkpoint as the poem's `diffusers pipeline` format reads it: every
                                    plane under `dit.`, `te.` and `vae.` in <dir>/model.safetensors, bf16
                                    (the VAE's norms and biases f32), and the Qwen3 tokenizer's files
                                    (--tokenizer <dir>)
  --text <id,id,...> <out.f32>      the text encoder's rows for those token ids: hidden_states[2] of
                                    Qwen3ForCausalLM (no padding), [T, 192] f32
  --gqa                             `flux2-mini-gqa`'s encoder instead (klein's in miniature: hidden 256, 4 q
                                    heads over 2 kv heads, three layers, all tapped): --text gives
                                    hidden_states[1..3] side by side through the context embedder (the poem's
                                    `te_context.<i>` column blocks), [T, 256] f32
  --vae <latent.f32> <h>,<w> <out.f32>
                                    AutoencoderKLFlux2's decode of a [h * w, 32] latent (the poem's `latents`
                                    input), the batch norm undone as Flux2KleinPipeline does; [8h * 8w, 3]
                                    f32 pixels in [-1, 1]
  --pipe <prompt> <noise.f32> <h>,<w> <s0,s1,...> <out>
                                    the pipeline: the prompt's chat template (no thinking) tokenized, the
                                    text encoder's rows, the denoise loop from the latent <noise.f32> (`draw`: the tests' drawn latent) over the
                                    sigmas (then 0; an Euler step a velocity, guidance 4 unless --guidance),
                                    the VAE's decode; <out>.latent.f32, <out>.pixels.f32 and <out>.png

Weights: the DiT's and the encoder's as the poem names its params (mk_tiny_flux2.py's `plane`: a norm's gain
near one, else uniform at the variance of 1 / fan-in), q / k / v and gate / up split as the poem fuses them;
the VAE's under the checkpoint's own names (`vae_plane`). Every model runs in f32 over its bf16 weights.
Needs diffusers >= 0.36 and transformers >= 4.51. Threads: 8."""
import json
import os
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mk_tiny_flux2 as dit  # noqa: E402

torch.set_num_threads(8)

MASK = (1 << 64) - 1
A, C = 6364136223846793005, 1
TE = dict(hidden=192, inter=384, heads=2, kv_heads=2, head_dim=128, layers=2, held=3, vocab=151936)
VAE = dict(blocks=(32, 32, 32, 32), layers=1, groups=8)


def units(name, count):
    """mk_tiny_flux2.units, a block of 4096 draws at a time (the generator's jump ahead)."""
    B = 4096
    seed = dit.fnv(name)
    first = np.empty(B, dtype=np.uint64)
    for i in range(B):
        seed = (seed * A + C) & MASK
        first[i] = seed
    ab, cb = 1, 0
    for _ in range(B):
        ab, cb = (ab * A) & MASK, (cb * A + C) & MASK
    ab, cb = np.uint64(ab), np.uint64(cb)
    out = np.empty(((count + B - 1) // B) * B, dtype=np.uint64)
    cur = first
    with np.errstate(over="ignore"):
        for k in range(len(out) // B):
            out[k * B:(k + 1) * B] = cur
            cur = cur * ab + cb
    v = (out[:count] >> np.uint64(40)).astype(np.int64) - (1 << 23)
    return v.astype(np.float32) / np.float32(1 << 24)


def plane(name, shape):
    u = units(name, int(np.prod(shape)))
    if name.endswith("norm"):
        v = np.float32(1) + np.float32(0.5) * u
    else:
        v = u * np.sqrt(np.float32(12) / np.float32(shape[-1]))
    return dit.bf16(v.reshape(shape))


def te_state_dict():
    """Qwen3ForCausalLM's state dict (`held` layers: the poem reads the first `layers`)."""
    h, hd, q, kv = TE["hidden"], TE["head_dim"], TE["heads"], TE["kv_heads"]
    sd = {"model.embed_tokens.weight": plane("te.embed", [TE["vocab"], h])}
    for l in range(TE["held"]):
        s, d = f"te.layers.{l}", f"model.layers.{l}"
        qkv = plane(f"{s}.qkv", [(q + 2 * kv) * hd, h])
        sd[f"{d}.self_attn.q_proj.weight"] = qkv[: q * hd]
        sd[f"{d}.self_attn.k_proj.weight"] = qkv[q * hd:(q + kv) * hd]
        sd[f"{d}.self_attn.v_proj.weight"] = qkv[(q + kv) * hd:]
        sd[f"{d}.self_attn.q_norm.weight"] = plane(f"{s}.q_norm", [hd])
        sd[f"{d}.self_attn.k_norm.weight"] = plane(f"{s}.k_norm", [hd])
        sd[f"{d}.self_attn.o_proj.weight"] = plane(f"{s}.o", [h, q * hd])
        sd[f"{d}.input_layernorm.weight"] = plane(f"{s}.attn_norm", [h])
        sd[f"{d}.post_attention_layernorm.weight"] = plane(f"{s}.mlp_norm", [h])
        gu = plane(f"{s}.gate_up", [2 * TE["inter"], h])
        sd[f"{d}.mlp.gate_proj.weight"] = gu[: TE["inter"]]
        sd[f"{d}.mlp.up_proj.weight"] = gu[TE["inter"]:]
        sd[f"{d}.mlp.down_proj.weight"] = plane(f"{s}.down", [h, TE["inter"]])
    sd["model.norm.weight"] = plane("te.norm", [h])
    return sd


def te_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=TE["vocab"], hidden_size=TE["hidden"], intermediate_size=TE["inter"],
        num_hidden_layers=TE["held"], num_attention_heads=TE["heads"], num_key_value_heads=TE["kv_heads"],
        head_dim=TE["head_dim"], rope_theta=1e6, rms_norm_eps=1e-6, max_position_embeddings=512,
        tie_word_embeddings=True, attention_bias=False,
    )
    m = Qwen3ForCausalLM(cfg).float().eval()
    m.load_state_dict(te_state_dict(), strict=False)
    return m


def vae_plane(name, shape):
    """A VAE plane by its checkpoint name: a norm's gain near one, its shift and every bias small, a batch
    norm's variance near one, else uniform at the variance of 1 / fan-in."""
    u = units(name, int(np.prod(shape))).reshape(shape)
    if name.endswith("running_var"):
        v = np.float32(1) + np.float32(0.5) * u
    elif name.endswith("running_mean"):
        v = np.float32(0.5) * u
    elif "norm" in name and name.endswith(".weight"):
        v = np.float32(1) + np.float32(0.5) * u
    elif name.endswith(".bias"):
        v = np.float32(0.2) * u
    else:
        fan = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        v = u * np.sqrt(np.float32(12) / np.float32(fan))
    return torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))


def vae_model():
    from diffusers import AutoencoderKLFlux2

    m = AutoencoderKLFlux2(
        block_out_channels=VAE["blocks"], layers_per_block=VAE["layers"], norm_num_groups=VAE["groups"],
        latent_channels=32, sample_size=64,
    ).float().eval()
    sd = {}
    for n, t in m.state_dict().items():
        if n.endswith("num_batches_tracked"):
            sd[n] = t
            continue
        v = vae_plane("vae." + n, list(t.shape))
        # the convolutions' and projections' kernels are bf16 in the checkpoint; norms and biases f32
        if not (n.endswith(".bias") or "norm" in n or n.startswith("bn.")):
            v = dit.bf16(v.numpy())
        sd[n] = v
    m.load_state_dict(sd, strict=True)
    return m


def write_ckpt(out, tokenizer):
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    tensors = {}
    for n, t in dit.state_dict(dit.poem_params()).items():
        tensors["dit." + n] = t.to(torch.bfloat16)
    for n, t in te_state_dict().items():
        tensors["te." + n] = t.to(torch.bfloat16)
    for n, t in vae_model().state_dict().items():
        if n.endswith("num_batches_tracked"):
            continue
        f32 = n.endswith(".bias") or "norm" in n or n.startswith("bn.")
        tensors["vae." + n] = t.to(torch.float32 if f32 else torch.bfloat16).contiguous()
    save_file(tensors, os.path.join(out, "model.safetensors"))
    # what a snapshot's loader reads its encoding from (no layer counts: the poem's are the mini's)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"_class_name": "Flux2KleinPipeline", "torch_dtype": "bfloat16"}, f)
    if tokenizer:
        for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
            if os.path.exists(os.path.join(tokenizer, f)):
                shutil.copyfile(os.path.join(tokenizer, f), os.path.join(out, f))
    print(f"{out}: {len(tensors)} planes")


def text_rows(te, ids):
    with torch.no_grad():
        o = te(input_ids=torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    taps = TE.get("taps")
    if not taps:
        return o.hidden_states[TE["layers"]][0].float()
    # each tapped layer's rows through its column block of the context embedder, summed
    out = 0
    for i, l in enumerate(taps):
        w = torch.as_tensor(plane(f"te_context.{i}", [256, TE["hidden"]])).float()
        out = out + o.hidden_states[l][0].float() @ w.T
    return out


def use_gqa():
    """`flux2-mini-gqa`'s encoder: four layers in the state dict (the poem reads three, its last tap
    before transformers' final norm)."""
    TE.update(hidden=256, heads=4, kv_heads=2, layers=3, held=4, taps=[1, 2, 3])


def decode(vae, latent, h, w):
    """Flux2KleinPipeline's decode of the poem's latent [h * w, 32]: packed into 2 x 2 patches, the batch
    norm undone, unpacked, the VAE's decode."""
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P

    x = latent.reshape(h, w, 32).permute(2, 0, 1)[None]
    x = P._patchify_latents(x)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    x = P._unpatchify_latents(x * std + mean)
    with torch.no_grad():
        img = vae.decode(x, return_dict=False)[0][0]
    return img.permute(1, 2, 0).reshape(-1, 3).contiguous()


def main():
    a = sys.argv
    tok = a[a.index("--tokenizer") + 1] if "--tokenizer" in a else None
    if "--gqa" in a:
        use_gqa()
    if "--ckpt" in a:
        write_ckpt(a[a.index("--ckpt") + 1], tok)
    if "--text" in a:
        i = a.index("--text")
        ids = [int(x) for x in a[i + 1].split(",")]
        y = text_rows(te_model(), ids).numpy().astype("<f4")
        y.tofile(a[i + 2])
        print(f"{a[i + 2]}: text rows {y.shape}, rms {float(np.sqrt((y ** 2).mean())):.4f}")
    if "--vae" in a:
        i = a.index("--vae")
        h, w = (int(x) for x in a[i + 2].split(","))
        lat = torch.from_numpy(np.fromfile(a[i + 1], dtype="<f4"))
        px = decode(vae_model(), lat, h, w).numpy().astype("<f4")
        px.tofile(a[i + 3])
        print(f"{a[i + 3]}: pixels {px.shape}, mean {float(px.mean()):.4f}, rms {float(np.sqrt((px ** 2).mean())):.4f}")
    if "--pipe" in a:
        from transformers import AutoTokenizer

        i = a.index("--pipe")
        prompt, noise, hw, sig, out = a[i + 1:i + 6]
        h, w = (int(x) for x in hw.split(","))
        g = float(a[a.index("--guidance") + 1]) if "--guidance" in a else 4.0
        t = AutoTokenizer.from_pretrained(tok)
        text = t.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
        ids = t(text)["input_ids"]
        ctx = dit.bf16(text_rows(te_model(), ids).numpy())
        dit.TXT, dit.H, dit.W, dit.GUIDANCE = len(ids), h, w, g
        m = dit.model(dit.state_dict(dit.poem_params()))
        if noise == "draw":  # the latent engine-cerebras' tests draw (`input.latents`)
            x = dit.inputs()[0].clone()
        else:
            x = torch.from_numpy(np.fromfile(noise, dtype="<f4")).reshape(h, w, 32)
        sigmas = [np.float32(s) for s in sig.split(",")] + [np.float32(0)]
        for k in range(len(sigmas) - 1):
            v = dit.step(m, x, ctx, torch.float32, float(sigmas[k]))
            x = x + torch.from_numpy(v).reshape(h, w, 32) * float(np.float32(sigmas[k + 1] - sigmas[k]))
        lat = x.reshape(h * w, 32).contiguous()
        lat.numpy().astype("<f4").tofile(out + ".latent.f32")
        px = decode(vae_model(), lat, h, w)
        px.numpy().astype("<f4").tofile(out + ".pixels.f32")
        from PIL import Image

        img = ((px.reshape(8 * h, 8 * w, 3) / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        Image.fromarray(img).save(out + ".png")
        print(json.dumps({"ids": ids, "rows": len(ids), "latent_rms": float(lat.pow(2).mean().sqrt()),
                          "out": out}))


if __name__ == "__main__":
    main()
