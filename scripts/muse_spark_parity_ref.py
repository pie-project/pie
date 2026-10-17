"""Muse Spark Open reference logits for a release-layout snapshot (the NVFP4
miniature `scratchpad`'s carve writes, or any cut of the release), written as
`transformers`' `modeling_muse_spark.py` computes its text model, so a pie
reading of the same snapshot can be read against it. The installed
`transformers` predates the model, so its text forward is restated here op
for op: the normed embedding, centered and input-scaled norms in f32 rounded
where the reference rounds, the fp32 router, the per-route post-expert norm,
the residual gates, NoPE full layers and the `output_multiplier` softcap.

NVFP4 experts read as modelopt dequantizes them for bf16 activations: e2m1
codes (the even element the low nibble) times their e4m3 scale a 16 times the
leg's `weight_scale_2`, rounded to bf16.

    python3 scripts/muse_spark_parity_ref.py <snapshot-dir> OUT [--steps 16] [--probes probes.json]

Writes, per probe NAME, `OUT/NAME.ref.{tf,gen}.f32` and `OUT/NAME.ref.json`
as `scripts/muse_glimmer_parity_ref.py` does.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

PROMPTS = [
    ("capital", "The capital of France is"),
    ("haiku", "Write a haiku about the sea."),
    ("code", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"),
    ("story", "Once upon a time, in a small village by the river,"),
]

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
TRUNK = "model.language_model."


class Weights:
    def __init__(self, snapshot, device):
        index = json.load(open(os.path.join(snapshot, "model.safetensors.index.json")))["weight_map"]
        self.where = index
        self.snapshot = snapshot
        self.device = device
        self.files = {}

    def get(self, name):
        shard = self.where[name]
        if shard not in self.files:
            self.files[shard] = safe_open(os.path.join(self.snapshot, shard), "pt", device="cpu")
        return self.files[shard].get_tensor(name).to(self.device)

    def has(self, name):
        return name in self.where


def nvfp4(w, stem, legs):
    codes = w.get(stem)  # [E, rows, cols / 2] u8
    scale = w.get(stem + "_weight_scale").float()  # [E, rows, cols / 16]
    tensor = w.get(stem + "_weight_scale_2").float().reshape(codes.shape[0], -1)  # [E, legs]
    table = torch.tensor(E2M1 + [-v for v in E2M1], device=codes.device)
    lo, hi = codes & 0x0F, (codes >> 4) & 0x0F
    vals = torch.stack([table[lo.long()], table[hi.long()]], dim=-1).flatten(-2)  # [E, rows, cols]
    vals = vals * scale.repeat_interleave(16, dim=-1)
    per_leg = vals.shape[1] // legs
    vals = vals * tensor.repeat_interleave(per_leg, dim=1)[..., None]
    return vals.to(torch.bfloat16)


def rms(x, eps):
    return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)


def norm_plain(x, eps, weight=None):  # MuseSparkRMSNorm
    y = rms(x.float(), eps)
    if weight is not None:
        y = y * weight.float()
    return y.type_as(x)


def norm_centered(x, weight, eps):  # MuseSparkTextCenteredRMSNorm
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + weight).float()).type_as(x)


def norm_input_scaled(x, weight, eps):  # MuseSparkTextInputScaledRMSNorm
    y = (x.float() * (1.0 + weight).float()).type_as(x).float()
    return (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)).type_as(x)


def residual_gate(res, h, gate, temperature):
    w = gate.float() / temperature
    beta = torch.sigmoid(w)
    alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
    return (alpha * res + beta * h).to(res.dtype)


def rope(x, positions, theta, dim):
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=x.device) / dim))
    freqs = positions[:, None].float() * inv[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    x1, x2 = x[..., : dim // 2], x[..., dim // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def forward(w, tc, ids):
    """Every position's logits for `ids` (one sequence), the cacheless forward."""
    dev = w.device
    H, nh, nkv, hd = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    eps, post_eps = tc["rms_norm_eps"], tc["post_norm_eps"]
    T = len(ids)
    pos = torch.arange(T, device=dev)
    x = F.embedding(torch.tensor(ids, device=dev), w.get(TRUNK + "embed_tokens.weight"))
    x = norm_plain(x, eps)
    for l in range(tc["num_hidden_layers"]):
        n = lambda s: f"{TRUNK}layers.{l}.{s}"  # noqa: E731
        res = x
        h = norm_centered(x, w.get(n("input_layernorm.weight")), eps)
        q = (h @ w.get(n("self_attn.q_proj.weight")).T).view(T, nh, hd)
        k = (h @ w.get(n("self_attn.k_proj.weight")).T).view(T, nkv, hd)
        v = (h @ w.get(n("self_attn.v_proj.weight")).T).view(T, nkv, hd)
        q = norm_plain(q, eps) * tc["qk_scale_factor"]
        k = norm_plain(k, eps)
        theta = tc["layer_rope_theta"][l]
        if theta:
            q, k = rope(q, pos, theta, hd), rope(k, pos, theta, hd)
        kk = k.repeat_interleave(nh // nkv, dim=1)
        vv = v.repeat_interleave(nh // nkv, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q.float(), kk.float()) * hd**-0.5
        mask = pos[None, :] <= pos[:, None]
        if tc["layer_types"][l] == "sliding_attention":
            mask = mask & (pos[:, None] - pos[None, :] < tc["sliding_window"])
        scores = scores.masked_fill(~mask[None], float("-inf"))
        p = torch.softmax(scores, dim=-1).to(vv.dtype)
        a = torch.einsum("hqk,khd->qhd", p, vv)
        a = norm_plain(a, eps).reshape(T, nh * hd)
        a = a * torch.sigmoid(h @ w.get(n("self_attn.gate_proj.weight")).T)
        o = a @ w.get(n("self_attn.o_proj.weight")).T
        o = norm_plain(o, post_eps)
        x = residual_gate(res, o, w.get(n("post_attention_residual_gate.gate")), tc["residual_gate_temperature"])

        res = x
        h = norm_centered(x, w.get(n("pre_feedforward_layernorm.weight")), eps)
        logits = F.linear(h.float(), w.get(n("mlp.gate.weight")).float()) * tc["output_multiplier"]
        probs = torch.sigmoid(logits)
        choice = probs + w.get(n("mlp.gate.e_score_correction_bias")).float()
        top = torch.topk(choice, k=tc["num_experts_per_tok"], dim=-1, sorted=False)[1]
        weights = probs.gather(1, top)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-15) * tc["routed_scaling_factor"]
        hp = h @ w.get(n("mlp.pre_expert_proj.weight")).T
        hp = norm_centered(hp, w.get(n("mlp.pre_expert_norm.weight")), eps)
        gu_name, dn_name = n("mlp.experts.gate_up_proj"), n("mlp.experts.down_proj")
        if w.has(gu_name + "_weight_scale"):
            gate_up, down = nvfp4(w, gu_name, 2), nvfp4(w, dn_name, 1)
        else:
            gate_up, down = w.get(gu_name), w.get(dn_name)
        post_norm = w.get(n("mlp.experts.post_expert_norm.weight"))
        out = torch.zeros_like(hp)
        for e in range(gate_up.shape[0]):
            rows, slot = torch.where(top == e)
            if rows.numel() == 0:
                continue
            g, u = F.linear(hp[rows], gate_up[e]).chunk(2, dim=-1)
            y = F.linear(F.silu(g) * u, down[e])
            y = norm_input_scaled(y, post_norm, post_eps)
            y = y * weights[rows, slot, None]
            out.index_add_(0, rows, y.to(out.dtype))
        f = out @ w.get(n("mlp.post_expert_proj.weight")).T
        f = norm_input_scaled(f, w.get(n("post_feedforward_layernorm.weight")), post_eps)
        x = residual_gate(res, f, w.get(n("post_feedforward_residual_gate.gate")), tc["residual_gate_temperature"])
    x = norm_plain(x, eps, w.get(TRUNK + "norm.weight"))
    logits = (x @ w.get("lm_head.weight").T).float()
    cap = tc["final_logit_softcapping"]
    return torch.tanh(logits * tc["output_multiplier"] / cap) * cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--probes", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = json.load(open(os.path.join(args.snapshot, "config.json")))
    tc = cfg.get("text_config", cfg)
    w = Weights(args.snapshot, args.device)
    if args.probes:
        probes = json.load(open(args.probes))["probes"]
    else:
        from transformers import PreTrainedTokenizerFast

        tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(args.snapshot, "tokenizer.json"))
        bos = tc.get("bos_token_id")
        probes = [{"name": n, "ids": ([bos] if bos is not None else []) + tok.encode(t, add_special_tokens=False)}
                  for n, t in PROMPTS]
        json.dump({"probes": probes}, open(os.path.join(args.out, "probes.json"), "w"), indent=1)
    torch.set_grad_enabled(False)
    for probe in probes:
        name, ids = probe["name"], list(probe["ids"])
        started = time.time()
        tf = forward(w, tc, ids).cpu().numpy()
        rows = [tf[-1].copy()]
        seq = list(ids)
        produced = []
        last = int(rows[0].argmax())
        for _ in range(args.steps):
            produced.append(last)
            seq.append(last)
            row = forward(w, tc, seq)[-1].cpu().numpy()
            rows.append(row)
            last = int(row.argmax())
        vocab = tf.shape[-1]
        tf.astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        np.stack(rows).astype(np.float32).tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump({"ids": ids, "argmax": [int(a) for a in tf.argmax(-1)], "gen": produced, "vocab": vocab},
                  open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"))
        print(f"  {name}: {len(ids)} tokens, gen={produced[:12]}  ({time.time() - started:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
