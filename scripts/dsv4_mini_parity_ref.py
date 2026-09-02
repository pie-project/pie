"""DeepSeek-V4-Flash reference logits for the `mini-l5-e16` snapshot.

An MLX transcription of `ssd-moe/deepseek-v4-flash-mlx`'s `oracle/decode_engine.py`
`layer_fwd` prefill path (itself graded against the official ds4 dumps), reading the
MLX per-tensor affine checkpoint directly (`config.json`'s `quantization` map: a
default pair plus per-tensor overrides) and driven by the mini's own config: five
layers, sixteen experts, `compress_ratios` [0, 0, 4, 128, 4], three hash layers.

Every organ is the reference's (`v4mlx/{hc,mla,norm_rope,compressor,sparse_attn}.py`),
vendored below so this file has no dependency beyond `mlx`, `numpy`, `safetensors`,
`ml_dtypes`. Where the official model and the reference agree on a rule pie's text
does not follow, the rule is a switch (`--pie ogroups,rope,hash,trunk`) so a pie run
can be attributed organ by organ WITHOUT per-layer dumps from pie: the reference with
a deviation switched on should meet pie at the bf16 floor if that deviation is the
whole difference.

    python3 scripts/dsv4_mini_parity_ref.py --probes probes.json --out OUT [--steps 24]
        [--sim] [--pie ogroups,rope,hash,trunk]

Outputs, per probe NAME in `probes.json` (`{"probes": [{"name", "ids"}]}`):
    OUT/NAME.ref.tf.f32   float32 [len(ids), vocab]   teacher-forced logits at every position
    OUT/NAME.ref.gen.f32  float32 [steps + 1, vocab]  logits at the last prompt position and
                                                       after each greedy step
    OUT/NAME.ref.json     {"argmax": [...], "gen": [...]}

Limits, stated: the reference models no compressed rows on ratio-128 layers and no
indexer (`index_topk` 512 is beyond every sequence here), so prompt + steps must stay
under 128 tokens for the ratio-128 layer to be honestly reproduced.
"""

import argparse
import json
import math
import os
import time

import ml_dtypes
import mlx.core as mx
import numpy as np
from safetensors import safe_open

mx.set_default_device(mx.gpu)

# ----------------------------------------------------------------------------- organs
# v4mlx/norm_rope.py


def rmsnorm(x, weight, eps=1e-6):
    x = x.astype(mx.float32)
    var = x.square().mean(-1, keepdims=True)
    return weight.astype(mx.float32) * (x * mx.rsqrt(var + eps))


def precompute_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow):
    def corr_dim(num_rot):
        return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    freqs = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(corr_dim(beta_fast)), 0)
        high = min(math.ceil(corr_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        lin = (mx.arange(dim // 2).astype(mx.float32) - low) / (high - low)
        ramp = mx.clip(lin, 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def rope_cos_sin(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = precompute_freqs(dim, original_seq_len, base, factor, beta_fast, beta_slow)
    t = mx.arange(seqlen).astype(mx.float32)
    ang = t[:, None] * freqs[None, :]
    return mx.cos(ang), mx.sin(ang)


ROPE_HALF = [False]  # `--pie rope_half`: pair lane i with i + d/2 (rotate-half) instead of 2i with 2i+1


def apply_rotary_emb(x, cos, sin, inverse=False):
    """x: [..., seq, (heads,) d]; cos/sin: [seq, d/2]; interleaved pairs."""
    *lead, d = x.shape
    if ROPE_HALF[0]:
        xf = x.astype(mx.float32)
        x0, x1 = xf[..., : d // 2], xf[..., d // 2 :]
    else:
        xp = x.astype(mx.float32).reshape(*lead, d // 2, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
    if x.ndim == 3:  # [seq, heads, d]
        c = cos[:, None, :]
        s = sin[:, None, :]
    else:  # [seq, d]
        c, s = cos, sin
    if inverse:
        s = -s
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    if ROPE_HALF[0]:
        return mx.concatenate([o0, o1], axis=-1)
    return mx.stack([o0, o1], axis=-1).reshape(*lead, d)


# v4mlx/hc.py


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, eps):
    lead = mixes.shape[:-1]
    pre = mx.sigmoid(mixes[..., 0:hc] * hc_scale[0] + hc_base[0:hc]) + eps
    post = 2.0 * mx.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.reshape(*lead, hc, hc)
    comb = mx.softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def hc_pre(x, hc_fn, hc_scale, hc_base, hc, norm_eps, sinkhorn_iters, hc_eps):
    """x: [s, hc, dim] -> y [s, dim], post [s, hc], comb [s, hc, hc]."""
    s = x.shape[0]
    xf = x.reshape(s, -1).astype(mx.float32)
    rsqrt = mx.rsqrt(xf.square().mean(-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, sinkhorn_iters, hc_eps)
    y = mx.sum(pre[..., None] * x, axis=1)
    return y, post, comb


def hc_post(x, residual, post, comb):
    """y[s, j, :] = post[s, j] * x[s, :] + sum_i comb[s, i, j] * residual[s, i, :]."""
    return post[..., None] * x[:, None, :] + mx.sum(comb[..., None] * residual[:, :, None, :], axis=1)


def hc_head(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps):
    s = x.shape[0]
    xf = x.reshape(s, -1).astype(mx.float32)
    rsqrt = mx.rsqrt(xf.square().mean(-1, keepdims=True) + norm_eps)
    mixes = (xf @ hc_fn.T) * rsqrt
    pre = mx.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    return mx.sum(pre[..., None] * x, axis=1)


# v4mlx/compressor.py


def act_quant_sim(x, block=64):
    a = np.array(x.astype(mx.float32))
    sh = a.shape
    fp8_max = 448.0
    ab = a.reshape(*sh[:-1], sh[-1] // block, block)
    amax = np.maximum(np.abs(ab).max(-1, keepdims=True), 1e-4)
    s = 2.0 ** np.ceil(np.log2(amax / fp8_max))
    q = np.clip(ab / s, -fp8_max, fp8_max).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * s
    return mx.array(q.reshape(sh))


def overlap_transform(t, ratio, d, value):
    """t: [g, ratio, 2d] -> [g, 2*ratio, d]: previous block's first half, this block's second."""
    g = t.shape[0]
    new = mx.full((g, 2 * ratio, d), value, dtype=t.dtype)
    new[:, ratio:] = t[:, :, d:]
    new[1:, :ratio] = t[:-1, :, :d]
    return new


def compressor_prefill(x, wkv, wgate, ape, norm_w, cos, sin, head_dim, ratio, rope_head_dim, sim):
    """x: [s, dim] -> compressed rows [s // ratio, head_dim], roped at block starts."""
    s = x.shape[0]
    d, rd = head_dim, rope_head_dim
    x = x.astype(mx.float32)
    kv = x @ wkv.T
    score = x @ wgate.T
    cutoff = s - s % ratio
    g = cutoff // ratio
    if g == 0:
        return mx.zeros((0, d), dtype=mx.float32)
    kv = kv[:cutoff].reshape(g, ratio, -1)
    score = score[:cutoff].reshape(g, ratio, -1) + ape
    kv = overlap_transform(kv, ratio, d, 0.0)
    score = overlap_transform(score, ratio, d, -mx.inf)
    kv = (kv * mx.softmax(score, axis=1)).sum(axis=1)
    kv = rmsnorm(kv, norm_w)
    rows = mx.arange(0, cutoff, ratio)
    roped = apply_rotary_emb(kv[..., -rd:], cos[rows], sin[rows])
    kv = mx.concatenate([kv[..., :-rd], roped], axis=-1)
    if sim:
        kv = mx.concatenate([act_quant_sim(kv[..., :-rd], 64), kv[..., -rd:]], axis=-1)
    return kv


# v4mlx/sparse_attn.py, batched over the query rows with a visibility mask — the
# same masked-dense arithmetic the reference states, with -inf where its gather
# would not read.


def windowed_attention(q, keys, sink, mask, scale):
    """q: [s, h, d], keys: [n, d], sink: [h], mask: [s, n] bool -> o [s, h, d]."""
    scores = scale * mx.einsum("shd,nd->shn", q, keys)
    scores = mx.where(mask[:, None, :], scores, -mx.inf)
    rowmax = scores.max(axis=-1, keepdims=True)
    p = mx.exp(scores - rowmax)
    denom = p.sum(axis=-1, keepdims=True) + mx.exp(sink[None, :, None] - rowmax)
    return mx.einsum("shn,nd->shd", p, keys) / denom


# ----------------------------------------------------------------------------- checkpoint


class Checkpoint:
    def __init__(self, snapshot):
        self.dir = snapshot
        self.config = json.load(open(os.path.join(snapshot, "config.json")))
        quant = self.config.get("quantization") or self.config.get("quantization_config")
        self.default_q = (quant["bits"], quant["group_size"])
        self.q = {k: (v["bits"], v["group_size"]) for k, v in quant.items() if isinstance(v, dict)}
        files = sorted(f for f in os.listdir(snapshot) if f.endswith(".safetensors"))
        self.handles = [safe_open(os.path.join(snapshot, f), framework="numpy") for f in files]
        self.where = {}
        for h in self.handles:
            for k in h.keys():
                self.where[k] = h
        self.cache = {}

    def raw(self, name):
        h = self.where[name]
        a = h.get_tensor(name)
        if a.dtype == ml_dtypes.bfloat16 or str(a.dtype) == "bfloat16":
            return mx.array(a.astype(np.float32))
        return mx.array(a)

    def has(self, name):
        return name in self.where or (name + ".weight") in self.where

    def dense(self, name):
        """A quantized or plain tensor at `name`, as float32."""
        if name in self.cache:
            return self.cache[name]
        if (name + ".scales") in self.where:
            bits, gs = self.q.get(name, self.default_q)
            w = self.raw(name + ".weight")
            sc = self.raw(name + ".scales")
            bi = self.raw(name + ".biases")
            out = mx.dequantize(w, sc, bi, group_size=gs, bits=bits).astype(mx.float32)
        elif (name + ".weight") in self.where:
            out = self.raw(name + ".weight").astype(mx.float32)
        else:
            out = self.raw(name).astype(mx.float32)
        mx.eval(out)
        self.cache[name] = out
        return out

    def embed_rows(self, ids):
        name = "model.embed_tokens"
        bits, gs = self.q.get(name, self.default_q)
        idx = mx.array(np.asarray(ids, dtype=np.int32))
        w = self.raw(name + ".weight")[idx]
        sc = self.raw(name + ".scales")[idx]
        bi = self.raw(name + ".biases")[idx]
        return mx.dequantize(w, sc, bi, group_size=gs, bits=bits).astype(mx.float32)


# ----------------------------------------------------------------------------- model


class Reference:
    def __init__(self, ck, sim=False, pie=()):
        self.ck = ck
        c = ck.config
        self.sim = sim
        self.pie = set(pie)
        self.layers = c["num_hidden_layers"]
        self.ratios = list(c["compress_ratios"])[: self.layers]
        self.hash_layers = c["num_hash_layers"]
        self.hidden = c["hidden_size"]
        self.heads = c["num_attention_heads"]
        self.head_dim = c["head_dim"]
        self.rd = c["qk_rope_head_dim"]
        self.hc = c["hc_mult"]
        self.hc_eps = c["hc_eps"]
        self.sinkhorn = c["hc_sinkhorn_iters"]
        self.eps = c["rms_norm_eps"]
        self.window = c["sliding_window"]
        self.top_k = c["num_experts_per_tok"]
        self.experts = c["n_routed_experts"]
        self.limit = c["swiglu_limit"]
        self.scaling = c["routed_scaling_factor"]
        self.groups = c["o_groups"]
        self.o_lora = c["o_lora_rank"]
        self.scale = self.head_dim**-0.5
        rs = c["rope_scaling"]
        self.max_seq = 4096
        # ROPE[0]: no compressor -> base theta, no YaRN. ROPE[1]: compressor layers -> the
        # compress theta WITH YaRN (official `Attention.__init__`; decode_engine `ROPE`).
        self.rope0 = rope_cos_sin(self.rd, self.max_seq, 0, c["rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"])
        self.rope1 = rope_cos_sin(
            self.rd, self.max_seq, rs["original_max_position_embeddings"], c["compress_rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"]
        )

    def W(self, name):
        return self.ck.dense(name)

    def layer(self, L, x4, ids):
        """x4: [s, hc, hidden] -> [s, hc, hidden]."""
        ck, n = self.ck, lambda k: f"model.layers.{L}.{k}"
        s = x4.shape[0]
        r = self.ratios[L]
        pos = mx.arange(s)
        cos, sin = (self.rope1 if (r and "rope" not in self.pie) else self.rope0)
        ccos, csin = self.rope0 if "comp_rope0" in self.pie else self.rope1

        y, post, comb = hc_pre(x4, self.W(n("attn_hc.fn")), self.W(n("attn_hc.scale")), self.W(n("attn_hc.base")), self.hc, self.eps, self.sinkhorn, self.hc_eps)
        xn = rmsnorm(y, self.W(n("attn_norm")), self.eps)

        q = rmsnorm(xn @ self.W(n("attn.wq_a")).T, self.W(n("attn.q_norm")), self.eps)
        q_lora = q
        q = (q @ self.W(n("attn.wq_b")).T).reshape(s, self.heads, self.head_dim)
        q = q * mx.rsqrt(q.square().mean(-1, keepdims=True) + self.eps)
        q = mx.concatenate([q[..., : -self.rd], apply_rotary_emb(q[..., -self.rd :], cos[pos], sin[pos])], axis=-1)

        kv = rmsnorm(xn @ self.W(n("attn.wkv")).T, self.W(n("attn.kv_norm")), self.eps)
        kv = mx.concatenate([kv[..., : -self.rd], apply_rotary_emb(kv[..., -self.rd :], cos[pos], sin[pos])], axis=-1)
        if self.sim:
            kv = mx.concatenate([act_quant_sim(kv[..., : -self.rd], 64), kv[..., -self.rd :]], axis=-1)

        # Keys: the per-token latent window, then every compressed row of a ratio-4 layer.
        comp = None
        if r == 4:
            comp = compressor_prefill(
                xn, self.W(n("attn.compressor.wkv")), self.W(n("attn.compressor.wgate")), self.W(n("attn.compressor.ape")),
                self.W(n("attn.compressor.norm")), ccos, csin, self.head_dim, r, self.rd, self.sim,
            )
        elif r and s >= r:
            raise SystemExit(f"layer {L} has ratio {r} and the sequence ({s}) reaches its first compressed row, which this reference does not model")
        i = np.arange(s)[:, None]
        j = np.arange(s)[None, :]
        win_mask = (j <= i) & (j >= i - (self.window - 1))
        if comp is not None and comp.shape[0] > 0:
            g = comp.shape[0]
            c = np.arange(g)[None, :]
            comp_mask = c < ((i + 1) // r)  # row c visible once its block has closed
            keys = mx.concatenate([kv, comp], axis=0)
            mask = np.concatenate([win_mask, comp_mask], axis=1)
        else:
            keys = kv
            mask = win_mask
        o = windowed_attention(q, keys, self.W(n("attn.attn_sink")), mx.array(mask), self.scale)
        if "no_orope" not in self.pie:
            # The value carries the key's rope (kv is both), so the output is un-roped.
            o = mx.concatenate([o[..., : -self.rd], apply_rotary_emb(o[..., -self.rd :], cos[pos], sin[pos], inverse=True)], axis=-1)

        # o-projection: block-diagonal over `o_groups` (official einsum "bsgd,grd->bsgr").
        wo_a, wo_b = self.W(n("attn.wo_a")), self.W(n("attn.wo_b"))
        og = o.reshape(s, self.groups, -1)
        if "ogroups" in self.pie:
            ob = og.sum(axis=1) @ wo_a.T
        else:
            wa = wo_a.reshape(self.groups, self.o_lora, -1)
            ob = mx.concatenate([og[:, g, :] @ wa[g].T for g in range(self.groups)], axis=-1)
        attn_out = ob @ wo_b.T
        h = hc_post(attn_out, x4, post, comb)

        y2, post2, comb2 = hc_pre(h, self.W(n("ffn_hc.fn")), self.W(n("ffn_hc.scale")), self.W(n("ffn_hc.base")), self.hc, self.eps, self.sinkhorn, self.hc_eps)
        xn2 = rmsnorm(y2, self.W(n("ffn_norm")), self.eps)

        # Router: sqrt-softplus scores; hash layers pick by table, weights still from scores.
        logits = xn2 @ self.W(n("ffn.gate")).T
        sc = np.array(mx.sqrt(mx.log1p(mx.exp(-mx.abs(logits))) + mx.maximum(logits, 0)))
        if L < self.hash_layers:
            tid = np.array(ck.raw(n("ffn.gate.tid2eid")))
            sel = tid[np.asarray(ids)]
        else:
            bias = np.array(self.W(n("ffn.gate.e_score_correction_bias")))
            sel = np.argsort(-(sc + bias[None, :]), axis=-1, kind="stable")[:, : self.top_k]
        w = np.take_along_axis(sc, sel, axis=-1)
        w = w / w.sum(-1, keepdims=True) * self.scaling
        if "hash" in self.pie and L < self.hash_layers:
            w = np.full_like(w, 1.0 / self.top_k)

        xn2_np = xn2
        gate_b, up_b, down_b = self.W(n("ffn.switch_mlp.gate_proj")), self.W(n("ffn.switch_mlp.up_proj")), self.W(n("ffn.switch_mlp.down_proj"))
        moe = mx.zeros((s, self.hidden), dtype=mx.float32)
        for e in range(self.experts):
            rows, slots = np.nonzero(sel == e)
            if len(rows) == 0:
                continue
            xe = xn2_np[mx.array(rows)]
            gg = mx.minimum(xe @ gate_b[e].T, self.limit)
            uu = mx.clip(xe @ up_b[e].T, -self.limit, self.limit)
            act = (gg * mx.sigmoid(gg)) * uu
            ye = (act @ down_b[e].T) * mx.array(w[rows, slots])[:, None]
            moe = moe.at[mx.array(rows)].add(ye)
        sg, su, sd = self.W(n("ffn.shared_experts.gate_proj")), self.W(n("ffn.shared_experts.up_proj")), self.W(n("ffn.shared_experts.down_proj"))
        g = mx.minimum(xn2 @ sg.T, self.limit)
        shared = ((g * mx.sigmoid(g)) * mx.clip(xn2 @ su.T, -self.limit, self.limit)) @ sd.T
        return hc_post(moe + shared, h, post2, comb2)

    def logits(self, ids):
        """Teacher-forced logits at every position: [len(ids), vocab]."""
        ck = self.ck
        emb = ck.embed_rows(ids)
        x4 = mx.repeat(emb[:, None, :], self.hc, axis=1)
        for L in range(self.layers):
            x4 = self.layer(L, x4, ids)
            mx.eval(x4)
        if "trunk" in self.pie:
            f = x4.sum(axis=1)
        else:
            f = hc_head(x4, self.W("model.hc_head.fn"), self.W("model.hc_head.scale"), self.W("model.hc_head.base"), self.eps, self.hc_eps)
        f = rmsnorm(f, self.W("model.norm"), self.eps)
        out = f @ self.W("lm_head").T
        mx.eval(out)
        return out


# ----------------------------------------------------------------------------- driver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.expanduser("~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/mini-l5-e16"))
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--sim", action="store_true", help="the official fp8 round-trip on cached kv and compressed rows")
    ap.add_argument("--pie", default="", help="comma list of pie deviations to emulate: ogroups,rope,rope_half,hash,trunk,no_orope,comp_rope0")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()
    pie = [p for p in args.pie.split(",") if p]
    ROPE_HALF[0] = "rope_half" in pie
    os.makedirs(args.out, exist_ok=True)
    ck = Checkpoint(args.snapshot)
    ref = Reference(ck, sim=args.sim, pie=pie)
    probes = json.load(open(args.probes))["probes"]
    print(f"reference over {ref.layers} layers, ratios {ref.ratios}, sim={args.sim}, pie={pie}")
    for probe in probes:
        name, ids = probe["name"], list(probe["ids"])
        t0 = time.time()
        tf = ref.logits(ids)
        tf_np = np.array(tf, dtype=np.float32)
        gen_rows = [tf_np[-1]]
        gen = []
        cur = list(ids)
        for _ in range(args.steps):
            nxt = int(np.argmax(gen_rows[-1]))
            gen.append(nxt)
            cur.append(nxt)
            gen_rows.append(np.array(ref.logits(cur), dtype=np.float32)[-1])
        gen_np = np.stack(gen_rows[:-1]) if args.steps > 0 else np.stack(gen_rows)
        # gen row k is the logits that chose gen[k]; the last computed row (after the final
        # step) is dropped so rows and tokens align one to one, matching the pie dump.
        tf_np.tofile(os.path.join(args.out, f"{name}.{args.tag}.tf.f32"))
        gen_np.tofile(os.path.join(args.out, f"{name}.{args.tag}.gen.f32"))
        json.dump(
            {"ids": ids, "argmax": tf_np.argmax(-1).tolist(), "gen": gen, "vocab": int(tf_np.shape[1])},
            open(os.path.join(args.out, f"{name}.{args.tag}.json"), "w"),
        )
        print(f"  {name}: {len(ids)} tokens, {args.steps} steps, {time.time() - t0:.1f}s, gen={gen[:12]}")


if __name__ == "__main__":
    main()
