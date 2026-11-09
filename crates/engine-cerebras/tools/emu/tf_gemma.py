"""tf_gemma.py <golden json> <mode,mode,...> <ntok> <outdir> [tag]  (env GEMMA_MODEL: the E2B checkpoint dir)

Teacher-forced emulation of the Gemma 4 E2B wafer kernel's number formats (host only) against an HF f32 golden.

  writes <outdir>/tf_<tag>_<mode>.json, and <outdir>/lse_<tag>.json from the ref pass

Own forward over prompt + golden tokens (full sequence, causal, eager), weights from the bf16 checkpoint in f32.
mode = base [_flag ...]:
  ref      f32 everywhere (the golden's arithmetic)
  f16acc   matvecs: x, W in IEEE f16, each weight PE's chunk accumulated in f16 (rounded after every FMA,
           sequential over the chunk), partials summed in f32. Chunks: hidden-input projections 12, down_proj 24
           (layers 0-14) / 48 (15-34), per_layer_projection 1, o_proj the kernel's column order (8 / 16 inputs)
  _dfr     deferred norm inputs: rr * (W f16(g base) + s W f16(g p)) at every hidden-input projection
  _kv16    q, k, v f16; scores f32 accumulation; probabilities f16; values f32 accumulation
  _ple16   the per-layer input rounded to f16 once
  _g16     MLP act: f16(gelu(f16 gate) * f16 up)        _g16full: every gelu intermediate in f16
  _h16     head f16(x_normed) . f16(W_embed), f32 accumulation
Logits compared after the final softcap 30 (monotonic)."""
import json, sys, time
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText

torch.set_num_threads(8)
torch.set_grad_enabled(False)
import os
P = os.environ["GEMMA_MODEL"]
G, modes, ntok, outdir = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), sys.argv[4]
tag = sys.argv[5] if len(sys.argv) > 5 else "g"
gold = json.load(open(G))
ntok = min(ntok, len(gold["tokens"]))
ids = gold["prompt_ids"] + gold["tokens"][:ntok]
NPR, T = len(gold["prompt_ids"]), len(gold["prompt_ids"]) + ntok
want = gold["tokens"][:ntok]

t0 = time.time()
m = AutoModelForImageTextToText.from_pretrained(P, dtype=torch.float32, attn_implementation="eager")
m.eval()
tm = m.model.language_model
cfg = tm.config
EPS = cfg.rms_norm_eps
NL, H, NH = cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads
print("loaded", type(m).__name__, round(time.time() - t0), "s", flush=True)


def rsq(x):
    return torch.pow(x.pow(2).mean(-1, keepdim=True) + EPS, -0.5)


def opermute(D):
    if D == 256:  # sliding, Dc 2: column p of copy c: dims (p, p+128) of each of its 4 heads
        return [(c * 4 + h) * D + d for c in range(2) for p in range(128) for h in range(4) for d in (p, p + 128)]
    return [(c * 4 + h) * D + d for c in range(2) for p in range(128) for h in range(4)
            for d in (2 * p, 2 * p + 256, 2 * p + 1, 2 * p + 257)]


class Lin:
    """one projection: f32 weight (ref) and the f16 chunked layout (f16acc)"""
    def __init__(self, lin, chunk, name, perm=None):
        self.w = lin.weight.detach().float()           # (out, K)
        self.name, self.chunk = name, chunk
        wt = self.w.t()
        if perm is not None:
            self.perm = torch.tensor(perm)
            wt = wt[self.perm]
        else:
            self.perm = None
        K, out = wt.shape
        assert K % chunk == 0, (name, K, chunk)
        self.nch = K // chunk
        self.w16 = wt.half().reshape(self.nch, chunk, out).contiguous()

    def ref(self, x):
        return x @ self.w.t()

    def f16acc(self, x, tagp=""):
        if self.perm is not None:
            x = x[:, self.perm]
        xh = x.half().reshape(x.shape[0], self.nch, self.chunk)
        out = self.w16.shape[2]
        tb = max(1, (1 << 24) // (self.nch * out))
        res, mp, inf = [], 0.0, 0
        for t in range(0, x.shape[0], tb):
            xb = xh[t:t + tb]
            acc = torch.zeros(xb.shape[0], self.nch, out, dtype=torch.half)
            for k in range(self.chunk):
                acc.addcmul_(xb[:, :, k, None], self.w16[None, :, k, :])
                a = acc.abs()
                fin = torch.isfinite(a)
                if not fin.all():
                    inf += int((~fin).sum()); a = a[fin]
                if a.numel():
                    mp = max(mp, float(a.max()))
            res.append(acc.float().sum(1))
        key = self.name + tagp
        STATS["max_abs_partial"][key] = max(STATS["max_abs_partial"].get(key, 0.0), mp)
        if inf:
            STATS["f16_overflow"][key] = STATS["f16_overflow"].get(key, 0) + inf
        return torch.cat(res, 0)


LAYERS = []
for i, L in enumerate(tm.layers):
    a = L.self_attn
    D = a.head_dim
    d = {"D": D, "sliding": a.is_sliding, "shared": a.is_kv_shared_layer, "store": a.store_full_length_kv,
         "type": cfg.layer_types[i],
         "q": Lin(a.q_proj, 12, "q_proj"), "o": Lin(a.o_proj, len(opermute(D)) // 256, "o_proj", opermute(D)),
         "qn": a.q_norm.weight.float(), "g_in": L.input_layernorm.weight.float(),
         "w_pa": L.post_attention_layernorm.weight.float(), "g_ff": L.pre_feedforward_layernorm.weight.float(),
         "w_pf": L.post_feedforward_layernorm.weight.float(), "w_ppl": L.post_per_layer_input_norm.weight.float(),
         "ls": L.layer_scalar.float(),
         "gate": Lin(L.mlp.gate_proj, 12, "gate_proj"), "up": Lin(L.mlp.up_proj, 12, "up_proj"),
         "down": Lin(L.mlp.down_proj, L.mlp.intermediate_size // 256, "down_proj"),
         "pg": Lin(L.per_layer_input_gate, 12, "per_layer_input_gate"),
         "pp": Lin(L.per_layer_projection, 1, "per_layer_projection")}
    if not a.is_kv_shared_layer:
        d["k"] = Lin(a.k_proj, 12, "k_proj"); d["v"] = Lin(a.v_proj, 12, "v_proj"); d["kn"] = a.k_norm.weight.float()
    LAYERS.append(d)
WE = tm.embed_tokens.weight.detach().float()            # tied head (bf16 values in f32)
WE16 = None
NORMW = tm.norm.weight.float()
SOFTCAP = cfg.final_logit_softcapping
print("prepped", round(time.time() - t0), "s", "chunks o_proj", LAYERS[0]["o"].chunk, LAYERS[4]["o"].chunk,
      "down", LAYERS[0]["down"].chunk, LAYERS[20]["down"].chunk, flush=True)

idt = torch.tensor([ids])
pos = torch.arange(T)[None]
E = tm.embed_tokens(idt)[0].float()
PLE = tm.project_per_layer_inputs(E[None], tm.get_per_layer_inputs(idt, E[None]))[0].float()   # (T, NL, 256)
ROPE = {lt: tuple(t[0].float() for t in tm.rotary_emb(E[None], pos, lt)) for lt in set(cfg.layer_types)}
qi, kj = torch.arange(T)[:, None], torch.arange(T)[None, :]
MASK = {"full_attention": kj <= qi, "sliding_attention": (kj <= qi) & (qi - kj < cfg.sliding_window)}


def rot_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), -1)


def rope(x, lt):  # x (T, h, D)
    c, s = ROPE[lt]
    return x * c[:, None] + rot_half(x) * s[:, None]


C16 = torch.tensor([1.5957691216, 0.044715, 1.0], dtype=torch.half)


def gelu16full(x, u):
    c, k3, one = C16
    x = x.half(); u = u.half()
    x2 = x * x; x3 = x2 * x; t = k3 * x3; a = x + t; y = c * a
    e = torch.exp(-y); dd = one + e; sg = one / dd          # each op on f16 tensors rounds to f16
    return ((x * sg) * u).float()


def gelu16k(g, u):
    """The f16 kernel form (ctl_cy_gemma.csl GELU16, proposed), op for op: y' = g (1 + 0.044715 g^2) 1.5958 log2e in
    f32; then f16: e = 2^-|y'| by silu16's machinery (clamp -15, m = t + 1536, 2^r degree-3, 2^n by the exponent
    bits: 0 at n = -15), 1/(1+e) by the quadratic and one Newton step, act = max(G e, G) U q."""
    f32 = torch.float32
    e = g * g; e = e * 0.044715; e = e + 1.0; e = e * g
    y = e * torch.tensor(1.5957691216057308 * 1.4426950408889634, dtype=f32)
    H = lambda v: torch.as_tensor(v, dtype=torch.half)
    Y, G, U = y.half(), g.half(), u.half()
    T = -Y.abs()
    T = torch.maximum(T, H(-15.0))
    M = T + H(1536.0)
    Nn = M - H(1536.0)
    R = T - Nn
    P = R * H(0.05516966520599117); P = P + H(0.24260644929856207); P = P * R
    P = P + H(0.6932608957863087); P = P * R; P = P + H(0.9999283788785166)
    n = Nn.float()
    pw = torch.where(n <= -15, torch.zeros_like(n), torch.pow(2.0, n)).half()
    E = P * pw
    D = E + H(1.0)
    Q = D * H(0.323112507040746); Q = Q + H(-1.4540415401729079); Q = Q * D; Q = Q + H(2.1207294161729284)
    W = D * Q; W = H(2.0) - W; Q = Q * W
    A = E * G; A = torch.maximum(A, G); A = A * U; A = A * Q
    return A.float()


def gelu16h(g, u, variant):
    """Hybrids of gelu16k: A f16 e and 1/(1+e), the products max(g, g e) u q in f32; B f16 e only, 1/(1+e) in f32
    (vsilu's quadratic + two Newton steps) and the products in f32."""
    f32 = torch.float32
    e = g * g; e = e * 0.044715; e = e + 1.0; e = e * g
    y = e * torch.tensor(1.5957691216057308 * 1.4426950408889634, dtype=f32)
    H = lambda v: torch.as_tensor(v, dtype=torch.half)
    Y = y.half()
    T = torch.maximum(-Y.abs(), H(-15.0))
    M = T + H(1536.0); Nn = M - H(1536.0); R = T - Nn
    P = R * H(0.05516966520599117); P = P + H(0.24260644929856207); P = P * R
    P = P + H(0.6932608957863087); P = P * R; P = P + H(0.9999283788785166)
    n = Nn.float()
    E = P * torch.where(n <= -15, torch.zeros_like(n), torch.pow(2.0, n)).half()
    if variant == "A":
        D = E + H(1.0)
        Q = D * H(0.323112507040746); Q = Q + H(-1.4540415401729079); Q = Q * D; Q = Q + H(2.1207294161729284)
        W = D * Q; W = H(2.0) - W; Q = (Q * W).float()
    else:
        d = E.float() + 1.0
        Q = d * 0.323112507040746; Q = Q - 1.4540415401729079; Q = Q * d; Q = Q + 2.1207294161729284
        for _ in range(2):
            W = 2.0 - d * Q; Q = Q * W
    Ef = E.float()
    a = torch.maximum(g * Ef, g)
    return a * u * Q


def run(mode):
    base_, *fl = mode.split("_")
    fl = set(fl)
    f16 = base_ == "f16acc"
    dfr = f16 and "dfr" in fl
    kv16, ple16 = "kv16" in fl, "ple16" in fl
    ple = PLE.half().float() if ple16 else PLE
    lse_all = []

    def mv(lin, x, tagp=""):
        return lin.f16acc(x, tagp) if f16 else lin.ref(x)

    def proj_n(lins, g, base, p, s):
        """hidden-input projections of the normed x_true = base + s p (g None: no pre-norm)"""
        xt = base if p is None else base + s * p
        if dfr:
            rr = rsq(xt) if g is not None else 1.0
            gg = g if g is not None else 1.0
            outs = []
            for lin in lins:
                y = mv(lin, gg * base)
                if p is not None:
                    y = y + s * mv(lin, gg * p, "_p")
                outs.append(rr * y)
            return outs, xt
        xn = (xt * rsq(xt)) * g if g is not None else xt
        return [mv(lin, xn) for lin in lins], xt

    base, p, s = E, None, None
    shared = {}
    for li, d in enumerate(LAYERS):
        D, lt = d["D"], d["type"]
        lins = [d["q"]] if d["shared"] else [d["q"], d["k"], d["v"]]
        outs, xt = proj_n(lins, d["g_in"], base, p, s)
        q = rope((outs[0].view(T, NH, D) * rsq(outs[0].view(T, NH, D))) * d["qn"], lt)
        if d["shared"]:
            k, v = shared[lt]
        else:
            k = rope((outs[1].view(T, 1, D) * rsq(outs[1].view(T, 1, D))) * d["kn"], lt)[:, 0]
            v = (outs[2].view(T, 1, D) * rsq(outs[2].view(T, 1, D)))[:, 0]
            if kv16:
                k, v = k.half().float(), v.half().float()
            if d["store"]:
                shared[lt] = (k, v)
        if kv16:
            q = q.half().float()
        sc = torch.einsum("thd,sd->hts", q, k)        # scale 1
        sc = sc.masked_fill(~MASK[lt][None], float("-inf"))
        if base_ == "ref":
            lse_all.append(torch.logsumexp(sc, -1))       # (NH, T)
        pr = torch.softmax(sc, -1)
        if kv16:
            pr = pr.half().float()
        att = torch.einsum("hts,sd->thd", pr, v).reshape(T, NH * D)
        o = mv(d["o"], att)
        s1, p1 = rsq(o), d["w_pa"] * o
        # MLP: base = exact x_true of the attention input, pending = post-attn-normed o
        (gt, up), xa_base = proj_n([d["gate"], d["up"]], d["g_ff"], xt, p1, s1)
        x_attn = xt + s1 * p1
        if "g16hA" in fl or "g16hB" in fl:
            act = gelu16h(gt, up, "A" if "g16hA" in fl else "B")
        elif "g16k" in fl:
            act = gelu16k(gt, up)
        elif "g16full" in fl:
            act = gelu16full(gt, up)
        elif "g16" in fl:
            act = (F.gelu(gt.half().float(), approximate="tanh") * up.half().float()).half().float()
        else:
            act = F.gelu(gt, approximate="tanh") * up
        dn = mv(d["down"], act)
        s2, p2 = rsq(dn), d["w_pf"] * dn
        (pg,), x_mlp = proj_n([d["pg"]], None, x_attn, p2, s2)
        hp = mv(d["pp"], F.gelu(pg, approximate="tanh") * ple[:, li])
        s3, p3 = rsq(hp), d["w_ppl"] * hp
        ls = d["ls"]
        base, p, s = ls * x_mlp, ls * p3, s3
    xf = base + s * p
    xn = (xf * rsq(xf)) * NORMW
    xn = xn[NPR - 1:NPR - 1 + ntok]
    if "h16" in fl:
        global WE16
        if WE16 is None:
            WE16 = WE.half().float()
        lg = xn.half().float() @ WE16.t()
    else:
        lg = xn @ WE.t()
    lg = torch.tanh(lg / SOFTCAP) * SOFTCAP
    return lg, lse_all


def lse_redo(lse_all):
    lo, hi = -5.545, 10.397
    res = []
    for li, l in enumerate(lse_all):
        prev = torch.cat([torch.zeros(l.shape[0], 1), l[:, :-1]], 1)
        dl = l - prev
        bad = (dl < lo) | (dl >= hi)
        res.append({"layer": li, "type": cfg.layer_types[li], "redo_total": int(bad.sum()),
                    "redo_decode": int(bad[:, NPR:].sum()), "redo_at_t0": int(bad[:, 0].sum()),
                    "redo_tokens_any_head": int(bad.any(0).sum()), "redo_tokens_any_head_decode": int(bad[:, NPR:].any(0).sum()),
                    "per_head": bad.sum(1).tolist(), "dlse_min": float(dl[:, 1:].min()),
                    "dlse_max": float(dl[:, 1:].max()), "lse_t0_max": float(l[:, 0].max())})
    return res


refpred, refmarg = None, None
gm = gold.get("margins")
for mode in modes:
    STATS = {"max_abs_partial": {}, "f16_overflow": {}}
    t1 = time.time()
    if mode == "hf":
        pred, lse_all = m(idt, use_cache=False).logits[0, NPR - 1:NPR - 1 + ntok].float(), None
    else:
        pred, lse_all = run(mode)
    top = torch.topk(pred, 2, -1)
    got = top.indices[:, 0].tolist()
    marg = (top.values[:, 0] - top.values[:, 1]).tolist()
    if mode == "ref":
        refpred, refmarg = pred.clone(), marg
        json.dump({"T": T, "NPR": NPR, "window": [-5.545, 10.397], "layers": lse_redo(lse_all)},
                  open(f"{outdir}/lse_{tag}.json", "w"), indent=1)
    mis = [{"i": i, "got": g, "want": w_, "golden_margin": round(gm[i], 4) if gm else None,
            "our_margin": round(marg[i], 4), "logit_gap": round(float(pred[i, g] - pred[i, w_]), 4)}
           for i, (g, w_) in enumerate(zip(got, want)) if g != w_]
    tm_ = gm if gm else refmarg
    tight = sorted(range(ntok), key=lambda i: tm_[i])[:5]

    def gap(i):
        row = pred[i].clone(); gw = float(row[want[i]]); row[want[i]] = -1e30
        return gw - float(row.max())
    rep = {"mode": mode, "tag": tag, "ntok": ntok, "n_mismatch": len(mis), "mismatches": mis,
           "tight_source": "golden margins" if gm else "ref-mode margins (golden has min_margin only)",
           "tight": [{"i": i, "golden_margin": round(tm_[i], 4), "gap": round(gap(i), 4)} for i in tight],
           "min_gap": round(min(gap(i) for i in range(ntok)), 4), "stats": STATS, "seconds": round(time.time() - t1)}
    if refpred is not None:
        dd = pred - refpred
        rep["vs_ref"] = {"rms": float(dd.pow(2).mean().sqrt()), "max": float(dd.abs().max()),
                         "argmax_diff_ref": int((pred.argmax(-1) != refpred.argmax(-1)).sum())}
    json.dump(rep, open(f"{outdir}/tf_{tag}_{mode}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("mismatches",)}), flush=True)
