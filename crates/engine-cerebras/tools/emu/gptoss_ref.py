"""gpt-oss in numpy f32, one token at a time (prompt included).

The golden for the wafer's gpt-oss program: every op in f32 from the
checkpoint (bf16 attention/router/embed/head widened, MXFP4 experts decoded
exactly: fp4 x 2^(scale - 127) is exact in f32). Attention sinks, the
sliding window (128, even layers), YaRN RoPE (neox halves), q/k/v/o biases,
the router's top-4 then softmax, clamped SwiGLU (alpha 1.702, limit 7) and
the down biases weighted by the routes. `--layers N` truncates the model.
"""
import argparse, json, math, os, struct
from collections import OrderedDict
import numpy as np

P = os.environ.get("GPTOSS_MODEL", "gpt-oss-20b")   # the checkpoint dir (main's default)
FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], np.float32)


class Ckpt:
    """Tensors by name off the safetensors files, read on demand."""

    def __init__(self, path=P):
        import glob
        self.where = {}
        for f in sorted(glob.glob(f"{path}/*.safetensors")):
            with open(f, "rb") as h:
                n = struct.unpack("<Q", h.read(8))[0]
                hdr = json.loads(h.read(n))
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.where[k] = (f, 8 + n, v)

    def raw(self, k):
        f, base, v = self.where[k]
        s, e = v["data_offsets"]
        with open(f, "rb") as h:
            h.seek(base + s)
            buf = h.read(e - s)
        return v["dtype"], v["shape"], buf

    def f32(self, k):
        dt, shape, buf = self.raw(k)
        if dt == "BF16":
            a = (np.frombuffer(buf, np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dt == "F32":
            a = np.frombuffer(buf, np.float32)
        else:
            raise ValueError(f"{k}: {dt}")
        return a.reshape(shape).copy()

    def u8(self, k):
        dt, shape, buf = self.raw(k)
        assert dt == "U8", (k, dt)
        return np.frombuffer(buf, np.uint8).reshape(shape)


def mxfp4(blocks, scales):
    """[..., G, 16] code bytes (low nibble first) and [..., G] E8M0 -> [..., G*32] f32."""
    lo = FP4[blocks & 0x0F]
    hi = FP4[blocks >> 4]
    v = np.stack([lo, hi], axis=-1).reshape(*blocks.shape[:-1], 32)
    v = v * np.ldexp(np.float32(1.0), scales.astype(np.int32) - 127)[..., None].astype(np.float32)
    return v.reshape(*blocks.shape[:-2], blocks.shape[-2] * 32).astype(np.float32)


def yarn_inv_freq(cfg):
    """HF's _compute_yarn_parameters (truncate false), in f32 like torch."""
    rs = cfg["rope_scaling"]
    base, dim = float(cfg["rope_theta"]), cfg["head_dim"]
    factor, bf, bs = rs["factor"], rs["beta_fast"], rs["beta_slow"]
    omax = rs["original_max_position_embeddings"]

    def corr(rot):
        return (dim * math.log(omax / (rot * 2 * math.pi))) / (2 * math.log(base))

    low, high = corr(bf), corr(bs)
    if rs.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1)
    pos_freqs = np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim))
    extra = np.float32(1.0) / pos_freqs
    inter = np.float32(1.0) / (np.float32(factor) * pos_freqs)
    ef = np.float32(1.0) - ramp
    inv = (inter * (1 - ef) + extra * ef).astype(np.float32)
    att = rs.get("attention_factor") or (0.1 * math.log(factor) + 1.0)
    return inv, np.float32(att)


def rms(x, w, eps):
    v = np.mean(x * x, dtype=np.float32)
    return (x / np.sqrt(v + np.float32(eps))).astype(np.float32) * w


class Model:
    def __init__(self, path=P, layers=None, cache=160):
        self.cfg = cfg = json.load(open(f"{path}/config.json"))
        self.ck = Ckpt(path)
        self.L = layers or cfg["num_hidden_layers"]
        self.H, self.KV, self.D = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.E, self.K = cfg["num_local_experts"], cfg["num_experts_per_tok"]
        self.I = cfg["intermediate_size"]
        self.win = cfg["sliding_window"]
        self.types = cfg["layer_types"]
        self.eps = cfg["rms_norm_eps"]
        self.limit, self.alpha = cfg["swiglu_limit"], 1.702
        self.inv_freq, self.att = yarn_inv_freq(cfg)
        g = self.ck.f32
        self.embed = g("model.embed_tokens.weight")
        self.norm = g("model.norm.weight")
        self.head = g("lm_head.weight")
        self.W = []
        for l in range(self.L):
            p = f"model.layers.{l}."
            w = {k: g(p + k) for k in (
                "input_layernorm.weight", "post_attention_layernorm.weight",
                "self_attn.q_proj.weight", "self_attn.q_proj.bias",
                "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                "self_attn.o_proj.weight", "self_attn.o_proj.bias", "self_attn.sinks",
                "mlp.router.weight", "mlp.router.bias",
                "mlp.experts.gate_up_proj_bias", "mlp.experts.down_proj_bias")}
            self.W.append(w)
        self.xcache = OrderedDict()
        self.cap = cache
        self.reset()

    def reset(self):
        self.k = [[] for _ in range(self.L)]
        self.v = [[] for _ in range(self.L)]
        self.pos = 0

    def expert(self, l, e):
        """(gate [I, hidden], up [I, hidden], down [hidden, I]) of expert e, f32."""
        key = (l, e)
        if key in self.xcache:
            self.xcache.move_to_end(key)
            return self.xcache[key]
        p = f"model.layers.{l}.mlp.experts."
        if not hasattr(self, "_mm"):
            self._mm = {}
        def mm(n):
            if (l, n) not in self._mm:
                f, base, v = self.ck.where[p + n]
                s, _ = v["data_offsets"]
                self._mm[(l, n)] = np.memmap(f, np.uint8, "r", base + s, tuple(v["shape"]))
            return self._mm[(l, n)]
        gu = mxfp4(np.asarray(mm("gate_up_proj_blocks")[e]), np.asarray(mm("gate_up_proj_scales")[e]))
        dn = mxfp4(np.asarray(mm("down_proj_blocks")[e]), np.asarray(mm("down_proj_scales")[e]))
        val = (np.ascontiguousarray(gu[0::2]), np.ascontiguousarray(gu[1::2]), dn)
        self.xcache[key] = val
        if len(self.xcache) > self.cap:
            self.xcache.popitem(last=False)
        return val

    def rope(self, x, pos):
        f = np.float32(pos) * self.inv_freq
        c, s = np.cos(f) * self.att, np.sin(f) * self.att
        h = self.D // 2
        a, b = x[..., :h], x[..., h:]
        return np.concatenate([a * c - b * s, b * c + a * s], axis=-1).astype(np.float32)

    def moe(self, l, x, trace=None):
        w = self.W[l]
        logits = w["mlp.router.weight"] @ x + w["mlp.router.bias"]
        top = np.argsort(-logits, kind="stable")[: self.K]
        tv = logits[top]
        p = np.exp(tv - tv.max())
        p = (p / p.sum()).astype(np.float32)
        out = np.zeros_like(x)
        gub = w["mlp.experts.gate_up_proj_bias"]
        db = w["mlp.experts.down_proj_bias"]
        for e, pe in zip(top, p):
            G, U, Dn = self.expert(l, int(e))
            gate = G @ x + gub[e, 0::2]
            up = U @ x + gub[e, 1::2]
            gate = np.minimum(gate, self.limit)
            up = np.clip(up, -self.limit, self.limit)
            glu = gate / (1 + np.exp(-gate * self.alpha))
            act = ((up + 1) * glu).astype(np.float32)
            out += pe * (Dn @ act + db[e])
        if trace is not None:
            trace.append({"layer": l, "experts": top.tolist(), "weights": p.tolist(),
                          "logits": logits.tolist()})
        return out.astype(np.float32)

    def step(self, tok, trace=None):
        pos = self.pos
        y = self.embed[tok].copy()
        H, KV, D = self.H, self.KV, self.D
        for l in range(self.L):
            w = self.W[l]
            x = rms(y, w["input_layernorm.weight"], self.eps)
            q = (w["self_attn.q_proj.weight"] @ x + w["self_attn.q_proj.bias"]).reshape(H, D)
            k = (w["self_attn.k_proj.weight"] @ x + w["self_attn.k_proj.bias"]).reshape(KV, D)
            v = (w["self_attn.v_proj.weight"] @ x + w["self_attn.v_proj.bias"]).reshape(KV, D)
            q, k = self.rope(q, pos), self.rope(k, pos)
            self.k[l].append(k)
            self.v[l].append(v)
            lo = 0
            if self.types[l] == "sliding_attention":
                lo = max(0, pos + 1 - self.win)
            Kc = np.stack(self.k[l][lo:], 1)  # [KV, T, D]
            Vc = np.stack(self.v[l][lo:], 1)
            g = H // KV
            qg = q.reshape(KV, g, D)
            sc = np.einsum("kgd,ktd->kgt", qg, Kc) * np.float32(D ** -0.5)
            sink = w["self_attn.sinks"].reshape(KV, g, 1)
            m = np.maximum(sc.max(-1, keepdims=True), sink)
            e = np.exp(sc - m)
            den = e.sum(-1, keepdims=True) + np.exp(sink - m)
            a = np.einsum("kgt,ktd->kgd", e / den, Vc).reshape(H * D).astype(np.float32)
            o = w["self_attn.o_proj.weight"] @ a + w["self_attn.o_proj.bias"]
            y = (y + o).astype(np.float32)
            x = rms(y, w["post_attention_layernorm.weight"], self.eps)
            y = (y + self.moe(l, x, trace)).astype(np.float32)
        self.pos += 1
        x = rms(y, self.norm, self.eps)
        return (self.head @ x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--golden", default=None)
    ap.add_argument("--prompt-ids", default=None, help="comma-separated ids")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    m = Model(layers=a.layers)
    g = json.load(open(a.golden)) if a.golden else None
    ids = g["prompt_ids"] if g else [int(t) for t in a.prompt_ids.split(",")]
    for t in ids:
        logits = m.step(t)
    first_logits = logits
    toks, margins = [], []
    for i in range(a.n):
        t = int(np.argmax(logits))
        srt = np.sort(logits)
        margins.append(float(srt[-1] - srt[-2]))
        toks.append(t)
        if i + 1 < a.n:
            logits = m.step(t)
    print("tokens", toks)
    print("min margin", min(margins))
    if g:
        same = sum(1 for x, y in zip(toks, g["tokens"]) if x == y)
        first = next((i for i, (x, y) in enumerate(zip(toks, g["tokens"])) if x != y), None)
        print(f"vs golden: {same}/{len(toks)} equal, first diff {first}")
    if a.out:
        json.dump({"prompt_ids": ids, "tokens": toks, "margins": margins,
                   "first_logits_top": np.argsort(-first_logits)[:8].tolist()}, open(a.out, "w"))


if __name__ == "__main__":
    main()
