"""The forwards: the torch function the interpreter runs for each Poem op of
a dense model, by the op's `group.name`.

Each forward reads its operands from a replay (`run.get`) and writes its
outputs (`run.put`). Values are `[rows, T, width]`: each batch row is one lane
of a training prefill at positions 0..T-1. Products accumulate in f32 and land
at the value's dtype, as the engines do. A forward refuses, naming what it
cannot run, any option it has no torch form for.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def need(holds: bool, what: str) -> None:
    """Refuse an op option the trainer has no torch forward for."""
    if not holds:
        raise NotImplementedError(f"the trainer has no torch forward for {what}")


def _embed(run, a):
    need(a["weights"] is None and a["scales"] is None and not a["concat"], "a weighted or concatenated embed")
    run.put(a["y"], F.embedding(run.get(a["ids"]), run.get(a["table"])).to(run.dtype))


def _rmsnorm(run, a):
    weight = None
    if a["weight"] != "None":
        (kind, wid), = a["weight"].items()
        weight = (run.get(wid), 1.0 if kind == "PlusOne" else 0.0)
    run.put(a["y"], rmsnorm(run.get(a["x"]), weight, a["axis"], a["eps"]))


def _rmsnorm_gated(run, a):
    # Each head normed, times the weight, times act(gate) (Qwen3.5's GDN norm).
    x, gate = run.get(a["x"]), run.get(a["gate"])
    weight = None if a["weight"] is None else (run.get(a["weight"]), 0.0)
    y = rmsnorm(x.float(), weight, {"Head": a["head_dim"]}, a["eps"])
    acts = {"Silu": F.silu, "Sigmoid": torch.sigmoid}
    run.put(a["y"], (y * acts[a["act"]](gate.float())).to(x.dtype))


def _matmul(run, a):
    need(a["quant"] is None and a["scales"] is None and a["input_scales"] is None, "a quantized matmul")
    run.put(a["y"], F.linear(run.get(a["act"]), run.get(a["w"])))


def _lora_correct(run, a):
    # y += B[0] (A[0] x), the waist (A x) held at f32 as the Metal kernel holds it.
    x, A, B, y = run.get(a["x"]), run.get(a["bank_a"]), run.get(a["bank_b"]), run.get(a["y"])
    waist = F.linear(x.float(), A[0].float())
    run.put(a["y_out"], (y.float() + F.linear(waist, B[0].float())).to(y.dtype))


def _rope(run, a):
    pos = run.get(a["positions"])
    run.put(a["q_out"], rope(run.get(a["q"]), pos, a["head_dim"], a["theta"], a["turn"], a["freqs"]))
    if a["k"] is not None:
        run.put(a["k_out"], rope(run.get(a["k"]), pos, a["head_dim"], a["theta"], a["turn"], a["freqs"]))


def _kv_append(run, a):
    cache = run.interp.values[a["cache"]]["def"]["Cache"]
    run.kv[cache] = (run.get(a["k"]), run.get(a["v"]) if a["v"] is not None else run.get(a["k"]))


def _plan(run, a):
    run.vals[a["plan"]] = None


def _prefill(run, a):
    need(a["mask"] is None and a["causal"], "a masked or non-causal prefill")
    _attend(run, a)


def _decode(run, a):
    # Each readout row's query against its lane's keys, as one token's: every
    # row is a readout row here, so each reads the keys up to its position.
    need(run.get(a["q"]).shape[:2] == run.shape, "a decode over some rows of a prefill")
    _attend(run, a)


def _attend(run, a):
    for key in ("selection", "bias", "floor", "selected", "rel"):
        need(a.get(key) is None, f"attention with `{key}`")
    k, v = run.kv[run.interp.values[a["cache"]]["def"]["Cache"]]
    q, hd = run.get(a["q"]), a["head_dim"]
    B, T = run.shape
    qh = q.reshape(B, T, -1, hd).transpose(1, 2)
    kh = k.reshape(B, T, -1, hd).transpose(1, 2)
    vh = v.reshape(B, T, -1, hd).transpose(1, 2)
    window = a["window"]
    if window is not None and window < T:
        # Each query reads the `window` keys that end at its own position.
        at = torch.arange(T, device=q.device)
        gap = at[:, None] - at[None, :]
        mask = (gap >= 0) & (gap < window)
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, scale=a["sm_scale"], enable_gqa=True)
    else:
        o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=a["sm_scale"], enable_gqa=True)
    run.put(a["o"], o.transpose(1, 2).reshape(B, T, -1))


def _residual_add(run, a):
    need(a["gate"] is None, "a gated residual add")
    run.put(a["y_out"], run.get(a["x"]) + run.get(a["y"]))


def _binary(run, a):
    x, y = run.get(a["x"]), run.get(a["y"])
    ops = {"Add": torch.add, "Mul": torch.mul, "Div": torch.div}
    need(a["op"] in ops, f"binary `{a['op']}`")
    run.put(a["z"], ops[a["op"]](x.float(), y.float()).to(x.dtype))


def _add_bias(run, a):
    run.put(a["out_out"], run.get(a["out"]) + run.get(a["bias"]))


def _act(run, a):
    # act(x * k): k the op's scale, the first element of the scalar `s`, or 1.
    x = run.get(a["x"])
    if a["scale"] is not None:
        k = a["scale"]
    elif a["s"] is not None:
        k = run.get(a["s"]).float().reshape(-1)[0]
    else:
        k = 1.0
    act = a["act"]
    kind = act if isinstance(act, str) else next(iter(act))
    acts = {
        "Identity": lambda v: v,
        "Silu": F.silu,
        "Gelu": lambda v: F.gelu(v, approximate="tanh" if act["Gelu"]["tanh"] else "none"),
        "Tanh": torch.tanh,
        "Sqrt": torch.sqrt,
    }
    need(kind in acts, f"activation `{kind}`")
    run.put(a["x_out"], acts[kind](x.float() * k).to(x.dtype))


def _softcap(run, a):
    x = run.get(a["x"])
    run.put(a["x_out"], softcap(x.float(), a["cap"]).to(x.dtype))


def _gate_sigmoid_mul(run, a):
    need(a["heads"] is None, "a sigmoid gate of one logit a head")
    x = run.get(a["x"])
    run.put(a["x_out"], (x.float() * torch.sigmoid(run.get(a["gate"]).float())).to(x.dtype))


def _glu(run, a):
    need(a["weights"] is None and a["limit"] is None, "a weighted or limited glu")
    if a["packed"] is not None:
        p = run.get(a["packed"])
        gate, up = p[..., : a["intermediate"]], p[..., a["intermediate"] :]
    else:
        gate, up = run.get(a["gate"]), run.get(a["up"])
    acts = {"Swiglu": F.silu, "GegluTanh": lambda g: F.gelu(g, approximate="tanh")}
    need(a["act"] in acts, f"glu `{a['act']}`")
    run.put(a["y"], (acts[a["act"]](gate.float()) * up.float()).to(gate.dtype))


def _gelu_tanh(run, a):
    x = run.get(a["x"])
    run.put(a["y"], F.gelu(x.float(), approximate="tanh").to(x.dtype))


def _rows(run, a):
    rows, x = run.get(a["rows"]), run.get(a["x"])
    run.put(a["y"], x if rows is None else x.reshape(-1, x.shape[-1])[rows])


def _select(run, a):
    # Layer `layer`'s `width` columns of a stacked table (Gemma's per-layer inputs).
    at = a["layer"] * a["width"]
    run.put(a["y"], run.get(a["table"])[..., at : at + a["width"]])


def _concat_cols(run, a):
    run.put(a["y"], torch.cat([run.get(x) for x in a["xs"]], dim=-1))


def _split_q_gate(run, a):
    # Each head's packed row is its query, then its output gate.
    p, hd = run.get(a["packed"]), a["head_dim"]
    heads = p.reshape(*p.shape[:-1], -1, 2 * hd)
    run.put(a["q"], heads[..., :hd].reshape(*p.shape[:-1], -1))
    run.put(a["gate"], heads[..., hd:].reshape(*p.shape[:-1], -1))


FORWARDS = {
    "Gather.Embed": _embed,
    "Gather.Rows": _rows,
    "Gather.Select": _select,
    "Gather.ConcatCols": _concat_cols,
    "Gather.SplitQGate": _split_q_gate,
    "Elemwise.Rmsnorm": _rmsnorm,
    "Elemwise.RmsnormGated": _rmsnorm_gated,
    "Elemwise.Rope": _rope,
    "Elemwise.ResidualAdd": _residual_add,
    "Elemwise.Binary": _binary,
    "Elemwise.AddBias": _add_bias,
    "Elemwise.Act": _act,
    "Elemwise.Softcap": _softcap,
    "Elemwise.GateSigmoidMul": _gate_sigmoid_mul,
    "Linear.Matmul": _matmul,
    "Linear.LoraCorrect": _lora_correct,
    "Linear.Glu": _glu,
    "Linear.GeluTanh": _gelu_tanh,
    "Attn.KvAppend": _kv_append,
    "Attn.PlanPrefill": _plan,
    "Attn.PlanDecode": _plan,
    "Attn.Prefill": _prefill,
    "Attn.Decode": _decode,
}
"""Each dense op's forward, by `group.name`."""


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """`cap * tanh(x / cap)`."""
    return cap * torch.tanh(x / cap)


def rope(x, pos, head_dim, theta, turn, freqs):
    """Rotate each head of x [..., T, H*hd] at positions pos [..., T]."""
    *lead, w = x.shape
    xs = x.float().reshape(*lead, w // head_dim, head_dim)
    (kind, cfg), = turn.items()
    if kind == "Full":
        pairs, span, inter = head_dim // 2, head_dim // 2, cfg["interleaved"]
    elif kind == "Partial":
        pairs, span, inter = cfg["rotary_dim"] // 2, cfg["span"], False
    else:
        raise NotImplementedError(f"the trainer has no torch forward for rope turn `{kind}`")
    if freqs is not None:
        inv = torch.tensor(freqs["inv_freq"], dtype=torch.float32, device=x.device)[:pairs]
        mscale = freqs["mscale"]
    else:
        inv = theta ** (-torch.arange(pairs, dtype=torch.float32, device=x.device) / span)
        mscale = 1.0
    ang = pos.float()[..., None] * inv
    cos = (ang.cos() * mscale)[..., None, :]
    sin = (ang.sin() * mscale)[..., None, :]
    out = xs.clone()
    if inter:
        a, b = xs[..., 0 : 2 * pairs : 2], xs[..., 1 : 2 * pairs : 2]
        out[..., 0 : 2 * pairs : 2] = a * cos - b * sin
        out[..., 1 : 2 * pairs : 2] = b * cos + a * sin
    else:
        a, b = xs[..., :pairs], xs[..., span : span + pairs]
        out[..., :pairs] = a * cos - b * sin
        out[..., span : span + pairs] = b * cos + a * sin
    return out.reshape(*lead, w).to(x.dtype)


def rmsnorm(x, weight, axis, eps):
    """`x / sqrt(mean(x^2) + eps) * (w + plus)` over each row, or over each
    head of width d when `axis` is `{"Head": d}`."""
    xf = x.float()
    shape = None
    if axis != "Row":
        (kind, d), = axis.items()
        if kind != "Head":
            raise NotImplementedError(f"the trainer has no torch forward for rmsnorm axis {axis}")
        shape = xf.shape
        xf = xf.reshape(*shape[:-1], shape[-1] // d, d)
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        w, plus = weight
        y = y * (w.float() + plus)
    if shape is not None:
        y = y.reshape(shape)
    return y.to(x.dtype)
