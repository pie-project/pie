"""The interpreter: it replays a `poem_ir::Trace` in PyTorch, so autograd
differentiates the very forward pie serves.

Guards in the trace test bits of a lane-class word. The interpreter picks one
word per run and takes every arm whose guard holds, as
`crates/engine-cpu/src/forward.rs` does. A training run is a full-sequence
causal prefill: each batch row is one lane at positions 0..T-1 with an empty
cache. Adapted is on when the stanzas apply and off for the plain arm; Masked,
SingleToken, Scores and Drafts are off. Paged-KV ops collapse to "keep this
step's k and v". Every adapted row reads slot 0, so the trainer hands over
one-slot banks.

The run stops at the logits matmul and returns its operands, so the head can
be computed in chunks. Any op the interpreter does not know raises and names
the op.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class Interpreter:
    """One trace, ready to replay."""

    trace: dict
    """The `poem_ir::Trace` as parsed JSON."""
    values: list[dict]
    """The trace's values, indexed by value id."""
    facts: dict[str, int]
    """The word bit of each builtin fact, such as Adapted."""
    nodes: list[tuple]
    """Each node as (op name, torch forward or None, args, guard)."""
    inputs: dict[int, str]
    """The runtime input each input value reads, by value id."""

    def __init__(self, trace: dict):
        self.trace = trace
        self.values = trace["values"]
        self.facts = {
            fact["Builtin"]: bit for bit, fact in enumerate(trace["facts"]["table"]) if isinstance(fact, dict) and "Builtin" in fact
        }
        self.nodes = []
        for node in trace["nodes"]:
            (group, inner), = node["op"].items()
            (op, args), = inner.items() if isinstance(inner, dict) else ((inner, {}),)
            self.nodes.append((f"{group}.{op}", _OPS.get(f"{group}.{op}"), args, node["guard"]))
        self.inputs = {
            vid: v["def"]["Input"] for vid, v in enumerate(self.values) if isinstance(v["def"], dict) and "Input" in v["def"]
        }

    def word(self, adapted: bool) -> int:
        """The lane-class word of a training prefill, with Adapted on or off."""
        if not adapted:
            return 0
        if "Adapted" not in self.facts:
            raise ValueError("this trace has no Adapted fact, so no stanza can attach")
        return 1 << self.facts["Adapted"]

    def hidden(self, params: dict, tokens: torch.Tensor, *, adapted: bool, dtype: torch.dtype
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the trace over `tokens` [B, T] and return the logits matmul's
        operands: the readout rows [B, T, D] and the head weight [V, D]."""
        run = _Run(self, params, self.word(adapted), dtype, tokens)
        for name, op, a, guard in self.nodes:
            if not _holds(guard, run.word):
                run.skip(a)
                continue
            if name == "Linear.Matmul" and a["logits"]:
                return run.get(a["act"]), run.get(a["w"])
            if op is None:
                raise NotImplementedError(f"the trainer has no torch forward for `{name}`")
            op(run, a)
        raise ValueError("the trace has no logits matmul")


class _Run:
    """The values of one replay."""

    def __init__(self, interp: Interpreter, params: dict, word: int, dtype: torch.dtype, tokens: torch.Tensor):
        self.interp = interp
        self.params = params
        self.word = word
        self.dtype = dtype
        self.vals: dict[int, object] = {}
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        B, T = tokens.shape
        self.shape = (B, T)
        for vid, inp in interp.inputs.items():
            if inp == "Tokens":
                self.vals[vid] = tokens
            elif inp == "Positions":
                self.vals[vid] = torch.arange(T, device=tokens.device).expand(B, T)
            elif inp == "AdapterRoutes":
                self.vals[vid] = torch.zeros_like(tokens)
            else:
                self.vals[vid] = None  # readout rows (all rows), paged geometry, masks

    def get(self, vid):
        values = self.interp.values
        while True:
            d = values[vid]["def"]
            if isinstance(d, dict) and "Merge" in d:
                for arm, g in d["Merge"]:
                    if _holds(g, self.word):
                        vid = arm
                        break
                else:
                    raise RuntimeError(f"no arm of merge %{vid} holds")
                continue
            if isinstance(d, dict) and "Weight" in d:
                return self.params[self.interp.trace["params"][d["Weight"]]["name"]]
            return self.vals[vid]

    def put(self, vid, x):
        # A bf16-typed value rounds to bf16 when the run is bf16; an f32 run
        # keeps every value f32 (the reference mode for parity).
        if self.dtype == torch.bfloat16 and isinstance(x, torch.Tensor) and x.is_floating_point():
            if self.interp.values[vid]["ty"].get("Tensor", {}).get("dtype") == "bf16":
                x = x.to(torch.bfloat16)
        self.vals[vid] = x

    def skip(self, a):
        # A skipped in-place op leaves its operand as it found it: each
        # `<f>_out` output (aliasing field `<f>`) reads as `<f>`.
        for field, vid in a.items():
            src = a.get(field[:-4]) if field.endswith("_out") else None
            if isinstance(vid, int) and src is not None:
                try:
                    self.vals[vid] = self.get(src)
                except KeyError:
                    pass


def _holds(g, word: int) -> bool:
    if g == "Always":
        return True
    (kind, arg), = g.items()
    if kind == "Fact":
        return bool(word >> arg & 1)
    if kind == "Not":
        return not _holds(arg, word)
    if kind == "And":
        return _holds(arg[0], word) and _holds(arg[1], word)
    if kind == "Or":
        return _holds(arg[0], word) or _holds(arg[1], word)
    raise ValueError(f"guard {g}")


def _need(holds: bool, what: str):
    if not holds:
        raise NotImplementedError(f"the trainer has no torch forward for {what}")


def _embed(run, a):
    _need(a["weights"] is None and a["scales"] is None and not a["concat"], "a weighted or concatenated embed")
    run.put(a["y"], F.embedding(run.get(a["ids"]), run.get(a["table"])).to(run.dtype))


def _rmsnorm(run, a):
    weight = None
    if a["weight"] != "None":
        (kind, wid), = a["weight"].items()
        weight = (run.get(wid), 1.0 if kind == "PlusOne" else 0.0)
    run.put(a["y"], rmsnorm(run.get(a["x"]), weight, a["axis"], a["eps"]))


def _matmul(run, a):
    _need(a["quant"] is None and a["scales"] is None and a["input_scales"] is None, "a quantized matmul")
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
    for key in ("selection", "bias", "floor", "window", "selected", "rel"):
        _need(a.get(key) is None, f"a prefill with `{key}`")
    _need(a["mask"] is None and a["causal"], "a masked or non-causal prefill")
    k, v = run.kv[run.interp.values[a["cache"]]["def"]["Cache"]]
    q, hd = run.get(a["q"]), a["head_dim"]
    B, T = run.shape
    qh = q.reshape(B, T, -1, hd).transpose(1, 2)
    kh = k.reshape(B, T, -1, hd).transpose(1, 2)
    vh = v.reshape(B, T, -1, hd).transpose(1, 2)
    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=a["sm_scale"], enable_gqa=True)
    run.put(a["o"], o.transpose(1, 2).reshape(B, T, -1))


def _residual_add(run, a):
    _need(a["gate"] is None, "a gated residual add")
    run.put(a["y_out"], run.get(a["x"]) + run.get(a["y"]))


def _binary(run, a):
    x, y = run.get(a["x"]), run.get(a["y"])
    ops = {"Add": torch.add, "Mul": torch.mul, "Div": torch.div}
    _need(a["op"] in ops, f"binary `{a['op']}`")
    run.put(a["z"], ops[a["op"]](x.float(), y.float()).to(x.dtype))


def _add_bias(run, a):
    run.put(a["out_out"], run.get(a["out"]) + run.get(a["bias"]))


def _glu(run, a):
    _need(a["weights"] is None and a["limit"] is None, "a weighted or limited glu")
    if a["packed"] is not None:
        p = run.get(a["packed"])
        gate, up = p[..., : a["intermediate"]], p[..., a["intermediate"] :]
    else:
        gate, up = run.get(a["gate"]), run.get(a["up"])
    acts = {"Swiglu": F.silu, "GegluTanh": lambda g: F.gelu(g, approximate="tanh")}
    _need(a["act"] in acts, f"glu `{a['act']}`")
    run.put(a["y"], (acts[a["act"]](gate.float()) * up.float()).to(gate.dtype))


def _gelu_tanh(run, a):
    x = run.get(a["x"])
    run.put(a["y"], F.gelu(x.float(), approximate="tanh").to(x.dtype))


def _rows(run, a):
    rows, x = run.get(a["rows"]), run.get(a["x"])
    run.put(a["y"], x if rows is None else x.reshape(-1, x.shape[-1])[rows])


_OPS = {
    "Gather.Embed": _embed,
    "Elemwise.Rmsnorm": _rmsnorm,
    "Linear.Matmul": _matmul,
    "Linear.LoraCorrect": _lora_correct,
    "Elemwise.Rope": _rope,
    "Attn.KvAppend": _kv_append,
    "Attn.PlanPrefill": _plan,
    "Attn.PlanDecode": _plan,
    "Attn.Prefill": _prefill,
    "Elemwise.ResidualAdd": _residual_add,
    "Elemwise.Binary": _binary,
    "Elemwise.AddBias": _add_bias,
    "Linear.Glu": _glu,
    "Linear.GeluTanh": _gelu_tanh,
    "Gather.Rows": _rows,
}


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
