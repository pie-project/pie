"""The interpreter: it replays a `poem_ir::Trace` in PyTorch, so autograd
differentiates the very forward pie serves.

Guards in the trace test bits of a lane-class word. The interpreter picks one
word per run and takes every arm whose guard holds, as
`crates/engine-cpu/src/forward.rs` does. A training run is a full-sequence
causal prefill: each batch row is one lane at positions 0..T-1 with an empty
cache. Adapted is on when the stanzas apply and off for the plain arm; Masked,
SingleToken, Scores and Drafts are off. Paged-KV ops collapse to "keep this
step's k and v". Every row is a readout row, so a layer that reads its keys
as one token's (Gemma's KV-shared layers) reads each row's keys up to its own
position. Every adapted row reads slot 0, so the trainer hands over one-slot
banks. A node whose operand no holding arm computes is dead in this word
(it feeds only other arms) and is skipped.

The run stops at the logits matmul and returns its operands, and the softcap
the trace puts on the logits, if any, so the head can be computed in chunks.
Any op the interpreter does not know raises and names the op.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import experts, forwards, recurrence

_OPS = forwards.FORWARDS | recurrence.FORWARDS | experts.FORWARDS


@dataclass(frozen=True)
class Readout:
    """What the logits are made of: `logits = cap * tanh(rows @ head.T / cap)`,
    or `rows @ head.T` with no cap."""

    rows: torch.Tensor
    """The readout rows [B, T, D]."""
    head: torch.Tensor
    """The head weight [V, D]."""
    cap: float | None
    """The softcap the trace puts on the logits (Gemma's), or None."""


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

    def hidden(self, params: dict, tokens: torch.Tensor, *, adapted: bool, dtype: torch.dtype) -> Readout:
        """Replay the trace over `tokens` [B, T] up to the logits matmul and
        return what the logits are made of."""
        run = _Run(self, params, self.word(adapted), dtype, tokens)
        for at, (name, op, a, guard) in enumerate(self.nodes):
            if not _holds(guard, run.word):
                run.skip(a)
                continue
            if name == "Linear.Matmul" and a["logits"]:
                return Readout(run.get(a["act"]), run.get(a["w"]), self._cap(at + 1, a["y"], run.word))
            if op is None:
                raise NotImplementedError(f"the trainer has no torch forward for `{name}`")
            try:
                op(run, a)
            except Undefined:
                run.skip(a)
        raise ValueError("the trace has no logits matmul")

    def _cap(self, start: int, logits: int, word: int) -> float | None:
        """The softcap the ops after the logits matmul put on `logits`;
        refuses any other op that reads them."""
        cap = None
        for name, _, a, guard in self.nodes[start:]:
            if not _holds(guard, word) or logits not in _reads(a):
                continue
            if name != "Elemwise.Softcap" or cap is not None:
                raise NotImplementedError(f"the trainer reads logits through `{name}`, which it has no head for")
            cap, logits = a["cap"], a["x_out"]
        return cap


class Undefined(KeyError):
    """A value no arm that holds in the replay's word computes."""


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
            if vid not in self.vals:
                raise Undefined(f"%{vid} is computed by no arm that holds in this word")
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
                except Undefined:
                    pass


def _reads(a: dict) -> list[int]:
    """The value ids an op's arguments name (outputs included)."""
    out = []
    for v in a.values():
        if isinstance(v, int) and not isinstance(v, bool):
            out.append(v)
        elif isinstance(v, list):
            out += [x for x in v if isinstance(x, int) and not isinstance(x, bool)]
    return out


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
