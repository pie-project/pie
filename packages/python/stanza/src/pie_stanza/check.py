"""What `stanza check` prints: a map of the mounted program's stanzas.

It reads the run directory (`trace.json`, `banks.json`, `lowered.json`) and
the stanza files, and states for each stanza how many lines of Poem it is,
at which taps and layers it attaches, and what one tenant costs. With weights
it adds four checks against the checkpoint, run through the trainer on CPU at
f32: a zero slot equals the base model, every bank receives a gradient, the
plain program matches transformers (P0), and LoRA and IA3 match PEFT (P1).
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .mount import STANZAS

TAPS = ("q", "k", "v", "o", "gate", "up", "down")
"""The taps a dense model offers, in the order the table shows them; any
other tap a stanza attaches at (a block's `attn`) follows them."""
DTYPE_BYTES = {"bf16": 2, "f16": 2, "f32": 4}
"""Bytes per element of each bank dtype."""
PROMPTS = ["The capital of France is", "Triage this ticket: my brakes failed on a descent.",
           "def add(a, b):", "Larkspur Bikes ships every order from"]
"""The prompts the weight checks run."""
TOLERANCE = 2e-4
"""The largest log-prob difference P0 and P1 accept, at f32 on CPU."""


@dataclass
class Row:
    """One mounted stanza's line of the map."""

    name: str
    """The stanza name."""
    lines: int
    """Lines of Poem, not counting blank lines and comments."""
    sites: dict[str, list[int]]
    """For each tap it attaches at, the layers it attaches in."""
    banks: int
    """How many banks it registers."""
    params: int
    """Trainable values per tenant (one slot of every bank)."""
    bytes: int
    """Bytes per tenant at the banks' dtype."""


@dataclass
class Report:
    """The map of a mounted program, and the weight checks when asked for."""

    package: str
    """The Poem package, such as `qwen3_dense`."""
    model: str
    """The model id."""
    layers: int
    """The model's layer count."""
    slots: int
    """Adapters resident at once (axis 0 of every bank)."""
    mount: str
    """The mount hash, which names the run directory."""
    stanzas: dict[str, Row]
    """One row per mounted stanza, in mount order."""
    lowered: dict
    """`lowered.json`: per platform, banks kept by name, or why it was refused."""
    weights: list[tuple[str, bool]] = field(default_factory=list)
    """The weight-check lines and whether each passed, when the checks ran."""

    def __str__(self) -> str:
        taps = [*TAPS, *sorted({s for r in self.stanzas.values() for s in r.sites} - set(TAPS))]
        head = ["stanza", "lines", *taps, "banks", "params/tenant", "bytes/tenant"]
        body = [[r.name, str(r.lines), *(_cell(r.sites.get(t, []), self.layers) for t in taps),
                 str(r.banks), f"{r.params:,}", _size(r.bytes)] for r in self.stanzas.values()]
        widths = [max(len(row[i]) for row in [head, *body]) for i in range(len(head))]
        table = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths)))
                 for row in [head, *body]]
        width = len(table[0])
        title = f"{self.package} / {self.model}   {self.layers} layers   {self.slots} slots   mount {self.mount}"
        refused = [f"        {p} refused: {e['refused']}" for p, e in self.lowered.items() if "refused" in e]
        lines = [title, *table, _status(self._lowers(), self._lowers_ok(), width), *refused]
        lines += [_status(text, ok, width) for text, ok in self.weights]
        return "\n".join(lines)

    def _lowers(self) -> str:
        if not self.lowered:
            return "lowers  (not traced with --lowered)"
        kept = [p for p, e in self.lowered.items() if "refused" not in e]
        counts = {(self.lowered[p]["banks_kept"], self.lowered[p]["banks_total"]) for p in kept}
        if len(counts) == 1:
            k, t = counts.pop()
            return f"lowers  {' '.join(kept)}: {k} of {t} banks kept by name"
        return "lowers  " + "  ".join(f"{p} {self.lowered[p]['banks_kept']}/{self.lowered[p]['banks_total']}"
                                      for p in kept)

    def _lowers_ok(self) -> bool:
        kept = [e for e in self.lowered.values() if "refused" not in e]
        return bool(kept) and all(e["banks_kept"] == e["banks_total"] for e in kept)

    @property
    def ok(self) -> bool:
        """Whether lowering kept every bank and every weight check passed."""
        return self._lowers_ok() and all(ok for _, ok in self.weights)


def _cell(layers: list[int], total: int) -> str:
    if not layers:
        return "-"
    return str(len(layers)) if len(layers) == total else ",".join(map(str, layers))


def _size(n: int) -> str:
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def _status(text: str, ok: bool, width: int) -> str:
    mark = "ok" if ok else "FAIL"
    return text + " " * max(2, width - len(text) - len(mark)) + mark


def code_lines(path: Path) -> int:
    """Lines of Poem in a file, not counting blank lines and comments."""
    return sum(1 for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#"))


def report(model, *, weights: bool = False) -> Report:
    """The map of `model`'s mounted program, with the weight checks if asked."""
    run = model.run_dir()
    banks = json.loads((run / "banks.json").read_text())
    trace = json.loads((run / "trace.json").read_text())
    lowered_file = run / "lowered.json"
    lowered = json.loads(lowered_file.read_text()) if lowered_file.exists() else {}
    layers = 1 + max(int(m.group(1)) for p in trace["params"] if (m := re.match(r"layers\.(\d+)\.", p["name"])))
    mounted = json.loads((run / "mount.json").read_text())["stanzas"]
    rows = {}
    for name in mounted:
        own = [b for b in banks["banks"] if b["stanza"] == name]
        sites: dict[str, set[int]] = {}
        for b in own:
            sites.setdefault(b["site"], set()).add(b["layer"])
        params = sum(math.prod(b["shape"][1:]) for b in own)
        size = sum(math.prod(b["shape"][1:]) * DTYPE_BYTES[b["dtype"]] for b in own)
        lines = code_lines(model.models / STANZAS / f"{name}.poem")
        rows[name] = Row(name, lines, {s: sorted(ls) for s, ls in sites.items()}, len(own), params, size)
    out = Report(banks["package"], banks["model"], layers, banks["slots"], run.parent.name, rows, lowered)
    if weights:
        out.weights = weight_checks(model, mounted)
    return out


def weight_checks(model, stanzas: list[str]) -> list[tuple[str, bool]]:
    """The four checks against the checkpoint, as (line, passed)."""
    from . import train

    if not stanzas:
        return [("weight checks need a mounted stanza", False)]
    rows = [model.tokenizer.encode(p) for p in PROMPTS]
    zero, reached, p1, p0 = {}, [0, 0], {}, None
    for s in stanzas:
        t = model.trainer(s, device="cpu", dtype="f32")
        plain = t.logprobs(rows, adapted=False)[0]
        zero[s] = _gap(t.logprobs(rows)[0], plain)
        p0 = _gap(plain, _hf_logprobs(model, rows)) if p0 is None else p0
        got, total = _reached(t, rows)
        reached = [reached[0] + got, reached[1] + total]
        if s in ("lora", "ia3"):
            with tempfile.TemporaryDirectory() as tmp:
                peft = _peft(model, s, Path(tmp))
                t.inner.load(train.from_peft(tmp, model.program(), stanza=s))
                p1[s] = _gap(t.logprobs(rows)[0], _logprobs_of(peft, rows))
    return [
        ("zero slot equals base   " + "  ".join(f"{s} {d:.1e}" for s, d in zero.items()),
         all(d == 0.0 for d in zero.values())),
        (f"every bank gets a gradient   {reached[0]} of {reached[1]}", reached[0] == reached[1]),
        (f"P0 vs transformers   max |logp| {p0:.1e} on {len(rows)} prompts", p0 <= TOLERANCE),
        ("P1 vs PEFT   " + ("  ".join(f"{s} {d:.1e}" for s, d in p1.items()) or "no LoRA or IA3 mounted"),
         all(d <= TOLERANCE for d in p1.values())),
    ]


def _gap(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def _reached(t, rows) -> tuple[int, int]:
    """How many of the stanza's banks change in two optimizer steps, of how many."""
    from . import train

    with tempfile.TemporaryDirectory() as tmp:
        before = train.read_bundle(t.inner.save(Path(tmp) / "before", name="check"))[0]
        batch = train.Batch(tokens=rows, weights=[[1.0] * (len(r) - 1) for r in rows])
        for _ in range(2):
            t.forward_backward(batch)
            t.optim_step(1e-2)
        after = train.read_bundle(t.inner.save(Path(tmp) / "after", name="check"))[0]
    return sum(1 for k in before if not before[k].equal(after[k])), len(before)


def _logprobs_of(hf, rows) -> list[list[float]]:
    import torch

    out = []
    with torch.no_grad():
        for r in rows:
            ids = torch.tensor([r])
            logp = hf(ids).logits[0, :-1].float().log_softmax(-1)
            out.append(logp.gather(-1, ids[0, 1:, None])[:, 0].tolist())
    return out


def _hf_logprobs(model, rows) -> list[list[float]]:
    import torch
    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(model.weights, dtype=torch.float32).eval()
    return _logprobs_of(hf, rows)


def _peft(model, stanza: str, out: Path):
    """A random PEFT adapter of this form on the checkpoint, saved to `out`."""
    import torch
    from peft import IA3Config, LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    hf = AutoModelForCausalLM.from_pretrained(model.weights, dtype=torch.float32)
    if stanza == "lora":
        config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                  "gate_proj", "up_proj", "down_proj"])
    else:
        config = IA3Config(target_modules=["k_proj", "v_proj", "up_proj"], feedforward_modules=[])
    hf = get_peft_model(hf, config)
    for name, p in hf.named_parameters():
        if "lora_" in name or "ia3_" in name:
            base = 1.0 if "ia3_" in name else 0.0
            torch.nn.init.normal_(p, mean=base, std=0.03)
            p.data = base + (p.data - base).bfloat16().float()   # the trainer reads bf16 planes
    hf.save_pretrained(out)
    return hf.eval()
