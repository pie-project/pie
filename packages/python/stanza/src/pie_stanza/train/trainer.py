"""The trainer: it trains one stanza of a program by replaying the trace pie
serves and differentiating it.

Slot 0 of the stanza's banks trains as f32 masters, one slot each. The forward
reads each master through a straight-through cast to the dtype the trace
serves it at (bf16 for every bank), so training sees the exact bytes the
bundle will carry to pie. Every other registered bank is zero and frozen,
which is the identity for every stanza. Checkpoint params stay frozen unless
a `trainable` glob names them; a param the trace serves at f32 (such as
Qwen3.5's `a_log`) stays f32 whatever the trainer's dtype.

The calls follow Tinker's shape: `forward_backward` accumulates gradients,
`optim_step` applies them with AdamW and appends one line to `log`.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import time
from collections.abc import Callable
from fnmatch import fnmatch

import torch

from . import checkpoint, head
from .batch import Batch, pad, padded
from .bundle import write_bundle
from .interpreter import Interpreter
from .loss import FUNCTIONS, REFERENCED
from .program import Bank, Program

DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
"""The dtypes a trainer runs frozen weights and activations at."""
SERVED = DTYPES | {"f16": torch.float16}
"""The trace dtypes a trained param can be served at."""


class Trainer:
    """Trains one stanza's banks (or checkpoint params a glob names) on one device."""

    program: Program
    """The program it replays."""
    stanza: str | None
    """The stanza it trains, or None when only `trainable` params train."""
    device: str
    """The torch device, such as `mps` or `cpu`."""
    dtype: str
    """The dtype of frozen weights and activations: `bf16` or `f32`."""
    masters: dict[str, torch.Tensor]
    """The f32 master of every trained weight: one slot (the bank shape without
    axis 0) per bank of the stanza, and each checkpoint param a glob names."""
    log: list[dict]
    """One `{"step", "loss", "tokens", "seconds"}` per `optim_step`."""

    def __init__(self, program: Program, weights: str | os.PathLike, *, stanza: str | None,
                 trainable: list[str] | None = None, device: str = "mps", dtype: str = "bf16", seed: int = 0):
        if dtype not in DTYPES:
            raise ValueError(f"dtype {dtype!r} is not one of {sorted(DTYPES)}")
        if stanza is None and not trainable:
            raise ValueError("nothing to train: name a stanza or a trainable glob")
        self.program, self.stanza, self.device, self.dtype = program, stanza, device, dtype
        self._dtype = DTYPES[dtype]
        self._interp = Interpreter(program.trace)
        self._served = {p["name"]: SERVED[p["dtype"]] for p in program.trace["params"] if p["dtype"] in SERVED}
        frozen = checkpoint.load(program, weights, dtype=self._dtype, device=device)
        gen = torch.Generator().manual_seed(seed)
        trained = program.banks_of(stanza) if stanza is not None else []
        self._banks = {b.name for b in trained}
        self.masters = {b.name: _start(b, gen).to(device).requires_grad_() for b in trained}
        tuned = sorted(n for n in frozen if any(fnmatch(n, g) for g in trainable or []))
        if trainable and not tuned:
            raise ValueError(f"trainable {trainable} names no checkpoint param")
        odd = [n for n in [*self._banks, *tuned] if n not in self._served]
        if odd:
            raise ValueError(f"param `{odd[0]}`: the trace serves it at a dtype the trainer cannot train")
        self.masters.update({n: frozen.pop(n).float().requires_grad_() for n in tuned})
        self._frozen = frozen
        self._zeros = {
            b.name: torch.zeros((1, *b.shape[1:]), dtype=self._dtype, device=device)
            for b in program.banks if b.name not in self._banks
        }
        self.log = []
        self._opt = None
        self._lr = None
        self._losses: list[float] = []
        self._tokens = 0
        self._since = None
        self._data = hashlib.sha256()

    def forward_backward(self, batch: Batch, loss: str | Callable = "cross_entropy", *, reference: bool = False) -> float:
        """Add the gradient of `loss` on `batch` to the masters; return the loss.
        `reference=True`, or a loss that reads one (`dpo`), first fills
        `batch.ref_logp` with plain-arm log-probs."""
        fn = loss if callable(loss) else FUNCTIONS.get(loss)
        if fn is None:
            raise ValueError(f"loss {loss!r} is not one of {sorted(FUNCTIONS)} or a function")
        if self._since is None:
            self._since = time.perf_counter()
        view, real = padded(batch, self.device)
        if reference or loss in REFERENCED:
            with torch.no_grad():
                batch.ref_logp = view.ref_logp = self._logp(view.tokens, real, adapted=False)
        value = fn(self._logp(view.tokens, real, adapted=True), view)
        value.backward()
        self._losses.append(value.item())
        self._tokens += sum(len(row) for row in batch.tokens)
        self._data.update(json.dumps([batch.tokens, batch.weights]).encode())
        return self._losses[-1]

    def optim_step(self, lr: float, *, betas=(0.9, 0.999), weight_decay: float = 0.0, clip: float = 1.0) -> None:
        """Clip the accumulated gradient to norm `clip`, take one AdamW step, clear it."""
        if not self._losses:
            raise RuntimeError("optim_step before any forward_backward")
        params = list(self.masters.values())
        if self._opt is None:
            self._opt = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        for group in self._opt.param_groups:
            group.update(lr=lr, betas=betas, weight_decay=weight_decay)
        torch.nn.utils.clip_grad_norm_(params, clip)
        self._opt.step()
        self._opt.zero_grad(set_to_none=True)
        _sync(self.device)
        self.log.append({
            "step": len(self.log) + 1,
            "loss": sum(self._losses) / len(self._losses),
            "tokens": self._tokens,
            "seconds": round(time.perf_counter() - self._since, 4),
        })
        self._lr, self._losses, self._tokens, self._since = lr, [], 0, None

    def logprobs(self, tokens: list[list[int]], *, adapted: bool = True, dtype: str | None = None
                 ) -> tuple[list[list[float]], list[list[int]], list[list[float]]]:
        """Per row, for each prediction of tokens[i+1]: its log-prob, the argmax
        token and the gap between the top two log-probs. `adapted=False` reads the
        plain arm; `dtype` runs at another dtype than the trainer's."""
        run = DTYPES[dtype or self.dtype]
        with torch.no_grad():
            ids, real = pad(tokens, self.device)
            out = self._interp.hidden(self._params(run), ids, adapted=adapted, dtype=run)
            lp, top, gap = head.readout(out.rows[:, :-1][real], out.head, ids[:, 1:][real], out.cap)
        counts = [len(row) - 1 for row in tokens]
        return tuple([t.tolist() for t in x.cpu().split(counts)] for x in (lp, top, gap))

    def save(self, out_dir: str | os.PathLike, *, name: str, meta: dict | None = None) -> pathlib.Path:
        """Write the trained stanza's banks as a bundle; `meta` adds to `stanza.json`."""
        if self.stanza is None:
            raise ValueError("this trainer trains no stanza, so it has no bundle to save")
        provenance = {
            "data_sha256": self._data.hexdigest() if self.log else None,
            "steps": len(self.log),
            "lr": self._lr,
            "loss_curve": [e["loss"] for e in self.log],
            "train_seconds": round(sum(e["seconds"] for e in self.log), 3),
            "device": self.device,
        }
        planes = {n: self.masters[n] for n in self._banks}
        return write_bundle(out_dir, planes, self.program, stanza=self.stanza, name=name, meta=provenance | (meta or {}))

    def load(self, planes: dict[str, torch.Tensor]) -> None:
        """Set the trained slot to `planes`, one per bank of the stanza, such as
        a bundle's planes or what `from_peft` returns."""
        if set(planes) != self._banks:
            odd = sorted(set(planes) ^ self._banks)
            raise ValueError(f"bank `{odd[0]}`: the planes and stanza `{self.stanza}`'s banks differ")
        with torch.no_grad():
            for n, plane in planes.items():
                if plane.shape != self.masters[n].shape:
                    raise ValueError(f"bank `{n}`: plane is {list(plane.shape)}, one slot is {list(self.masters[n].shape)}")
                self.masters[n].copy_(plane.float())

    def _params(self, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        params = {n: t.to(self._at(n, dtype)) for n, t in self._frozen.items()}
        params.update((n, t.to(dtype)) for n, t in self._zeros.items())
        for n, m in self.masters.items():
            served = (m + (m.to(self._served[n]).float() - m).detach()).to(self._at(n, dtype))
            params[n] = served[None] if n in self._banks else served
        return params

    def _at(self, name: str, dtype: torch.dtype) -> torch.dtype:
        """The dtype a param runs at in a `dtype` replay: f32 where the trace serves it so."""
        return torch.float32 if self._served.get(name) == torch.float32 else dtype

    def _logp(self, ids: torch.Tensor, real: torch.Tensor, *, adapted: bool) -> torch.Tensor:
        out = self._interp.hidden(self._params(self._dtype), ids, adapted=adapted, dtype=self._dtype)
        lp = head.logp(out.rows[:, :-1][real], out.head, ids[:, 1:][real], out.cap)
        return torch.zeros(real.shape, dtype=torch.float32, device=ids.device).masked_scatter(real, lp)


def _start(bank: Bank, gen: torch.Generator) -> torch.Tensor:
    """Slot 0 of `bank` as its init says: zeros, PEFT's kaiming_uniform_(a=sqrt(5)), or N(0, 0.02)."""
    t = torch.zeros(bank.shape[1:])
    if bank.init == "kaiming":
        if t.dim() < 2:
            raise ValueError(f"bank `{bank.name}`: kaiming init needs a matrix, one slot is {list(t.shape)}")
        torch.nn.init.kaiming_uniform_(t, a=math.sqrt(5), generator=gen)
    elif bank.init == "normal":
        t.normal_(0.0, 0.02, generator=gen)
    return t


def _sync(device: str) -> None:
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
