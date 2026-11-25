"""Skew: how far the served model is from the trained one.

`skew` scores the same token rows twice, through the trainer and through pie
with the bundle the trainer saved last, and compares them on answer tokens
only. It reports P3, the mean and the largest log-prob gap; P4, top-1
agreement where the trainer's top two tokens are more than 0.05 apart; and,
for scale, the bf16 floor: the trainer at bf16 against itself at f32.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DECISIVE = 0.05
"""The top-2 log-prob gap above which the trainer's top-1 counts for P4."""
CHUNK = 8
"""Rows the trainer scores at once."""


@dataclass(frozen=True)
class Skew:
    """Trained against served log-probs on the answer tokens of a dataset."""

    mean: float
    """P3: mean |trained - served| log-prob, in nats."""
    max: float
    """P3: the largest |trained - served| log-prob."""
    top1_agree: float
    """P4: the share of decisive answer tokens where pie's argmax is the trainer's."""
    floor_mean: float
    """Mean |bf16 - f32| log-prob of the trainer itself: the rounding floor."""
    tokens: int
    """How many answer tokens were compared."""
    pairs: list[tuple[float, float]] = field(repr=False, compare=False)
    """(trained, served) log-prob of every answer token, for a parity scatter."""

    def summary(self) -> dict:
        """The five numbers, as `results.json` and `stanza.json` record them."""
        return {"mean": self.mean, "max": self.max, "top1_agree": self.top1_agree,
                "floor_mean": self.floor_mean, "tokens": self.tokens}


def skew(trainer, pie, data) -> Skew:
    """Compare `trainer` with pie serving `trainer.adapter` on `data`'s answer tokens."""
    if trainer.adapter is None:
        raise ValueError("the trainer has saved no bundle to compare pie against")
    gaps, floor, pairs, decisive, agree = [], [], [], 0, 0
    for start in range(0, len(data.tokens), CHUNK):
        rows = data.tokens[start : start + CHUNK]
        logp, top, margin = trainer.logprobs(rows)
        logp32 = trainer.logprobs(rows, dtype="f32")[0]
        for r, row in enumerate(rows):
            served, served_top = pie.score(row, trainer.adapter)
            for i, w in enumerate(data.weights[start + r]):
                if w <= 0:
                    continue
                gaps.append(abs(logp[r][i] - served[i]))
                floor.append(abs(logp[r][i] - logp32[r][i]))
                pairs.append((logp[r][i], served[i]))
                if margin[r][i] > DECISIVE:
                    decisive += 1
                    agree += top[r][i] == served_top[i]
    return Skew(
        mean=sum(gaps) / len(gaps),
        max=max(gaps),
        top1_agree=agree / decisive if decisive else float("nan"),
        floor_mean=sum(floor) / len(floor),
        tokens=len(gaps),
        pairs=pairs,
    )
