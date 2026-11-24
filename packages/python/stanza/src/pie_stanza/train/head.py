"""The head: log-probs of the next token, computed from the readout rows in
chunks of 256 rows.

The full logits of a batch, `[rows*T, 151936]` in f32, would not fit in
memory for long batches. `logp` keeps only one chunk of logits alive at a
time, in the forward and again in the backward, where it recomputes them.
Logits are rounded to the head weight's dtype first, as the trace types them,
and then normalized in f32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

CHUNK = 256


def logp(h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(targets[i] | row i) for readout rows h [N, D] through the head
    w [V, D]; differentiable in h and in w."""
    return _Logp.apply(h, w, targets)


@torch.no_grad()
def readout(h: torch.Tensor, w: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each row: the target's log-prob, the argmax token, and the gap
    between the top two log-probs."""
    lps, tops, gaps = [], [], []
    for i in range(0, h.shape[0], CHUNK):
        logits = _logits(h[i : i + CHUNK], w)
        lps.append(_target(logits, targets[i : i + CHUNK]))
        top = logits.topk(2, dim=-1)
        tops.append(top.indices[:, 0])
        gaps.append(top.values[:, 0] - top.values[:, 1])
    return torch.cat(lps), torch.cat(tops), torch.cat(gaps)


def _logits(h, w):
    return F.linear(h, w).float()


def _target(logits, targets):
    # logits - logsumexp: on CPU this is 30 times closer to f64 than f32
    # log_softmax over a 151936-wide row (8e-6 against 2.3e-4, measured).
    return logits.gather(-1, targets[:, None])[:, 0] - logits.logsumexp(-1)


class _Logp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w, targets):
        ctx.save_for_backward(h, w, targets)
        out = []
        for i in range(0, h.shape[0], CHUNK):
            logits = _logits(h[i : i + CHUNK], w)
            out.append(_target(logits, targets[i : i + CHUNK]))
        return torch.cat(out) if out else h.new_zeros(0, dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad):
        h, w, targets = ctx.saved_tensors
        dh = torch.empty_like(h) if ctx.needs_input_grad[0] else None
        dw = torch.zeros_like(w, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        for i in range(0, h.shape[0], CHUNK):
            hc, tc = h[i : i + CHUNK], targets[i : i + CHUNK]
            # d logp[t] / d logits = onehot(t) - softmax(logits)
            dlogits = -_logits(hc, w).softmax(-1)
            dlogits.scatter_add_(-1, tc[:, None], torch.ones_like(tc[:, None], dtype=dlogits.dtype))
            dlogits *= grad[i : i + CHUNK, None]
            if dh is not None:
                dh[i : i + CHUNK] = (dlogits.to(w.dtype) @ w).to(h.dtype)
            if dw is not None:
                dw += dlogits.T @ hc.float()
        return dh, None if dw is None else dw.to(w.dtype), None
