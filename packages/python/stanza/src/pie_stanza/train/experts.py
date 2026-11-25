"""The experts: the torch forwards of a mixture-of-experts layer's routing,
expert products and weighted sum (Qwen3 MoE).

A token's `top_k` routes are rows `t * top_k + j` of the routed values, as the
engines lay them out. The route picks are integers and take no gradient; the
route weights, the expert banks and the activations do.
"""

from __future__ import annotations

import torch

from .forwards import need


def _route(run, a):
    for key in ("ids", "table", "bias", "scale", "hint"):
        need(a[key] is None, f"an expert route with `{key}`")
    need(a["score"] == "Softmax", f"expert scores `{a['score']}`")
    logits = run.get(a["logits"]).float()[..., : a["experts"]]
    top = logits.topk(a["top_k"], dim=-1)
    # A softmax over the picked logits: the full softmax, renormalized over the top k.
    run.vals[a["routes"]] = top.indices
    run.vals[a["weights"]] = top.values.softmax(-1)


def _select(run, a):
    need(a["bias"] is None and not a["quantized"] and a["quant"] is None and a["input_scales"] is None,
         "a biased or quantized expert product")
    x, bank, routes = run.get(a["x"]), run.get(a["bank"]), run.get(a["routes"])
    B, T, k = routes.shape
    flat = routes.reshape(-1)
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] == B * T:
        rows = rows.repeat_interleave(k, dim=0)
    out = rows.new_zeros(flat.shape[0], bank.shape[1], dtype=torch.float32)
    for e in flat.unique().tolist():
        at = (flat == e).nonzero()[:, 0]
        out = out.index_put((at,), rows[at].float() @ bank[e].float().T)
    run.put(a["y"], out.reshape(B, T * k, -1).to(x.dtype))


def _combine(run, a):
    need(a["weights"] is not None and a["shared"] is None, "an expert sum without weights or with a shared row")
    routed, weights = run.get(a["routed"]), run.get(a["weights"])
    B, T, k = weights.shape
    y = (routed.float().reshape(B, T, k, -1) * weights[..., None]).sum(2)
    run.put(a["y"], y.to(routed.dtype))


FORWARDS = {
    "Linear.MoeRoute": _route,
    "Linear.MoeSelect": _select,
    "Linear.MoeCombine": _combine,
}
"""Each expert op's forward, by `group.name`."""
