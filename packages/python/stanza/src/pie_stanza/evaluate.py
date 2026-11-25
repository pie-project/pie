"""Evaluation: how well a served adapter answers held-out examples.

`evaluate` asks pie for a reply to every example of a dataset, on the base
model or with an adapter, and averages a metric over the replies.
`json_exact` is the metric for answers that are JSON objects: whether the
reply parses as one, whether every field equals the reference, and per field.
"""

from __future__ import annotations

import json

BATCH = 16
"""How many requests are launched together."""


def json_exact(reply: str, answer: str) -> dict:
    """Score one reply against a reference JSON object: `json_valid`,
    `all_exact`, and per field whether it equals the reference's."""
    try:
        got = json.loads(reply)
    except json.JSONDecodeError:
        got = None
    valid = isinstance(got, dict)
    fields = {k: valid and got.get(k) == v for k, v in json.loads(answer).items()}
    return {"json_valid": valid, "all_exact": valid and all(fields.values()), "fields": fields}


def evaluate(pie, adapter, data, metric) -> dict:
    """`metric` averaged over the replies pie gives to every example of
    `data` with `adapter` (None for the base model), plus `n`."""
    replies = []
    for start in range(0, len(data), BATCH):
        ids = range(start, min(start + BATCH, len(data)))
        replies += pie.chat_many([(data.messages(i), adapter) for i in ids])
    return _mean([metric(r, data.answers[i]) for i, r in enumerate(replies)]) | {"n": len(replies)}


def _mean(scores: list[dict]) -> dict:
    return {k: _mean([s[k] for s in scores]) if isinstance(v, dict) else sum(float(s[k]) for s in scores) / len(scores)
            for k, v in scores[0].items()}
