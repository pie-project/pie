"""Axis smoke: every composition axis's lane completes sanely.

The AC campaign's repo-side regression floor. One lane per axis —
hooked snapkv, masked dense, lora (span-grouped correction),
layer-truncated draft — plus a plain anchor. Sequential on the
harness client (the concurrent-census instrument with fire-trace
verdicts lives in .wiki/tart/ac5_census.py; per-lane connections and
PIE_FIRE_TRACE need a hand-driven boot). What THIS catches: any
seriation/relax/gate regression that breaks an axis's fire shape
outright — the failure mode every AC increment hit first.

Run from the repo root::

    uv run python tests/inferlets/test_axis_smoke.py --model Qwen/Qwen3-0.6B
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet  # noqa: E402

PROMPT = "The old clockmaker examined the strange timepiece carefully"
LONG = (
    "The capital of France is Paris. "
    + "Paris is a large European city with a long history. " * 24
)

LANES = [
    ("plain", "naive-baseline", {"prompt": PROMPT, "max_tokens": 32, "seed": 900}),
    (
        "hook",
        "trackb-snapkv",
        {
            "prompt": LONG,
            "max_tokens": 32,
            "temperature": 0.001,
            "seed": 3,
            "page_budget": 4096,
        },
    ),
    ("lora", "lora-probe", {"max_tokens": 32, "seed": 5, "adapter_scale": 0.7}),
    (
        "mask",
        "naive-masked",
        {"prompt": PROMPT, "max_tokens": 32, "seed": 12, "mask_mode": "dense"},
    ),
    (
        "mask-policy",
        "naive-masked",
        {
            "prompt": PROMPT,
            "max_tokens": 32,
            "seed": 12,
            "mask_mode": "doc-isolation",
        },
    ),
    (
        "depth",
        "naive-baseline",
        {"prompt": PROMPT, "max_tokens": 32, "seed": 7, "max_layers": 8},
    ),
    (
        "mask-x-depth",
        "naive-masked",
        {
            "prompt": PROMPT,
            "max_tokens": 32,
            "seed": 12,
            "mask_mode": "dense",
            "max_layers": 8,
        },
    ),
]


async def test_axis_smoke(client, args):
    for tag, name, lane_args in LANES:
        out = await run_inferlet(client, name, lane_args, timeout=args.timeout)
        payload = json.loads(out[out.find("{"):])
        assert payload.get("count", 0) > 0, f"{tag}: empty generation"


if __name__ == "__main__":
    from conftest import run_tests

    run_tests([test_axis_smoke], "axis smoke")
