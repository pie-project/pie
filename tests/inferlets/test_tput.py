"""Mixed-fleet high-concurrency throughput: the polymorphic-batching number.

`test_polymorph.py` proves a heterogeneous fleet FINISHES correctly. This file
measures what it costs: the same request volume is run twice — once as a
homogeneous fleet (every request the same program) and once as a heterogeneous
mix — at increasing concurrency, and the two are compared. The number In Gim
cares about is the mix's throughput relative to the homogeneous baseline: a
batcher that serializes on shape changes shows up as the ratio collapsing as
concurrency grows, long before any single request fails.

Correctness stays armed but light: every output must be non-empty and the
prompt-attending programs must still attend (a fast batcher that corrupts
neighbour rows is not fast, it is broken).

Usage::

    sdk/server/python/.venv/bin/python tests/inferlets/test_tput.py \
        --model Qwen/Qwen3.5-0.8B --timeout 600
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time

os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from conftest import run_tests  # noqa: E402
from test_polymorph import (  # noqa: E402
    ATTEND, _ensure_installed, _gate_attends, _gate_nonempty, _launch,
)

# Prompt pool: distinct prefixes so batching sees real diversity, not one
# shared prefix over and over.
PROMPTS = [
    "The capital of France is",
    "Water is made of two elements:",
    "In the year 1969, humans first",
    "The fastest land animal is the",
    "Photosynthesis is the process by which",
    "A haiku about winter:",
    "The primary colors are",
    "Newton's first law states that",
]

TOKENS = 32  # every request asks for the same volume, so arms are comparable


def _homog_fleet(n: int):
    """`n` text-completion requests over the prompt pool."""
    fleet = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        gate = _gate_attends if prompt == ATTEND else _gate_nonempty
        fleet.append(("text-completion", {"prompt": prompt, "max_tokens": TOKENS}, gate))
    return fleet


# The heterogeneous rotation. Every entry generates ~TOKENS tokens so the two
# arms carry the same request volume. Entries are added here as the engine
# gains the features (top_k, per-row masks, row readout, copy_kv).
MIX_ROTATION = [
    lambda p: ("text-completion", {"prompt": p, "max_tokens": TOKENS},
               _gate_attends if p == ATTEND else _gate_nonempty),
    lambda p: ("naive-baseline", {"max_tokens": TOKENS}, _gate_nonempty),
    lambda p: ("chat-completion", {"prompt": p, "max_tokens": TOKENS}, _gate_nonempty),
    lambda p: ("repetition-penalty",
               {"max_tokens": TOKENS, "repetition_penalty": 1.5,
                "frequency_penalty": 0.1}, _gate_nonempty),
    lambda p: ("token-healing", {"prompt": "The capital of Fra", "max_tokens": TOKENS},
               _gate_nonempty),
    lambda p: ("dry-repetition-penalty", {"max_tokens": TOKENS, "multiplier": 0.8},
               _gate_nonempty),
    lambda p: ("locally-typical-sampling", {"max_tokens": TOKENS}, _gate_nonempty),
    lambda p: ("cacheback-speculative-decoding",
               {"prompt": p, "max_tokens": TOKENS, "max_ngram": 4, "draft_length": 4},
               _gate_nonempty),
]


def _mixed_fleet(n: int):
    fleet = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        fleet.append(MIX_ROTATION[i % len(MIX_ROTATION)](prompt))
    return fleet


async def _run_arm(client, args, fleet, *, label: str) -> dict:
    for name, _inputs, _gate in fleet:
        await _ensure_installed(client, name)
    t0 = time.time()
    records = await asyncio.gather(*[
        _launch(client, name, inputs, timeout=args.timeout)
        for name, inputs, _gate in fleet
    ])
    wall = time.time() - t0

    failures = []
    for (name, _inputs, gate), record in zip(fleet, records):
        if record["status"] != "ok":
            failures.append(f"{name}: {record['detail'][:200]}")
            continue
        try:
            gate(record)
        except AssertionError as e:
            failures.append(f"{name}: GATE {e}")

    walls = sorted(r["wall"] for r in records)
    stats = {
        "label": label,
        "n": len(fleet),
        "wall": wall,
        "failures": failures,
        "req_per_s": len(fleet) / wall if wall > 0 else 0.0,
        "p50": statistics.median(walls),
        "p95": walls[max(0, int(len(walls) * 0.95) - 1)],
        "max": walls[-1],
    }
    print(f"    [{label:14s}] n={stats['n']:3d} wall={wall:7.1f}s "
          f"req/s={stats['req_per_s']:6.2f} p50={stats['p50']:6.1f}s "
          f"p95={stats['p95']:6.1f}s max={stats['max']:6.1f}s "
          f"fail={len(failures)}")
    return stats


async def test_tput_sweep(client, args):
    print()
    # Warm every program once before any timed arm: first registration of a
    # program is an NVRTC compile, and a fleet of first-registrations measured
    # 31s of wall on a mix whose steady state is 6s — a compile-storm number,
    # not a batching number. (That the storm STALLS unrelated lanes is a real
    # finding, tracked separately; this sweep measures steady state.)
    warm = {}
    for maker in MIX_ROTATION:
        name, inputs, _g = maker(PROMPTS[0])
        warm.setdefault(name, inputs)
    t0 = time.time()
    for name, inputs in warm.items():
        await _ensure_installed(client, name)
        await _launch(client, name, {**inputs, "max_tokens": 2}, timeout=args.timeout)
    print(f"    [warmup        ] {len(warm)} programs compiled in {time.time()-t0:.1f}s")
    results = []
    for n in (16, 48, 96):
        homog = await _run_arm(client, args, _homog_fleet(n), label=f"homog-{n}")
        mixed = await _run_arm(client, args, _mixed_fleet(n), label=f"mixed-{n}")
        ratio = (mixed["req_per_s"] / homog["req_per_s"]) if homog["req_per_s"] else 0.0
        print(f"    [ratio-{n}] mixed/homog req/s = {ratio:.2f}")
        results.append((n, homog, mixed, ratio))

    lines = []
    for n, homog, mixed, ratio in results:
        for arm in (homog, mixed):
            if arm["failures"]:
                lines.append(f"{arm['label']}: {len(arm['failures'])} failed — "
                             + "; ".join(arm["failures"][:3]))
    assert not lines, "\n".join(lines)


def tests():
    return [test_tput_sweep]


if __name__ == "__main__":
    run_tests(tests(), description="Mixed-fleet concurrency throughput")
