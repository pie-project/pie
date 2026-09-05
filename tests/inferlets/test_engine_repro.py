"""Engine reproducibility probes — NOT part of `run_all.py`.

These are diagnostics for the engine, driven through the Python `eta` port
(`rs-fork-probe-py`): they launch N copies of a greedy program at once and
assert every copy produced the same tokens. A greedy program is a pure
function of its prompt, so any spread is the engine's, not the guest's.

Known state (2026-09-05, engine-cuda): the UNFORKED decode is batch-invariant
on both models, but any decode from a working set produced by
`WorkingSet.fork` diverges across concurrent processes (one fork suffices;
sequential runs are stable) — on gemma-4-E4B (attention-only, KV fork only)
as well as on Qwen3.5-0.8B (hybrid), so the second test fails on both and
points at the KV copy-on-write path — the page-aligned control (third test)
is stable, so it is the CoW copy of a shared page, not the shared read.

    uv run python tests/inferlets/test_engine_repro.py --attach ws://127.0.0.1:8080
"""

from __future__ import annotations

import asyncio

from conftest import run_inferlet, run_tests

PROBE = "rs-fork-probe-py"


async def _distinct(client, args, inputs: dict, n: int) -> set[str]:
    outs = await asyncio.gather(*[run_inferlet(client, PROBE, inputs, timeout=args.timeout) for _ in range(n)])
    return {o.strip() for o in outs}


async def test_unforked_greedy_decode_is_batch_invariant(client, args):
    got = await _distinct(client, args, {"num_tokens": 8, "leaves": 1, "depth": 0, "repeats": 1}, 8)
    assert len(got) == 1, f"{len(got)} distinct outputs from 8 identical unforked greedy decodes:\n" + "\n".join(got)


async def test_forked_greedy_decode_is_batch_invariant(client, args):
    got = await _distinct(client, args, {"num_tokens": 6, "leaves": 2, "depth": 1, "repeats": 1}, 8)
    assert len(got) == 1, (
        f"{len(got)} distinct outputs from 8 identical greedy decodes after one working-set fork "
        "(engine: forked KV state is not reproducible under concurrency):\n" + "\n".join(got)
    )


async def test_forked_decode_onto_fresh_pages_is_batch_invariant(client, args):
    """The control for the test above: the same fork, but the root prompt is
    padded to a KV page boundary so the child's first write lands on a FRESH
    page and the shared prefix pages are only read. This is stable, which
    pins the divergence to the copy-on-write of a shared page."""
    got = await _distinct(client, args, {"num_tokens": 6, "leaves": 2, "depth": 1, "repeats": 1, "align": True}, 8)
    assert len(got) == 1, f"{len(got)} distinct outputs even with page-aligned forks:\n" + "\n".join(got)


def tests():
    return [
        test_unforked_greedy_decode_is_batch_invariant,
        test_forked_greedy_decode_is_batch_invariant,
        test_forked_decode_onto_fresh_pages_is_batch_invariant,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Engine reproducibility probes")
