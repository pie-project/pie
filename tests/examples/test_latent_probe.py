"""The float-port loop closes (imagegen design D1/D3/D4) — a skeleton.

`latent-probe` drives one `forward` pass in a token-less, KV-less reading:
a latent channel seeded on the device by a keyed Normal draw, bound as
`input("latents")` beside `input("timestep")` and whatever else the reading
declares, stepped by an Euler epilogue over `velocity()`, and read back once
a fire. This gate asks only what a probe can ask without the family's
parity harness: the loop CLOSES. Every fire hands back `[rows, width]`
finite values, the loop-carried cells advance (the trace moves after the
seed), and the seed fire itself lands a unit-variance draw.

**WANTS A LOAD WITH A LATENT READING, AND SAYS SO RATHER THAN FAILING.** A
text row declares no readings, and the probe refuses it by name; that is the
correct answer for that load, reported as a skip. The run that means to
exercise the door names the mini-DiT row (the family lands in a later round):

    tests/examples/test_latent_probe.py --model <mini-dit checkpoint> \\
        --sku mini-dit-bf16-kv-bf16

Run from the repo root with PYTHONPATH=python/server/python.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet, run_tests  # noqa: E402

# Refusals that mean "this load has no latent reading", not "the loop broke".
LATENTLESS = (
    "declares no readings",
    "no token-less reading",
    "pass kind is not attention",
)

# The model this probe is curated against. `test_curated.py` carries the
# entry marked with this so a census run on a text row reports a skip.
REQUIRES_MODEL = "mini-dit"


async def _probe(client, args, **inputs) -> dict:
    """One probe run, or a skip when this load has no latent reading."""
    try:
        output = await run_inferlet(client, "latent-probe", inputs, timeout=args.timeout)
    except RuntimeError as error:
        if any(mark in str(error) for mark in LATENTLESS):
            # conftest reports FileNotFoundError as SKIPPED.
            raise FileNotFoundError(
                f"the loaded model has no latent reading, so the float-port loop has "
                f"nothing to drive (wants `{REQUIRES_MODEL}`): {error}"
            ) from error
        raise
    start = output.find("{")
    assert start >= 0, f"latent-probe returned no JSON object: {output[:200]!r}"
    return json.loads(output[start:])


async def test_latent_probe_closes_the_loop(client, args):
    """Every fire reads back finite rows and the loop-carried cells advance."""
    steps, rows = 4, 16
    r = await _probe(client, args, steps=steps, rows=rows, seed=7)

    print(f"  reading = {r['reading']}  rows = {r['rows']}  width = {r['width']}")
    print(f"  ports   = {r['ports']}")
    print(f"  sigmas  = {[round(s, 4) for s in r['sigmas']]}")
    print(f"  trace   = {[round(m, 4) for m in r['trace']]}")
    print(f"  final   mean = {r['mean']:.4f}  std = {r['std']:.4f}  head = {r['head']}")

    assert r["rows"] == rows and r["steps"] == steps, r
    assert r["non_finite"] == 0, f"the final latent carries non-finite values: {r}"
    assert len(r["sigmas"]) == steps + 1 and r["sigmas"][-1] == 0.0, r["sigmas"]
    assert "latents" in r["ports"][0], r["ports"]
    # Fire 0 is the seed: a keyed Normal draw has mean-abs near sqrt(2/pi)
    # for any width worth probing.
    assert len(r["trace"]) == steps + 1, r["trace"]
    seed_mean_abs = r["trace"][0]
    assert 0.5 < seed_mean_abs < 1.1, (
        f"the seed fire did not land a unit-variance draw: mean|x| = {seed_mean_abs}"
    )
    # The loop-carried cells advance: an integrating step moves the latent.
    assert any(
        not math.isclose(a, b, rel_tol=1e-6) for a, b in zip(r["trace"], r["trace"][1:])
    ), f"the latent never moved after the seed; the epilogue's put is not being fed back: {r}"


async def test_latent_probe_is_deterministic_under_a_seed(client, args):
    """The same seed lands the same latent; another seed does not."""
    a = await _probe(client, args, steps=2, rows=8, seed=11)
    b = await _probe(client, args, steps=2, rows=8, seed=11)
    c = await _probe(client, args, steps=2, rows=8, seed=12)
    assert a["head"] == b["head"], (a["head"], b["head"])
    assert a["head"] != c["head"], (a["head"], c["head"])


def tests():
    return [
        test_latent_probe_closes_the_loop,
        test_latent_probe_is_deterministic_under_a_seed,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Float-port loop probe (wants a latent reading)")
