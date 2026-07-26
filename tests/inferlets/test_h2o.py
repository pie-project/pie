"""H2O — is the score tap actually stateful across fires, and does it enforce?

`tova-attention` re-seeds its accumulator every fire, so a tap that silently
lost its carry would look identical. H2O does not re-seed, which makes the
carry directly observable: the score mass must grow by one fire's worth of
layers on every step. A flat mass trace means the loop-carried channel is
being re-seeded behind the program's back; a jumpy one means it is reading
stale device memory.

Then the same two enforcement questions `test_mask_enforced.py` asks of Quest.

Run from the repo root with PYTHONPATH=sdk/python-server/python.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

import test_curated as tc  # noqa: E402
from conftest import run_inferlet  # noqa: E402


PROMPT = (
    "The capital of France is Paris. "
    + "Paris is a large European city with a long history. " * 24
)
# Near-greedy: see the note above `_COHERENCE_TAU` in test_curated.py. Token
# equality against the baseline is only a stable invariant when the decision
# margin is wider than the plan-level float residue.
COMMON = {"prompt": PROMPT, "max_tokens": 12, "temperature": 0.001, "seed": 90210}


async def _run(client, args, name, extra=None):
    out = await run_inferlet(
        client, name, {**COMMON, **(extra or {})}, timeout=args.timeout
    )
    return json.loads(out[out.find("{"):])


async def test_h2o_accumulates_across_fires(client, args):
    r = await _run(client, args, "trackb-h2o", {"page_budget": 4096})

    print(f"  live_pages  = {r['live_pages']}  budget = {r['page_budget']}")
    print(f"  layers      = {r['layers_observed']}  mass = {r['score_mass']:.4f}")
    print(f"  mass_trace  = {[round(m, 3) for m in r['mass_trace']]}")

    assert r["scores_nan"] == 0, r
    # H2O's seed ramp is non-zero everywhere and never re-seeded away, so the
    # tail cannot be asserted to be zero -- but no ATTENTION may land there.
    assert r["tail_polluted"] == 0, (
        "positions past the live prefix carry attention mass; either the "
        f"capture buffer's slack was not cleared or kv_len is wrong: {r}"
    )

    # Every layer of every fire contributes one distribution, so the cumulative
    # mass must equal the cumulative layer count.
    assert r["layers_observed"] > 0, r
    assert abs(r["score_mass"] - r["layers_observed"]) < 0.05 * r["layers_observed"], (
        f"score mass {r['score_mass']} should track the cumulative layer count "
        f"{r['layers_observed']} -- each layer-fire adds exactly 1.0: {r}"
    )

    # THE H2O SIGNATURE. TOVA's trace would be flat at L.
    trace = r["mass_trace"]
    assert len(trace) >= 3, r
    assert all(b > a for a, b in zip(trace, trace[1:])), (
        f"the accumulator is not carrying across fires -- H2O reduces to TOVA: {trace}"
    )
    steps = [b - a for a, b in zip(trace, trace[1:])]
    lo, hi = min(steps), max(steps)
    assert hi - lo < 0.05 * hi, (
        f"each fire must add the SAME layer count; uneven steps mean fires are "
        f"dropping or double-counting layers: {steps}"
    )
    print(f"  [ ok ] mass grows by {lo:.2f} per fire -> the carry is real")


async def test_h2o_enforces_its_keep_set(client, args):
    wide = await _run(client, args, "trackb-h2o", {"page_budget": 4096})
    tight = await _run(client, args, "trackb-h2o", {"page_budget": 1})
    base = await _run(client, args, "naive-baseline")

    print(f"  wide  = {wide['text']!r}")
    print(f"  tight = {tight['text']!r}")
    print(f"  base  = {base['text']!r}")

    assert tight["scores_nan"] == 0, tight
    assert tight["text"].strip(), tight
    assert wide["text"] != tight["text"], (
        "attn_page_mask is not enforced for H2O: a 1-page budget produced the "
        "same continuation as an all-page one"
    )
    assert wide["text"] == base["text"], (
        "an all-keep page mask perturbed the output; compaction is not "
        f"equivalent to the original page table\n  {wide['text']!r}\n"
        f"  {base['text']!r}"
    )
    print("  [ ok ] budget 1 diverges, full budget matches the baseline exactly")


async def test_h2o_keeps_what_it_scored(client, args):
    r = await _run(client, args, "trackb-h2o", {"page_budget": 3})
    masses = [float(m) for m in r["page_mass"]]
    kept = r["kept_pages"]

    print(f"  page_mass = {r['page_mass']}")
    print(f"  kept      = {kept}  evict_next = {r['evicted_first']}")

    assert len(kept) == min(r["page_budget"], r["live_pages"]), r
    # The host recomputes the ranking independently of the device; the keep-set
    # must be exactly the top-`budget` masses.
    ranked = sorted(range(len(masses)), key=lambda p: (-masses[p], -p))
    assert set(kept) == set(ranked[: len(kept)]), (
        f"kept set {kept} is not the top-{len(kept)} by mass {ranked}"
    )
    assert r["evicted_first"] == ranked[-1], r
    # A heavy hitter must actually be heavy: the top page has to carry more
    # than a uniform share, or the statistic is not discriminating at all.
    assert max(masses) > sum(masses) / len(masses), r
    print("  [ ok ] keep-set is the top-budget page mass, and it discriminates")


if __name__ == "__main__":
    from conftest import run_tests

    run_tests(
        [
            test_h2o_accumulates_across_fires,
            test_h2o_enforces_its_keep_set,
            test_h2o_keeps_what_it_scored,
        ],
        "H2O heavy-hitter accumulation + enforcement",
    )
