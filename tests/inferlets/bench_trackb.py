"""From what context does each Track B policy pay for itself?

`bench_quest.py` answers this for Quest and documents the method; this file
applies the same method to the eviction policies of Track B, whose cost/benefit
shapes are genuinely different from Quest's and from each other:

* **H2O** ranks a cumulative attention-mass row every layer of every decode
  step. Its per-step cost is closest to Quest's, but it ranks a row it already
  has rather than computing one, so it has no `envelope_dot` to pay for.
* **SnapKV** decides its keep-set ONCE, during prefill, and every decode step
  afterwards merely re-applies a fixed device-resident mask. Its per-step cost
  should therefore be the floor for any per-layer policy in this system: no
  scoring, no ranking, just the mask. What it still pays is the subject of
  §14.4 -- binding any per-layer hook takes the request off CUDA-graph replay,
  and that cost is the same whether the hook does a little or a lot.

That last point is the reason this benchmark exists as well as Quest's. If the
measurements come back with SnapKV's overhead close to Quest's despite doing a
fraction of the work, the constant in §14.4 is confirmed end-to-end from a
second direction -- and the ranking of what to optimise follows from it.

Method, identically to `bench_quest.py` (see its docstring for why each of
these is necessary rather than fussy):

  * difference two `max_tokens` endpoints to cancel prefill and start-up;
  * minimise the two endpoints INDEPENDENTLY before differencing;
  * interleave configurations across rounds rather than running each to
    completion;
  * take the minimum over rounds, not the mean, on a shared host;
  * pin `reserve_tokens` so both endpoints reserve the same page count;
  * turn `report` off so the telemetry readback is not timed.

Run from the repo root with PYTHONPATH=sdk/server/python/python:

    PIE_CUDA_KV_ENVELOPES=1 python tests/inferlets/bench_trackb.py \
        --engine cuda_native --model <path>
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from bench_quest import (  # noqa: E402
    LONG_TOKENS,
    REPS,
    SHORT_TOKENS,
    _parse,
    _prompt_for,
    _timed,
    assert_full_budget_is_overhead,
    assert_monotone_baseline,
)
from conftest import run_inferlet, run_tests  # noqa: E402

# Extends past 8192 because the one-shot prefill ceiling that used to cap this
# at 6144 is gone (section 17). That matters here more than for Quest: section
# 15 puts SnapKV's crossover at ~2.3K and H2O's at ~4.1K, so the range where
# these policies actually pay was almost entirely above where they could run.
CONTEXTS = [1024, 2048, 4096, 6144, 8192, 12288, 16384]
BUDGET_FRACTIONS = [1.0, 0.5, 0.25]

# H2O and SnapKV both keep `page_budget` pages; the backend force-keeps the
# request's last page on top of that, so a budget is never actually zero.
POLICIES = ["trackb-h2o", "trackb-snapkv"]


async def _endpoints(client, args, name, base_params):
    """One round: both endpoints of the difference, back to back."""
    pin = {} if name == "naive-baseline" else {"reserve_tokens": LONG_TOKENS}
    t_short, out_s = await _timed(
        client, args, name, {**base_params, **pin, "max_tokens": SHORT_TOKENS})
    t_long, out_l = await _timed(
        client, args, name, {**base_params, **pin, "max_tokens": LONG_TOKENS})
    return (t_short, out_s), (t_long, out_l)


async def _sweep(client, args, ctx, configs):
    best = {label: [None, None] for label, _ in configs}
    reports = {}
    for _ in range(REPS):
        for label, (name, params) in configs:
            (t_s, out_s), (t_l, out_l) = await _endpoints(
                client, args, name, params)
            slot = best[label]
            if slot[0] is None or t_s < slot[0]:
                slot[0] = t_s
            if slot[1] is None or t_l < slot[1]:
                slot[1] = t_l
            for out, want in ((out_s, SHORT_TOKENS), (out_l, LONG_TOKENS)):
                r = _parse(out)
                if r is None:
                    continue
                reports[label] = r
                assert r.get("count") == want, (
                    f"{label} generated {r.get('count')} tokens for "
                    f"max_tokens={want}; the difference method assumes the "
                    "step count is exactly max_tokens"
                )
    steps = LONG_TOKENS - SHORT_TOKENS
    return ({label: (v[1] - v[0]) / steps for label, v in best.items()},
            reports)


def _configs(prompt, pages):
    configs = [("baseline", ("naive-baseline", {"prompt": prompt}))]
    for policy in POLICIES:
        short = policy.removeprefix("trackb-")
        for frac in BUDGET_FRACTIONS:
            budget = max(1, int(pages * frac))
            configs.append((
                f"{short}@{frac:g}",
                (policy, {"prompt": prompt, "page_budget": budget,
                          "report": False}),
            ))
    return configs


async def test_trackb_crossover(client, args):
    print()
    header = ["ctx", "pages", "baseline"]
    for policy in POLICIES:
        short = policy.removeprefix("trackb-")
        header += [f"{short}@{f:g}" for f in BUDGET_FRACTIONS]
    print("  " + "  ".join(f"{h:>14s}" for h in header))

    ratios = {}
    baseline_ms = []
    for ctx in CONTEXTS:
        prompt = _prompt_for(ctx)
        # The real page count, probed -- NOT `(ctx + LONG_TOKENS) // PAGE_SIZE`,
        # which is what this used to do and which was quietly wrong. `ctx` is a
        # TARGET; `_prompt_for` lays down `round(ctx / 9)` repetitions of a unit
        # that tokenizes to a bit more than 9, so the real prompt runs ~25%
        # long (ctx=12288 measures 15032 tokens). The estimate therefore
        # UNDER-counts pages, and `budget=1.0` -- the column whose entire job is
        # to isolate the policy's cost with the benefit set to zero -- was
        # actually a ~80% budget that evicted. That is how a "pure overhead"
        # column came to report the policy running FASTER than the baseline.
        probe_out = await run_inferlet(
            client, "trackb-snapkv",
            {"prompt": prompt, "max_tokens": SHORT_TOKENS, "page_budget": 1 << 20},
            timeout=args.timeout)
        probe = _parse(probe_out)
        assert probe, "page-count probe returned no report"
        # Plus the pages the long endpoint will DECODE into. The budget has to
        # cover the request at its largest or `budget=1.0` still evicts -- and
        # the decoded tail is 104 tokens, which is 8% of the pages at ctx=1024.
        pages = probe["prompt_pages"] + -(-LONG_TOKENS // probe["page_size"])
        per_token, reports = await _sweep(
            client, args, ctx, _configs(prompt, pages))
        base = per_token["baseline"]
        baseline_ms.append(base)
        row = [f"{ctx}", f"{pages}", f"{base:.2f} ms"]
        for label, ms in per_token.items():
            if label == "baseline":
                continue
            row.append(f"{base / ms:.2f}x")
            ratios.setdefault(label, []).append(base / ms)
        print("  " + "  ".join(f"{c:>14s}" for c in row))
        sys.stdout.flush()
        assert_full_budget_is_overhead(
            {k: v for k, v in per_token.items() if k.endswith("@1")}, base)

    print()
    # Before reading anything off the table, check the table is readable at all.
    assert_monotone_baseline(CONTEXTS, baseline_ms)

    for label, series in ratios.items():
        print(f"  {label:>16s}: " + "  ".join(f"{r:.2f}x" for r in series))

    # A policy that evicts must get RELATIVELY better as the context grows: its
    # per-layer cost is fixed by the layer count while the attention it removes
    # grows with the prefix. If a policy's ratio does not improve from the
    # shortest context to the longest, either the eviction is not reaching the
    # attention kernel or the measurement is dominated by something that is not
    # the policy -- both of which have happened here before (§13.1, §14.2), and
    # neither of which changes the generated text, so nothing else catches it.
    for label, series in ratios.items():
        assert series[-1] > series[0], (
            f"{label} did not improve with context: {series}. An eviction "
            "policy whose advantage is flat in the context length is not "
            "evicting."
        )


if __name__ == "__main__":
    run_tests(
        [test_trackb_crossover],
        "Track B crossover benchmark",
        requires=("attn_score", "attn_page_mask"),
    )
