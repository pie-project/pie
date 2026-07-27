"""From what context does Quest pay for itself?

Quest trades work for work. It spends time it did not have to spend -- an
`envelope_dot` per layer, a top-k, a page-table compaction -- to make the
attention read fewer pages. Below some context the overhead is larger than the
saving and the feature is a net loss; above it, a win. That crossover is the
only number that decides whether to turn Quest on, and it is not derivable from
kernel microbenchmarks, because the overhead is per-layer-per-step while the
saving is proportional to the context that the budget removes.

Method. Wall-clock decode cost is measured by DIFFERENCING two runs that differ
only in `max_tokens`:

    ms_per_token = (t(long) - t(short)) / (long - short)

which cancels prefill, WASM instantiation, server round-trips and process
warm-up -- everything that does not scale with the number of decode steps.
Every configuration is run `reps` times and the MINIMUM is kept: this host is
shared, so a mean measures the neighbours, while the minimum measures the work.

Run from the repo root with PYTHONPATH=sdk/python-server/python:

    PIE_CUDA_KV_ENVELOPES=1 python tests/inferlets/bench_quest.py \
        --driver cuda_native --model <path>
"""

import asyncio
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from conftest import run_inferlet, run_tests  # noqa: E402

PAGE_SIZE = 16
# ~9 tokens per repetition for this tokenizer; measured, not assumed (the
# report carries the real kv_len, which is what the table prints).
_UNIT = "Paris is a large European city with a long history. "
_HEAD = "The capital of France is Paris. "

SHORT_TOKENS = 8
LONG_TOKENS = 104
REPS = 7

# Contexts to sweep. The 8192-token forward limit caps a one-shot prefill
# somewhere below 10K, so the sweep stops where a single fire still fits.
CONTEXTS = [1024, 2048, 4096, 6144]
# Fractions of the request's real page count to keep. 1.0 is "keep everything",
# which isolates the cost of running Quest from the benefit of evicting.
BUDGET_FRACTIONS = [1.0, 0.5, 0.25]


def _prompt_for(target_tokens):
    return _HEAD + _UNIT * max(1, round(target_tokens / 9))


async def _timed(client, args, name, params):
    t0 = time.perf_counter()
    out = await run_inferlet(client, name, params, timeout=args.timeout)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, out


async def _endpoints(client, args, name, base_params):
    """One round: both endpoints of the difference, back to back."""
    # `reserve_tokens` pins `p_max` to the long endpoint for both runs; the
    # baseline has no per-page work and ignores it.
    pin = {"reserve_tokens": LONG_TOKENS} if name == "quest-attention" else {}
    t_short, out_s = await _timed(
        client, args, name,
        {**base_params, **pin, "max_tokens": SHORT_TOKENS})
    t_long, out_l = await _timed(
        client, args, name,
        {**base_params, **pin, "max_tokens": LONG_TOKENS})
    return (t_short, out_s), (t_long, out_l)


def _parse(out):
    i = out.find("{")
    if i < 0:
        return None
    try:
        return json.loads(out[i:])
    except json.JSONDecodeError:
        return None


async def _sweep_context(client, args, ctx, configs):
    """Measure every configuration at one context, INTERLEAVED.

    All configurations are run once per round rather than one configuration to
    completion. A shared host drifts -- a neighbour starts, the clock boosts --
    and measuring configurations in blocks lets that drift land entirely on
    one of them. Interleaving spreads it, and the per-configuration minimum
    across rounds then picks each one's quietest observation.
    """
    best = {label: [None, None] for label, _ in configs}
    reports = {}
    for _ in range(REPS):
        for label, (name, params) in configs:
            (t_s, out_s), (t_l, out_l) = await _endpoints(client, args, name, params)
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
    span = LONG_TOKENS - SHORT_TOKENS
    return ({label: (v[1] - v[0]) / span for label, v in best.items()}, reports)


async def test_quest_crossover(client, args):
    print()
    print(f"  differencing max_tokens {SHORT_TOKENS} -> {LONG_TOKENS}, "
          f"min of {REPS} reps; quest runs with report=false so the "
          "instrumentation is not being timed")
    print()
    header = f"  {'ctx':>6} {'pages':>6} {'baseline':>10}"
    for f in BUDGET_FRACTIONS:
        header += f" {('quest ' + format(f, '.2f')):>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for ctx in CONTEXTS:
        prompt = _prompt_for(ctx)
        common = {"prompt": prompt, "temperature": 0.001, "seed": 7}

        # One instrumented probe to learn the request's real page count, so the
        # budgets below are fractions of something real rather than of a guess.
        probe_out = await run_inferlet(
            client, "quest-attention",
            {**common, "max_tokens": SHORT_TOKENS, "page_budget": 1 << 20},
            timeout=args.timeout)
        probe = _parse(probe_out)
        pages = -(-probe["kv_len_last"] // probe["page_size"]) if probe else 0

        configs = [("baseline", ("naive-baseline", dict(common)))]
        for frac in BUDGET_FRACTIONS:
            budget = max(1, int(pages * frac))
            configs.append((f"q{frac:.2f}", ("quest-attention", {
                **common, "page_budget": budget, "report": False})))

        ms, _ = await _sweep_context(client, args, ctx, configs)

        base_ms = ms["baseline"]
        line = f"  {ctx:>6} {pages:>6} {base_ms:>9.3f}ms"
        cells = []
        for frac in BUDGET_FRACTIONS:
            q_ms = ms[f"q{frac:.2f}"]
            ratio = q_ms / base_ms
            cells.append((max(1, int(pages * frac)), q_ms, ratio))
            line += f" {q_ms:>7.3f} {ratio:>4.2f}x"
        print(line)
        rows.append((ctx, pages, base_ms, cells))

    print()
    # The claim under test: the ratio must IMPROVE with context. Quest's
    # overhead is per layer per step and roughly context-independent, while its
    # saving grows with the pages the budget removes. If the ratio does not
    # trend down, the eviction is not reaching the attention kernel at all --
    # which is precisely the failure that a correctness test cannot see,
    # because a mask that is computed and then ignored still produces a
    # perfectly coherent continuation.
    tight = [(ctx, cells[-1][2]) for ctx, _, _, cells in rows]
    print(f"  tightest budget ({BUDGET_FRACTIONS[-1]:.2f}) vs baseline: " +
          ", ".join(f"{c}:{r:.2f}x" for c, r in tight))
    first, last = tight[0][1], tight[-1][1]
    print(f"  trend {first:.2f}x -> {last:.2f}x over "
          f"{tight[0][0]} -> {tight[-1][0]} tokens")
    assert last < first, (
        "Quest does not get relatively cheaper as context grows, so the page "
        "budget is not reducing attention work: " + repr(tight)
    )
    below_one = [c for c, r in tight if r < 1.0]
    print("  net win at: " + (", ".join(map(str, below_one)) if below_one
                              else "no context in this sweep"))


if __name__ == "__main__":
    run_tests([test_quest_crossover], "Quest cost/benefit crossover")
