"""SnapKV — does the PREFILL capture see the observation window, and does it enforce?

H2O and TOVA read the decode capture: one query row, every fire. SnapKV reads a
different tap entirely -- the prefill capture, which records the last `window`
rows of the prompt against the whole prefix, once. Everything downstream of that
tap is shared with H2O, so the only claims worth making here are the ones that
would still hold if the decode path were deleted:

  1. the row describes the PROMPT, not the decode step (`observed_live` must be
     the prompt length, and everything past it exactly zero -- there is no seed
     ramp in the captured row, unlike H2O);
  2. it is a real per-layer softmax, so the mass is the layer count;
  3. the mass concentrates at the END of the prompt, because that is where the
     observation window sits and attention is local. A tap that captured the
     wrong rows -- say the first `window` rows, or an unmasked dot product --
     would still produce a plausible-looking distribution, but not this shape;
  3b. and INSIDE the window the mass tapers toward the end, which is the causal
     mask being obeyed rather than a property of the prompt. This is a separate
     case on a separate prompt, and the note above `PROMPT` argues why the two
     window claims cannot share one;
  4. the resulting keep-set is enforced: a 1-page budget must change the text,
     an all-page budget must reproduce the baseline bit for bit.

Run from the repo root with PYTHONPATH=sdk/server/python/python.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from conftest import run_inferlet  # noqa: E402


# SnapKV selects among PROMPT pages, so a prompt has to span several of them or
# the policy has nothing to choose between. ~280 tokens is >= 15 pages at the
# usual page size, which both prompts below clear.
#
# THERE ARE TWO OF THEM, AND THE SPLIT IS STRUCTURAL RATHER THAN COSMETIC. The
# window claims in this file pull in opposite directions and one prompt cannot
# satisfy both:
#
#   * `min(win) > max(mid)` -- the window's pages outweigh EVERY ordinary
#     prefix page -- is a claim about CONTENT. It needs a middle that carries
#     nothing the window carries, because attention is local: the page
#     immediately before the window is adjacent to it and will out-mass the
#     window's own tail page on any prompt whose middle reads like its end.
#     Measured on the old single-sentence prompt (`_QUEST_PROMPT`'s shape):
#     window [0.671, 0.433] against a middle whose last page reached 0.512 --
#     a failure of the PROMPT, not of the capture, and the same reason
#     `test_curated.py`'s `_EVICT_PROMPT` exists.
#   * `win[-1] < max(win)` -- the taper inside the window -- is a claim about
#     the CAUSAL MASK, and it only reads cleanly when the window's own pages
#     are interchangeable. Put the question on the window's last page and that
#     page is heaviest for a reason that has nothing to do with masking, so the
#     mask claim would fail on a model that masks perfectly.
#
# So the content claims get a prompt with content to separate, and the mask
# claim gets one with none.

#: A fact at the HEAD the filler never repeats, asked for at the TAIL. The head
#: page and the question page are then the only two that carry information, and
#: that is the shape SnapKV claims to find -- so the window separates, and a
#: budget too small to hold the question's pages loses the continuation.
#: (Same construction as `test_curated.py`'s `_EVICT_PROMPT`, and for the
#: reason argued there.)
PROMPT = (
    "Remember this: the secret code word is banana. "
    + "The weather in the valley is mild and the roads are clear. " * 20
    + " What is the secret code word? The secret code word is"
)

#: ONE SENTENCE, REPEATED, AND NOTHING ELSE -- every page carries the same
#: information, so nothing but position can shape the profile. That is useless
#: for the separation claim (it is exactly why the claim used to fail) and it is
#: precisely what the taper claim wants: inside the window the only thing left
#: to explain a falling profile is the causal mask.
FILLER_PROMPT = "Paris is a large European city with a long history. " * 24
# Near-greedy: see the note above `_COHERENCE_TAU` in test_curated.py. Token
# equality against the baseline is only a stable invariant when the decision
# margin is wider than the plan-level float residue.
COMMON = {"prompt": PROMPT, "max_tokens": 12, "temperature": 0.001, "seed": 90210}


async def _run(client, args, name, extra=None):
    out = await run_inferlet(
        client, name, {**COMMON, **(extra or {})}, timeout=args.timeout
    )
    return json.loads(out[out.find("{"):])


def _window_pages(r):
    """The page span the engine's observation window covers, derived here rather
    than reported by the inferlet.

    The engine captures the last `window` query rows of the prompt
    (`default_attn_score_window()` in crates/driver-cuda/csrc/src/model/attn_score.hpp,
    overridable by PIE_ATTN_SCORE_WINDOW). Those rows start at token
    `prompt_len - window`, so the kv positions they weight most heavily span
    every page from that token onward.

    Computing it independently on this side is the point: the engine decides
    which rows to capture from the same env var, and the two derivations only
    agree if the capture really is anchored to the END of the prompt. Sweeping
    PIE_ATTN_SCORE_WINDOW over 16/32/64/128 moves the observed mass profile in
    lockstep with this function.
    """
    window = int(os.environ.get("PIE_ATTN_SCORE_WINDOW", "32"))
    win_lo = max(r["prompt_len"] - window, 0) // r["page_size"]
    return window, win_lo


async def test_snapkv_observes_the_prompt_window(client, args):
    r = await _run(client, args, "trackb-snapkv", {"page_budget": 4096})

    print(f"  prompt_len  = {r['prompt_len']}  pages = {r['prompt_pages']}")
    print(f"  layers      = {r['layers_observed']}  mass = {r['score_mass']:.4f}")
    print(f"  observed    = {r['observed_live']}  tail_nonzero = {r['tail_nonzero']}")
    print(f"  score_head  = {r['score_head']}")

    assert r["scores_nan"] == 0, r
    assert r["layers_observed"] > 0, (
        "the prefill fire reported no layers -- `on_attn` never bound, or the "
        f"capture path was refused and silently skipped: {r}"
    )

    # (1) THE PREFILL SIGNATURE. The decode capture writes one row per fire and
    # the row describes the position that was just appended; this row must
    # describe the prompt as it stood at prefill time. `observed_live` is one
    # past the last non-zero slot, computed on the host from the drained row.
    assert r["observed_live"] == r["prompt_len"], (
        f"the score row covers {r['observed_live']} positions but the prompt is "
        f"{r['prompt_len']} tokens; the capture is describing a different kv "
        f"length than the program thinks it fired on: {r}"
    )
    # Unlike H2O there is no seed ramp anywhere in this row, so slack is a bug,
    # not a policy choice: the normalize kernel writes exactly one entry per
    # live kv position and the fold reads exactly that region.
    assert r["tail_nonzero"] == 0, (
        f"{r['tail_nonzero']} slots past the prompt carry mass; the capture "
        f"buffer's slack was not cleared or kv_len is wrong: {r}"
    )
    assert r["live_scored"] == r["prompt_len"], (
        f"only {r['live_scored']} of {r['prompt_len']} prompt slots are finite "
        f"and non-negative: {r}"
    )

    # (2) Each layer folds one softmax distribution over the prefix -- averaged
    # over heads and over window rows, but NOT over layers -- so the total is
    # the layer count. This is the single strongest check that the fold divisor
    # is right: a divisor of `heads * window` instead of `heads * rows` would
    # scale a short window's mass down, and no normalization at all would land
    # at `heads * rows * layers`.
    assert abs(r["score_mass"] - r["layers_observed"]) < 0.02 * r["layers_observed"], (
        f"score mass {r['score_mass']} should equal the layer count "
        f"{r['layers_observed']} -- one folded distribution per layer: {r}"
    )
    print(f"  [ ok ] {r['layers_observed']} layers, mass {r['score_mass']:.4f}, "
          f"nothing past the prompt")


async def test_snapkv_window_is_at_the_end_of_the_prompt(client, args):
    r = await _run(client, args, "trackb-snapkv", {"page_budget": 4096})
    masses = [float(m) for m in r["page_mass"]]
    n = len(masses)
    window, win_lo = _window_pages(r)
    win_pages = n - win_lo

    print(f"  pages       = {n}  page_size = {r['page_size']}  window = {window}")
    print(f"  window      = pages {win_lo}..{n - 1}")
    print(f"  page_mass   = {r['page_mass']}")

    assert n >= 5, (
        f"the prompt only spans {n} pages -- SnapKV has nothing to select "
        "between; lengthen PROMPT"
    )
    assert 1 <= win_pages <= n - 2, (
        f"the observation window covers {win_pages} of {n} pages, leaving no "
        "middle to contrast it against -- adjust PROMPT or the window"
    )

    # (3) THE SNAPKV SIGNATURE. Causal attention from the last rows of the
    # prompt is strongly local, so the pages the observation window sits on must
    # carry more than the middle of the prompt. A tap that captured the FIRST
    # `window` rows, or one that dropped the causal mask, would spread mass
    # roughly uniformly over the prefix and fail here.
    #
    # The sink is held out of BOTH sides: it alone carries roughly half of all
    # attention mass, so a share-of-total figure says more about the sink than
    # about the window. What SnapKV actually relies on is that a window page
    # outweighs an ordinary prefix page.
    sink, mid, win = masses[0], masses[1:win_lo], masses[win_lo:]
    total = sum(masses)
    win_mean, mid_mean = sum(win) / len(win), sum(mid) / len(mid)
    assert win_mean > 1.3 * mid_mean, (
        f"a window page averages {win_mean:.5f} against {mid_mean:.5f} for an "
        f"ordinary prefix page -- only {win_mean / mid_mean:.2f}x. The window "
        f"is not concentrating attention, so ranking by it is not SnapKV: "
        f"{r['page_mass']}"
    )

    # The strong form of the same claim, but it is only meaningful when the
    # window is actually a window. Once it spans a third of the prompt the
    # "window" and the "middle" overlap in character and the contrast
    # legitimately washes out -- that is a statement about the CONFIGURATION,
    # not about the capture, so it is a precondition rather than a failure.
    if win_pages * 3 <= n:
        assert min(win) > max(mid), (
            f"the observation window's pages {win} do not separate from the "
            f"middle of the prompt (max {max(mid):.5f}); the capture is not "
            f"describing an END-of-prompt window: {r['page_mass']}"
        )
        assert win_mean > 3.0 * mid_mean, (
            f"a window page averages only {win_mean / mid_mean:.2f}x an "
            f"ordinary prefix page; at this window width the separation should "
            f"be far wider: {r['page_mass']}"
        )
    else:
        print(f"  [note] window spans {win_pages}/{n} pages -- too wide for the "
              "strong separation check; skipped")

    # ... and the attention sink at position 0 must still be visible, which is
    # the other half of the shape SnapKV relies on: it is why an evicting policy
    # must be told to keep page 0.
    assert sink > 3.0 * mid_mean, (
        f"page 0 ({sink:.5f}) carries no more than an ordinary prefix page "
        f"({mid_mean:.5f}); the attention sink is missing, which means these "
        f"are not real softmax rows: {r['page_mass']}"
    )
    print(f"  [ ok ] window page {win_mean:.4f} vs prefix page {mid_mean:.4f} "
          f"({win_mean / mid_mean:.1f}x), sink {sink:.4f} "
          f"({sink / total:.0%} of all mass)")


async def test_snapkv_window_mass_tapers_across_the_window(client, args):
    """(3b) THE CAUSAL MASK, READ OFF THE WINDOW'S OWN PROFILE -- on the
    filler-only prompt, because this is the claim `PROMPT` cannot carry.

    Within the window the mass must FALL toward the end, and that is not a
    quirk: window row `w` sees kv up to position `first_row + w`, so kv
    position `first_row + j` is weighted by only `window - j` of the `window`
    rows. The last page is structurally attended by the fewest rows, so a
    profile that PEAKED at the very last page would mean the mask was not
    applied to the captured rows.

    "Structurally" is the load-bearing word, and it is why this runs on
    `FILLER_PROMPT`. The argument counts ROWS, not tokens, so it only survives
    if every position in the window is worth the same to a row that can see
    it. On `PROMPT` the window's last page holds the question -- which the
    later rows attend to for its content -- and the taper would break on a
    model whose mask is perfect. One sentence repeated leaves the mask as the
    only thing that can shape the profile.
    """
    r = await _run(client, args, "trackb-snapkv",
                   {"prompt": FILLER_PROMPT, "page_budget": 4096})
    masses = [float(m) for m in r["page_mass"]]
    window, win_lo = _window_pages(r)
    win = masses[win_lo:]

    print(f"  pages       = {len(masses)}  page_size = {r['page_size']}  "
          f"window = {window}")
    print(f"  window mass = {[f'{m:.5f}' for m in win]}")

    # With a single-page window there is no interior to compare against, so the
    # claim would be vacuous rather than passing.
    assert len(win) >= 2, (
        f"the observation window covers {len(win)} page of {len(masses)} -- "
        "there is no interior to taper across; widen PIE_ATTN_SCORE_WINDOW or "
        "shorten the page size"
    )
    assert win[-1] < max(win), (
        f"the last page is the heaviest in the window {win}; the captured "
        "rows are not causally masked -- every row would have to see the "
        "end of the prompt for that to happen"
    )
    print(f"  [ ok ] the window tapers: last page {win[-1]:.4f} under the "
          f"window's peak {max(win):.4f}")


async def test_snapkv_enforces_its_keep_set(client, args):
    wide = await _run(client, args, "trackb-snapkv", {"page_budget": 4096})
    tight = await _run(client, args, "trackb-snapkv", {"page_budget": 1})
    base = await _run(client, args, "naive-baseline")

    print(f"  wide  = {wide['text']!r}")
    print(f"  tight = {tight['text']!r}")
    print(f"  base  = {base['text']!r}")

    assert tight["scores_nan"] == 0, tight
    assert tight["text"].strip(), tight
    # (4) The keep-set has to reach the kernel, not just the JSON.
    assert wide["text"] != tight["text"], (
        "the page-mask door is not enforced for SnapKV: a 1-page budget produced "
        "the same continuation as an all-page one"
    )
    assert wide["text"] == base["text"], (
        "an all-keep page mask perturbed the output; compaction is not "
        f"equivalent to the original page table\n  {wide['text']!r}\n"
        f"  {base['text']!r}"
    )
    print("  [ ok ] budget 1 diverges, full budget matches the baseline exactly")


async def test_snapkv_keeps_what_it_scored(client, args):
    budget = 3
    r = await _run(client, args, "trackb-snapkv", {"page_budget": budget})
    masses = [float(m) for m in r["page_mass"]]
    kept = r["kept_pages"]

    print(f"  kept      = {kept}  evict_next = {r['evicted_first']}")

    assert r["page_budget"] == min(budget, r["prompt_pages"]), r
    assert len(kept) == r["page_budget"], r
    # The host recomputes the ranking independently of the device; the keep-set
    # must be exactly the top-`budget` masses.
    ranked = sorted(range(len(masses)), key=lambda p: (-masses[p], -p))
    assert set(kept) == set(ranked[: len(kept)]), (
        f"kept set {kept} is not the top-{len(kept)} by mass {ranked}"
    )
    assert r["evicted_first"] == ranked[-1], r
    # SnapKV is a one-shot prefill decision: the pages it keeps are chosen
    # before a single token is generated, so the observation window -- the only
    # thing it looked at -- must be represented in the keep-set. Which
    # individual window page wins depends on the window width (the mass peak
    # moves toward the START of the window as it widens, because interior
    # positions are seen by more rows), so the claim is about the region.
    _, win_lo = _window_pages(r)
    assert any(p >= win_lo for p in kept), (
        f"no page of the observation window (pages {win_lo}..{len(masses) - 1}) "
        f"survived at budget {r['page_budget']}: kept {kept}"
    )
    print("  [ ok ] keep-set is the top-budget page mass and retains the window")


def tests():
    return [
        test_snapkv_observes_the_prompt_window,
        test_snapkv_window_is_at_the_end_of_the_prompt,
        test_snapkv_window_mass_tapers_across_the_window,
        test_snapkv_enforces_its_keep_set,
        test_snapkv_keeps_what_it_scored,
    ]


if __name__ == "__main__":
    from conftest import run_tests

    run_tests(
        tests(),
        "SnapKV prefill-window observation + enforcement",
        requires=("attn_score",),
    )
