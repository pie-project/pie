"""M-4 — the speculative mechanism, executed on the device: does the model's
draft head serve, and does speculating with it change the answer?

`mtp-speculative-decoding` runs the whole loop in the epilogue — verify,
commit, the next window and its geometry — and the host only drains committed
tokens. This file asks four things, in the order in which failing one makes
the next meaningless:

  1. THE HEAD IS THERE AND IT DRAFTED. `mtp_drafts` is bound only for a load
     whose text declares a draft head (`mtp_depth > 0`), so a run that
     completes has proved the head landed; the counters prove it drafted.
  2. THE LOOP WITHOUT DRAFTS IS THE BASELINE. `k = 0` is a one-row window
     through the same fire and the same commit arithmetic; its text must equal
     `naive-baseline`'s greedy text. A failure here is under the speculation,
     not in it.
  3. THE ANSWER IS THE ANSWER. Any `k` answers the `k = 0` text, character
     for character — true BY CONSTRUCTION for any head, since verification
     keeps the trunk's own argmax. A failure is the verify, the row alignment,
     or the loop-carried length being wrong, never the head being bad.
  4. AND IT HOLDS IN A CROWD. The same run beside plain lanes answers the same
     text, and so do the plain lanes.

Acceptance is REPORTED AND NOT ASSERTED: how often the head agrees with the
trunk is a property of the head (a synthetic head, or a real MTP module over
a truncated miniature, agrees rarely; the mechanism is what this gates).

**THIS FILE NEEDS A DRAFTING ARTIFACT** — a base checkpoint with the head
overlaid by `pie model import <base> --aux <head>`. Against a plain artifact
`mtp_depth` is zero and the run refuses at bind. Run it with `--model`
pointing at the overlay (a store name or a `.zt` path), e.g.

    uv run python tests/inferlets/test_eagle.py --engine metal \
        --model /tmp/warmstream/dsv4-mini-mtp.zt
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet  # noqa: E402

#: One experiment, restated once. Greedy on both sides — the identity is a
#: claim about the SAME decision rule reached two ways, so a temperature on
#: either would make the comparison meaningless rather than tolerant.
#:
#: `naive-baseline` DRAWS rather than argmaxes: its epilogue temperature-scales
#: and takes a Gumbel-max sample, so a default-temperature run is not the
#: decision rule speculation is verified against. 0.01 with a fixed seed is
#: what A-1's own parity gate used to pin the same model to its argmax, and it
#: is used here for the same reason and with the same numbers.
PROMPT = "The capital of France is"
MAX_TOKENS = 24
#: Asked for; the inferlet caps it at the model's depth and reports what it used.
K = 8
GREEDY = {"temperature": 0.01, "seed": 7}

#: How many plain lanes ride beside the speculative one in claim 3. Enough that
#: the fire is genuinely mixed and small enough that the gate stays a gate.
CROWD = 3


async def _spec(client, args, extra=None):
    out = await run_inferlet(
        client,
        "mtp-speculative-decoding",
        {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "k": K, "margin": True, **(extra or {})},
        timeout=args.timeout,
    )
    return json.loads(out[out.find("{"):])


def _same_prefix(control, base):
    """The loop stops at the template's stop tokens; `naive-baseline` does not.
    So the baseline is read up to the loop's length: the loop's tokens must be
    the baseline's first tokens, exactly."""
    n = len(control["tokens"])
    return n > 0 and base["tokens"][:n] == control["tokens"]


async def _base(client, args):
    out = await run_inferlet(
        client,
        "naive-baseline",
        {"prompt": PROMPT, "max_tokens": MAX_TOKENS, **GREEDY},
        timeout=args.timeout,
    )
    return json.loads(out[out.find("{"):])


#: Below this top-1/top-2 gap (bf16 logit units) an argmax is a tie: fires of
#: different widths take different kernel tilings on the routed two-bit path
#: and part by up to ~1.4 logits at the readout (measured in
#: `engine-metal/tests/a_drafting_neighbour_leaves_a_plain_lane_alone`, with
#: no head anywhere), so which side of a gap narrower than that they land on
#: is the floor, not the mechanism. The claim that owes no floor is the
#: shape-matched one below, asked exactly.
TIE = 1.5


def _round_of(run, index):
    """The verify round that committed token `index` (`0` is the prefill's seed)."""
    at = 1
    for r, n in enumerate(run["commits_trace"]):
        if index < at + n:
            return r
        at += n
    return None


def _same_up_to_a_tie(control, run, what):
    a, b = control["tokens"], run["tokens"]
    div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if div is None:
        return "identical"
    r = _round_of(control, div)
    margin = control["margin_trace"][r] if r is not None and r < len(control["margin_trace"]) else None
    assert margin is not None and margin < TIE, (
        f"{what}: the runs diverge at token {div} (round {r}) where the control's "
        f"top-2 margin is {margin}, wider than a tie ({TIE}) — that is the verify, the "
        f"row alignment or the loop-carried length, not the bf16 floor:\n"
        f"  control = {control['text']!r}\n  run     = {run['text']!r}"
    )
    return f"identical for {div} tokens, then a tie (margin {margin:.3f})"


async def test_the_overlaid_head_serves_and_drafts(client, args):
    r = await _spec(client, args)
    print(f"  rounds = {r['rounds']}  drafted = {r['drafted']}  accepted = {r['accepted']}")
    print(f"  acceptance_rate = {r['acceptance_rate']:.3f}  k = {r['k']}")

    assert r["count"] > 0, f"the speculative run generated nothing: {r}"
    assert r["rounds"] > 0, (
        "no verify round committed a token, so the loop never turned and every "
        "claim below is about a run that did not happen"
    )
    assert 0 < r["k"] <= r["depth"], f"k={r['k']} against depth={r['depth']}: {r}"
    assert r["drafted"] == r["rounds"] * r["k"], (
        f"{r['drafted']} drafts over {r['rounds']} rounds at k={r['k']}; a round that "
        f"drafted a different number is a window whose shape moved"
    )
    assert r["accepted"] <= r["drafted"], r
    print("  [ ok ] the overlaid head is bound, and it drafted")


async def test_the_loop_without_drafts_is_the_baseline(client, args):
    """The control, and the one that says WHERE a failure is.

    `k = 0` runs the same loop with a one-row window: nothing is drafted,
    nothing is verified, and what is left is the geometry, the pages, the
    positions, the commit arithmetic and the KV rollback — through the same
    `fire`, the same working set and the same code that speculation uses. If
    this passes and the widths below do not, the defect is in the speculation
    and provably not in the machinery under it; if this fails, nothing below it
    means anything. `cacheback-speculative-decoding` states the same trick for
    the same reason under the name `draft_length`.
    """
    control = await _spec(client, args, {"k": 0})
    base = await _base(client, args)
    print(f"  k=0      = {control['text']!r}")
    print(f"  baseline = {base['text']!r}")

    assert control["drafted"] == 0, f"k=0 drafted {control['drafted']} tokens"
    assert _same_prefix(control, base), (
        "the loop with NO drafts answered differently from the non-speculative "
        "run, so the defect is under the speculation and not in it — the "
        "window's geometry, its pages, or the length it carries.\n"
        f"  k=0      = {control['text']!r}\n"
        f"  baseline = {base['text']!r}"
    )
    print("  [ ok ] the machinery under the speculation is the machinery")


async def test_the_drafts_change_nothing_at_the_same_fire_shape(client, args):
    """The mechanism claim with the width held: `k = 0` padded to the window's
    width (pad rows carry the correction again and are rejected like any wrong
    draft) against `k = depth` — the same lanes, the same fires, the same
    kernel paths, drafts or no drafts. Verification keeps the trunk's argmax,
    so the two texts are the same text EXACTLY; nothing here is a tie.
    """
    wide = await _spec(client, args)
    padded = await _spec(client, args, {"k": 0, "pad": wide["k"]})
    print(f"  k={wide['k']}       = {wide['text']!r}")
    print(f"  k=0 pad={wide['k']} = {padded['text']!r}")
    assert padded["drafted"] == 0, f"a padded control drafted {padded['drafted']}"
    assert padded["tokens"] == wide["tokens"], (
        "the same fire shape answered differently with drafts in it than without, "
        "so the drafts are changing the answer — the verify, the row alignment or "
        "the loop-carried length:\n"
        f"  k={wide['k']}       = {wide['text']!r}\n  k=0 pad={wide['k']} = {padded['text']!r}"
    )
    print("  [ ok ] at one shape, the drafts buy fires and nothing else")


async def test_the_draft_width_does_not_change_the_answer(client, args):
    """Claim 3, asked without a second program: `k` is how many tokens a round
    proposes and nothing else, so `k = 0`, `k = 1` and `k = depth` must answer
    one string — up to a tie. A one-row window and a many-row one take
    different kernel paths and differ in the last bits, so an argmax the
    control decided by less than `TIE` may fall either way; a divergence at a
    wider margin is the window's geometry, the KV rollback or the commit
    arithmetic, and is the failure this asserts against.
    """
    control = await _spec(client, args, {"k": 0})
    narrow = await _spec(client, args, {"k": 1})
    wide = await _spec(client, args)
    print(f"  k=0 = {control['text']!r}")
    print(f"  k=1 = {narrow['text']!r}")
    print(f"  k={wide['k']} = {wide['text']!r}  (acceptance {wide['acceptance_rate']:.3f})")
    for name, run in (("k=1", narrow), (f"k={wide['k']}", wide)):
        print(f"  {name} against k=0: {_same_up_to_a_tie(control, run, name)}")
    print("  [ ok ] the width buys fires and nothing else")


async def test_greedy_speculation_answers_the_greedy_baseline(client, args):
    """The claim through a second program: the speculative run answers the
    non-speculative greedy baseline. Exact through `k = 0` (one row, the same
    kernel path as the baseline), and up to a tie at full width.
    """
    control = await _spec(client, args, {"k": 0})
    spec = await _spec(client, args)
    base = await _base(client, args)
    print(f"  baseline    = {base['text']!r}")
    print(f"  speculative = {spec['text']!r}")
    assert base["text"], f"the baseline generated no text: {base}"
    assert _same_prefix(control, base), (
        "the one-row loop answered a different string from the baseline:\n"
        f"  baseline = {base['text']!r}\n  k=0      = {control['text']!r}"
    )
    print(f"  speculative against k=0: {_same_up_to_a_tie(control, spec, 'speculative')}")
    print("  [ ok ] byte for byte to the floor, and the head only changed how fast it got there")


async def test_it_holds_beside_non_speculative_lanes(client, args):
    """Claim 4: mixed in one fire.

    The lanes are launched concurrently and the scheduler composes whatever
    arrives together; what this asserts is the speculative lane's answer with
    neighbours, against its answer alone, up to a tie. The plain lanes are
    REPORTED and not asserted: their fires are wider beside a window than
    alone, and a plain lane's logits move with the fire's width by up to the
    floor with no head anywhere (`a_drafting_neighbour_leaves_a_plain_lane_
    alone` holds a window against a plain crowd of the SAME width exactly);
    the widths of a crowd cannot be matched across two runs, since a window
    that accepts drafts ends its rounds sooner than one that does not.
    """
    import asyncio

    alone = await _base(client, args)
    spec_alone = await _spec(client, args)
    control = await _spec(client, args, {"k": 0})
    results = await asyncio.gather(
        _spec(client, args),
        *[_base(client, args) for _ in range(CROWD)],
    )
    spec, crowd = results[0], results[1:]
    print(f"  speculative in a crowd of {CROWD} = {spec['text']!r}")
    print(f"  mixed against alone: {_same_up_to_a_tie(spec_alone, spec, 'speculative in a crowd')}")
    assert _same_prefix(control, alone), (
        "the one-row loop and the baseline disagree, so the crowd has no control:\n"
        f"  baseline = {alone['text']!r}\n  k=0      = {control['text']!r}"
    )
    for at, plain in enumerate(crowd):
        a, b = control["tokens"], plain["tokens"]
        div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        r = _round_of(control, div) if div is not None else None
        margin = control["margin_trace"][r] if r is not None and r < len(control["margin_trace"]) else None
        print(f"  plain lane {at}: " + ("identical" if div is None else
              f"parts at token {div} (control margin {margin}) — the width floor, reported not asserted"))
    print("  [ ok ] the drafts window is a window")


def tests():
    return [
        test_the_overlaid_head_serves_and_drafts,
        test_the_loop_without_drafts_is_the_baseline,
        test_the_drafts_change_nothing_at_the_same_fire_shape,
        test_the_draft_width_does_not_change_the_answer,
        test_greedy_speculation_answers_the_greedy_baseline,
        test_it_holds_beside_non_speculative_lanes,
    ]


if __name__ == "__main__":
    from conftest import run_tests

    # `mtp_depth` is exactly "does this load's text declare a draft head".
    # Against a drafting artifact it is positive; against a plain one the
    # suite refuses at bind, which is the honest failure and not a skip.
    run_tests(
        tests(),
        description="M-4 speculative mechanism (needs a drafting artifact, "
        "e.g. the dsv4 `--aux` overlay)",
    )
