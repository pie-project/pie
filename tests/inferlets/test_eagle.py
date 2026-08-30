"""M-4 — the EAGLE mechanism, executed: does an overlaid draft head serve, and
does speculating with it change the answer?

The campaign's box (§2, M-4) is "overlay import (`--aux`) bakes a second
artifact; a SYNTHETIC head passes the identity gate (greedy outputs identical
to non-speculative, by verify construction); the draft-verify inferlet runs
mixed with non-spec lanes in one fire". This file asks that as three claims, in
the order in which failing one makes the next meaningless:

  1. THE HEAD IS THERE AND IT DRAFTED. `mtp_logits` is a capability the shell
     advertises only for a load whose text declares a draft head, so a run that
     completes at all has already proved the overlay landed and bound. What it
     has NOT proved is that anything was drafted, so the round and draft
     counters are read: a loop that proposed nothing would pass claim 2 by
     doing no speculation whatsoever.
  2. THE ANSWER IS THE ANSWER. The greedy speculative run's text equals the
     greedy non-speculative run's text, character for character. This is the
     claim, and it is true BY CONSTRUCTION for any head — verification keeps
     the target's own argmax and discards a draft that disagrees — which is
     exactly why a synthetic head gates the MECHANISM honestly. A failure here
     is never "the head is bad"; it is the verify loop, the row alignment, the
     kv rollback or the loop-carry being wrong.
  3. AND IT HOLDS IN A CROWD. The same run, launched concurrently with plain
     non-speculative lanes, answers the same text. Speculative lanes and
     ordinary ones compose into one fire — that is the polymorphic-batching
     thesis and the drafts window is one more axis of it — so a draft head that
     was right alone and wrong beside a neighbour would be a window that is not
     a window.

Acceptance is REPORTED AND NOT ASSERTED. The synthetic head is one decoder
layer over the trunk's final hidden (`tests/eagle/synthesize_head.py` argues
the construction); how often that agrees with the target is a property of the
head, and gating on it would make this file a quality measurement of an
artifact the campaign explicitly says is not the deliverable (§3: "a REAL EAGLE
head is out; M-4 gates the mechanism with a synthetic head").

**THIS FILE NEEDS THE EAGLE ARTIFACT**, not the plain one — a base checkpoint
with the head overlaid by `pie model import <base> --aux <head>`. Against a
plain artifact the load has no draft head, `mtp_logits` is not advertised, and
the run refuses at bind. Run it the way the other standalone gates are run,
with `--model` pointing at the overlay.
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
K = 4
GREEDY = {"temperature": 0.01, "seed": 7}

#: How many plain lanes ride beside the speculative one in claim 3. Enough that
#: the fire is genuinely mixed and small enough that the gate stays a gate.
CROWD = 3


async def _spec(client, args, extra=None):
    out = await run_inferlet(
        client,
        "mtp-speculative-decoding",
        {"prompt": PROMPT, "max_tokens": MAX_TOKENS, "k": K, **(extra or {})},
        timeout=args.timeout,
    )
    return json.loads(out[out.find("{"):])


async def _base(client, args):
    out = await run_inferlet(
        client,
        "naive-baseline",
        {"prompt": PROMPT, "max_tokens": MAX_TOKENS, **GREEDY},
        timeout=args.timeout,
    )
    return json.loads(out[out.find("{"):])


async def test_the_overlaid_head_serves_and_drafts(client, args):
    r = await _spec(client, args)
    print(f"  rounds = {r['rounds']}  drafted = {r['drafted']}  accepted = {r['accepted']}")
    print(f"  acceptance_rate = {r['acceptance_rate']:.3f}  k = {r['k']}")

    assert r["count"] > 0, f"the speculative run generated nothing: {r}"
    assert r["rounds"] > 0, (
        "no verify round committed a token, so the loop never turned and every "
        "claim below is about a run that did not happen"
    )
    assert r["drafted"] == r["rounds"] * K, (
        f"{r['drafted']} drafts over {r['rounds']} rounds at k={K}; a round that "
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
    assert control["text"] == base["text"], (
        "the loop with NO drafts answered differently from the non-speculative "
        "run, so the defect is under the speculation and not in it — the "
        "window's geometry, its pages, or the length it carries.\n"
        f"  k=0      = {control['text']!r}\n"
        f"  baseline = {base['text']!r}"
    )
    print("  [ ok ] the machinery under the speculation is the machinery")


async def test_the_fire_shape_does_not_change_the_answer(client, args):
    """The variable the width gate confounds, held on its own.

    `qo_one` is a fact, so a one-row window takes the decode arm and a
    multi-row one takes prefill. `k = 0` against `k = 4` therefore moves TWO
    things — the fire's shape and the presence of drafts — and cannot say which
    one a divergence belongs to. `pad` adds rows the loop ignores, so this asks
    the shape alone: a padded no-draft window must answer what an unpadded one
    answers.

    It passes at `pad = 1` — and that is the whole of what it proves, because
    `pad = 2` does NOT (`.wiki/alto/multimodal.md` §17). One passenger row
    carrying the pending token again is one extra fold of a token the recurrent
    state already holds; two are not. So read this as "the decode and prefill
    arms answer the same argmax" and nothing wider: the width gate below is red
    for the state and not for the drafts.
    """
    flat = await _spec(client, args, {"k": 0})
    padded = await _spec(client, args, {"k": 0, "pad": 1})
    print(f"  1 row  = {flat['text']!r}")
    print(f"  2 rows = {padded['text']!r}")

    assert padded["drafted"] == 0, f"a padded control drafted {padded['drafted']}"
    assert padded["text"] == flat["text"], (
        "the same loop with no drafts answered differently at two fire shapes, "
        "so the decode and prefill arms disagree on the same keys and every "
        "claim about speculation below is measuring that instead.\n"
        f"  1 row  = {flat['text']!r}\n"
        f"  2 rows = {padded['text']!r}"
    )
    print("  [ ok ] prefill and decode answer the same argmax")


async def test_the_draft_width_does_not_change_the_answer(client, args):
    """The claim, asked WITHOUT a second program.

    `k` is how many tokens a round proposes and nothing else: verification
    keeps the target's argmax at every position whatever the window's width, so
    two runs that differ only in `k` must answer the same string. If the
    window's geometry, the KV rollback or the accepted-count arithmetic were
    wrong, the two would diverge — and this asks it against the same program,
    the same artifact and the same decision rule, so nothing about a second
    inferlet's sampler can flatter or spoil it.

    **IT IS RED, AND §17 SAYS WHY — NOT §14.** The KV is not the carrier: on a
    NON-recurrent SKU (gemma-4-E4B, measured) the same fire shapes answer
    bit-identically at every `pad` width and every pad token, and on the hybrid
    default SKU masking the rewritten cell out does not remove the dependence.
    What carries it is the gated-delta state, which three layers in four keep:
    a round folds all `w` rows into it, the loop accepts one, and nothing
    retracts the rest — "overwritten by the next fire" is a sentence about
    addressed KV cells and says nothing about a fold. `pad=1` carrying the
    pending token again survives because it is one extra fold of a token that
    is already there; `pad=2` and up do not (`pad=2` diverges at round 5,
    which is what falsified §14's reading of the same knob).
    """
    wide = await _spec(client, args)
    narrow = await _spec(client, args, {"k": 1})
    print(f"  k={K} = {wide['text']!r}")
    print(f"  k=1 = {narrow['text']!r}")

    assert narrow["text"] == wide["text"], (
        "the same loop answered differently at two draft widths, so `k` is "
        "changing the answer and not only the number of fires:\n"
        f"  k={K} = {wide['text']!r}\n"
        f"  k=1  = {narrow['text']!r}"
    )
    print("  [ ok ] the width buys fires and nothing else")


async def test_greedy_speculation_answers_the_greedy_baseline(client, args):
    spec = await _spec(client, args)
    base = await _base(client, args)
    print(f"  baseline    = {base['text']!r}")
    print(f"  speculative = {spec['text']!r}")

    assert base["text"], f"the baseline generated no text: {base}"
    assert spec["text"] == base["text"], (
        "the speculative run answered a different string from the non-speculative "
        "one. Verification keeps the target's own argmax, so this is never the "
        "draft head being wrong — it is the verify comparison, the draft row "
        "alignment, or the loop-carried length.\n"
        f"  baseline    = {base['text']!r}\n"
        f"  speculative = {spec['text']!r}"
    )
    print("  [ ok ] byte for byte, and the head only changed how fast it got there")


async def test_it_holds_beside_non_speculative_lanes(client, args):
    """Claim 3: mixed in one fire.

    The lanes are launched concurrently and the scheduler composes whatever
    arrives together; what this asserts is the OUTCOME of that composition —
    the speculative lane's answer, and every plain lane's answer, are the
    answers each gives alone. A run where the two never met would pass, which
    is why the concurrency is what it is and the assertion is not about it.
    """
    import asyncio

    alone = await _base(client, args)
    results = await asyncio.gather(
        _spec(client, args),
        *[_base(client, args) for _ in range(CROWD)],
    )
    spec, crowd = results[0], results[1:]
    print(f"  speculative in a crowd of {CROWD} = {spec['text']!r}")

    assert spec["text"] == alone["text"], (
        "the speculative lane answered differently with neighbours in the fire "
        f"than without:\n  alone = {alone['text']!r}\n  mixed = {spec['text']!r}"
    )
    for at, plain in enumerate(crowd):
        assert plain["text"] == alone["text"], (
            f"plain lane {at} answered differently beside a speculative one:\n"
            f"  alone = {alone['text']!r}\n  mixed = {plain['text']!r}"
        )
    print("  [ ok ] the drafts window is a window, and the neighbours never saw it")


def tests():
    return [
        test_the_overlaid_head_serves_and_drafts,
        test_the_loop_without_drafts_is_the_baseline,
        test_the_fire_shape_does_not_change_the_answer,
        test_the_draft_width_does_not_change_the_answer,
        test_greedy_speculation_answers_the_greedy_baseline,
        test_it_holds_beside_non_speculative_lanes,
    ]


if __name__ == "__main__":
    from conftest import run_tests

    # `mtp_logits` is NOT in `conftest.UNADVERTISED`, and that is the point of
    # this wave: the cuda shell answers `has_mtp_logits: shell.drafts()`, so
    # the capability is exactly "does this load's text declare a draft head".
    # Against the overlay artifact it is true; against a plain one the suite
    # refuses at bind, which is the honest failure and not a skip.
    run_tests(tests(), description="M-4 EAGLE mechanism (needs the --aux overlay artifact)")
