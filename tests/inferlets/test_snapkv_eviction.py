"""S-4 — SnapKV eviction, executed: does the cut shrink the cache, and does the
needle survive it?

`test_snapkv.py` asks whether the SCORES are right. This asks the only question
that matters after that: if a program acts on them, does the model still work?

The gate is `.wiki/alto/campaign.md` §2's S-4 — "one eviction emulation (SnapKV
path) demonstrably shrinks the served mask AND still answers a needle prompt
correctly" — and it is asked as four claims, in the order in which failing one
makes the next meaningless:

  1. THE CUT HAPPENED. The decode is served strictly fewer prompt pages, and
     strictly fewer KV positions, than the prefill wrote. A gate that only
     checked the answer would pass on a program that evicted nothing.
  2. THE CUT WAS INFORMED. The kept set is the top of the device-folded page
     mass, plus the two pages that are not a choice (the sink and the write
     page), and the page the needle actually landed in is in it. That page is
     computed on this side from the tokenizer's own offsets, so the policy and
     the check are not reading the same number.
  3. THE ANSWER SURVIVED. The evicted run still produces the needle. This is
     the claim; everything above exists so that it means something.
  4. THE CUT IS WHAT DID IT. One page tighter, the needle's page falls off the
     bottom of the mass ranking and the answer must be LOST. Without this,
     claim 3 is satisfiable by a model that can answer from the question alone,
     and the whole file would be measuring nothing.

Run from the repo root with PYTHONPATH=sdk/server/python/python.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet  # noqa: E402


#: The answer the needle carries. Deliberately unlike anything in the filler,
#: so "did it survive" is a substring check and not a judgement call.
ANSWER = "banana"

#: The inferlet's own defaults, restated here so the gate reads as one
#: experiment rather than as four calls that happen to agree. `filler: 6` is
#: the model's recall limit and not the policy's (the inferlet's `filler` doc
#: argues it); `depth: 0.6` lands the needle wholly inside one middle page,
#: away from both regions a keep-set gets for free.
COMMON = {"filler": 6, "depth": 0.6}

#: The budget that keeps the needle's page, and the one that does not. Both
#: are cuts; only one of them is a cut the answer survives.
KEEPS = 5
LOSES = 4


async def _run(client, args, extra=None):
    out = await run_inferlet(
        client, "snapkv-eviction", {**COMMON, **(extra or {})}, timeout=args.timeout
    )
    return json.loads(out[out.find("{"):])


async def test_the_eviction_shrinks_the_served_cache(client, args):
    r = await _run(client, args, {"page_budget": KEEPS})
    print(f"  prompt_len   = {r['prompt_len']}  pages = {r['prompt_pages']}")
    print(f"  served pages = {r['served_pages']}  kept = {r['kept_pages']}")
    print(f"  served kv    = {r['served_kv']} of {r['full_kv']}")

    assert r["evicted"], r
    # The prompt has to span enough pages that the budget is a real cut.
    assert r["prompt_pages"] > KEEPS, (
        f"the haystack is only {r['prompt_pages']} pages against a {KEEPS}-page "
        f"budget; there is nothing to evict and this file is testing nothing"
    )
    assert r["served_pages"] < r["prompt_pages"], (
        f"the decode was served {r['served_pages']} of {r['prompt_pages']} prompt "
        f"pages, which is not an eviction"
    )
    assert r["served_kv"] < r["full_kv"], (
        f"the decode attends over {r['served_kv']} positions against a full "
        f"{r['full_kv']}; the page list shrank but the KV length did not, which "
        f"means the geometry is describing pages the list no longer names"
    )
    # The mass is self-validating: one distribution per exported layer.
    assert abs(r["score_mass"] - r["layers_observed"]) < 0.05 * r["layers_observed"], (
        f"the page fold carries {r['score_mass']} of mass against "
        f"{r['layers_observed']} layers; the rectangle it was folded from is not "
        f"a set of distributions"
    )
    print("  [ ok ] the cut happened, and the page fold is a distribution")


async def test_the_keep_set_is_the_mass_and_holds_the_needle(client, args):
    r = await _run(client, args, {"page_budget": KEEPS})
    masses = [float(m) for m in r["page_mass"]]
    kept = set(r["kept_pages"])
    print(f"  needle pages = {r['needle_pages']}  kept = {sorted(kept)}")

    # The two that are not a choice.
    assert 0 in kept, "the attention sink was evicted"
    assert r["prompt_pages"] - 1 in kept, "the write page was evicted"

    # Everything else the budget bought is the top of the mass. Ranked over the
    # pages that were NOT forced, so the forced pair does not flatter the claim.
    forced = {0, r["prompt_pages"] - 1}
    chosen = kept - forced
    if chosen:
        rest = [p for p in range(r["prompt_pages"]) if p not in forced]
        rest.sort(key=lambda p: (-masses[p], p))
        assert set(rest[: len(chosen)]) == chosen, (
            f"the keep-set is not the top of the page mass: chose {sorted(chosen)}, "
            f"the mass ranks {rest[: len(chosen)]}"
        )

    # AND THE POLICY KEPT WHAT IT NEEDED. The needle's page is derived from the
    # tokenizer's offsets on the guest side and reported; the policy never saw
    # it. A keep-set that dropped it would still satisfy every claim above.
    assert any(p in kept for p in r["needle_pages"]), (
        f"the needle lives in pages {r['needle_pages']} and the policy kept "
        f"{sorted(kept)}; SnapKV's own observation window did not find the fact "
        f"the question asks for"
    )
    print("  [ ok ] the keep-set is the mass, and the needle's page is in it")


async def test_the_needle_survives_the_cut(client, args):
    evicted = await _run(client, args, {"page_budget": KEEPS, "evict": True})
    whole = await _run(client, args, {"page_budget": KEEPS, "evict": False})
    print(f"  whole   ({whole['served_pages']} pages): {whole['text']!r}")
    print(f"  evicted ({evicted['served_pages']} pages): {evicted['text']!r}")

    # The control arm has to be able to answer at all, or the experiment has no
    # baseline: a model that cannot recall the needle from the FULL cache says
    # nothing about what eviction cost.
    assert ANSWER in whole["text"], (
        f"the control arm did not recall the needle from the whole prompt: "
        f"{whole['text']!r}. Nothing below can be measured against this."
    )
    assert ANSWER in evicted["text"], (
        f"the needle did not survive the eviction: served "
        f"{evicted['served_pages']} of {evicted['prompt_pages']} pages and "
        f"answered {evicted['text']!r}"
    )
    print("  [ ok ] a needle answered from a strictly smaller cache")


async def test_a_cut_that_drops_the_needles_page_loses_it(client, args):
    """THE NEGATIVE CONTROL, and without it claim 3 proves nothing.

    One page tighter, and the needle's page falls off the bottom of the mass
    ranking. If the model still answers, it is answering from the question and
    not from the cache, and every other assertion in this file is decoration.

    The two budgets differing by ONE page is the point: it is the same prompt,
    the same scores, the same policy, and the only thing that moved is whether
    the page SnapKV ranked third survived. That is as close as a gate can get
    to attributing the answer to the eviction itself.
    """
    tight = await _run(client, args, {"page_budget": LOSES})
    print(f"  tight   ({tight['served_pages']} pages): {tight['text']!r}")
    assert tight["served_pages"] <= LOSES, tight
    assert not any(p in set(tight["kept_pages"]) for p in tight["needle_pages"]), (
        f"a two-page budget still kept the needle's page {tight['needle_pages']}; "
        f"this control cannot say anything"
    )
    assert ANSWER not in tight["text"], (
        f"the model answered {tight['text']!r} from two pages that do not contain "
        f"the needle, so it is not reading the cache and the survival claim above "
        f"is not about eviction"
    )
    print("  [ ok ] the needle is lost when its page is, so the cache is what answered")


def tests():
    return [
        test_the_eviction_shrinks_the_served_cache,
        test_the_keep_set_is_the_mass_and_holds_the_needle,
        test_the_needle_survives_the_cut,
        test_a_cut_that_drops_the_needles_page_loses_it,
    ]


if __name__ == "__main__":
    from conftest import run_tests

    run_tests(
        tests(),
        "S-4: SnapKV eviction shrinks the cache and keeps the answer",
        requires=("attn_score",),
    )
