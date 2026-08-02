"""Does `attn_page_mask` actually change what attention reads?

Every Quest test until now asserted on the RANKING the program computed. That
proves the observation works; it says nothing about enforcement, because a
program that ranks pages perfectly and is then ignored produces exactly the
same statistics. This asks the only question that separates the two: with a
budget far below the request's page count, does the model produce a DIFFERENT
continuation than the same model attending over everything?

Run from the repo root with PYTHONPATH=sdk/server/python/python.
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


async def _quest(client, args, budget):
    out = await run_inferlet(
        client,
        "quest-attention",
        {
            "prompt": PROMPT,
            "max_tokens": 12,
            "page_budget": budget,
            "temperature": 0.001,
            "seed": 12345,
        },
        timeout=args.timeout,
    )
    report = json.loads(out[out.find("{"):])
    return report


async def _baseline(client, args):
    out = await run_inferlet(
        client,
        "naive-baseline",
        {
            "prompt": PROMPT,
            "max_tokens": 12,
            "temperature": 0.001,
            "seed": 12345,
        },
        timeout=args.timeout,
    )
    return json.loads(out[out.find("{"):])


async def test_mask_is_enforced(client, args):
    # A budget that keeps almost everything, and one that keeps almost nothing.
    wide = await _quest(client, args, 1024)
    tight = await _quest(client, args, 1)

    print(f"  max_pages    = {wide['max_pages']}")
    print(f"  wide  budget = {wide['page_budget']}  text = {wide['text']!r}")
    print(f"  tight budget = {tight['page_budget']}  text = {tight['text']!r}")

    # The tight run must still be a well-formed run: the tap fired, the pages
    # were scored, nothing went NaN. Enforcement must not corrupt the pass.
    assert tight["layers_observed"] > 0, tight
    assert tight["pages_nan"] == 0, tight
    assert tight["text"].strip(), tight

    # ...and it must actually differ. If these agree, the mask is being
    # computed and discarded -- the exact failure this test exists to catch.
    assert wide["text"] != tight["text"], (
        "attn_page_mask is NOT enforced: attending over 1 page produced the "
        f"same continuation as attending over {wide['page_budget']} pages\n"
        f"  {wide['text']!r}"
    )
    print("  [ ok ] budget=1 diverges from budget=1024 -> mask is enforced")

    # Divergence alone would also be produced by a mask that scrambles the page
    # table. The other half of the claim is that a mask which keeps everything
    # changes nothing: same prompt, same seed, no policy at all. If the
    # compaction reorders pages, drops the wrong one, or hands FlashInfer a
    # `last_page_len` that no longer matches its list, this is where it shows.
    base = await _baseline(client, args)
    print(f"  baseline           text = {base['text']!r}")
    assert wide["text"] == base["text"], (
        "attn_page_mask with an all-keep mask perturbed the output; the "
        "compacted page table is not equivalent to the original\n"
        f"  quest    {wide['text']!r}\n"
        f"  baseline {base['text']!r}"
    )
    print("  [ ok ] full budget reproduces the unmasked baseline exactly")


if __name__ == "__main__":
    from conftest import run_tests

    run_tests([test_mask_is_enforced], "attn_page_mask enforcement")
