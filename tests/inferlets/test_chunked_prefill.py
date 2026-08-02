"""Chunked prefill must equal one-shot prefill, and must lift the ceiling.

Every policy here exists for long contexts, but each one's prefill was a single
fire, and a single fire cannot exceed the driver's structural per-launch token
capacity (`max_embed_length()`, 8192 on this CUDA driver). That put a hard
ceiling on the context they could be run at -- *below* the range where sections
14 and 15 show they start to pay for themselves, which made the ceiling a
correctness-shaped hole in the features rather than a mere limit.

Two things have to be true of the fix.

**1. Equivalence.** Chunk `i` attends over the whole prefix written so far and
writes only its own tokens, so concatenating `ceil(n/C)` chunks should produce
what the one-shot fire produced. "Should" is not a test: forcing the chunk width
down runs the multi-chunk path on a short prompt, and the result must match the
one-shot run.

*What "match" can mean here is the subtle part, and getting it wrong is what
this file is really about.* Chunking changes the attention kernel's tile
decomposition -- 28 fires of 37 tokens do not reduce in the same order as one
fire of 1024 -- so the prompt's hidden states differ in their last bits, exactly
as section 11.4 records for decode batch shape. Those bits are written into the
KV cache and are still there during decoding.

So the assertion has to be pinned to a prompt whose continuation is **decisive**.
On an ambiguous prompt the next-token distribution is nearly flat, a 1-ulp logit
difference flips the argmax, and near-greedy decoding amplifies that into
completely different -- but equally coherent -- text. That is not a bug, and a
test that fails on it is measuring float associativity, not correctness. On a
prompt whose answer the model is sure of, the argmax has a real margin, the
1-ulp difference cannot flip it, and the texts agree exactly. They do, for 32
tokens, which is the assertion below.

**2. The ceiling is gone.** A prompt longer than `max_embed_length()` has to run
at all, produce coherent text, and still have its page budget enforced.

**3. SnapKV's observation window has to survive the split.** SnapKV is the only
*prefill* policy here, so chunking is not mechanical for it: its whole premise
is that it ranks pages by how much the prompt's **tail** attended to them, and
the capture hook's window is the last `window` rows *of the forward chunk it is
attached to*. Attach the tap to the wrong chunk and the policy silently ranks by
an observation of the middle of the prompt -- coherent-looking, completely
wrong, and invisible without a reference. So the tap is attached to the final
chunk only (which also means the earlier chunks do not pay the capture variant's
1.2-1.9x cost), and the chunking is made *even* rather than greedy so the final
chunk cannot be a 1-token sliver too short to hold the window.

`tail_page_share` -- the fraction of total observed mass sitting on the last
page of the whole prompt -- is the discriminator. Only a tap on the final chunk
can put mass there; a tap on any earlier chunk reports zero for it.

Run from the repo root with PYTHONPATH=sdk/server/python/python:

    PIE_CUDA_KV_ENVELOPES=1 python tests/inferlets/test_chunked_prefill.py \
        --driver cuda_native --model <path>
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from conftest import run_inferlet, run_tests  # noqa: E402

_UNIT = "Paris is a large European city with a long history. "
_HEAD = "The capital of France is Paris. "
# The prompt ends in a question the model answers with a large margin, so the
# argmax cannot be flipped by reduction order. See the module docstring.
_DECISIVE_TAIL = " In one word, the capital of France is"

# Comfortably above the 8192-token one-shot ceiling this test exists to remove.
_LONG_TARGET_TOKENS = 12288

# Widths chosen to straddle a KV page (16) rather than align to it, so a chunk
# boundary lands mid-page and the write offsets have to be right.
_WIDTHS = (37, 128, 999)


def _prompt_for(target_tokens, decisive=False):
    body = _HEAD + _UNIT * max(1, round(target_tokens / 9))
    return body + _DECISIVE_TAIL if decisive else body


def _parse(out):
    return json.loads(out.strip().splitlines()[-1])


async def _quest(client, **params):
    base = {
        "max_tokens": 24,
        "temperature": 0.0001,
        "page_budget": 1 << 20,
        "report": True,
    }
    out = await run_inferlet(client, "quest-attention", {**base, **params})
    return _parse(out)


async def test_chunking_is_exact(client, args):
    """The multi-chunk path must reproduce the one-shot path."""
    prompt = _prompt_for(1024, decisive=True)
    one = await _quest(client, prompt=prompt, seed=4242, max_tokens=32)
    kv = one["kv_len_last"]
    assert kv > 1000, f"prompt too short to be interesting: kv_len={kv}"
    assert one["text"].strip(), "one-shot produced nothing to compare against"

    for width in _WIDTHS:
        many = await _quest(
            client, prompt=prompt, seed=4242, max_tokens=32, prefill_chunk=width
        )
        assert many["kv_len_last"] == kv, (
            f"chunk={width}: kv_len {many['kv_len_last']} != one-shot {kv} -- "
            "the chunks did not write the prompt they were given"
        )
        assert many["text"] == one["text"], (
            f"chunk={width}: chunked prefill changed the output.\n"
            f"  one-shot: {one['text']!r}\n"
            f"  chunked : {many['text']!r}"
        )
    widths = "/".join(str(w) for w in _WIDTHS)
    print(f"    kv_len={kv}: 32 tokens identical at chunk widths {widths} vs one-shot")


async def test_above_the_one_shot_ceiling(client, args):
    """A prompt longer than `max_embed_length()` must run, and read coherently."""
    r = await _quest(
        client,
        prompt=_prompt_for(_LONG_TARGET_TOKENS, decisive=True),
        seed=7,
        max_tokens=16,
    )
    kv = r["kv_len_last"]
    assert kv > 8192, (
        f"prompt did not exceed the 8192-token one-shot ceiling (kv_len={kv}); "
        "the test is not testing what it claims to"
    )
    text = r["text"]
    assert text.strip(), f"empty generation at kv_len={kv}"
    # A KV cache stitched together wrongly across 2+ chunks does not answer the
    # question at the end of the prompt; it produces text unrelated to it.
    assert "Paris" in text, f"incoherent continuation at kv_len={kv}: {text!r}"
    print(f"    kv_len={kv} (> 8192 one-shot ceiling), answers: {text[:48]!r}")


async def test_quest_still_evicts_at_long_context(client, args):
    """The whole point: Quest must still enforce its budget past the ceiling."""
    common = {
        "prompt": _prompt_for(_LONG_TARGET_TOKENS, decisive=True),
        "seed": 11,
        "max_tokens": 16,
    }
    full = await _quest(client, **common, page_budget=1 << 20)
    tight = await _quest(client, **common, page_budget=1)
    kv = full["kv_len_last"]
    assert tight["kv_len_last"] == kv, "endpoints disagree on context"
    assert full["text"] != tight["text"], (
        f"a 1-page budget produced the same text as an unlimited one at "
        f"kv_len={kv} -- the mask is being computed and ignored"
    )
    print(
        f"    kv_len={kv}: budget=1 diverges from budget=inf, "
        "so the mask is still enforced past the ceiling"
    )


async def _policy(client, name, **params):
    base = {
        "max_tokens": 24,
        "temperature": 0.0001,
        "page_budget": 1 << 20,
        "report": True,
    }
    return _parse(await run_inferlet(client, name, {**base, **params}))


async def test_trackb_chunking_is_exact(client, args):
    """H2O and TOVA are decode policies, so chunking must not move their text."""
    prompt = _prompt_for(1024, decisive=True)
    for name in ("trackb-h2o", "tova-attention"):
        one = await _policy(client, name, prompt=prompt, seed=4242, max_tokens=32)
        assert one["text"].strip(), f"{name}: one-shot produced nothing"
        for width in _WIDTHS:
            many = await _policy(
                client, name, prompt=prompt, seed=4242, max_tokens=32,
                prefill_chunk=width,
            )
            assert many["text"] == one["text"], (
                f"{name} chunk={width}: chunked prefill changed the output.\n"
                f"  one-shot: {one['text']!r}\n"
                f"  chunked : {many['text']!r}"
            )
        print(f"    {name}: 32 tokens identical at every chunk width")


async def test_snapkv_observes_the_final_chunk(client, args):
    """SnapKV's window must land on the prompt's tail, not on a middle chunk.

    Two distinct failures have to be caught, and they need different probes.

    **Wrong chunk.** `tail_page_share` (mass on the prompt's last page / total)
    catches this outright: a tap attached to any chunk but the last cannot see
    the last page at all, so it reports 0.

    **Right chunk, truncated window.** The capture takes the last `window` rows
    *of the fire it is attached to*, so a final chunk shorter than `window`
    silently narrows the observation -- and a narrow window still puts mass on
    the last page, so the check above passes. `kept_pages` is a surprisingly
    weak probe for this; measured on this prompt at a 24-page budget, forcing
    the window down to 4 rows moved only 2 of 24 pages. `tail_page_share` is
    the sharp one, because a narrower window means fewer query rows and all of
    them sit near the last page, which inflates its share:

        final chunk   36    127    635  |   19     9     4
        tail_share  .0507  .0506  .0507 | .0852 .1801 .2702
                    <-- window whole -->|<-- window truncated -->

    Whole-window noise is 0.2%; the smallest truncation signal is +68%. A 2%
    band sits two orders of magnitude clear of the noise and still trips on the
    mildest truncation, so that is the assertion.
    """
    prompt = _prompt_for(1024, decisive=True)
    # ~82 pages of prompt; 24 forces the policy to actually discriminate.
    tight = {"page_budget": 24, "seed": 4242, "max_tokens": 32}
    one = await _policy(client, "trackb-snapkv", prompt=prompt, **tight)
    assert one["tail_page_share"] > 0.0, "one-shot baseline already sees no tail"
    assert one["prefill_chunks"] == 1, "the reference run was itself chunked"
    assert len(one["kept_pages"]) == 24, f"budget not binding: {len(one['kept_pages'])}"
    ref = one["tail_page_share"]

    for width in _WIDTHS:
        many = await _policy(
            client, "trackb-snapkv", prompt=prompt, prefill_chunk=width, **tight
        )
        assert many["prompt_len"] == one["prompt_len"], "endpoints disagree on prompt"
        assert many["prefill_chunks"] > 1, (
            f"chunk={width} did not actually split the prefill "
            f"({many['prefill_chunks']} chunk); the case is vacuous"
        )
        # Even chunking: the final chunk is the largest a last chunk can be.
        assert many["prefill_final"] == many["prompt_len"] // many["prefill_chunks"], (
            f"chunk={width}: final chunk {many['prefill_final']} is not "
            f"floor({many['prompt_len']}/{many['prefill_chunks']}) -- the "
            "remainder was not spread and the last chunk is a sliver"
        )
        assert many["layers_observed"] == one["layers_observed"], (
            f"chunk={width}: observed {many['layers_observed']} layers, "
            f"one-shot observed {one['layers_observed']}"
        )
        assert many["tail_nonzero"] == 0, (
            f"chunk={width}: {many['tail_nonzero']} scores past the live prefix -- "
            "the capture wrote outside the prompt"
        )
        share = many["tail_page_share"]
        assert share > 0.0, (
            f"chunk={width}: the last page of the prompt got no observed mass, "
            "so the tap fired on an earlier chunk and SnapKV is ranking by the "
            "wrong window"
        )
        assert abs(share - ref) <= 0.02 * ref, (
            f"chunk={width}: tail_page_share {share:.4f} vs one-shot {ref:.4f} "
            f"({100 * (share / ref - 1):+.1f}%) -- the observation window is not "
            "the same set of rows the one-shot fire saw"
        )
    print(
        f"    tail_page_share={ref:.4f} +/-2% at widths "
        f"{'/'.join(str(w) for w in _WIDTHS)}; {one['layers_observed']} layers, "
        "final chunk never a sliver"
    )


async def test_trackb_above_the_one_shot_ceiling(client, args):
    """Every Track B policy must run past the ceiling and still enforce a budget."""
    prompt = _prompt_for(_LONG_TARGET_TOKENS, decisive=True)
    for name, kv_field in (
        ("trackb-h2o", "kv_len"),
        ("trackb-snapkv", "prompt_len"),
    ):
        common = {"prompt": prompt, "seed": 11, "max_tokens": 16}
        full = await _policy(client, name, **common, page_budget=1 << 20)
        kv = full[kv_field]
        assert kv > 8192, (
            f"{name}: prompt did not exceed the 8192-token one-shot ceiling "
            f"({kv_field}={kv}); the test is not testing what it claims to"
        )
        assert "Paris" in full["text"], (
            f"{name}: incoherent continuation at {kv_field}={kv}: {full['text']!r}"
        )
        tight = await _policy(client, name, **common, page_budget=1)
        assert tight[kv_field] == kv, f"{name}: endpoints disagree on context"
        assert full["text"] != tight["text"], (
            f"{name}: a 1-page budget produced the same text as an unlimited one "
            f"at {kv_field}={kv} -- the mask is being computed and ignored"
        )
        print(f"    {name}: {kv_field}={kv} (> 8192), budget still enforced")


async def test_policies_agree_with_the_baseline_past_the_ceiling(client, args):
    """All five programs must write the same KV for a 15K-token prompt.

    With eviction disabled, every policy here is the baseline plus observation,
    so past the ceiling -- where all of them are splitting the prompt into
    multiple fires and stitching the results -- they must all still produce the
    baseline's continuation. An off-by-one in a chunk's write offsets, a
    `page_indptr` that does not cover the prefix, a `kv_len` that is not
    cumulative: none of those change whether the program *runs*, and all of them
    would show up here.

    Note what this does NOT test, because the docstring above explains why the
    prompt is decisive: a decisive prompt is deliberately insensitive to
    last-bit differences, and different chunk boundaries only move last bits.
    So this will not detect one program splitting 7516/7516 where another splits
    8192/6840 -- `test_snapkv_observes_the_final_chunk` and the shared
    `prefill_chunks` helper cover that. What it detects is a chunk that wrote
    the wrong thing, which is the failure that survives everything else.
    """
    prompt = _prompt_for(_LONG_TARGET_TOKENS, decisive=True)
    common = {"seed": 4242, "max_tokens": 24, "temperature": 0.0001}
    base = _parse(await run_inferlet(client, "naive-baseline", {**common, "prompt": prompt}))
    assert base["text"].strip(), "baseline produced nothing to compare against"
    assert "Paris" in base["text"], (
        "the baseline itself is incoherent past the ceiling, so agreeing with it "
        f"would prove nothing: {base['text']!r}"
    )

    for name in ("quest-attention", "trackb-h2o", "trackb-snapkv", "tova-attention"):
        r = await _policy(client, name, prompt=prompt, **common)
        assert r["text"] == base["text"], (
            f"{name} disagrees with the baseline past the ceiling, with nothing "
            f"evicted -- some chunk did not write the prompt it was given.\n"
            f"  baseline: {base['text']!r}\n"
            f"  {name}: {r['text']!r}"
        )
    print("    4 policies reproduce the baseline exactly on a >8192-token prompt")


run_tests(
    [
        test_chunking_is_exact,
        test_above_the_one_shot_ceiling,
        test_quest_still_evicts_at_long_context,
        test_trackb_chunking_is_exact,
        test_snapkv_observes_the_final_chunk,
        test_trackb_above_the_one_shot_ceiling,
        test_policies_agree_with_the_baseline_past_the_ceiling,
    ],
    description="Chunked prefill (Track A + Track B)",
)
