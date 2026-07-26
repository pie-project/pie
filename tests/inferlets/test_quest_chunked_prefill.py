"""Chunked prefill must equal one-shot prefill, and must lift the ceiling.

Quest exists for long contexts, but its prefill was a single fire, and a single
fire cannot exceed the driver's structural per-launch token capacity
(`max_embed_length()`, 8192 on this CUDA driver). That put a hard ceiling on the
context Quest could be run at -- *below* the range where section 14 shows it
starts to pay for itself, which made the ceiling a correctness-shaped hole in
the feature rather than a mere limit.

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

Run from the repo root with PYTHONPATH=sdk/python-server/python:

    PIE_CUDA_KV_ENVELOPES=1 python tests/inferlets/test_quest_chunked_prefill.py \
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


run_tests(
    [
        test_chunking_is_exact,
        test_above_the_one_shot_ceiling,
        test_quest_still_evicts_at_long_context,
    ],
    description="Quest chunked prefill",
)
