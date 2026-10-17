import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from conftest import run_inferlet, run_tests  # noqa: E402

_UNIT = "Paris is a large European city with a long history. "
_HEAD = "The capital of France is Paris. "
# Ends in a question answered with a large margin, so reduction order cannot flip the argmax.
_DECISIVE_TAIL = " In one word, the capital of France is"

# Comfortably above one fire's 8192 rows.
_LONG_TARGET_TOKENS = 12288


def _prompt_for(target_tokens, decisive=False):
    body = _HEAD + _UNIT * max(1, round(target_tokens / 9))
    return body + _DECISIVE_TAIL if decisive else body


def _parse(out):
    return json.loads(out.strip().splitlines()[-1])


async def _policy(client, name, **params):
    base = {
        "max_tokens": 24,
        "temperature": 0.0001,
        "page_budget": 1 << 20,
    }
    return _parse(await run_inferlet(client, name, {**base, **params}))


async def test_snapkv_observes_the_tail(client, args):
    prompt = _prompt_for(_LONG_TARGET_TOKENS, decisive=True)
    r = await _policy(
        client, "trackb-snapkv", prompt=prompt, page_budget=24, seed=4242, max_tokens=16
    )
    assert r["prompt_len"] > 8192, (
        f"prompt_len={r['prompt_len']} fits one fire; the case is vacuous"
    )
    assert len(r["kept_pages"]) == 24, f"budget not binding: {len(r['kept_pages'])}"
    assert r["layers_observed"] > 0, "the capture observed no layer"
    assert r["tail_nonzero"] == 0, (
        f"{r['tail_nonzero']} scores past the live prefix -- the capture wrote "
        "outside the prompt"
    )
    assert r["tail_page_share"] > 0.0, (
        "the last page of the prompt got no observed mass, so the capture rode "
        "an earlier piece and SnapKV is ranking by the wrong window"
    )
    print(
        f"    prompt_len={r['prompt_len']}: tail_page_share="
        f"{r['tail_page_share']:.4f}, {r['layers_observed']} layers observed"
    )


async def test_policies_agree_with_the_baseline_past_the_ceiling(client, args):
    prompt = _prompt_for(_LONG_TARGET_TOKENS, decisive=True)
    common = {"seed": 4242, "max_tokens": 24, "temperature": 0.0001}
    base = _parse(await run_inferlet(client, "naive-baseline", {**common, "prompt": prompt}))
    assert base["text"].strip(), "baseline produced nothing to compare against"
    assert "Paris" in base["text"], (
        "the baseline itself is incoherent past the ceiling, so agreeing with it "
        f"would prove nothing: {base['text']!r}"
    )

    for name in ("trackb-h2o", "trackb-snapkv", "tova-attention"):
        r = await _policy(client, name, prompt=prompt, **common)
        assert r["text"] == base["text"], (
            f"{name} disagrees with the baseline past the ceiling, with nothing "
            f"evicted -- some piece did not write the prompt it was given.\n"
            f"  baseline: {base['text']!r}\n"
            f"  {name}: {r['text']!r}"
        )
    print("    3 policies reproduce the baseline exactly on a >8192-token prompt")


run_tests(
    [
        test_snapkv_observes_the_tail,
        test_policies_agree_with_the_baseline_past_the_ceiling,
    ],
    description="Long prefill (Track A + Track B)",
    requires=("attn_score",),
)
