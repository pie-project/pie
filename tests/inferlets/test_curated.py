"""Smoke tests for the curated inference-time inferlets.

Native MTP is build-validated separately because the default test model does
not expose multi-token-prediction heads.

The inference-time-algorithm inferlets each report the statistic that proves
their rule fired, so these assert on that statistic rather than on the decoded
text, which is model- and seed-dependent. Two of them additionally get an
identity control: at the parameter value where the algorithm reduces to plain
sampling, the reported divergence must be exactly zero.
"""

import json
import os

# `quest-attention` needs the per-page key envelopes, which are an operator
# opt-in because they cost `2/page_size` of the KV pool and have to be
# allocated with the pages (driver/cuda/src/store/kv_cache.cpp). Set it before
# the engine boots — the driver reads it while sizing the cache. Enabling it
# for the whole run is deliberate: it also proves the envelope maintenance that
# now rides every KV append does not perturb the other inferlets.
os.environ.setdefault("PIE_CUDA_KV_ENVELOPES", "1")

from conftest import run_inferlet, run_tests  # noqa: E402


async def _nonempty(client, args, name: str, inputs: dict) -> str:
    output = await run_inferlet(client, name, inputs, timeout=args.timeout)
    assert output.strip(), f"{name} returned empty output"
    return output


# A non-empty continuation proves the plumbing carried tokens; it does NOT prove
# the pass attended the prompt. A decode loop that over-declares its page CSR (or
# otherwise reads uninitialised KV) still emits fluent-looking garbage, so any
# inferlet whose job is "continue this prompt" gets an attention gate as well.
ATTENDS_PROMPT = "The capital of France is"


async def _attends_prompt(client, args, name: str, inputs: dict) -> str:
    output = await _nonempty(
        client, args, name, {"prompt": ATTENDS_PROMPT, **inputs}
    )
    lowered = output.lower()
    assert "france" in lowered or "paris" in lowered, (
        f"{name} did not attend the prompt {ATTENDS_PROMPT!r} "
        f"(reads uninitialised KV?): {output[:200]!r}"
    )
    return output


async def _report(client, args, name: str, inputs: dict, *, expect_text: bool = True) -> dict:
    """Run an inferlet whose `Output` struct is serialized to JSON."""
    output = await _nonempty(client, args, name, inputs)
    start = output.find("{")
    assert start >= 0, f"{name} returned no JSON object: {output[:200]!r}"
    try:
        report = json.loads(output[start:])
    except json.JSONDecodeError as error:
        raise AssertionError(f"{name} returned malformed JSON: {output[:200]!r}") from error
    if expect_text:
        assert report.get("text", "").strip(), f"{name} generated no text"
    return report


async def test_chat_completion(client, args):
    await _nonempty(client, args, "chat-completion", {"prompt": "Say hello.", "max_tokens": 4})


async def test_sampling_primitives(client, args):
    output = await _nonempty(client, args, "sampling-primitives", {})
    assert "token=" in output and "entropy=" in output


async def test_consensus_decoding(client, args):
    await _nonempty(
        client,
        args,
        "consensus-decoding",
        {"question": "What is 2 + 2?", "num_candidates": 2, "max_tokens": 4},
    )


async def test_greenlist_watermarking(client, args):
    await _nonempty(client, args, "greenlist-watermarking", {"max_tokens": 4})


async def test_json_schema_constrained_decoding(client, args):
    output = await _nonempty(
        client,
        args,
        "json-schema-constrained-decoding",
        {
            "prompt": "Return an object with an integer field named value.",
            "schema": (
                '{"type":"object","properties":{"value":{"type":"integer"}},'
                '"required":["value"],"additionalProperties":false}'
            ),
            "max_tokens": 64,
        },
    )
    assert "value" in output


async def test_attention_sink(client, args):
    await _nonempty(
        client,
        args,
        "attention-sink",
        {"prompt": "Count upward.", "max_tokens": 4, "sink_size": 1, "window_size": 2},
    )


async def test_sliding_window_attention(client, args):
    await _nonempty(
        client,
        args,
        "sliding-window-attention",
        {"prompt": "Count upward.", "max_tokens": 4, "window_size": 2},
    )


async def test_prefix_tree_kv_cache(client, args):
    output = await _nonempty(client, args, "prefix-tree-kv-cache", {"num_tokens": 2})
    assert "city at dawn:" in output and "forest at night:" in output


async def test_cacheback_speculative_decoding(client, args):
    """Speculation must change the number of forward passes and nothing else.

    Verification is greedy, so a draft token survives only when it equals what
    the target model would have produced anyway. `draft_length=0` short-circuits
    the drafter and decodes sequentially through the identical prompt, stop
    tokens and `verify()` call, which makes it an exact control rather than an
    approximate one. A divergence here is a real bug -- most likely rejected
    draft KV leaking into the next window.

    Rejection is the path under test, so the test asserts that rejections
    actually happened. A run with a perfect acceptance rate would pass while
    never exercising the code that discards draft state. Both a repetitive and
    a prose prompt are used because they reject at different points.
    """
    for label, prompt in (
        ("repetitive", "Repeat exactly: red green blue, red green blue, red green"),
        ("prose", "Explain in detail why the sky appears blue during the day."),
    ):
        base = {"prompt": prompt, "max_tokens": 24, "max_ngram": 4}
        sequential = await _report(
            client, args, "cacheback-speculative-decoding", {**base, "draft_length": 0}
        )
        speculative = await _report(
            client, args, "cacheback-speculative-decoding", {**base, "draft_length": 4}
        )

        # With no draft, every generated token costs exactly one forward pass.
        assert sequential["drafted"] == 0, (label, sequential)
        assert sequential["verification_steps"] == sequential["count"], (label, sequential)

        # The comparison is only meaningful if speculation fired *and* was
        # rejected at least once.
        rejected = speculative["drafted"] - speculative["accepted"]
        assert speculative["accepted"] > 0, (label, "no draft token was ever accepted")
        assert rejected > 0, (label, "no draft was ever rejected; the isolation path is untested")
        assert speculative["verification_steps"] < sequential["verification_steps"], (
            label, "speculation ran no fewer forward passes than sequential decoding"
        )

        assert speculative["tokens"] == sequential["tokens"], (
            f"[{label}] speculative decoding changed the output:\n"
            f"  sequential  = {sequential['tokens']}\n"
            f"  speculative = {speculative['tokens']}"
        )


async def test_constrained_speculative_decoding(client, args):
    """Grammar constraints and speculation must compose without changing output.

    Verification is greedy *and* grammar-masked at every readout row, so a draft
    token survives only when it is what the target model would have produced
    anyway under that row's mask. Speculation is therefore a pure latency
    optimization on top of constrained decoding.

    `draft_length=0` short-circuits the drafter and decodes sequentially through
    the identical prompt, schema and `verify()` call, which makes it an exact
    control. A divergence means the per-row masks are misaligned with the
    positions they gate, or that rejected draft KV is leaking forward.

    The inferlet also asserts internally, on every step, that rolling the
    grammar back over a drafted window restores the exact state a `fork()` taken
    beforehand reports. That check reaching a nonzero count here is the
    end-to-end evidence that the matcher's fork/rollback ABI works through wasm.
    """
    # The inferlet's default prompt and schema are used verbatim: how far the
    # model runs before closing the document is prompt-sensitive, and the test
    # needs both arms to reach termination rather than the token cap.
    base = {"max_tokens": 256, "max_ngram": 4}
    sequential = await _report(
        client, args, "constrained-speculative-decoding", {**base, "draft_length": 0}
    )
    speculative = await _report(
        client, args, "constrained-speculative-decoding", {**base, "draft_length": 4}
    )

    # Both arms must have produced schema-valid JSON (the inferlet parses it
    # before returning, so reaching here already proves that) ...
    json.loads(sequential["text"])
    json.loads(speculative["text"])

    # ... and both must have exercised the fork/rollback invariant check.
    assert sequential["rollback_checks"] > 0, sequential
    assert speculative["rollback_checks"] > 0, speculative

    # With no draft, every generated token costs exactly one forward pass.
    assert sequential["drafted"] == 0, sequential
    assert sequential["verification_steps"] == sequential["count"], sequential

    # The comparison is only meaningful if speculation fired and was rejected.
    rejected = speculative["drafted"] - speculative["accepted"]
    assert speculative["accepted"] > 0, "no draft token was ever accepted"
    assert rejected > 0, "no draft was ever rejected; the reject path is untested"
    assert speculative["verification_steps"] < sequential["verification_steps"], (
        "speculation ran no fewer forward passes than sequential decoding"
    )

    assert speculative["tokens"] == sequential["tokens"], (
        "speculation changed the constrained output:\n"
        f"  sequential  = {sequential['tokens']}\n"
        f"  speculative = {speculative['tokens']}"
    )


async def test_mirostat_v2_sampling(client, args):
    output = await _nonempty(client, args, "mirostat-v2-sampling", {"max_tokens": 4})
    assert "mirostat-v2" in output


async def test_beam_search(client, args):
    output = await _nonempty(client, args, "beam-search", {"max_tokens": 2})
    assert "[beam] width=2" in output, output


async def test_beam_search_greedy_identity(client, args):
    """Width 1 is exactly greedy, so no step may diverge from the raw argmax."""
    output = await _nonempty(client, args, "beam-search", {"max_tokens": 6, "beams": 1})
    assert "greedy_mismatches=0" in output, output


async def test_beam_search_width_explores(client, args):
    """Width 3 must actually leave the greedy path, or the search is a no-op."""
    output = await _nonempty(client, args, "beam-search", {"max_tokens": 6, "beams": 3})
    assert "[beam] width=3" in output, output
    mismatches = int(output.split("greedy_mismatches=")[1].split()[0])
    assert mismatches > 0, output


async def test_contrastive_decoding(client, args):
    await _nonempty(
        client,
        args,
        "contrastive-decoding",
        {"prompt": "Say hello.", "max_tokens": 4, "amateur_window": 2},
    )


async def test_locally_typical_sampling(client, args):
    report = await _report(
        client, args, "locally-typical-sampling", {"max_tokens": 4, "k_max": 64}
    )
    assert 0 < report["mean_kept"] <= 64, report
    assert report["min_kept"] >= 1, "the typical set is never empty"
    assert 0 < report["mean_mass"] <= 1.0, report


async def test_eta_epsilon_sampling(client, args):
    for mode in ("eta", "epsilon"):
        report = await _report(
            client, args, "eta-epsilon-sampling", {"max_tokens": 4, "mode": mode}
        )
        assert report["mode"] == mode, report
        assert report["min_kept"] >= 1, "truncation never empties the candidate set"


async def test_tail_free_sampling(client, args):
    report = await _report(client, args, "tail-free-sampling", {"max_tokens": 4, "k_max": 64})
    assert 0 < report["mean_kept"] <= 64, report
    assert report["min_kept"] >= 1, report


async def test_top_a_sampling(client, args):
    loose = await _report(client, args, "top-a-sampling", {"max_tokens": 4, "a": 0.0001})
    tight = await _report(client, args, "top-a-sampling", {"max_tokens": 4, "a": 1.0})
    # The floor is a*p_max^2, so a larger `a` can only shrink the kept set.
    assert tight["mean_kept"] <= loose["mean_kept"], (tight, loose)
    assert tight["min_kept"] >= 1, tight


async def test_xtc_sampling(client, args):
    never = await _report(client, args, "xtc-sampling", {"max_tokens": 4, "probability": 0.0})
    always = await _report(client, args, "xtc-sampling", {"max_tokens": 4, "probability": 1.0})
    assert never["fire_rate"] == 0.0, never
    assert always["fire_rate"] == 1.0, always
    assert never["mean_dropped"] == 0.0, "a gate that never fires drops nothing"


async def test_repetition_penalty(client, args):
    report = await _report(
        client,
        args,
        "repetition-penalty",
        {"max_tokens": 4, "repetition_penalty": 1.5, "frequency_penalty": 0.1},
    )
    assert report["mean_penalized"] > 0, "the prompt alone makes some token penalized"
    assert 0 < report["unique_ratio"] <= 1.0, report


async def test_dry_repetition_penalty(client, args):
    off = await _report(client, args, "dry-repetition-penalty", {"max_tokens": 4, "multiplier": 0.0})
    on = await _report(client, args, "dry-repetition-penalty", {"max_tokens": 4, "multiplier": 2.0})
    assert off["peak_penalty"] == 0.0, "multiplier=0 disables DRY entirely"
    assert on["peak_penalty"] >= 0.0, on


async def test_entropy_adaptive_temperature(client, args):
    report = await _report(
        client, args, "entropy-adaptive-temperature", {"max_tokens": 4, "t0": 1.0, "theta": 0.3}
    )
    # T = t0 * n^(theta/H) with 0<n<1, so the derived temperature never exceeds t0.
    assert 0 < report["mean_temperature"] <= report["t0"], report
    assert report["mean_entropy"] > 0, report


async def test_gumbel_watermark(client, args):
    # Detection *power* needs hundreds of tokens to separate from the null, so a
    # short smoke test asserts the detector's structure rather than its verdict.
    # The power curve versus n is measured in the faithfulness audit instead.
    report = await _report(client, args, "gumbel-watermark", {"max_tokens": 8, "secret": 7})
    assert report["watermark"] is True, report
    assert 1 <= report["unique_contexts"] <= report["count"], report
    assert report["mean_score"] > 0 and report["mean_null_score"] > 0, report


async def test_synthid_tournament_sampling(client, args):
    report = await _report(
        client, args, "synthid-tournament-sampling", {"max_tokens": 8, "secret": 7, "depth": 4}
    )
    assert report["watermark"] is True, report
    assert report["depth"] == 4, report
    # g is Bernoulli-valued, so both means must land in [0,1] by construction.
    assert 0.0 <= report["mean_score"] <= 1.0, report
    assert 0.0 <= report["mean_null_score"] <= 1.0, report


async def test_classifier_free_guidance(client, args):
    output = await _nonempty(
        client,
        args,
        "classifier-free-guidance",
        {"max_tokens": 4, "guidance": 2.0, "negative_prompt": "Talk about the weather."},
    )
    assert "[cfg]" in output and "mean_kl=" in output, output


async def test_classifier_free_guidance_identity(client, args):
    """guidance=1.0 is plain conditional sampling, so the KL must be exactly 0."""
    output = await _nonempty(
        client, args, "classifier-free-guidance", {"max_tokens": 4, "guidance": 1.0}
    )
    assert "mean_kl=0.0000" in output, output


async def test_context_aware_decoding(client, args):
    output = await _nonempty(
        client,
        args,
        "context-aware-decoding",
        {"max_tokens": 4, "alpha": 0.5, "context": "The tallest mountain is Mt. Nowhere.",
         "query": "What is the tallest mountain?"},
    )
    assert "[cad]" in output and "mean_kl=" in output, output


async def test_context_aware_decoding_identity(client, args):
    """alpha=0 is plain context-conditioned decoding, so the KL must be exactly 0."""
    output = await _nonempty(
        client,
        args,
        "context-aware-decoding",
        {"max_tokens": 4, "alpha": 0.0, "context": "The sky is green.",
         "query": "What colour is the sky?"},
    )
    assert "mean_kl=0.0000" in output, output


async def test_asap_grammar_aligned_decoding(client, args):
    report = await _report(
        client,
        args,
        "asap-grammar-aligned-decoding",
        {
            "prompt": "Return an object with an integer field named value.",
            "schema": (
                '{"type":"object","properties":{"value":{"type":"integer"}},'
                '"required":["value"],"additionalProperties":false}'
            ),
            "rounds": 3,
            "max_tokens": 32,
        },
        expect_text=False,
    )
    assert report["rounds"] == 3, report
    # ASAp's guarantee: the root approximation mass is non-decreasing in the
    # round index. A regression here means the trie bookkeeping is wrong.
    assert report["monotone"] is True, report["root_alpha_trace"]


async def test_token_healing(client, args):
    report = await _report(
        client, args, "token-healing", {"prompt": "The capital of Fra", "max_tokens": 4}
    )
    assert report["healed"] is True, report
    # Healing must re-emit the exact prompt bytes, never a shorter prefix.
    assert report["prompt_preserved"] is True, report
    assert report["prefix_candidates"] >= 1, report


# Long enough to fill several pages, so the tap has something to rank. The
# answer sits in the FIRST page, which is what makes the score meaningful:
# Quest must rank that page above the filler that follows it.
_QUEST_PROMPT = (
    "The capital of France is Paris. "
    + "Paris is a large European city with a long history. " * 24
)


async def test_quest_attention(client, args):
    report = await _report(
        client, args, "quest-attention",
        {"prompt": _QUEST_PROMPT, "max_tokens": 8, "page_budget": 4},
    )
    # The tap has to fire once per layer, on every layer.
    assert report["layers_observed"] > 0, report
    # Every page the request has already filled must carry a real bound. Only
    # the in-flight last page is allowed to be pinned (+inf, "always keep"),
    # and nothing may be NaN or missing.
    assert report["pages_finite"] == report["max_pages"] - 1, report
    assert report["pages_pinned"] == 1, report
    assert report["pages_absent"] == 0, report
    assert report["pages_nan"] == 0, report
    # The budget must be honoured and the in-flight page force-kept.
    assert len(report["kept_pages"]) == report["page_budget"], report
    assert report["max_pages"] - 1 in report["kept_pages"], report
    # The criticality bound must actually discriminate: the page holding the
    # answer outranks the repeated filler.
    scores = [float(s) for s in report["page_scores"][:-1]]
    assert scores[0] == max(scores), report


async def test_naive_baseline(client, args):
    report = await _report(client, args, "naive-baseline", {"max_tokens": 4})
    assert report["sampler"] == "naive-baseline", report


async def test_chat_completion_attends_prompt(client, args):
    await _attends_prompt(client, args, "chat-completion", {"max_tokens": 14, "temperature": 0.0})


async def test_attention_sink_attends_prompt(client, args):
    await _attends_prompt(
        client, args, "attention-sink", {"max_tokens": 14, "sink_size": 4, "window_size": 64}
    )


async def test_sliding_window_attention_attends_prompt(client, args):
    await _attends_prompt(
        client, args, "sliding-window-attention", {"max_tokens": 14, "window_size": 64}
    )


def tests():
    return [
        test_chat_completion,
        test_sampling_primitives,
        test_consensus_decoding,
        test_greenlist_watermarking,
        test_json_schema_constrained_decoding,
        test_attention_sink,
        test_sliding_window_attention,
        test_chat_completion_attends_prompt,
        test_attention_sink_attends_prompt,
        test_sliding_window_attention_attends_prompt,
        test_prefix_tree_kv_cache,
        test_cacheback_speculative_decoding,
        test_constrained_speculative_decoding,
        test_mirostat_v2_sampling,
        test_beam_search,
        test_beam_search_greedy_identity,
        test_beam_search_width_explores,
        test_contrastive_decoding,
        test_locally_typical_sampling,
        test_eta_epsilon_sampling,
        test_tail_free_sampling,
        test_top_a_sampling,
        test_xtc_sampling,
        test_repetition_penalty,
        test_dry_repetition_penalty,
        test_entropy_adaptive_temperature,
        test_gumbel_watermark,
        test_synthid_tournament_sampling,
        test_classifier_free_guidance,
        test_classifier_free_guidance_identity,
        test_context_aware_decoding,
        test_context_aware_decoding_identity,
        test_asap_grammar_aligned_decoding,
        test_token_healing,
        test_naive_baseline,
        test_quest_attention,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Curated inferlet E2E tests")
