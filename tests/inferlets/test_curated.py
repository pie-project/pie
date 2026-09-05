"""Smoke tests for the curated inference-time inferlets.

Native MTP is build-validated separately because the default test model does
not expose multi-token-prediction heads.

The inference-time-algorithm inferlets each report the statistic that proves
their rule fired, so these assert on that statistic rather than on the decoded
text, which is model- and seed-dependent. Two of them additionally get an
identity control: at the parameter value where the algorithm reduces to plain
sampling, the reported divergence must be exactly zero.
"""

import asyncio
import re
import json
import os

# `quest-attention` needs the per-page key envelopes, which are an operator
# opt-in because they cost `2/page_size` of the KV pool and have to be
# allocated with the pages (crates/driver-cuda/csrc/src/store/kv_cache.cpp). Set it before
# the engine boots — the engine reads it while sizing the cache. Enabling it
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
        # 48, not 24: the current default SKU (qwen35-d0.8b) spends its first
        # ~20 tokens on a `<think>` preamble where prompt-lookup drafting can
        # never hit, so a 24-token window proves the identity but starves the
        # acceptance/rejection assertions below. At 48 the run reaches
        # repeatable text on both prompts (measured: accepted=14 repetitive,
        # rejected=43) while both arms still terminate well under the timeout.
        base = {"prompt": prompt, "max_tokens": 48, "max_ngram": 4}
        sequential = await _report(
            client, args, "cacheback-speculative-decoding", {**base, "draft_length": 0}
        )
        speculative = await _report(
            client, args, "cacheback-speculative-decoding", {**base, "draft_length": 4}
        )

        # With no draft, every generated token costs exactly one forward pass —
        # plus one for the stop token when the model ends the run itself, since
        # that token is produced by a pass and then left out of `count`. A run
        # truncated by `max_tokens` never pays it, so a bare
        # `verification_steps == count` holds only where the budget runs out
        # first, which is a property of the model and the prompt rather than of
        # the loop. gemma-4-26B-A4B answers the repetitive prompt in 14 tokens
        # and stops, and read the bare way that is 15 != 14.
        assert sequential["drafted"] == 0, (label, sequential)
        assert sequential["verification_steps"] == sequential["count"] + int(
            sequential["stopped"]
        ), (label, sequential)

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
    # **THE SCHEMA IS STATED HERE, NOT TAKEN FROM THE INFERLET'S DEFAULT.**
    # How far a model runs before closing the document is a property of the
    # model and the schema, and the inferlet's default (a profile with a name,
    # an age and a skills array) is rich enough that gemma-4-26B-A4B never
    # closes it: it writes two good fields and then degenerates into
    # "abilityon abilityon ..." past 384 tokens. Its non-speculative sibling
    # `json-schema-constrained-decoding` fails the same way on the same
    # schema, so that is the model, not this loop — and that sibling's own
    # case passes only because it states a SMALL schema here too.
    #
    # A small schema proves everything this case is for: the sequential and
    # speculative arms come back token-identical, and speculation still fires,
    # is rejected, and saves passes (measured: accepted 2 of 4 drafted,
    # 19 verification steps against 21).
    base = {
        "prompt": (
            "Return an object with an integer field named value and a string "
            "field named name."
        ),
        "schema": (
            '{"type":"object","properties":'
            '{"value":{"type":"integer"},"name":{"type":"string"}},'
            '"required":["value","name"],"additionalProperties":false}'
        ),
        "max_tokens": 128,
        "max_ngram": 4,
    }
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


def _assert_kl_identity(output: str) -> None:
    """The report's mean KL is zero to the tolerance the inferlet itself uses.

    Asserting the STRING `mean_kl=0.0000` looked equivalent and was not. KL is
    non-negative by Jensen, so at an exact identity every term is float error
    and the total lands either side of zero at about 1e-9. Printed to four
    places that is `0.0000` or `-0.0000`, and the string form accepts one and
    rejects the other -- so the sign bit of a value indistinguishable from
    zero decided the result. It passed on Qwen3-0.6B and failed on the second
    model for no reason but which way the last rounding went.

    The tolerance is 5e-5 and not the 1e-3 the inferlets use for their own
    `IDENTITY-VIOLATION` marker, because 5e-5 is what the string check
    actually enforced -- four decimal places round to `0.0000` below exactly
    that -- and a rewrite of a check should not quietly loosen it by twenty
    times. The float error being corrected for is around 1e-9, so this still
    has four orders of margin. The marker is asserted too, since it is the
    inferlet's own statement about its own arithmetic.
    """
    found = re.search(r"mean_kl=(-?\d+\.\d+)", output)
    assert found, f"no mean_kl in the report: {output}"
    mean_kl = float(found.group(1))
    assert abs(mean_kl) < 5e-5, f"mean_kl={mean_kl} is not an identity: {output}"
    assert "IDENTITY-VIOLATION" not in output, output


async def test_classifier_free_guidance_identity(client, args):
    """guidance=1.0 is plain conditional sampling, so the KL must be exactly 0."""
    output = await _nonempty(
        client, args, "classifier-free-guidance", {"max_tokens": 4, "guidance": 1.0}
    )
    _assert_kl_identity(output)


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
    _assert_kl_identity(output)


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


# Coherence is compared at NEAR-GREEDY, not at a sampling temperature.
# Running a policy program changes the decode batch's shape and therefore the
# attention kernel's split/reduction plan, so its logits differ from the plain
# baseline's in the last bits or two. That is expected and benign -- the CUDA
# test `test_attn_score_capture` already proves the capture variant leaves the
# attention output BIT-identical, so the residue is plan-level, not policy-level
# -- but under Gumbel sampling at tau=0.1 those last bits are amplified 10x and
# occasionally decide a near-tied token. Measured over 24 tokens x 5 seeds x 3
# programs: 15/15 exact at tau=0.001, scattered single-token flips at tau=0.1
# that hit the mask-only program (quest) as readily as the capture ones. So the
# invariant worth asserting is the one that is actually true: with an all-keep
# mask the policy reproduces the baseline's ARGMAX path exactly.
_COHERENCE_TAU = 0.001


# The EVICTION pair's prompt, and it is not `_QUEST_PROMPT` for a measured
# reason. `_QUEST_PROMPT` is one sentence repeated twenty-four times: every
# page carries the same information, so a policy that drops sixteen of
# seventeen loses nothing the model was going to say, and both `trackb-snapkv`
# and `trackb-h2o` return the baseline continuation VERBATIM at a one-page
# budget. Under it "the policy is enforced" is untestable through the text --
# not because enforcement fails but because there is nothing to lose. (It is
# also why its end-of-prompt attention profile does not separate: with the
# same sentence everywhere, the observation window's mass is spread by
# position rather than by content.)
#
# This one puts a fact at the HEAD that the filler never repeats and asks for
# it at the TAIL. The head page and the question page are then the only two
# that matter, which is exactly the Λ shape (attention sink + recency) that
# H2O and SnapKV both claim to find -- so a keep-set that finds it answers,
# and a budget too small to hold both does not.
_EVICT_PROMPT = (
    "Remember this: the secret code word is banana. "
    + "The weather in the valley is mild and the roads are clear. " * 20
    + " What is the secret code word? The secret code word is"
)


async def test_quest_attention(client, args):
    # `max_tokens` is deliberately large relative to the prompt so the page
    # channel's DECLARED capacity runs ahead of the pages the request actually
    # holds. The two coincide at small `max_tokens`, and while they coincided
    # this test could not tell a correct page slice from one taken with the
    # host's bound instead of the device's real counts.
    report = await _report(
        client, args, "quest-attention",
        {"prompt": _QUEST_PROMPT, "max_tokens": 64, "page_budget": 4},
    )
    # The tap has to fire once per layer, on every layer.
    assert report["layers_observed"] > 0, report

    # The exact slot census, derived from the fire's own kv_len rather than
    # assumed. `envelope_dot` has three outcomes per slot: a real bound for a
    # page the envelopes already describe, `+inf` for a page still being
    # filled (never evict what we cannot bound), and `-inf` for a slot past
    # the end of the request. Every page must land in exactly one of them.
    page_size = report["page_size"]
    kv_len = report["kv_len_last"]
    real_pages = -(-kv_len // page_size)
    scored = min((kv_len - 1) // page_size, real_pages)  # qo_len == 1
    assert report["pages_absent"] == report["max_pages"] - real_pages, report
    assert report["pages_absent"] > 0, (
        "test is vacuous: the declared page bound equals the real page count, "
        f"so a slice taken with either CSR would agree\n{report}"
    )
    assert report["pages_finite"] == scored, report
    assert report["pages_pinned"] == real_pages - scored, report
    assert report["pages_nan"] == 0, report

    # The budget must be honoured and the in-flight page force-kept.
    assert len(report["kept_pages"]) == report["page_budget"], report
    assert real_pages - 1 in report["kept_pages"], report
    # A kept page must belong to this request; the slots past the end score
    # `-inf` precisely so a top-k consumer cannot reach into a neighbour's.
    assert max(report["kept_pages"]) < real_pages, report
    # The criticality bound must actually discriminate: the page holding the
    # answer outranks the repeated filler.
    scores = [float(s) for s in report["page_scores"][:scored]]
    assert scores[0] == max(scores), report

    # Everything above is about the RANKING, and a ranking the engine computes
    # and then discards looks exactly like one it honours. This is the part
    # that separates them: with a budget of one page the continuation has to
    # change, and with a budget of everything it has to be the unmasked answer
    # verbatim. `test_mask_enforced.py` covers the same ground in isolation;
    # keeping a version here means the matrix cannot go green on a build where
    # `attn_page_mask` silently stopped being applied.
    common = {"prompt": _QUEST_PROMPT, "max_tokens": 8,
              "temperature": _COHERENCE_TAU, "seed": 4242}
    wide = await _report(
        client, args, "quest-attention", {**common, "page_budget": 4096})
    tight = await _report(
        client, args, "quest-attention", {**common, "page_budget": 1})
    base = await _report(client, args, "naive-baseline", common)
    assert tight["pages_nan"] == 0, tight
    assert wide["text"] != tight["text"], (
        "attn_page_mask is not enforced: a 1-page budget produced the same "
        f"continuation as a {wide['page_budget']}-page one"
    )
    assert wide["text"] == base["text"], (
        "an all-keep page mask perturbed the output; compaction is not "
        f"equivalent to the original page table\n  {wide['text']!r}\n"
        f"  {base['text']!r}"
    )


_TOVA_PROMPT = (
    "The capital of France is Paris. "
    + "Paris is a large European city with a long history. " * 8
)


async def test_tova_attention(client, args):
    report = await _report(
        client, args, "tova-attention",
        {"prompt": _TOVA_PROMPT, "max_tokens": 8, "cache_size": 16},
    )
    # The tap has to fire once per layer, on every layer.
    assert report["layers_observed"] > 0, report
    # Every live KV position must carry a finite, non-negative probability, and
    # every slot past the live length must be EXACTLY zero.
    assert report["live_scored"] == report["kv_len"], report
    assert report["tail_nonzero"] == 0, report
    assert report["scores_nan"] == 0, report
    # The row must describe the positions the program THINKS it describes. The
    # host-side page CSR the engine hands a model body is allowed to be a
    # conservative upper bound (graph-lattice padding, the decode-envelope KV
    # bound), while the device CSR the attention kernel reads is exact. If the
    # capture ever sizes itself off the bound without zeroing the slack, the
    # tail of the row is live garbage — and because it is garbage rather than
    # absence, an eviction policy keeps a position that does not exist and
    # drops a real one. This ran clean for the first few fires and only then
    # diverged, so it is checked on EVERY fire, not just the last.
    for declared, observed in report["trace"]:
        assert declared == observed, report
    # Each layer contributes a softmax distribution over the live prefix, so
    # the folded row must sum to exactly the number of layers observed. This is
    # what makes the drained row self-validating: it fails if the capture is
    # mis-normalized, mis-strided, or truncated.
    assert abs(report["score_mass"] - report["layers_observed"]) < 0.05, report
    # The keep-set is well formed and honours the budget.
    assert len(report["kept_positions"]) == min(
        report["cache_size"], report["kv_len"]), report
    assert len(set(report["kept_positions"])) == len(
        report["kept_positions"]), report
    assert max(report["kept_positions"]) < report["kv_len"], report
    assert report["evicted_first"] is not None, report
    assert report["evicted_first"] not in report["kept_positions"], report
    # The scores must DISCRIMINATE, not merely be well formed. A uniform row
    # would satisfy every check above. TOVA on a real model reliably keeps two
    # things: the attention sink at the start of the sequence, and a recency
    # window at the end. Requiring both is what separates "the plumbing works"
    # from "the plumbing carries the signal the algorithm needs".
    kept = set(report["kept_positions"])
    assert 0 in kept, report
    assert any(p >= report["kv_len"] - 4 for p in kept), report
    # ...and it must drop something in the middle, or it is not a policy.
    assert 4 < report["evicted_first"] < report["kv_len"] - 4, report


async def test_h2o_attention(client, args):
    """H2O: a cumulative heavy-hitter statistic, and a mask that serves it.

    H2O is the one Track B policy that wants to OBSERVE and ENFORCE on the same
    fire, and it cannot: a lane that brings a mask takes the masked attention
    arm and keeps no scores (`crates/model/src/qwen_3/forward.rs` orders
    `masked` ahead of `captures_scores` as a correctness ruling). So the
    program splits its generation -- `observe_fires` unmasked capturing fires
    accumulate the statistic, then `enforce_tokens` masked fires are served the
    keep-set it chose. What that costs against the paper is the per-step
    re-ranking; what it keeps, and what the staircase below measures, is the
    CUMULATIVE ranking that is H2O's entire difference from TOVA.
    """
    common = {"prompt": _EVICT_PROMPT, "max_tokens": 12,
              "temperature": _COHERENCE_TAU, "seed": 4242}
    report = await _report(
        client, args, "trackb-h2o", {**common, "page_budget": 4096})
    # The split is real: both halves ran, and together they are the generation.
    assert report["observe_fires"] > 0, report
    assert report["enforce_tokens"] > 0, report
    assert report["observe_fires"] + report["enforce_tokens"] + 1 == report["count"], report
    # H2O's whole claim is the CARRY: the score row is accumulated across fires
    # rather than re-seeded from the latest one. Each fire adds one softmax row
    # per layer, so the mass must grow by exactly `layers_per_fire` every time.
    # This is the single assertion that separates H2O from TOVA -- with the
    # accumulator re-seeded the trace would be flat instead of a staircase.
    trace = [float(m) for m in report["mass_trace"]]
    assert len(trace) >= 2, report
    steps = [b - a for a, b in zip(trace, trace[1:])]
    per_fire = report["layers_per_fire"]
    for step in steps:
        assert abs(step - per_fire) < 0.05, (
            f"H2O is not accumulating: mass grew by {step}, not {per_fire}\n"
            f"  trace = {trace}"
        )
    # No attention may land past the live prefix. (Unlike TOVA this cannot be
    # "exactly zero" -- H2O seeds a cold-start ramp it never re-seeds away --
    # so the bar is the seed ceiling.)
    assert report["tail_polluted"] == 0, report
    assert report["scores_nan"] == 0, report
    # The heavy hitters must be HEAVY: a uniform row would pass everything
    # above. H2O accumulates DECODE attention, so what a real model produces is
    # the Λ that StreamingLLM named and that H2O's own keep-sets converge to --
    # an attention sink at the head, a recency window at the tail, and a thin
    # middle. Asserting the shape is what separates "the carry works" from "the
    # carry carries the signal the algorithm needs"; `tova-attention` is
    # checked the same way on its own single-step row.
    page_mass = [float(m) for m in report["page_mass"]]
    assert len(page_mass) > 8, report
    tail = len(page_mass) - 3
    # The recency window is the heaviest thing in the row...
    assert page_mass.index(max(page_mass)) >= tail, report
    # ...the sink is the heaviest thing that is NOT recent...
    assert page_mass[0] == max(page_mass[:tail]), report
    # ...and it stands clear of the middle it is being distinguished from.
    assert page_mass[0] > 2 * sorted(page_mass)[len(page_mass) // 2], report
    # The page H2O drops first is in the middle, where the Λ is thin -- not the
    # sink and not the recency window, or it is not this policy.
    assert 2 < report["evicted_first"] < tail, report
    # ...and the policy must be ENFORCED, not merely computed. The mask IS the
    # measurement: the all-keep arm serves every live page, the tight arm
    # serves strictly fewer KV positions than an unmasked decode attends over.
    tight = await _report(
        client, args, "trackb-h2o", {**common, "page_budget": 1})
    base = await _report(client, args, "naive-baseline", common)
    assert report["served_pages"] == report["selected_pages"], report
    assert report["served_kv"] == report["full_kv"], report
    assert tight["served_pages"] == 1, tight
    assert tight["served_kv"] < tight["full_kv"], tight
    # And enforcement has to reach the text: a one-page budget changes the
    # continuation, an all-keep mask reproduces the unmasked one verbatim. Only
    # the enforcement half can differ -- the observation half is unmasked in
    # both arms and decodes the baseline's own tokens.
    assert report["text"] != tight["text"], (
        "the keep-set is not enforced under H2O: a 1-page budget produced "
        "the same continuation as an all-keep one"
    )
    assert report["text"] == base["text"], (
        "an all-keep mask perturbed the output; the masked attention arm is "
        "not reproducing the causal one\n"
        f"  {report['text']!r}\n  {base['text']!r}"
    )


async def test_snapkv_attention(client, args):
    common = {"prompt": _EVICT_PROMPT, "max_tokens": 12,
              "temperature": _COHERENCE_TAU, "seed": 4242}
    report = await _report(
        client, args, "trackb-snapkv", {**common, "page_budget": 4096})
    # SnapKV reads a DIFFERENT TAP from every other policy here: the prefill
    # capture, which records the last `window` rows of the prompt against the
    # whole prefix, once, before any token is generated. So the row must
    # describe the prompt -- not the decode step H2O and TOVA see.
    assert report["observed_live"] == report["prompt_len"], (
        f"the score row covers {report['observed_live']} positions but the "
        f"prompt is {report['prompt_len']} tokens; the capture fired on a "
        f"different kv length than the program thinks\n  {report}"
    )
    # Unlike H2O there is no cold-start seed ramp in this row -- the prefill
    # capture writes exactly one entry per live kv position -- so slack past the
    # prompt is a bug, not a policy choice, and the bar is exactly zero.
    assert report["tail_nonzero"] == 0, report
    assert report["scores_nan"] == 0, report
    # Every layer folds one softmax distribution over the prefix, averaged over
    # heads and over window rows but NOT over layers, so the total mass is the
    # layer count. This pins the fold divisor: `heads * window` instead of
    # `heads * rows` would scale a short window down, and no normalization at
    # all would land at `heads * rows * layers`.
    layers = report["layers_observed"]
    assert layers > 0, report
    assert abs(report["score_mass"] - layers) < 0.02 * layers, (
        f"score mass {report['score_mass']} should equal the layer count "
        f"{layers} -- one folded distribution per layer\n  {report}"
    )
    # The mass has to sit where SnapKV says it does. `test_snapkv.py` sweeps the
    # window width to show the profile tracks it; here the cheap invariant is
    # that the observation window's own pages outweigh the middle of the prompt,
    # with the attention sink held out of both sides.
    page_mass = [float(m) for m in report["page_mass"]]
    win_lo = max(report["prompt_len"] - 32, 0) // report["page_size"]
    mid, win = page_mass[1:win_lo], page_mass[win_lo:]
    assert win and mid, report
    assert min(win) > max(mid), (
        f"the observation window {win} does not separate from the middle of "
        f"the prompt (max {max(mid):.5f}); the capture is not describing an "
        f"END-of-prompt window\n  {report['page_mass']}"
    )
    # ...and the policy must be ENFORCED, not merely computed. The eviction
    # goes through the attention mask (`.wiki/alto/attn-score.md` §4's first
    # honest delta: "evict" executes as CUSTOM MASK updates), so the mask IS
    # the measurement -- a count of set bits in the row the decode carried.
    tight = await _report(
        client, args, "trackb-snapkv", {**common, "page_budget": 1})
    base = await _report(client, args, "naive-baseline", common)
    assert report["served_pages"] == report["prompt_pages"], report
    assert report["served_kv"] == report["full_kv"] == report["prompt_len"], report
    assert tight["served_pages"] == 1, tight
    assert tight["served_kv"] < tight["full_kv"], tight
    # And the shrink has to reach the text: a one-page budget changes the
    # continuation, an all-keep mask reproduces the unmasked one verbatim.
    assert report["text"] != tight["text"], (
        "the keep-set is not enforced under SnapKV: a 1-page budget produced "
        "the same continuation as an all-keep one"
    )
    assert report["text"] == base["text"], (
        "an all-keep mask perturbed the output; the masked attention arm is "
        "not reproducing the causal one\n"
        f"  {report['text']!r}\n  {base['text']!r}"
    )


async def test_naive_baseline(client, args):
    report = await _report(client, args, "naive-baseline", {"max_tokens": 4})
    assert report["sampler"] == "naive-baseline", report


async def test_lora_probe(client, args):
    """A-1: a zero-B adapter is the base model, bit for bit; a live one moves.

    `lora-probe` IS `naive-baseline` plus one `Pass::adapter` at the
    mixer-output site -- same prompt, same seed, same temperature, same
    Gumbel-max draw -- so the two are comparable by construction, which is what
    makes this the one LoRA claim checkable with no reference implementation.

    At `adapter_scale: 0.0` the B plane is all zeros, so the correction
    `B(Ax)` is EXACTLY zero (adding a bf16 zero is exact) and the two strings
    must be equal. That fails on every way the plumbing can be wrong: an
    adapter that never landed, a mis-strided bank, a routes vector read at a
    lane offset instead of a row offset, a lane whose fact word missed the
    correction window.

    The temperature is collapsed for the reason
    `crates/worker/tests/cuda_lora_parity.rs` collapses it: at 1.0 this
    fixture's seeded Gumbel-max draw wanders on about one fire in sixteen over
    logits that are bit-identical, and asserting the sampler here would make an
    adapter gate red for a sampler defect.

    The live scale is 8.0 and not 0.5 because the fixture's A and B are
    random: a small correction is a real delta that is simply entitled not to
    move the argmax on this prompt. What is asserted is that a nonzero adapter
    CHANGES the answer -- without which the parity above passes just as well
    when the adapter was never staged at all.
    """
    fixed = {
        "prompt": "The capital of France is",
        "max_tokens": 6,
        "temperature": 0.01,
        "seed": 7,
    }
    base = await _report(client, args, "naive-baseline", dict(fixed))
    zero = await _report(client, args, "lora-probe", {**fixed, "adapter_scale": 0.0})
    assert zero["text"] == base["text"], (
        "a zero-B adapter changed the answer -- the correction term is exactly "
        "zero, so the adapter path wrote something it should not have, or read "
        "an operand that was not the one the lowering named\n"
        f"  base    = {base['text']!r}\n  adapted = {zero['text']!r}"
    )
    live = await _report(client, args, "lora-probe", {**fixed, "adapter_scale": 8.0})
    assert live["text"] != base["text"], (
        "a nonzero adapter left the answer unchanged -- the correction is not "
        "reaching the projections, and the parity above proves nothing"
    )
    again = await _report(client, args, "lora-probe", {**fixed, "adapter_scale": 8.0})
    assert again["text"] == live["text"], (
        "a nonzero adapter answered differently twice -- the correction is not "
        "a function of its inputs\n"
        f"  {live['text']!r}\n  {again['text']!r}"
    )


# ── THE MEDIA DOOR (`.wiki/alto/media-door.md`, M-1(b) / M-1(c)) ────────────
#
# **THESE TWO WANT A LOAD WITH A TOWER, AND THEY SAY SO RATHER THAN FAILING.**
# A vision checkpoint fits its family's text row AND its own, and the load
# identifies the text row first -- deliberately, because a two-unit load stands
# the fold down. So a census run that did not name a sku serves a trunk with no
# patch axis, and a fire carrying spans is refused there by name
# (`Fault::Towerless`, "this artifact declares no patch axis"). That is the
# CORRECT answer for that load, not a failure of this gate, so it is reported
# as a skip -- and the run that means to exercise the door names the row:
#
#     tests/inferlets/test_curated.py --model Qwen/Qwen3.5-0.8B \
#         --sku qwen35-d0.8b-vision-bf16-kv-bf16
TOWERLESS = ("no patch axis", "Towerless", "NoVisionFrontEnd")


async def _caption(client, args, color: str, **extra) -> dict:
    """One caption run, or a skip when this load has no vision tower."""
    try:
        return await _report(
            client, args, "image-captioning", {"color": color, **extra}
        )
    except RuntimeError as error:
        if any(mark in str(error) for mark in TOWERLESS):
            # conftest reports FileNotFoundError as SKIPPED.
            raise FileNotFoundError(
                f"the loaded model has no vision tower, so the media door has "
                f"nothing to open onto: {error}"
            ) from error
        raise


async def test_image_captioning(client, args):
    """M-1(b): a deterministic, sensible caption on a real image, e2e.

    Three assertions, and they are three different strengths of claim.

    1. **SENSIBLE** -- the caption names the colour. This is the weakest of the
       three and the one that is about the MODEL: a 0.8-billion-parameter
       trunk describing a flat colour field is not a hard vision task, but it
       is still a model's taste, and a checkpoint that answered "a solid block
       of colour" would fail this while being perfectly correct. That is the
       honest limit of "sensible" here, and it is why the image is a solid
       square rather than a photograph: it is the one picture whose content is
       a sentence this file can state.

    2. **DETERMINISTIC** -- greedy decoding, so the same image and the same
       prompt answer the same tokens twice. This is about the ENGINE: a patch
       payload staged into a slot the previous fire left dirty, or a route
       computed off a stale offset, shows up here as a caption that moves.

    3. **THE TOWER CONDITIONED THE TRUNK** -- a red square and a blue square
       answer DIFFERENTLY. This is the claim that holds whatever the model's
       taste is, and it is the one that catches the failure the whole door
       exists to prevent: a pass that scattered nothing, or scattered into the
       wrong rows, answers fluently about an image it never saw -- and answers
       the same fluent thing for every image.
    """
    red = await _caption(client, args, "red")
    assert red["soft_tokens"] > 0, "the span occupied no token rows"
    assert red["prompt_tokens"] > red["soft_tokens"], (
        "the prefill carried the span and no text around it"
    )
    lowered = red["text"].lower()
    assert "red" in lowered, (
        "the caption of a solid red square does not mention red: "
        f"{red['text']!r}"
    )

    again = await _caption(client, args, "red")
    assert again["text"] == red["text"], (
        "greedy captioning of one image answered differently twice\n"
        f"  first  = {red['text']!r}\n  second = {again['text']!r}"
    )
    assert again["digest"] == red["digest"], (
        "one image preprocessed to two different spans"
    )

    blue = await _caption(client, args, "blue")
    assert blue["digest"] != red["digest"], (
        "two different images share a span digest -- media-door §5's statute "
        "rests on exactly this not happening"
    )
    assert blue["text"] != red["text"], (
        "a red square and a blue square got the same caption, so the patch "
        "rows did not reach the trunk\n"
        f"  red  = {red['text']!r}\n  blue = {blue['text']!r}"
    )


async def test_image_captioning_beside_text_lanes(client, args):
    """M-1(c): a caption lane co-batched with text lanes, guest level.

    The door is open per submission; this asks whether it is open per LANE.
    A caption running alone and a caption running beside three text programs
    must answer the same tokens -- the media rows are keyed by lane and the
    batcher rebases them onto the step it co-batches into, so a rebase off by
    one lane would scatter this lane's patches into a neighbour's rows.

    The text neighbours are checked too, in the other direction: a text lane
    beside a media lane must still attend its own prompt, which is what fails
    if the seriated fire put the second row axis somewhere the text lanes'
    rectangle overlapped.
    """
    solo = await _caption(client, args, "green")

    ask = {"prompt": ATTENDS_PROMPT, "max_tokens": 12, "temperature": 0.0}
    results = await asyncio.gather(
        _caption(client, args, "green"),
        run_inferlet(client, "chat-completion", ask, timeout=args.timeout),
        run_inferlet(client, "chat-completion", ask, timeout=args.timeout),
        run_inferlet(client, "chat-completion", ask, timeout=args.timeout),
    )
    mixed, *texts = results
    assert mixed["text"] == solo["text"], (
        "a caption lane answered differently beside text lanes than alone\n"
        f"  alone = {solo['text']!r}\n  mixed = {mixed['text']!r}"
    )
    for i, text in enumerate(texts):
        lowered = text.lower()
        assert "france" in lowered or "paris" in lowered, (
            f"text lane {i} beside a media lane did not attend its prompt: "
            f"{text[:200]!r}"
        )


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


async def test_greedy_decoding_is_the_same_alone_and_in_a_crowd(client, args):
    """A greedy program answers the same whether or not it has neighbours.

    Everything else in this file runs one program at a time against a server
    that batches, pages, and grows a KV pool for many at once -- so the whole
    sweep can be green while the interesting half of the engine has never been
    asked a question. This asks it: the same prompt at temperature 0, run alone
    and then run 2, 4 and 8 ways at once, must give back the same tokens every
    time. Anything a request can see of another -- a page, a row of scores, a
    slot in a batch -- shows up here as a lane that diverged.

    It also covers the thing that made this test hard to write. Two of these
    launching together install the same program twice at once, which the
    server used to key by the program's hash and therefore treat as one
    upload.
    """
    ask = {"prompt": "Explain why the sky appears blue.", "max_tokens": 24, "temperature": 0.0}

    async def once():
        return await run_inferlet(client, "chat-completion", ask, timeout=300)

    solo = await once()
    assert solo.strip(), "the greedy run said nothing"
    for width in (2, 4, 8):
        lanes = await asyncio.gather(*[once() for _ in range(width)])
        for i, lane in enumerate(lanes):
            assert lane == solo, (
                f"at width {width}, lane {i} diverged from the same prompt run alone:\n"
                f"  alone = {solo!r}\n  crowd = {lane!r}"
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
        test_lora_probe,
        test_greedy_decoding_is_the_same_alone_and_in_a_crowd,
        test_quest_attention,
        test_tova_attention,
        test_h2o_attention,
        test_snapkv_attention,
        test_image_captioning,
        test_image_captioning_beside_text_lanes,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Curated inferlet E2E tests")
