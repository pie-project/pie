"""E2E test for sampler-suite: exercises every output-distribution sampler
in a single forward pass and cross-checks the values.

Asserts:
  * 6 slots returned, in the order attached
  * argmax(decoded raw_logits) == greedy token == distribution[0].id
  * top-8 distribution probs are sorted descending and sum to <= 1
  * logprob(t) value matches the same `t`'s entry in logprobs([t, ...])
  * entropy is in [0, ln(vocab)]

Run::

    uv run python tests/inferlets/test_sampler_suite.py --model Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import re

from conftest import run_inferlet, run_tests


_KV_RE = re.compile(r"^([A-Z0-9_]+)=(.+)$", re.MULTILINE)


def _parse(output: str) -> dict[str, str]:
    return dict(_KV_RE.findall(output))


def _is_true(s: str) -> bool:
    return s.strip().lower() == "true"


async def test_sampler_suite(client, args):
    output = await run_inferlet(
        client,
        "sampler-suite",
        timeout=args.timeout,
    )
    fields = _parse(output)

    required = {
        "VOCAB_SIZE", "SLOT_COUNT",
        "GREEDY_TOKEN", "RAW_ARGMAX_TOKEN", "ARGMAX_MATCHES_GREEDY",
        "DIST_FIRST_ID", "DIST_FIRST_PROB", "DIST_FIRST_MATCHES_GREEDY",
        "DIST_PROBS_SORTED", "DIST_PROBS_TOP8_SUM",
        "LOGPROB_CAND_A", "LOGPROBS_CAND_A", "LOGPROBS_CONSISTENT",
        "LOGPROBS_LEN",
        "ENTROPY", "ENTROPY_MAX", "ENTROPY_IN_BOUNDS",
    }
    missing = required - set(fields)
    assert not missing, f"Missing keys {missing}\n----\n{output}"

    vocab_size = int(fields["VOCAB_SIZE"])
    assert vocab_size > 1000, f"implausible vocab_size={vocab_size}"
    assert int(fields["SLOT_COUNT"]) == 6, fields["SLOT_COUNT"]

    # ---- Composability: all 6 sampler kinds returned in attachment order ---
    assert _is_true(fields["ARGMAX_MATCHES_GREEDY"]), (
        "argmax(raw_logits) != greedy token — raw_logits and the sampler "
        "kernel are seeing different tensors"
    )
    assert _is_true(fields["DIST_FIRST_MATCHES_GREEDY"]), (
        "distribution[0].id != greedy token — distribution and sampling "
        "paths disagree"
    )

    # ---- Distribution shape sanity ----
    assert _is_true(fields["DIST_PROBS_SORTED"]), (
        "Distribution probs were not sorted descending"
    )
    sum_top8 = float(fields["DIST_PROBS_TOP8_SUM"])
    assert 0.0 < sum_top8 <= 1.0 + 1e-3, f"top-8 sum implausible: {sum_top8}"

    # ---- Logprob consistency: single == multi[0] ----
    assert _is_true(fields["LOGPROBS_CONSISTENT"]), (
        f"logprob({fields['LOGPROB_CAND_A']}) != logprobs[0] "
        f"({fields['LOGPROBS_CAND_A']}) — log-softmax computed differently"
    )
    assert int(fields["LOGPROBS_LEN"]) == 3, fields["LOGPROBS_LEN"]

    # ---- Entropy in bounds [0, ln(vocab)] ----
    assert _is_true(fields["ENTROPY_IN_BOUNDS"]), (
        f"entropy {fields['ENTROPY']} outside [0, {fields['ENTROPY_MAX']}]"
    )
    h = float(fields["ENTROPY"])
    h_max = float(fields["ENTROPY_MAX"])
    assert 0.0 <= h <= h_max + 1e-3

    # Informational summary
    print(
        f"\n  greedy_token={fields['GREEDY_TOKEN']}  "
        f"top-8 sum={sum_top8:.3f}  H={h:.3f}/{h_max:.3f}  "
        f"logprob(cand_a)={fields['LOGPROB_CAND_A']}"
    )


def tests():
    return [test_sampler_suite]


if __name__ == "__main__":
    run_tests(tests(), description="Sampler-suite E2E test")
