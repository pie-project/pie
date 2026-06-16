"""E2E tests for the raw-completion inferlet.

raw-completion tokenises the prompt verbatim (no chat template) and runs
a top-p sampler directly, making it useful for probing base/pretrained
models.  These tests verify:

  1. Basic completion returns non-empty text for a simple prompt.
  2. Optional sampling parameters (temperature, top_p) are accepted and
     do not cause an error.
  3. A longer max_tokens budget produces at least as many tokens as a
     short one (generation isn't silently truncated to zero).

Run::

    uv run python tests/inferlets/test_raw_completion.py --dummy
    uv run python tests/inferlets/test_raw_completion.py --model Qwen/Qwen3-0.6B
"""
from __future__ import annotations

from conftest import run_inferlet, run_tests


async def test_raw_completion_basic(client, args):
    """Prompt completes and returns non-empty text."""
    output = await run_inferlet(
        client,
        "raw-completion",
        {"prompt": "The capital of France is", "max_tokens": 16},
        timeout=args.timeout,
    )
    assert len(output.strip()) > 0, "Output is empty"


async def test_raw_completion_optional_params(client, args):
    """Optional temperature and top_p are accepted without error."""
    output = await run_inferlet(
        client,
        "raw-completion",
        {
            "prompt": "Once upon a time",
            "max_tokens": 16,
            "temperature": 0.8,
            "top_p": 0.9,
        },
        timeout=args.timeout,
    )
    assert len(output.strip()) > 0, "Output is empty with custom sampling params"


async def test_raw_completion_longer_budget_produces_more_tokens(client, args):
    """A larger max_tokens budget yields a longer (or equal) completion."""
    short = await run_inferlet(
        client,
        "raw-completion",
        {"prompt": "List the planets:", "max_tokens": 8},
        timeout=args.timeout,
    )
    long = await run_inferlet(
        client,
        "raw-completion",
        {"prompt": "List the planets:", "max_tokens": 64},
        timeout=args.timeout,
    )
    assert len(long) >= len(short), (
        f"Longer budget produced fewer chars ({len(long)} < {len(short)})"
    )


def tests():
    return [
        test_raw_completion_basic,
        test_raw_completion_optional_params,
        test_raw_completion_longer_budget_produces_more_tokens,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="raw-completion inferlet E2E tests")
