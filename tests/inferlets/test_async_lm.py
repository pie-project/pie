"""E2E tests for the async-lm inferlet — AsyncLM async function calling (arXiv:2412.07017).

These are REAL behavioral tests: they assert that the model actually drives the
CML control protocol ([CALL] -> [TRAP] -> runtime-injected [INTR]) end to end.
That requires a model large enough to emit well-formed CML — in practice >=8B.
The suite default (Qwen/Qwen3-0.6B) and the --dummy driver produce random/short
tokens that never form valid CML, so these assertions will (correctly) FAIL
there. This file therefore pins Qwen/Qwen3-8B by default; override with
--model <repo> / --driver <d>.

Every asserted string is the inferlet's own stdout (see inferlets/async-lm/src/lib.rs),
verified to appear in 14/14 real 8B runs under asyncLM/benchmarking/results/sweep_final_*:
  - `[AsyncLM] dispatching id=<id> code=<tool(...)>`   non-blocking [CALL] dispatch
  - `[AsyncLM] (mid|keep)-inject: [INTR] <id> [HEAD] <result> [END]`  result injected back
  - `[TRACE] v=async t=<ms> kind=dispatch|inject|run_end ...`  machine-readable timeline

What each test covers:
  1. dispatch           — a tool-needing prompt emits >=1 `dispatching id=`
  2. parallel dispatch  — independent asks dispatch >=2 distinct ids in one round
  3. result round-trip  — a dispatched call's result is injected ([INTR]) back into context
  4. midstream injection — enable_midstream runs the injection path to completion
  5. trace schema       — enable_trace emits the [TRACE] kind=dispatch/run_end stream

Prompts are prefixed with `/no_think` (Qwen3 think-mode can loop/hang and only
adds decode latency; this matches the apples-to-apples benchmark config).
"""

import re

from conftest import run_inferlet, run_tests

# Capped tokens: the model needs only ~100-150 tokens to dispatch the calls and
# answer. Temperature is left at the inferlet default (the reference 8B sweeps ran
# at default temp and hit every asserted marker; assertions here are structural,
# not exact-text, so they tolerate sampling). disable_checkpointing=True is the
# single-session/always-Keep mode those sweeps used (and where injection fires
# most reliably).
_BASE = {
    "max_tokens": 512,
    "disable_checkpointing": True,
}

# A multi-tool ask that reliably dispatches three independent calls
# (get_stock_price, get_weather, get_time) — the canonical sweep prompt.
_MULTI_PROMPT = (
    "/no_think What is the stock price of AAPL, the weather in Tokyo, "
    "and the current time in London?"
)

_DISPATCH_RE = re.compile(r"\[AsyncLM\] dispatching id=(\S+)")
_INJECT_RE = re.compile(r"\[AsyncLM\] (?:mid|keep)-inject: \[INTR\]")


def _dispatched_ids(output: str) -> set[str]:
    return set(_DISPATCH_RE.findall(output))


async def test_async_lm_dispatch(client, args):
    """A tool-needing prompt makes the model emit at least one non-blocking [CALL]."""
    output = await run_inferlet(
        client, "async-lm",
        {**_BASE, "prompt": _MULTI_PROMPT},
        timeout=args.timeout,
    )
    ids = _dispatched_ids(output)
    assert ids, (
        "Expected at least one '[AsyncLM] dispatching id=' (a CML [CALL] dispatch); "
        f"got none. Model likely too small to drive CML. Output tail: {output[-400:]}"
    )


async def test_async_lm_parallel_dispatch(client, args):
    """Independent asks are dispatched as multiple distinct calls in one round."""
    output = await run_inferlet(
        client, "async-lm",
        {**_BASE, "prompt": _MULTI_PROMPT},
        timeout=args.timeout,
    )
    ids = _dispatched_ids(output)
    assert len(ids) >= 2, (
        f"Expected >=2 distinct dispatched call ids (parallel dispatch); got {sorted(ids)}. "
        f"Output tail: {output[-400:]}"
    )


async def test_async_lm_result_injection(client, args):
    """A dispatched call's result is injected back into the context as an [INTR] frame."""
    output = await run_inferlet(
        client, "async-lm",
        {**_BASE, "prompt": _MULTI_PROMPT},
        timeout=args.timeout,
    )
    assert _dispatched_ids(output), "No call was dispatched, so nothing to inject."
    assert _INJECT_RE.search(output), (
        "Expected an injected result frame ('[AsyncLM] mid-/keep-inject: [INTR] ...') "
        f"after dispatch. Output tail: {output[-500:]}"
    )


async def test_async_lm_midstream_injection(client, args):
    """enable_midstream runs the mid-stream injection path through to a delivered result."""
    output = await run_inferlet(
        client, "async-lm",
        {
            **_BASE,
            "prompt": (
                "/no_think Get the weather in Paris and the current time in Tokyo, then "
                "tell me whether it's a reasonable hour to video-call Tokyo from Paris."
            ),
            "enable_midstream": True,
        },
        timeout=args.timeout,
    )
    assert _dispatched_ids(output), (
        f"Expected a dispatch with enable_midstream. Output tail: {output[-400:]}"
    )
    assert _INJECT_RE.search(output), (
        "Expected a result to be injected on the midstream path "
        f"('mid-/keep-inject: [INTR]'). Output tail: {output[-500:]}"
    )


async def test_async_lm_trace_schema(client, args):
    """enable_trace emits the machine-readable [TRACE] timeline (dispatch + run_end kinds)."""
    output = await run_inferlet(
        client, "async-lm",
        {**_BASE, "prompt": _MULTI_PROMPT, "enable_trace": True},
        timeout=args.timeout,
    )
    assert "[TRACE]" in output, (
        f"Expected [TRACE] events when enable_trace=True. Output tail: {output[-400:]}"
    )
    assert "kind=dispatch" in output, (
        f"Expected a 'kind=dispatch' trace event. Output tail: {output[-400:]}"
    )
    assert "kind=run_end" in output, (
        f"Expected a 'kind=run_end' trace event (run completed). Output tail: {output[-400:]}"
    )


if __name__ == "__main__":
    import sys

    # AsyncLM's CML protocol needs a capable model; default to >=8B unless the
    # caller pinned a model explicitly. Skip the default for a dummy driver
    # (it ignores weights anyway) to avoid an 8B download for a run that can't pass.
    _has_model = any(a == "--model" or a.startswith("--model=") for a in sys.argv)
    _is_dummy = "--dummy" in sys.argv or ("--driver" in sys.argv and "dummy" in sys.argv)
    if not _has_model and not _is_dummy:
        sys.argv += ["--model", "Qwen/Qwen3-8B"]

    run_tests([
        test_async_lm_dispatch,
        test_async_lm_parallel_dispatch,
        test_async_lm_result_injection,
        test_async_lm_midstream_injection,
        test_async_lm_trace_schema,
    ])
