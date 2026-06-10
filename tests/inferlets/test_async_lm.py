"""E2E tests for async-lm inferlet (AsyncLM: arXiv:2412.07017).

Tests cover:
1. Basic completion — prompt with no tool calls returns non-empty text
2. Tool call dispatch — weather prompt triggers a [CALL] and gets an [INTR] result
3. Midstream injection — enable_midstream=True completes without error
4. Trace output — enable_trace=True emits [TRACE] events
5. Low token budget — max_tokens=32 returns output without crash
"""

from conftest import run_inferlet, run_tests


async def test_async_lm_basic_completion(client, args):
    """A simple factual prompt with no tool calls returns non-empty text."""
    output = await run_inferlet(
        client,
        "async-lm",
        {
            "prompt": "What is the capital of France?",
            "max_tokens": 64,
            "temperature": 0.0,
            "disable_checkpointing": True,
        },
        timeout=args.timeout,
    )
    assert output.strip(), "Expected non-empty output"
    assert "[AsyncLM]" in output, (
        f"Expected [AsyncLM] header in output, got: {output[:200]}"
    )


async def test_async_lm_weather_tool_call(client, args):
    """A weather prompt triggers a [CALL] dispatch and returns a result."""
    output = await run_inferlet(
        client,
        "async-lm",
        {
            "prompt": "What is the current weather in Tokyo?",
            "max_tokens": 256,
            "temperature": 0.0,
            "disable_checkpointing": True,
            "emulate_unknown_tools": True,
        },
        timeout=args.timeout,
    )
    assert output.strip(), "Expected non-empty output"
    assert "[AsyncLM]" in output, (
        f"Expected [AsyncLM] header in output, got: {output[:200]}"
    )


async def test_async_lm_midstream_injection(client, args):
    """enable_midstream=True with a tool-triggering prompt completes without error."""
    output = await run_inferlet(
        client,
        "async-lm",
        {
            "prompt": "What is the stock price of Apple?",
            "max_tokens": 256,
            "temperature": 0.0,
            "disable_checkpointing": True,
            "enable_midstream": True,
            "emulate_unknown_tools": True,
        },
        timeout=args.timeout,
    )
    assert output.strip(), "Expected non-empty output with midstream injection"
    assert "[AsyncLM]" in output, (
        f"Expected [AsyncLM] header in output, got: {output[:200]}"
    )


async def test_async_lm_trace_output(client, args):
    """enable_trace=True emits machine-readable [TRACE] events."""
    output = await run_inferlet(
        client,
        "async-lm",
        {
            "prompt": "What time is it in London?",
            "max_tokens": 128,
            "temperature": 0.0,
            "disable_checkpointing": True,
            "enable_trace": True,
            "emulate_unknown_tools": True,
        },
        timeout=args.timeout,
    )
    assert output.strip(), "Expected non-empty output"
    assert "[TRACE]" in output, (
        f"Expected [TRACE] events in output when enable_trace=True, got: {output[:200]}"
    )


async def test_async_lm_low_token_budget(client, args):
    """max_tokens=32 returns output without crash."""
    output = await run_inferlet(
        client,
        "async-lm",
        {
            "prompt": "Say hello.",
            "max_tokens": 32,
            "temperature": 0.0,
            "disable_checkpointing": True,
        },
        timeout=args.timeout,
    )
    assert output.strip(), "Expected non-empty output even with low token budget"
    assert "[AsyncLM]" in output, (
        f"Expected [AsyncLM] header in output, got: {output[:200]}"
    )


if __name__ == "__main__":
    run_tests([
        test_async_lm_basic_completion,
        test_async_lm_weather_tool_call,
        test_async_lm_midstream_injection,
        test_async_lm_trace_output,
        test_async_lm_low_token_budget,
    ])
