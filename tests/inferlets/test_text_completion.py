"""E2E test for text-completion inferlet."""
from conftest import run_inferlet, run_tests


async def test_text_completion(client, args):
    output = await run_inferlet(
        client, "text-completion",
        {
            "prompt": "What is the capital of France?",
            "max_tokens": 64,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        timeout=args.timeout,
    )
    assert len(output) > 0, "Output is empty"
    assert "Paris" in output, f"Expected a factual answer about Paris, got: {output[:200]!r}"


if __name__ == "__main__":
    run_tests([test_text_completion])
