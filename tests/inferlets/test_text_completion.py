"""E2E test for text-completion inferlet."""
from conftest import run_inferlet, run_tests


async def test_text_completion(client, args):
    output = await run_inferlet(
        client, "text-completion",
        {"p": "Hello, world!", "max_tokens": 32},
        timeout=args.timeout,
    )
    assert len(output) > 0, "Output is empty"


if __name__ == "__main__":
    run_tests([test_text_completion])
