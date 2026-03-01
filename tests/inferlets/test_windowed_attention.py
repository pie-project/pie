"""E2E test for windowed-attention inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_windowed_attention(client, args):
    output = await run_inferlet(
        client, "windowed-attention",
        {"max_tokens": 64},
        timeout=args.timeout,
    )
    assert "Windowed Attention" in output, "Missing header"
    assert "Output:" in output, "Missing 'Output:' section"

    match = re.search(r"Generated (\d+) tokens", output)
    assert match, "Missing 'Generated N tokens' line"
    assert int(match.group(1)) > 0, "Generated 0 tokens"


if __name__ == "__main__":
    run_tests([test_windowed_attention])
