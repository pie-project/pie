"""E2E test for attention-sink inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_attention_sink(client, args):
    output = await run_inferlet(
        client, "attention-sink",
        ["--max-tokens", "64"],
        timeout=args.timeout,
    )
    assert "Attention Sink" in output, "Missing header"
    assert "sink=" in output and "window=" in output, "Missing sink/window params"
    assert "Output:" in output, "Missing 'Output:' section"

    match = re.search(r"Generated (\d+) tokens", output)
    assert match, "Missing 'Generated N tokens' line"
    assert int(match.group(1)) > 0, "Generated 0 tokens"


if __name__ == "__main__":
    run_tests([test_attention_sink])
