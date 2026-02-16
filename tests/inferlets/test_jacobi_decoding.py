"""E2E test for jacobi-decoding inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_jacobi_decoding(client, args):
    output = await run_inferlet(
        client, "jacobi-decoding",
        ["--max-tokens", "32"],
        timeout=args.timeout,
    )
    assert "Jacobi Decoding" in output, "Missing header"
    assert "Output:" in output, "Missing 'Output:' section"
    assert "tokens/s" in output, "Missing throughput metric"

    match = re.search(r"Generated (\d+) tokens", output)
    assert match, "Missing 'Generated N tokens' line"
    assert int(match.group(1)) > 0, "Generated 0 tokens"


if __name__ == "__main__":
    run_tests([test_jacobi_decoding])
