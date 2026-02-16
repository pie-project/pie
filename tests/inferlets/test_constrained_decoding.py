"""E2E test for constrained-decoding inferlet."""
import re

from conftest import run_inferlet, run_tests


async def test_constrained_decoding(client, args):
    output = await run_inferlet(
        client, "constrained-decoding",
        ["--num-tokens", "128"],
        timeout=args.timeout,
    )
    assert "Generated (constrained):" in output, "Missing constrained output header"
    assert "Elapsed:" in output, "Missing elapsed time"

    match = re.search(r"Generated \(constrained\):\s*\n(.+?)(?:\n\nElapsed|\Z)", output, re.DOTALL)
    assert match and match.group(1).strip(), "Generated text is empty"


if __name__ == "__main__":
    run_tests([test_constrained_decoding])
