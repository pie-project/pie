"""E2E test for watermarking inferlet."""
from conftest import run_inferlet, run_tests


async def test_watermarking(client, args):
    output = await run_inferlet(
        client, "watermarking",
        {"max_tokens": 32},
        timeout=args.timeout,
    )
    assert "Output:" in output, "Missing 'Output:' section"


if __name__ == "__main__":
    run_tests([test_watermarking])
