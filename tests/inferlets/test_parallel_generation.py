"""E2E test for parallel-generation inferlet."""
from conftest import run_inferlet, run_tests


async def test_parallel_generation(client, args):
    output = await run_inferlet(
        client, "parallel-generation",
        ["--max-tokens", "32"],
        timeout=args.timeout,
    )
    assert "Output 1:" in output, "Missing Output 1"
    assert "Output 2:" in output, "Missing Output 2"


if __name__ == "__main__":
    run_tests([test_parallel_generation])
