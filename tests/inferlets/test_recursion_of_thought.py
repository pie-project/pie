"""E2E test for recursion-of-thought inferlet."""
from conftest import run_inferlet, run_tests


async def test_recursion_of_thought(client, args):
    output = await run_inferlet(
        client, "recursion-of-thought",
        {"max_tokens": 64, "max_depth": 2},
        timeout=args.timeout,
    )
    assert "RoT Complete" in output, "Missing 'RoT Complete' message"
    assert "Final solution:" in output, "Missing 'Final solution:'"


if __name__ == "__main__":
    run_tests([test_recursion_of_thought])
