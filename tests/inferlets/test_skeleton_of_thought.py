"""E2E test for skeleton-of-thought inferlet."""
from conftest import run_inferlet, run_tests


async def test_skeleton_of_thought(client, args):
    output = await run_inferlet(
        client, "skeleton-of-thought",
        {"max_tokens": 64},
        timeout=args.timeout,
    )
    assert "Completed in" in output, "Missing 'Completed in' timing line"


if __name__ == "__main__":
    run_tests([test_skeleton_of_thought])
