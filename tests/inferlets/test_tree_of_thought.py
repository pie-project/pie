"""E2E test for tree-of-thought inferlet."""
from conftest import run_inferlet, run_tests


async def test_tree_of_thought(client, args):
    output = await run_inferlet(
        client, "tree-of-thought",
        ["--max-tokens", "64"],
        timeout=args.timeout,
    )
    assert len(output) > 0, "Output is empty"


if __name__ == "__main__":
    run_tests([test_tree_of_thought])
