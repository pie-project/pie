"""E2E test for best-of-n inferlet."""
from conftest import run_inferlet, run_tests


async def test_best_of_n(client, args):
    output = await run_inferlet(
        client, "best-of-n",
        {"num_candidates": 3, "max_tokens": 256},
        timeout=args.timeout,
    )
    assert "Generating" in output, "Missing generation header"
    assert "Candidate Rankings" in output, "Missing candidate rankings"
    assert "Final Answer:" in output, "Missing final answer"


if __name__ == "__main__":
    run_tests([test_best_of_n])
