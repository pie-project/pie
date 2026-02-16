"""E2E test for graph-of-thought inferlet."""
from conftest import run_inferlet, run_tests


async def test_graph_of_thought(client, args):
    output = await run_inferlet(
        client, "graph-of-thought",
        ["--proposal-tokens", "64,64,64,64"],
        timeout=args.timeout,
    )
    assert "Aggregation complete" in output, "Missing 'Aggregation complete' message"
    assert "Final aggregated solution" in output, "Missing final solution"


if __name__ == "__main__":
    run_tests([test_graph_of_thought])
