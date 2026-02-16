"""E2E test for agent-react inferlet."""
from conftest import run_inferlet, run_tests


async def test_agent_react(client, args):
    output = await run_inferlet(
        client, "agent-react",
        ["--num-function-calls", "5", "--tokens-between-calls", "256"],
        timeout=args.timeout,
    )
    # The agent should either find a final answer or report none
    has_final = "Final answer:" in output
    has_no_final = "No final answer found" in output
    assert has_final or has_no_final, "Missing expected output"


if __name__ == "__main__":
    run_tests([test_agent_react])
