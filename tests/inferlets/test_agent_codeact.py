"""E2E test for agent-codeact inferlet."""
from conftest import run_inferlet, run_tests


async def test_agent_codeact(client, args):
    output = await run_inferlet(
        client, "agent-codeact",
        ["--num-function-calls", "5", "--tokens-between-calls", "256"],
        timeout=args.timeout,
    )
    # Should either reach a final answer or report none
    has_final = "Final answer:" in output
    has_no_final = "No final answer found" in output
    assert has_final or has_no_final, "Missing expected output"


if __name__ == "__main__":
    run_tests([test_agent_codeact])
