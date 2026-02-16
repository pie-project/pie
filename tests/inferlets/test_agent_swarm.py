"""E2E test for agent-swarm inferlet."""
from conftest import run_inferlet, run_tests


async def test_agent_swarm(client, args):
    output = await run_inferlet(
        client, "agent-swarm",
        ["idea_generator", "--prompt",
         "A detective story set in a cyberpunk city where AI and humans coexist"],
        timeout=args.timeout,
    )
    assert "Broadcasted story to channel" in output, "Missing broadcast message"
    assert "concept_to_plot" in output, "Missing 'concept_to_plot' topic"


if __name__ == "__main__":
    run_tests([test_agent_swarm])
