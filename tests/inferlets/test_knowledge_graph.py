"""E2E test for knowledge-graph inferlet."""
from conftest import run_inferlet, run_tests


async def test_knowledge_graph(client, args):
    output = await run_inferlet(
        client, "knowledge-graph",
        {"max_tokens": 2048},
        timeout=max(args.timeout, 300),
    )
    assert "Stage 1: Extracting knowledge triples" in output, "Missing stage 1"
    assert "Stage 2: Building knowledge graph" in output, "Missing stage 2"
    assert "Stage 3: Querying graph" in output, "Missing stage 3"
    assert "Stage 4: Generating answer" in output, "Missing stage 4"
    assert "Graph:" in output, "Missing graph stats"
    assert "Answer:" in output, "Missing answer"


if __name__ == "__main__":
    run_tests([test_knowledge_graph])
