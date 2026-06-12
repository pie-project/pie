"""E2E interface test for the unified reasoning benchmark inferlet."""
import json

from conftest import run_inferlet, run_tests


async def test_reasoning_benchmark(client, args):
    for pattern in (
        "direct",
        "best_of_n",
        "tree_of_thought",
        "graph_of_thought",
    ):
        output = await run_inferlet(
            client,
            "reasoning-benchmark",
            {
                "pattern": pattern,
                "question": "Mia has 7 apples and buys 5 more. How many apples?",
                "num_candidates": 2,
                "beam_width": 1,
                "max_tokens": 48,
                "score_tokens": 8,
            },
            timeout=args.timeout,
        )
        result = json.loads(output)
        assert result["pattern"] == pattern
        assert isinstance(result["candidates"], list)
        assert result["candidates"]
        assert result["stats"]["generation_calls"] >= 1
        assert result["stats"]["context_forks"] >= 1


if __name__ == "__main__":
    run_tests([test_reasoning_benchmark])
