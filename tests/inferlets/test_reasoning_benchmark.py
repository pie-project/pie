"""E2E interface tests for benchmark reasoning inferlets."""
import json

from conftest import run_inferlet, run_tests


REASONING_INFERLETS = {
    "reasoning-direct": "direct",
    "reasoning-best-of-n": "best_of_n",
    "reasoning-tree-of-thought": "tree_of_thought",
    "reasoning-graph-of-thought": "graph_of_thought",
}


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


async def test_reasoning_base(client, args):
    output = await run_inferlet(
        client,
        "reasoning-base",
        {
            "prompt": "Say the word ready.",
            "max_tokens": 16,
        },
        timeout=args.timeout,
    )
    result = json.loads(output)
    assert isinstance(result["completion"], str)
    assert result["stats"]["generator_steps"] >= 0
    assert result["stats"]["generated_tokens"] >= 0


async def test_separate_reasoning_inferlets(client, args):
    for inferlet, pattern in REASONING_INFERLETS.items():
        output = await run_inferlet(
            client,
            inferlet,
            {
                "pattern": "direct",
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
    run_tests(
        [test_reasoning_benchmark, test_reasoning_base, test_separate_reasoning_inferlets]
    )
