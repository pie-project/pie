"""Smoke tests for the curated inference-time inferlets.

Native MTP is build-validated separately because the default test model does
not expose multi-token-prediction heads.
"""

from conftest import run_inferlet, run_tests


async def _nonempty(client, args, name: str, inputs: dict) -> str:
    output = await run_inferlet(client, name, inputs, timeout=args.timeout)
    assert output.strip(), f"{name} returned empty output"
    return output


async def test_chat_completion(client, args):
    await _nonempty(client, args, "chat-completion", {"prompt": "Say hello.", "max_tokens": 4})


async def test_sampling_primitives(client, args):
    output = await _nonempty(client, args, "sampling-primitives", {})
    assert "token=" in output and "entropy=" in output


async def test_consensus_decoding(client, args):
    await _nonempty(
        client,
        args,
        "consensus-decoding",
        {"question": "What is 2 + 2?", "num_candidates": 2, "max_tokens": 4},
    )


async def test_greenlist_watermarking(client, args):
    await _nonempty(client, args, "greenlist-watermarking", {"max_tokens": 4})


async def test_json_schema_constrained_decoding(client, args):
    output = await _nonempty(
        client,
        args,
        "json-schema-constrained-decoding",
        {
            "prompt": "Return an object with an integer field named value.",
            "schema": (
                '{"type":"object","properties":{"value":{"type":"integer"}},'
                '"required":["value"],"additionalProperties":false}'
            ),
            "max_tokens": 64,
        },
    )
    assert "value" in output


async def test_attention_sink(client, args):
    await _nonempty(
        client,
        args,
        "attention-sink",
        {"prompt": "Count upward.", "max_tokens": 4, "sink_size": 1, "window_size": 2},
    )


async def test_sliding_window_attention(client, args):
    await _nonempty(
        client,
        args,
        "sliding-window-attention",
        {"prompt": "Count upward.", "max_tokens": 4, "window_size": 2},
    )


async def test_prefix_tree_kv_cache(client, args):
    output = await _nonempty(client, args, "prefix-tree-kv-cache", {"num_tokens": 2})
    assert "city at dawn:" in output and "forest at night:" in output


async def test_cacheback_speculative_decoding(client, args):
    await _nonempty(
        client,
        args,
        "cacheback-speculative-decoding",
        {"max_tokens": 4, "draft_length": 2, "max_ngram": 4},
    )


async def test_mirostat_v2_sampling(client, args):
    output = await _nonempty(client, args, "mirostat-v2-sampling", {"max_tokens": 4})
    assert "mirostat-v2" in output


async def test_beam_search(client, args):
    await _nonempty(client, args, "beam-search", {"max_tokens": 2})


async def test_contrastive_decoding(client, args):
    await _nonempty(
        client,
        args,
        "contrastive-decoding",
        {"prompt": "Say hello.", "max_tokens": 4, "amateur_window": 2},
    )


def tests():
    return [
        test_chat_completion,
        test_sampling_primitives,
        test_consensus_decoding,
        test_greenlist_watermarking,
        test_json_schema_constrained_decoding,
        test_attention_sink,
        test_sliding_window_attention,
        test_prefix_tree_kv_cache,
        test_cacheback_speculative_decoding,
        test_mirostat_v2_sampling,
        test_beam_search,
        test_contrastive_decoding,
    ]


if __name__ == "__main__":
    run_tests(tests(), description="Curated inferlet E2E tests")
