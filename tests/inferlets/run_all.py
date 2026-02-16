"""Run all inferlet E2E tests.

Usage::

    uv run python tests/inferlets/run_all.py --dummy
    uv run python tests/inferlets/run_all.py --model Qwen/Qwen3-0.6B --device cuda:0
"""
from conftest import run_tests

from test_helloworld import test_helloworld
from test_text_completion import test_text_completion
from test_parallel_generation import test_parallel_generation
from test_skeleton_of_thought import test_skeleton_of_thought
from test_tree_of_thought import test_tree_of_thought
from test_graph_of_thought import test_graph_of_thought
from test_recursion_of_thought import test_recursion_of_thought
from test_agent_react import test_agent_react
from test_agent_codeact import test_agent_codeact
from test_image_fetch import test_image_fetch
from test_prefix_tree import test_prefix_tree
from test_agent_swarm import test_agent_swarm
from test_output_validation import test_output_validation
from test_constrained_decoding import test_constrained_decoding
from test_watermarking import test_watermarking
from test_windowed_attention import test_windowed_attention
from test_attention_sink import test_attention_sink
from test_cacheback_decoding import test_cacheback_decoding
from test_jacobi_decoding import test_jacobi_decoding
from test_openresponses import test_openresponses


ALL_TESTS = [
    # Tier 1: Basic runtime
    test_helloworld,
    test_text_completion,
    # Tier 2: High-level SDK
    test_parallel_generation,
    test_skeleton_of_thought,
    test_tree_of_thought,
    test_graph_of_thought,
    test_recursion_of_thought,
    test_agent_react,
    test_agent_codeact,
    test_image_fetch,
    # Tier 3: Low-level APIs
    test_prefix_tree,
    test_agent_swarm,
    test_output_validation,
    test_constrained_decoding,
    test_watermarking,
    test_windowed_attention,
    test_attention_sink,
    test_cacheback_decoding,
    # Run last: jacobi may trigger CUDA errors that poison subsequent tests
    test_jacobi_decoding,
    # HTTP daemon test (runs after all process-based tests)
    test_openresponses,
]

if __name__ == "__main__":
    run_tests(ALL_TESTS, description="All Inferlet E2E Tests")
