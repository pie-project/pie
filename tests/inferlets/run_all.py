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
from test_draft_model_decoding import test_draft_model_decoding
from test_jacobi_decoding import test_jacobi_decoding
from test_best_of_n import test_best_of_n
from test_json_schema_validation import test_json_schema_validation
from test_knowledge_graph import test_knowledge_graph
from test_template_generation import test_template_generation
from test_js_example import test_js_example
from test_js_concurrency import test_js_concurrency
from test_js_constrained_decoding import test_js_constrained_decoding
from test_py_concurrency import test_py_concurrency
from test_python_example import test_python_example
from test_python_constrained_decoding import test_python_constrained_decoding
from test_openresponses import test_openresponses
from test_http_server import test_http_server


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
    test_best_of_n,
    test_knowledge_graph,
    test_agent_react,
    test_agent_codeact,
    test_image_fetch,
    # Tier 3: Low-level APIs
    test_prefix_tree,
    test_agent_swarm,
    test_output_validation,
    test_constrained_decoding,
    test_json_schema_validation,
    test_template_generation,
    test_watermarking,
    test_windowed_attention,
    test_attention_sink,
    test_draft_model_decoding,
    # Tier 4: JS/Python SDK
    test_js_example,
    test_js_concurrency,
    test_js_constrained_decoding,
    test_python_example,
    test_py_concurrency,
    test_python_constrained_decoding,
    # Run last: jacobi may trigger CUDA errors that poison subsequent tests
    test_jacobi_decoding,
    # HTTP daemon tests (run after all process-based tests)
    test_openresponses,
    test_http_server,
]

if __name__ == "__main__":
    run_tests(ALL_TESTS, description="All Inferlet E2E Tests")
