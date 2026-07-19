"""Run all inferlet E2E tests.

Usage::

    uv run python tests/inferlets/run_all.py --dummy
    uv run python tests/inferlets/run_all.py --model Qwen/Qwen3-0.6B --device cuda:0
"""
from conftest import run_tests
from test_curated import tests

if __name__ == "__main__":
    run_tests(tests(), description="Curated Inferlet E2E Tests")
