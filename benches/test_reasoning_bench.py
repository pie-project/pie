"""Unit tests for the reasoning benchmark's dataset and answer evaluator."""
import json
import tempfile
import unittest
from pathlib import Path

from argparse import Namespace

from benches.reasoning_bench import (
    Problem,
    PATTERN_INFERLETS,
    load_problems,
    normalize_number,
    payload_for,
    reference_answer,
)


class ReasoningBenchTests(unittest.TestCase):
    def test_normalize_number(self):
        self.assertEqual(normalize_number("$1,234.00"), "1234")
        self.assertEqual(normalize_number("Final Answer: -2.5"), "-2.5")
        self.assertIsNone(normalize_number("no numeric answer"))

    def test_gsm8k_reference_answer(self):
        self.assertEqual(
            reference_answer("Reasoning with intermediate numbers.\n#### 1,250"),
            "1250",
        )

    def test_payload_disables_thinking_by_default(self):
        args = Namespace(
            num_candidates=4,
            beam_width=2,
            max_tokens=256,
            score_tokens=16,
            temperature=0.7,
            top_p=0.95,
            thinking=False,
        )
        payload = payload_for(Problem("p1", "How many?", "42"), "direct", args)
        self.assertIs(payload["thinking"], False)

    def test_payload_can_enable_thinking(self):
        args = Namespace(
            num_candidates=4,
            beam_width=2,
            max_tokens=256,
            score_tokens=16,
            temperature=0.7,
            top_p=0.95,
            thinking=True,
        )
        payload = payload_for(Problem("p1", "How many?", "42"), "direct", args)
        self.assertIs(payload["thinking"], True)

    def test_has_separate_inferlet_for_each_pattern(self):
        self.assertEqual(
            set(PATTERN_INFERLETS),
            {"direct", "best_of_n", "tree_of_thought", "graph_of_thought"},
        )

    def test_loads_official_gsm8k_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question": "How many?",
                        "answer": "Compute carefully.\n#### 42",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            problems = load_problems(path, None)
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0].answer, "42")


if __name__ == "__main__":
    unittest.main()
