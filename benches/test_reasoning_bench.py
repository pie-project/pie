"""Unit tests for the reasoning benchmark's dataset and answer evaluator."""
import json
import tempfile
import unittest
from pathlib import Path

from benches.reasoning_bench import load_problems, normalize_number, reference_answer


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
