"""The public API, exercised the way the documentation says to use it.

Everything else in this suite tests the engine through `gpugrammar._engine`,
which is the research entry point. These tests go through `gpugrammar.Engine`
and `gpugrammar.Batch` instead, because that is what an artifact reviewer will
type and what a serving integration will call, and an API that is only ever
exercised by its own internals is not an API.
"""

from __future__ import annotations

import json
import unittest

import gpugrammar

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAVE_CUDA = False

# Small enough to compile quickly, wide enough that a mask is not trivial.
VOCABULARY = [
    b"{", b"}", b"[", b"]", b":", b",", b'"', b" ",
    b'{"', b'":', b'","', b'":"', b'"}', b"true", b"false", b"null",
    b"0", b"1", b"2", b"12", b"-", b".",
    b"a", b"b", b"ab", b"name", b"id", b'"name"', b'"id"',
]

SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["name", "id"],
    }
)


def _requirements():
    if not HAVE_CUDA:
        raise unittest.SkipTest("no CUDA device")


class PublicApi(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = gpugrammar.Engine(VOCABULARY)

    def test_the_shortest_useful_program(self):
        """The example in the module docstring has to actually run."""
        grammar = self.engine.compile_json_schema(SCHEMA)
        batch = self.engine.batch(4)
        batch.set_grammars([grammar] * 4)

        mask = batch.fill_mask()
        self.assertEqual(tuple(mask.shape), (4, self.engine.mask_words))
        self.assertTrue(mask.is_cuda)
        # A document must start with `{`, in one form or another, and nothing
        # else - so the mask is neither empty nor everything.
        allowed = int(mask[0].cpu().numpy().astype("uint32").sum() != 0)
        self.assertEqual(allowed, 1)

        opening = VOCABULARY.index(b"{")
        batch.advance(torch.full((4,), opening, dtype=torch.int32, device="cuda"))
        terminated, overflow = batch.problems()
        self.assertEqual(int(terminated.sum()), 0)
        self.assertEqual(int(overflow.sum()), 0)

    def test_the_mask_is_the_reference_parser_s(self):
        grammar = self.engine.compile_json_schema(SCHEMA)
        batch = self.engine.batch(1)
        batch.set_grammars([grammar])
        matcher = grammar.matcher(0)

        reference = torch.zeros(self.engine.mask_words, dtype=torch.int32)
        for piece in (b'{"', b"name", b'":"', b"ab", b'","'):
            token = VOCABULARY.index(piece)
            reference.zero_()
            matcher.fill_bitmask(reference)
            self.assertTrue(torch.equal(batch.fill_mask()[0].cpu(), reference))
            self.assertTrue(matcher.accept_token(token))
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))

    def test_a_regex_is_a_grammar_too(self):
        """Not only JSON - the same interface takes a pattern."""
        engine = gpugrammar.Engine([bytes([b]) for b in range(256)])
        grammar = engine.compile_regex(r"[0-9]{2}-[0-9]{2}")
        batch = engine.batch(1)
        batch.set_grammars([grammar])
        matcher = grammar.matcher(0)
        for byte in b"12-34":
            reference = torch.zeros(engine.mask_words, dtype=torch.int32)
            matcher.fill_bitmask(reference)
            self.assertTrue(torch.equal(batch.fill_mask()[0].cpu(), reference))
            self.assertTrue(matcher.accept_token(byte))
            batch.advance(torch.tensor([byte], dtype=torch.int32, device="cuda"))
        self.assertTrue(matcher.can_terminate())

    def test_one_batch_under_several_grammars(self):
        """The case a serving batch has by default."""
        grammars = [
            self.engine.compile_json_schema(SCHEMA),
            self.engine.compile_regex(r"[0-9]+"),
        ]
        batch = self.engine.batch(4)
        batch.set_grammars([grammars[i % 2] for i in range(4)])
        masks = batch.fill_mask().cpu()
        # Rows under different grammars must differ: one starts a document,
        # the other a number.
        self.assertFalse(torch.equal(masks[0], masks[1]))
        self.assertTrue(torch.equal(masks[0], masks[2]))

    def test_a_captured_batch_replays_the_same_mask(self):
        """The property the whole design is for."""
        grammar = self.engine.compile_json_schema(SCHEMA)
        batch = self.engine.batch(2)
        batch.set_grammars([grammar, grammar])
        before = batch.fill_mask().clone()
        batch.capture()
        self.assertTrue(torch.equal(batch.fill_mask(), before))

    def test_the_engine_reports_what_it_costs(self):
        self.assertEqual(self.engine.resident_bytes, self.engine.pool.resident_bytes())
        grammar = self.engine.compile_json_schema(SCHEMA)
        self.engine.admit(grammar)
        self.assertGreater(self.engine.resident_bytes, 0)
        # Admitting the same grammar twice must not admit it twice.
        first = self.engine.admit(grammar)
        self.assertEqual(first, self.engine.admit(grammar))

    def test_a_refused_schema_says_why(self):
        with self.assertRaises(ValueError) as refusal:
            self.engine.compile_json_schema(json.dumps({"type": "object",
                                                        "patternProperties": {
                                                            "^a{0,70000}$": {}}}))
        # The stage is the answer to what a caller should do next: a budget can
        # be raised and retried, a lowering failure cannot.
        self.assertIsInstance(refusal.exception, gpugrammar.CompileError)
        self.assertIn(refusal.exception.stage,
                      {"lowering", "lexer", "productions", "conflict", "emit"})

    def test_a_pattern_whose_automaton_explodes_is_refused_not_run(self):
        # A regex is the one grammar a request hands over directly, and its
        # DFA is exponential in the worst case. This path used to build it
        # unbounded, which spends the server's memory instead of refusing.
        with self.assertRaises(gpugrammar.CompileError) as refusal:
            self.engine.compile_regex("(a|b)*a" + "(a|b)" * 24)
        self.assertEqual(refusal.exception.stage, "lexer")

    def test_the_budget_can_be_raised_deliberately(self):
        self.assertIsNotNone(
            self.engine.compile_regex("(a|b)*a" + "(a|b)" * 8, lexer_states=100_000))




class Approximations(unittest.TestCase):
    """A widened mask is only safe if the caller is told how it widened."""

    @classmethod
    def setUpClass(cls):
        cls.engine = gpugrammar.Engine([bytes([b]) for b in range(256)])

    def accepts(self, grammar, text: str) -> bool:
        matcher = grammar.matcher(0)
        for byte in text.encode():
            if not matcher.accept_token(byte):
                return False
        return matcher.can_terminate()

    def test_a_schema_it_enforces_exactly_declares_nothing(self):
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "additionalProperties": False,
            }
        )
        self.assertEqual(self.engine.compile_json_schema(schema).approximations, [])

    def test_an_open_object_declares_that_a_declared_type_may_not_hold(self):
        schema = json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
        grammar = self.engine.compile_json_schema(schema)
        # The declaration has to match what the mask actually does.
        self.assertEqual(len(grammar.approximations), 1)
        self.assertTrue(self.accepts(grammar, '{"a":1}'))

    def test_exact_enforces_it_and_then_declares_nothing(self):
        schema = json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
        grammar = self.engine.compile_json_schema(schema, exact=True)
        self.assertEqual(grammar.approximations, [])
        self.assertFalse(self.accepts(grammar, '{"a":1}'))
        self.assertTrue(self.accepts(grammar, '{"a":"x"}'))
        # Widening is what a level may do; narrowing is not. A name that only
        # shares a prefix with a declared one is still an additional property.
        self.assertTrue(self.accepts(grammar, '{"ab":1}'))

    def test_keywords_no_level_enforces_are_declared(self):
        schema = json.dumps(
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
        )
        declared = self.engine.compile_json_schema(schema).approximations
        self.assertTrue(any("uniqueItems" in text for text in declared))

    def test_a_schema_that_never_mentions_them_is_not_warned_about_them(self):
        schema = json.dumps({"type": "array", "items": {"type": "integer"}})
        self.assertEqual(self.engine.compile_json_schema(schema).approximations, [])


class FusedStepThroughTheLibrary(unittest.TestCase):
    """`Batch.step` is the decode path, so the library has to expose it."""

    def test_step_agrees_with_advance_then_fill(self):
        import torch

        engine = gpugrammar.Engine([bytes([b]) for b in range(256)])
        schema = json.dumps(
            {"type": "object", "properties": {"a": {"type": "integer"}}}
        )
        grammar = engine.compile_json_schema(schema)

        apart = engine.batch(2)
        together = engine.batch(2)
        for batch in (apart, together):
            batch.set_grammars([grammar, grammar])
            batch.fill_mask()
            batch.capture()

        steps = 0
        for byte in b'{"a": 7}':
            self.assertTrue(
                torch.equal(apart.fill_mask(), together.fill_mask()),
                f"masks diverge at step {steps}",
            )
            token = torch.full((2,), byte, dtype=torch.int32, device="cuda")
            apart.advance(token)
            together.step(token)
            steps += 1
        self.assertEqual(steps, 8)


if __name__ == "__main__":
    unittest.main()
