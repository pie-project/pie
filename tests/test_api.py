"""The API a caller types, exercised the way the documentation says to use it.

An API that is only ever exercised by its own internals is not an API, so
nothing here reaches past `engrain.Engine`, `engrain.Slots` and the objects
they hand back. What the engine must *refuse* is in `test_internals.py`, one
layer down, where the failures are visible.
"""

from __future__ import annotations

import json
import os
import unittest

import engrain

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAVE_CUDA = False

# Small enough to compile quickly, wide enough that a mask is not trivial.
VOCABULARY = [
    b"{",
    b"}",
    b"[",
    b"]",
    b":",
    b",",
    b'"',
    b" ",
    b'{"',
    b'":',
    b'","',
    b'":"',
    b'"}',
    b"true",
    b"false",
    b"null",
    b"0",
    b"1",
    b"2",
    b"12",
    b"-",
    b".",
    b"a",
    b"b",
    b"ab",
    b"name",
    b"id",
    b'"name"',
    b'"id"',
]

SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
        "required": ["name", "id"],
    }
)

NEEDS_CAPTURE = unittest.skipIf(
    os.environ.get("ENGRAIN_BACKEND", "").strip().lower() == "differential",
    "differential mode compares on the host and cannot capture",
)


def _requirements():
    if not HAVE_CUDA:
        raise unittest.SkipTest("no CUDA device")


def _bits(mask, row: int, vocabulary: int) -> set[int]:
    """The token ids a mask row holds, read on the host."""
    words = mask[row].tolist()
    return {
        token for token in range(vocabulary) if (words[token >> 5] >> (token & 31)) & 1
    }


class TheShortestUsefulProgram(unittest.TestCase):
    """What the module docstring promises, run."""

    def setUp(self):
        _requirements()
        self.engine = engrain.Engine(VOCABULARY)

    def test_compile_admit_sample_commit(self):
        grammar = self.engine.compile(json_schema=SCHEMA)
        slots = self.engine.slots(4)
        for slot in range(4):
            slots.admit(slot, grammar)

        logits = torch.zeros(4, len(VOCABULARY), device="cuda")
        tokens = slots.sample(logits)
        verdict = slots.commit(tokens)

        self.assertTrue(verdict.ok)
        self.assertEqual(verdict.failures(), [])
        # A JSON object can only start one way in this vocabulary.
        self.assertTrue(all(VOCABULARY[t].startswith(b"{") for t in tokens.tolist()))

    def test_no_capture_call_is_needed(self):
        """The recording is the library's business, not the caller's."""
        grammar = self.engine.compile(json_schema=SCHEMA)
        slots = self.engine.slots(2)
        slots.admit(0, grammar)
        slots.admit(1, grammar)
        first = slots.mask().clone()
        tokens = slots.sample(torch.zeros(2, len(VOCABULARY), device="cuda"))
        slots.commit(tokens)
        slots.admit(0, grammar)
        slots.admit(1, grammar)
        # Back at the start, and the replayed graph gives the same answer as
        # the eager first call did.
        self.assertTrue(torch.equal(first, slots.mask()))


class OneVerbForEveryFrontEnd(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])

    def test_a_regex_is_a_grammar_too(self):
        grammar = self.engine.compile(regex=r"[0-9]{3}")
        slots = self.engine.slots(1)
        slots.admit(0, grammar)
        self.assertEqual(
            _bits(slots.mask(), 0, 256), set(range(ord("0"), ord("9") + 1))
        )

    def test_exactly_one_source_is_required(self):
        with self.assertRaises(TypeError):
            self.engine.compile()
        with self.assertRaises(TypeError):
            self.engine.compile(json_schema=SCHEMA, regex="a")

    def test_ebnf_needs_a_root(self):
        with self.assertRaises(TypeError):
            self.engine.compile(ebnf="start ::= 'a'")

    def test_a_refused_schema_says_which_stage_refused_it(self):
        with self.assertRaises(engrain.CompileError) as caught:
            self.engine.compile(json_schema='{"type": "object", "$ref": "#/nope"}')
        self.assertIn(
            caught.exception.stage,
            {"lowering", "lexer", "productions", "conflict", "emit"},
        )


class RelaxationsAreDeclared(unittest.TestCase):
    """The mask may admit more than the source. It has to say when."""

    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])

    def test_a_closed_object_declares_nothing(self):
        grammar = self.engine.compile(
            json_schema=json.dumps(
                {
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["a"],
                    "additionalProperties": False,
                }
            )
        )
        self.assertEqual(grammar.relaxations, [])

    def test_an_open_object_declares_the_type_it_stops_enforcing(self):
        grammar = self.engine.compile(
            json_schema=json.dumps(
                {"type": "object", "properties": {"a": {"type": "integer"}}}
            )
        )
        [note] = grammar.relaxations
        self.assertEqual(note["keyword"], "additionalProperties")
        self.assertEqual(note["at"], "#")
        self.assertIn("additionalProperties", note["remedy"])

    def test_exact_enforces_it_and_then_declares_nothing(self):
        source = json.dumps(
            {"type": "object", "properties": {"a": {"type": "integer"}}}
        )
        self.assertEqual(
            self.engine.compile(json_schema=source, exact=True).relaxations, []
        )


class SlotsAreWhereRequestsLive(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])
        self.digits = self.engine.compile(regex=r"[0-9]{3}")
        self.letters = self.engine.compile(regex=r"[a-c]{3}")

    def test_live_and_free_account_for_every_slot(self):
        slots = self.engine.slots(4)
        self.assertEqual(slots.free, frozenset(range(4)))
        self.assertFalse(slots)
        slots.admit(1, self.digits)
        self.assertEqual(slots.live, frozenset({1}))
        self.assertEqual(slots.free, frozenset({0, 2, 3}))
        self.assertTrue(slots)
        slots.release(1)
        self.assertEqual(slots.live, frozenset())

    def test_one_slot_can_be_replaced_without_disturbing_the_others(self):
        """This is continuous batching, and it is what the old API could not say."""
        slots = self.engine.slots(3)
        for slot in range(3):
            slots.admit(slot, self.digits)
        tokens = torch.tensor([ord("1")] * 3, dtype=torch.int32, device="cuda")
        slots.commit(tokens)
        before = slots.mask().clone()

        # One request finishes and another arrives, under a different grammar.
        slots.release(1)
        slots.admit(1, self.letters)
        after = slots.mask()

        self.assertEqual(_bits(after, 1, 256), set(range(ord("a"), ord("c") + 1)))
        for untouched in (0, 2):
            self.assertEqual(
                _bits(before, untouched, 256), _bits(after, untouched, 256)
            )

    def test_a_slot_outside_the_range_is_refused(self):
        slots = self.engine.slots(2)
        with self.assertRaises(IndexError):
            slots.admit(2, self.digits)

    def test_stepping_before_admitting_anything_is_refused(self):
        slots = self.engine.slots(2)
        with self.assertRaises(RuntimeError):
            slots.mask()

    def test_slots_before_a_grammar_is_refused(self):
        engine = engrain.Engine([bytes([b]) for b in range(256)])
        with self.assertRaises(RuntimeError):
            engine.slots(2)


class ApplyIsTheMask(unittest.TestCase):
    """The last mile the surface used to leave to the caller."""

    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])
        self.grammar = self.engine.compile(regex=r"[0-9]{3}")
        self.slots = self.engine.slots(2)
        self.slots.admit(0, self.grammar)
        self.slots.admit(1, self.grammar)

    def _survivors(self, logits, row):
        return {
            token
            for token, value in enumerate(logits[row].tolist())
            if value != float("-inf")
        }

    def test_it_leaves_exactly_the_tokens_the_mask_holds(self):
        wanted = _bits(self.slots.mask(), 0, 256)
        logits = torch.zeros(2, 256, device="cuda")
        self.slots.apply(logits)
        self.assertEqual(self._survivors(logits, 0), wanted)

    def test_float16_agrees_with_float32(self):
        wanted = _bits(self.slots.mask(), 0, 256)
        half = torch.zeros(2, 256, device="cuda", dtype=torch.float16)
        self.slots.apply(half)
        self.assertEqual(self._survivors(half, 0), wanted)

    def test_a_padded_row_keeps_its_tail(self):
        """A model's vocabulary is padded and a tokenizer's is not.

        Assuming they matched left the tail of every row as whatever it held,
        which is a bug this integration has already met once.
        """
        logits = torch.zeros(2, 300, device="cuda")
        self.slots.apply(logits)
        self.assertTrue(torch.isfinite(logits[:, 256:]).all())

    def test_a_wrong_shape_is_refused(self):
        with self.assertRaises(ValueError):
            self.slots.apply(torch.zeros(5, 256, device="cuda"))


class SampleDrawsOnlyWhatTheGrammarAllows(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])
        self.grammar = self.engine.compile(regex=r"[0-9]{3}")
        self.slots = self.engine.slots(8)
        for slot in range(8):
            self.slots.admit(slot, self.grammar)

    def test_every_draw_is_admissible(self):
        digits = set(range(ord("0"), ord("9") + 1))
        for temperature, top_p, top_k in ((1.0, 1.0, 0), (0.7, 0.9, 0), (1.0, 1.0, 5)):
            logits = torch.randn(8, 256, device="cuda")
            drawn = self.slots.sample(
                logits, temperature=temperature, top_p=top_p, top_k=top_k
            )
            self.assertTrue(set(drawn.tolist()) <= digits, (temperature, top_p, top_k))
            self.assertEqual(drawn.dtype, torch.int32)

    def test_zero_temperature_is_the_best_allowed_token(self):
        logits = torch.full((8, 256), -1.0, device="cuda")
        logits[:, ord("7")] = 5.0
        logits[:, ord("Z")] = 9.0  # higher, and forbidden
        drawn = self.slots.sample(logits, temperature=0.0)
        self.assertEqual(set(drawn.tolist()), {ord("7")})


class CommitReportsWhatHappened(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])
        self.grammar = self.engine.compile(regex=r"[0-9]{3}")

    def test_a_refused_token_is_named(self):
        slots = self.engine.slots(2)
        slots.admit(0, self.grammar)
        slots.admit(1, self.grammar)
        # `a` is not a digit; the parser must refuse it and say so.
        verdict = slots.commit(
            torch.tensor([ord("0"), ord("a")], dtype=torch.int32, device="cuda")
        )
        self.assertFalse(verdict.ok)
        self.assertEqual(verdict.failures(), [1])

    def test_the_next_mask_is_ready_without_asking(self):
        slots = self.engine.slots(1)
        slots.admit(0, self.grammar)
        slots.commit(torch.tensor([ord("0")], dtype=torch.int32, device="cuda"))
        self.assertEqual(
            _bits(slots.mask(), 0, 256), set(range(ord("0"), ord("9") + 1))
        )
        slots.commit(torch.tensor([ord("1")], dtype=torch.int32, device="cuda"))
        slots.commit(torch.tensor([ord("2")], dtype=torch.int32, device="cuda"))
        # Three digits and the pattern is satisfied: nothing may follow.
        self.assertEqual(_bits(slots.mask(), 0, 256), set())


class TheDeviceAgreesWithTheReferenceParser(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine(VOCABULARY)
        self.grammar = self.engine.compile(json_schema=SCHEMA)

    def test_the_mask_is_the_matcher_s(self):
        slots = self.engine.slots(1)
        slots.admit(0, self.grammar)
        matcher = self.grammar.matcher(0)
        for _ in range(6):
            wanted = set(matcher.allowed_tokens())
            self.assertEqual(_bits(slots.mask(), 0, len(VOCABULARY)), wanted)
            if not wanted:
                break
            token = min(wanted)
            matcher.accept_token(token)
            slots.commit(torch.tensor([token], dtype=torch.int32, device="cuda"))

    def test_parses_agree_with_the_matcher(self):
        slots = self.engine.slots(1)
        slots.admit(0, self.grammar)
        matcher = self.grammar.matcher(0)
        mine = {(state, tuple(stack)) for state, stack in slots.parses(0)}
        theirs = {(state, tuple(stack)) for state, stack in matcher.configurations()}
        self.assertEqual(mine, theirs)

    def test_resume_takes_a_matcher_s_state(self):
        matcher = self.grammar.matcher(0)
        for token, piece in enumerate(VOCABULARY):
            if piece.startswith(b"{"):
                matcher.accept_token(token)
                break
        slots = self.engine.slots(1)
        slots.resume(0, matcher)
        self.assertEqual(
            _bits(slots.mask(), 0, len(VOCABULARY)), set(matcher.allowed_tokens())
        )


class LookaheadBuysRollback(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine([bytes([b]) for b in range(256)])
        self.grammar = self.engine.compile(regex=r"[0-9]{3}")

    def test_a_committed_token_can_be_undone(self):
        slots = self.engine.slots(1, lookahead=2)
        slots.admit(0, self.grammar)
        before = slots.mask().clone()
        slots.commit(torch.tensor([ord("5")], dtype=torch.int32, device="cuda"))
        slots.rollback(1)
        self.assertTrue(torch.equal(before, slots.mask()))


class WhatTheEngineCosts(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = engrain.Engine(VOCABULARY)

    def test_it_reports_its_own_size(self):
        self.assertEqual(self.engine.mask_words, 0)
        self.engine.compile(json_schema=SCHEMA)
        self.assertEqual(self.engine.mask_words, (len(VOCABULARY) + 31) // 32)
        self.assertGreater(self.engine.resident_bytes, 0)

    def test_compiling_the_same_grammar_twice_costs_one_slot(self):
        grammar = self.engine.compile(json_schema=SCHEMA)
        held = self.engine.resident_bytes
        self.engine._admit(grammar)
        self.assertEqual(held, self.engine.resident_bytes)


if __name__ == "__main__":
    unittest.main()
