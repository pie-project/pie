"""The layer under `engrain.Engine`, and the failures only it can show.

These were the public API's tests until the surface was rebuilt around slots.
The behaviours they cover are still the engine's - a grammar from the wrong
tokenizer, a token past the end of the vocabulary, a pool that outgrew a live
batch, the allowed set agreeing with the mask bit for bit - and they are worth
more than the surface that happened to be in front of them, so they were kept
and pointed one layer down.

`tests/test_api.py` covers what a caller types. This covers what the engine
must refuse.
"""

from __future__ import annotations

import json
import os
import unittest

import engrain
from support import Engine as _Engine

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


def _requirements():
    if not HAVE_CUDA:
        raise unittest.SkipTest("no CUDA device")


# A differential run compares the two backends on the host, so nothing in it
# can be recorded into a graph. Tests whose subject *is* the recording have
# nothing to say in that mode.
NEEDS_CAPTURE = unittest.skipIf(
    os.environ.get("ENGRAIN_BACKEND", "").strip().lower() == "differential",
    "differential mode compares on the host and cannot capture",
)


class PublicApi(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = _Engine(VOCABULARY)

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
        engine = _Engine([bytes([b]) for b in range(256)])
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

    @NEEDS_CAPTURE
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
            self.engine.compile_json_schema(
                json.dumps(
                    {"type": "object", "patternProperties": {"^a{0,70000}$": {}}}
                )
            )
        # The stage is the answer to what a caller should do next: a budget can
        # be raised and retried, a lowering failure cannot.
        self.assertIsInstance(refusal.exception, engrain.CompileError)
        self.assertIn(
            refusal.exception.stage,
            {"lowering", "lexer", "productions", "conflict", "emit"},
        )

    def test_a_pattern_whose_automaton_explodes_is_refused_not_run(self):
        # A regex is the one grammar a request hands over directly, and its
        # DFA is exponential in the worst case. This path used to build it
        # unbounded, which spends the server's memory instead of refusing.
        with self.assertRaises(engrain.CompileError) as refusal:
            self.engine.compile_regex("(a|b)*a" + "(a|b)" * 24)
        self.assertEqual(refusal.exception.stage, "lexer")

    def test_the_budget_can_be_raised_deliberately(self):
        self.assertIsNotNone(
            self.engine.compile_regex("(a|b)*a" + "(a|b)" * 8, lexer_states=100_000)
        )


class Approximations(unittest.TestCase):
    """A widened mask is only safe if the caller is told how it widened."""

    @classmethod
    def setUpClass(cls):
        cls.engine = _Engine([bytes([b]) for b in range(256)])

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
        self.assertEqual(self.engine.compile_json_schema(schema).relaxations, [])

    def test_an_open_object_declares_that_a_declared_type_may_not_hold(self):
        """The default keeps the shared string, and says so.

        Excluding the declared names is exact and regular and `exact=True` does
        it. It is not the default because it is not affordable: one extra
        string-shaped terminal does not add one group, it refines the whole
        lexer state space, and over 40 corpus schemas that took the median
        group count from 18,944 to 85,366 and the median compile from 60.1 ms
        to 469.2.
        """
        schema = json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
        grammar = self.engine.compile_json_schema(schema)
        self.assertEqual(len(grammar.relaxations), 1)
        self.assertTrue(self.accepts(grammar, '{"a":1}'))

    def test_exact_excludes_each_object_s_own_names(self):
        """A declared name has one reading under the object that declares it.

        And a name another object declares is still an additional property
        here, which is what keeps the exclusion from narrowing - the one thing
        a lowering may not do.
        """
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "p": {"type": "object", "properties": {"a": {"type": "string"}}},
                    "q": {"type": "object", "properties": {"b": {"type": "integer"}}},
                },
            }
        )
        grammar = self.engine.compile_json_schema(schema, exact=True)
        self.assertEqual(grammar.relaxations, [])
        self.assertTrue(self.accepts(grammar, '{"p":{"a":"x"}}'))
        self.assertFalse(self.accepts(grammar, '{"p":{"a":1}}'))
        # `a` belongs to `p`, so under `q` it is an additional property and its
        # declared type does not apply to it.
        self.assertTrue(self.accepts(grammar, '{"q":{"a":"x"}}'))
        self.assertTrue(self.accepts(grammar, '{"p":{"b":1}}'))

    def test_the_caller_may_bound_what_json_leaves_unbounded(self):
        """Three constructs a model will run to the token limit inside.

        JSON permits a number of any length, a string of any length and
        whitespace of any length, so the grammar does too - and a model handed
        a mask that still admits a digit, a character or a newline emits one.
        Of 32 requests that ran to the token limit on a corpus of real schemas,
        19 were inside a string and 13 between tokens.

        Narrowing is the one direction the compiler will not take on its own,
        so all three are off until a caller asks.
        """
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        )
        loose = self.engine.compile_json_schema(schema)
        self.assertTrue(self.accepts(loose, '{"a":"abcdefgh"}'))
        self.assertTrue(self.accepts(loose, '{    "a":"x"}'))

        bounded = self.engine.compile_json_schema(
            schema, max_string=6, max_whitespace=2
        )
        self.assertTrue(self.accepts(bounded, '{"a":"abc"}'))
        self.assertFalse(self.accepts(bounded, '{"a":"abcdefgh"}'))
        self.assertTrue(self.accepts(bounded, '{  "a":"x"}'))
        self.assertFalse(self.accepts(bounded, '{    "a":"x"}'))

    def test_a_schema_that_bounds_a_string_itself_is_taken_at_its_word(self):
        """`maxLength` wins over the caller's default, in both directions."""
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"a": {"type": "string", "maxLength": 3}},
                "required": ["a"],
            }
        )
        grammar = self.engine.compile_json_schema(schema, max_string=64)
        self.assertTrue(self.accepts(grammar, '{"a":"abc"}'))
        self.assertFalse(self.accepts(grammar, '{"a":"abcd"}'))

    def test_exact_enforces_it_and_then_declares_nothing(self):
        schema = json.dumps({"type": "object", "properties": {"a": {"type": "string"}}})
        grammar = self.engine.compile_json_schema(schema, exact=True)
        self.assertEqual(grammar.relaxations, [])
        self.assertFalse(self.accepts(grammar, '{"a":1}'))
        self.assertTrue(self.accepts(grammar, '{"a":"x"}'))
        # Widening is what a level may do; narrowing is not. A name that only
        # shares a prefix with a declared one is still an additional property.
        self.assertTrue(self.accepts(grammar, '{"ab":1}'))

    def test_keywords_no_level_enforces_are_declared(self):
        schema = json.dumps(
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
        )
        declared = self.engine.compile_json_schema(schema).relaxations
        self.assertTrue(any("uniqueItems" in text for text in declared))

    def test_a_schema_that_never_mentions_them_is_not_warned_about_them(self):
        schema = json.dumps({"type": "array", "items": {"type": "integer"}})
        self.assertEqual(self.engine.compile_json_schema(schema).relaxations, [])


class FusedStepThroughTheLibrary(unittest.TestCase):
    """`Batch.step` is the decode path, so the library has to expose it."""

    @NEEDS_CAPTURE
    def test_step_agrees_with_advance_then_fill(self):
        import torch

        engine = _Engine([bytes([b]) for b in range(256)])
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


class WhatARehearsalMustNotLeaveBehind(unittest.TestCase):
    """`capture` and `warmup` run the advance. A caller must not see it.

    Recording a graph rehearses the advance on a synthetic token, and the
    grammar refuses that token, so before this was fixed every sequence
    reported itself terminated the moment the documented flow ran `capture()`
    - and a serving engine polling `problems` to retire sequences would have
    retired the whole batch on its first step.
    """

    def setUp(self):
        _requirements()
        self.engine = _Engine(VOCABULARY)
        self.grammar = self.engine.compile_json_schema(SCHEMA)

    @NEEDS_CAPTURE
    def test_capture_leaves_no_sequence_terminated(self):
        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammar] * 4)
        batch.capture()
        terminated, overflow = batch.problems()
        self.assertEqual(terminated.tolist(), [0, 0, 0, 0])
        self.assertEqual(overflow.tolist(), [0, 0, 0, 0])

    def test_warmup_leaves_no_sequence_terminated(self):
        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammar] * 4)
        batch.raw.warmup()
        self.assertEqual(batch.problems()[0].tolist(), [0, 0, 0, 0])

    def test_set_grammars_clears_what_the_last_parse_reported(self):
        batch = self.engine.batch(size=2)
        batch.set_grammars([self.grammar] * 2)
        batch.raw.terminated.fill_(1)
        batch.raw.overflow.fill_(1)
        batch.set_grammars([self.grammar] * 2)
        self.assertEqual(batch.problems()[0].tolist(), [0, 0])
        self.assertEqual(batch.problems()[1].tolist(), [0, 0])


class TheEngineDoesNotConfuseTwoGrammars(unittest.TestCase):
    def setUp(self):
        _requirements()
        self.engine = _Engine(VOCABULARY)

    def test_a_dropped_grammar_does_not_lend_its_slot_to_the_next(self):
        # The admission cache is keyed on `id`, and CPython reuses the address
        # of a dropped object - so unless a reference is held, a caller who let
        # one schema go out of scope could compile another, land on the same
        # address, and be handed the first one's tables.
        import gc

        first = self.engine.compile_json_schema(json.dumps({"type": "string"}))
        first_id, address = self.engine.admit(first), id(first)
        del first
        gc.collect()
        for _ in range(200):
            other = self.engine.compile_json_schema(json.dumps({"type": "integer"}))
            if self.engine.admit(other) == first_id:
                self.fail("a different grammar was given an occupied slot")
            if id(other) == address:
                return


class TheMatcherRefusesABufferItCannotWriteTo(unittest.TestCase):
    """`fill_bitmask` writes through a pointer the caller supplies."""

    def setUp(self):
        _requirements()
        self.engine = _Engine(VOCABULARY)
        self.grammar = self.engine.compile_json_schema(SCHEMA)
        self.matcher = self.grammar.matcher(32)
        self.words = self.grammar.bitset_words

    def test_a_device_buffer_is_refused(self):
        # Its `data_ptr` is a device address and this write is on the host.
        with self.assertRaises(ValueError):
            self.matcher.fill_bitmask(
                torch.zeros(self.words, dtype=torch.int32, device="cuda")
            )

    def test_a_narrower_element_is_refused(self):
        # `numel` passes while the buffer is a quarter of the bytes needed.
        with self.assertRaises(ValueError):
            self.matcher.fill_bitmask(torch.zeros(self.words, dtype=torch.uint8))

    def test_a_strided_buffer_is_refused(self):
        # At least two elements, or the stride has nothing to skip over and a
        # single element is contiguous by definition.
        stride = torch.zeros(max(2, self.words) * 2, dtype=torch.int32)[::2]
        self.assertFalse(stride.is_contiguous())
        with self.assertRaises(ValueError):
            self.matcher.fill_bitmask(stride)

    def test_the_buffer_it_documents_is_accepted(self):
        self.matcher.fill_bitmask(torch.zeros(self.words, dtype=torch.int32))


class ABatchThePoolOutgrew(unittest.TestCase):
    """Admitting a grammar can raise a ceiling a live batch was sized from.

    Buffers do not overflow, they read past themselves - the kernels index
    what they are given - so this has to be refused rather than flagged.
    """

    def setUp(self):
        _requirements()
        self.engine = _Engine(VOCABULARY)

    def _wide(self):
        return self.engine.compile_json_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {
                        name: {"type": "string"} for name in "abcdefghijklmnop"
                    },
                    "required": list("abcdefgh"),
                }
            )
        )

    @NEEDS_CAPTURE
    def test_a_wider_grammar_makes_a_live_batch_refuse(self):
        narrow = self.engine.compile_json_schema(json.dumps({"type": "boolean"}))
        batch = self.engine.batch(size=4)
        batch.set_grammars([narrow] * 4)
        batch.capture()
        self.assertFalse(batch.outgrown)
        wide = self._wide()
        self.assertTrue(batch.outgrown)
        with self.assertRaises(RuntimeError):
            batch.set_grammars([wide] * 4)
        with self.assertRaises(RuntimeError):
            batch.fill_mask()

    def test_and_a_new_batch_serves_it(self):
        narrow = self.engine.compile_json_schema(json.dumps({"type": "boolean"}))
        stale = self.engine.batch(size=4)
        stale.set_grammars([narrow] * 4)
        wide = self._wide()
        fresh = self.engine.batch(size=4)
        fresh.set_grammars([wide] * 4)
        self.assertFalse(fresh.outgrown)
        mask = fresh.fill_mask()
        reference = torch.zeros(wide.bitset_words, dtype=torch.int32)
        wide.matcher(32).fill_bitmask(reference)
        self.assertTrue(bool((mask[0].cpu()[: wide.bitset_words] == reference).all()))


class WrongMasksThatWouldNotRaise(unittest.TestCase):
    """Three ways to get a mask that is simply about something else.

    None of them is an overflow - the parse is perfectly happy - so none is
    reachable by the flag that reports narrowing. They have to be refused.
    """

    def setUp(self):
        _requirements()
        self.engine = _Engine(VOCABULARY)

    def test_a_grammar_from_another_tokenizer_is_refused(self):
        # Same size is not the same tokenizer: a grammar's groups are token
        # ids, so a permuted vocabulary gives a mask wrong token by token.
        other = _Engine(list(reversed(VOCABULARY)))
        foreign = other.compile_json_schema(SCHEMA)
        ours = self.engine.compile_json_schema(SCHEMA)
        self.assertNotEqual(foreign.vocabulary_digest, ours.vocabulary_digest)
        with self.assertRaises(ValueError):
            self.engine.admit(foreign)

    def test_an_unassigned_batch_is_refused_once_the_pool_is_mixed(self):
        first = self.engine.compile_json_schema(json.dumps({"type": "string"}))
        # One grammar: slot 0 is right for every sequence, so this is allowed.
        single = self.engine.batch(size=2)
        self.assertEqual(single.fill_mask().shape[0], 2)
        self.engine.compile_json_schema(json.dumps({"type": "integer"}))
        mixed = self.engine.batch(size=2)
        with self.assertRaises(RuntimeError):
            mixed.fill_mask()
        mixed.set_grammars([first, first])
        self.assertEqual(mixed.fill_mask().shape[0], 2)

    def test_a_released_grammar_cannot_be_assigned(self):
        keep = self.engine.compile_json_schema(json.dumps({"type": "string"}))
        gone = self.engine.compile_json_schema(json.dumps({"type": "integer"}))
        batch = self.engine.batch(size=2)
        identifier = self.engine.admit(gone)
        self.engine.pool.release(identifier)
        # The slot is still inside `count`; it holds nothing.
        with self.assertRaises(ValueError):
            batch.set_grammars([identifier, identifier])
        batch.set_grammars([keep, keep])

    def test_a_negative_id_is_refused(self):
        self.engine.compile_json_schema(SCHEMA)
        batch = self.engine.batch(size=2)
        with self.assertRaises(ValueError):
            batch.raw.set_grammars([-1, 0])


if __name__ == "__main__":
    unittest.main()


class ATokenOutsideTheVocabulary(unittest.TestCase):
    """A sampled id the vocabulary does not have must refuse, not corrupt.

    The tokens are never read on the host - that is the whole design - so a
    caller that passes an id past the end of the vocabulary cannot be caught
    there. A dense group set is indexed *by* the id, so one too large read
    past its payload and took the CUDA context with it: not a wrong mask, a
    dead process, and every other sequence in the batch with it.
    """

    def _batch(self):
        _requirements()
        engine = _Engine(VOCABULARY)
        grammar = engine.compile_json_schema(SCHEMA)
        batch = engine.batch(size=4)
        batch.set_grammars([grammar] * 4)
        return engine, grammar, batch

    def test_a_huge_id_refuses_and_leaves_the_others_alone(self):
        _, grammar, batch = self._batch()
        good = grammar.matcher(0).allowed_tokens()[0]
        batch.advance(
            torch.tensor([good, 10**9, -5, good], dtype=torch.int32, device="cuda")
        )
        torch.cuda.synchronize()
        terminated, overflow = batch.problems()
        self.assertEqual(terminated.tolist(), [0, 1, 1, 0])
        # Not an overflow: nothing hit a ceiling, the token simply is not in
        # the grammar. Reporting it as one would send a caller looking for a
        # buffer to enlarge.
        self.assertEqual(overflow.tolist(), [0, 0, 0, 0])

    def test_the_batch_still_fills_afterwards(self):
        _, grammar, batch = self._batch()
        good = grammar.matcher(0).allowed_tokens()[0]
        batch.advance(
            torch.tensor(
                [good, 2**31 - 1, good, good], dtype=torch.int32, device="cuda"
            )
        )
        mask = batch.fill_mask()
        torch.cuda.synchronize()
        rows = mask.to(torch.int32).view(4, -1).ne(0).sum(1).tolist()
        self.assertGreater(rows[0], 0)
        self.assertEqual(rows[0], rows[2])


class TheAllowedSetIsTheMask(unittest.TestCase):
    """`allowed` must name exactly the tokens the mask admits, in order.

    The set is what a fused sampler draws from, so a bit the compaction drops
    is a token the model can never choose and a bit it invents is one the
    grammar forbids. Checked against the mask itself rather than against a
    second implementation of the same idea.
    """

    def _filled(self, size=6):
        _requirements()
        engine = _Engine(VOCABULARY)
        grammar = engine.compile_json_schema(SCHEMA)
        batch = engine.batch(size=size)
        batch.set_grammars([grammar] * size)
        matchers = []
        for row in range(size):
            matcher = grammar.matcher(0)
            for _ in range(row):
                allowed = matcher.allowed_tokens()
                if not allowed or not matcher.accept_token(allowed[0]):
                    break
            matchers.append(matcher)
        batch.set_matchers(matchers)
        return engine, batch, batch.fill_mask()

    def test_it_names_exactly_the_bits_the_mask_sets(self):
        engine, batch, mask = self._filled()
        ids, counts = batch.allowed(4096)
        torch.cuda.synchronize()
        words = mask.view(batch.size, -1).cpu()
        for row in range(batch.size):
            wanted = [
                token
                for token in range(len(VOCABULARY))
                if (int(words[row][token >> 5]) >> (token & 31)) & 1
            ]
            self.assertEqual(ids[row, : int(counts[row])].tolist(), wanted)

    def test_no_bit_above_the_vocabulary_is_named(self):
        # A row is a whole number of words, so its last one runs past the
        # vocabulary. Those bits are nobody's token.
        engine, batch, _ = self._filled()
        ids, counts = batch.allowed(4096)
        torch.cuda.synchronize()
        for row in range(batch.size):
            held = ids[row, : int(counts[row])]
            if held.numel():
                self.assertLess(int(held.max()), len(VOCABULARY))

    def test_the_count_is_true_even_when_the_buffer_is_not(self):
        # A caller that cannot tell a truncated list from a complete one
        # samples from a prefix and never knows.
        engine, batch, mask = self._filled()
        _, full = batch.allowed(4096)
        torch.cuda.synchronize()
        expected = [int(x) for x in full]
        ids, counts = batch.allowed(2)
        torch.cuda.synchronize()
        self.assertEqual([int(x) for x in counts], expected)
        self.assertEqual(ids.shape[1], 2)

    def test_a_capacity_of_zero_is_refused(self):
        _, batch, _ = self._filled(size=2)
        with self.assertRaises(ValueError):
            batch.allowed(0)


class TheShortlistIsWhicheverListIsShorter(unittest.TestCase):
    """`shortlist` must name the forbidden tokens when those are fewer.

    A row inside a JSON string body admits nearly the whole vocabulary, so its
    allowed list is useless and its forbidden list is small. Getting the two
    the wrong way round would hand a sampler the complement of what the
    grammar permits, which is the worst possible failure and would still look
    like a plausible list.
    """

    def _batch(self, admits_most: bool):
        _requirements()
        wide = [bytes([byte]) for byte in range(256)]
        engine = _Engine(wide)
        grammar = engine.compile_json_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"s": {"type": "string"}},
                    "required": ["s"],
                }
            )
        )
        batch = engine.batch(size=3)
        batch.set_grammars([grammar] * 3)
        matcher = grammar.matcher(0)
        if admits_most:
            # Into the string body, where almost every byte is a legal
            # continuation.
            for piece in b'{"s": "':
                matcher.accept_token(piece)
        batch.set_matchers([matcher] * 3)
        return engine, batch, batch.fill_mask(), len(wide)

    def _check(self, admits_most):
        _, batch, mask, size = self._batch(admits_most)
        ids, counts, kind = batch.raw.compact(4096, both=True)
        torch.cuda.synchronize()
        words = mask.view(batch.size, -1).cpu()
        for row in range(batch.size):
            bits = [
                token
                for token in range(size)
                if (int(words[row][token >> 5]) >> (token & 31)) & 1
            ]
            wanted = (
                bits
                if int(kind[row]) == 0
                else [token for token in range(size) if token not in set(bits)]
            )
            self.assertEqual(ids[row, : int(counts[row])].tolist(), wanted)
            # And it really is the shorter one.
            self.assertLessEqual(int(counts[row]) * 2, size + 1)
        return int(kind[0]), int(counts[0])

    def test_a_structural_position_lists_what_it_admits(self):
        kind, _ = self._check(admits_most=False)
        self.assertEqual(kind, 0)

    def test_a_string_body_lists_what_it_forbids(self):
        kind, count = self._check(admits_most=True)
        self.assertEqual(kind, 1)
        self.assertGreater(count, 0)
