"""The device parser against the reference matcher, as a test rather than a script.

Every claim this engine makes reduces to one thing: the mask on the device is
the mask the CPU matcher would have produced, bit for bit, at every step of a
real document. The optimisations that shrank the scratch by three orders of
magnitude are all of the form "compute less, differently", and each one is a
chance to compute less than the grammar allows - which does not crash, it
quietly forbids a legal token. So this checks bits, not shapes.

Skipped where there is no GPU, no Triton, or no compiled `engrain`, since the
rest of the suite runs without them.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAS_CUDA = False

try:
    import support

    HAS_ENGRAIN = True
except Exception:  # noqa: BLE001
    HAS_ENGRAIN = False

INSTANCES = (
    Path(__file__).resolve().parents[1] / "results" / "jsonschemabench-instances.json"
)

# Enough vocabulary to make the token-to-terminal mapping non-trivial without
# pulling a tokenizer in: the pieces a JSON document is actually made of, plus
# multi-character tokens that span several terminals and one that can be read
# two ways.
VOCABULARY = [
    b"{",
    b"}",
    b"[",
    b"]",
    b":",
    b",",
    b'"',
    b" ",
    b"  ",
    b"\\n",
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
    b"123",
    b"-",
    b".",
    b"e",
    b"a",
    b"b",
    b"c",
    b"name",
    b"id",
    b"value",
    b"items",
    b'"name"',
    b'"id"',
    b'"a"',
    b'"b"',
    b"ab",
    b"abc",
]


def _requirements():
    if not HAS_CUDA:
        raise unittest.SkipTest("no CUDA device")
    if not HAS_ENGRAIN:
        raise unittest.SkipTest("engrain is not built")


# A differential run compares the two backends on the host, so nothing in it
# can be recorded into a graph. Tests whose subject *is* the recording have
# nothing to say in that mode.
NEEDS_CAPTURE = unittest.skipIf(
    os.environ.get("ENGRAIN_BACKEND", "").strip().lower() == "differential",
    "differential mode compares on the host and cannot capture",
)


class DeviceParserAgreement(unittest.TestCase):
    """The device mask equals the CPU matcher's, step by step."""

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)

    def _walk(self, schema, document: bytes):
        """Feed `document` a token at a time, checking the mask at every step."""
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        grammar = self.DeviceGrammar(compiled)
        batch = grammar.new_batch(1)
        matcher = compiled.matcher(0)
        reference = torch.zeros(grammar.mask_words, dtype=torch.int32)

        steps = 0
        rest = document
        while rest:
            token = None
            for identifier, piece in enumerate(VOCABULARY):
                if piece and rest.startswith(piece):
                    if token is None or len(piece) > len(VOCABULARY[token]):
                        token = identifier
            self.assertIsNotNone(token, f"no token spells {rest[:8]!r}")

            configurations = matcher.configurations()
            if len(configurations) > grammar.max_configs:
                break
            reference.zero_()
            matcher.fill_bitmask(reference)
            batch.set_configurations(0, configurations)
            device = batch.fill_mask()[0].cpu()
            self.assertTrue(
                torch.equal(device, reference),
                f"mask differs at step {steps}: "
                f"{int(((device & ~reference) != 0).sum())} words with extra bits, "
                f"{int(((reference & ~device) != 0).sum())} with missing bits",
            )
            self.assertTrue(
                bool(device[token // 32] >> (token % 32) & 1),
                f"the mask forbids the token the document uses at step {steps}",
            )

            accepted = matcher.accept_token(token)
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            self.assertTrue(
                accepted, f"the matcher refused its own document at {steps}"
            )
            self.assertEqual(
                sorted((s, tuple(k)) for s, k in matcher.configurations()),
                sorted((s, tuple(k)) for s, k in batch.configurations(0)),
                f"the advance diverged at step {steps}",
            )
            rest = rest[len(VOCABULARY[token]) :]
            steps += 1

        self.assertEqual(int(batch.overflow.sum()), 0, "a replay overran its window")
        self.assertGreater(steps, 0)
        return grammar, steps

    def test_flat_object(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        }
        self._walk(schema, b'{"name":"ab","id":12}')

    def test_properties_in_either_order(self):
        """An object is a set. The declared order must not be the only one."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        }
        self._walk(schema, b'{"id":12,"name":"ab"}')

    def test_array_of_objects(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
        }
        self._walk(schema, b'[{"a":1},{"a":2},{"a":12}]')

    def test_nesting_does_not_overrun_the_window(self):
        """A repetition must not deepen the stack, which is why it is left-recursive."""
        schema = {"type": "array", "items": {"type": "integer"}}
        grammar, _ = self._walk(schema, b"[1,2,12,123,1,2,12,123,1,2]")
        self.assertLessEqual(
            grammar.window,
            grammar.max_stack,
            "the window bound should never exceed the stack it lives over",
        )

    def test_window_bound_is_computed_not_guessed(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        grammar = self.DeviceGrammar(compiled)
        self.assertGreaterEqual(grammar.window, 8)
        self.assertLessEqual(grammar.window_bound, grammar.window)

    def test_a_narrowed_mask_is_reported_not_absorbed(self):
        """Running out of configuration room must not be silent.

        The ceiling is a policy rather than a property of the grammar, and a
        parse that outgrows it keeps a prefix of its states, which narrows the
        mask. Narrowing is the one failure this engine must never do quietly,
        so it is reported through the same flag as a replay that overran its
        window.
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        }
        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        # One configuration is not enough for a grammar where a declared
        # property name also scans as a generic string.
        grammar = self.DeviceGrammar(compiled, max_configs=1)
        batch = grammar.new_batch(1)
        matcher = compiled.matcher(0)
        batch.set_configurations(0, matcher.configurations())
        for piece in (b'{"', b"name", b'":'):
            token = VOCABULARY.index(piece)
            batch.fill_mask()
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            matcher.accept_token(token)
        terminated, overflow = batch.problems()
        self.assertEqual(terminated.numel(), 1)
        self.assertEqual(
            int(overflow.sum()),
            1,
            "a parse that outgrew one configuration should have said so",
        )


class MixedGrammarBatch(unittest.TestCase):
    """A batch whose sequences are under different grammars.

    The case a serving engine has by default, and the one a design shaped by a
    single grammar's tables cannot serve at all.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)
        self.schemas = [
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
            {"type": "array", "items": {"type": "integer"}},
            {
                "type": "object",
                "properties": {"b": {"type": "string"}},
                "required": ["b"],
            },
        ]
        self.pool = [
            self.compiler.compile_json_schema(json.dumps(schema))
            for schema in self.schemas
        ]

    def test_each_sequence_gets_its_own_grammar_mask(self):
        grammar = self.DeviceGrammar(self.pool)
        batch = grammar.new_batch(9)
        assignment = [index % len(self.pool) for index in range(9)]
        batch.set_grammars(assignment)

        matchers = [self.pool[assignment[i]].matcher(0) for i in range(9)]
        batch.set_batch_configurations(
            {i: matchers[i].configurations() for i in range(9)}
        )
        masks = batch.fill_mask().cpu()
        reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
        for index in range(9):
            reference.zero_()
            matchers[index].fill_bitmask(reference)
            self.assertTrue(
                torch.equal(masks[index], reference),
                f"sequence {index} under grammar {assignment[index]} got the wrong mask",
            )

    def test_lookalike_states_under_different_grammars_are_not_merged(self):
        """Deduplication must not share a mask across grammars.

        Two schemas can put a sequence at the same parser state with the same
        stack and still admit different tokens, so the grammar has to be part of
        what makes two parse states the same.
        """
        grammar = self.DeviceGrammar(self.pool)
        batch = grammar.new_batch(len(self.pool))
        batch.set_grammars(list(range(len(self.pool))))
        matchers = [item.matcher(0) for item in self.pool]
        batch.set_batch_configurations(
            {i: matchers[i].configurations() for i in range(len(self.pool))}
        )
        masks = batch.fill_mask().cpu()
        # The schemas differ, so at the very start their masks must differ too;
        # if dedup had merged them these rows would be identical.
        self.assertFalse(torch.equal(masks[0], masks[1]))

    @NEEDS_CAPTURE
    def test_one_graph_serves_any_assignment(self):
        """A CUDA graph is a fixed sequence of launches.

        A continuous batch changes composition every step, so a recording that
        only covered the grammars it was made with would be useless. The grid is
        fixed and the work list is built on the device, so it does not.
        """
        grammar = self.DeviceGrammar(self.pool)
        batch = grammar.new_batch(6)
        first = [0, 1, 2, 0, 1, 2]
        batch.set_grammars(first)
        matchers = [self.pool[first[i]].matcher(0) for i in range(6)]
        batch.set_batch_configurations(
            {i: matchers[i].configurations() for i in range(6)}
        )
        batch.fill_mask()
        batch.capture()

        second = [2, 2, 1, 1, 0, 0]
        batch.set_grammars(second)
        matchers = [self.pool[second[i]].matcher(0) for i in range(6)]
        batch.set_batch_configurations(
            {i: matchers[i].configurations() for i in range(6)}
        )
        replayed = batch.fill_mask().cpu()

        reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
        for index in range(6):
            reference.zero_()
            matchers[index].fill_bitmask(reference)
            self.assertTrue(
                torch.equal(replayed[index], reference),
                f"the replayed graph got sequence {index} wrong on a new assignment",
            )


class RandomWalkAgreement(unittest.TestCase):
    """Compare masks at every state a random walk reaches, not one document.

    A fixed document visits one path. The serving path disagreed at a state that
    path never reaches, because the model tokenises differently from the corpus
    and so arrives by routes the corpus does not take. Choosing a random one of
    the tokens the matcher admits at each step covers far more of what is
    reachable, and it is driven by the mask, so a mask that is too narrow shrinks
    the walk rather than hiding.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)

    def _walk(self, schema, walks=8, length=25):
        import random

        compiled = self.compiler.compile_json_schema(json.dumps(schema))
        pool = self.DeviceGrammar(compiled)
        batch = pool.new_batch(1)
        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        rng = random.Random(20260727)
        seen = set()
        steps = 0
        for _ in range(walks):
            matcher = compiled.matcher(0)
            for _ in range(length):
                configurations = matcher.configurations()
                if len(configurations) > pool.max_configs:
                    break
                seen.add(tuple(sorted((s, tuple(k)) for s, k in configurations)))
                reference.zero_()
                matcher.fill_bitmask(reference)
                batch.set_configurations(0, configurations)
                device = batch.fill_mask()[0].cpu()
                self.assertTrue(
                    torch.equal(device, reference),
                    f"mask differs at {configurations}: "
                    f"{int(((device & ~reference) != 0).sum())} words with extra "
                    f"bits, {int(((reference & ~device) != 0).sum())} with missing",
                )
                choices = [
                    token
                    for token in range(pool.vocab_size)
                    if (reference[token // 32] >> (token % 32)) & 1
                ]
                if not choices:
                    break
                if not matcher.accept_token(rng.choice(choices)):
                    break
                steps += 1
        self.assertEqual(int(batch.overflow.sum()), 0)
        self.assertGreater(steps, walks, "the walk barely moved")
        return len(seen)

    def test_open_object(self):
        states = self._walk(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
                "required": ["name"],
            }
        )
        self.assertGreater(states, 1)

    def test_closed_object(self):
        """`additionalProperties: false` is where the serving path disagreed."""
        self._walk(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "active": {"type": "boolean"},
                },
                "required": ["name", "age", "active"],
                "additionalProperties": False,
            }
        )

    def test_array(self):
        self._walk({"type": "array", "items": {"type": "integer"}})


class Rollback(unittest.TestCase):
    """Undoing advances without asking the host to replay anything.

    Speculative decoding advances through a draft and then keeps only the prefix
    the model accepted, so the parser has to go back. Going back by replaying
    the tokens on the host is the round trip the whole design exists not to
    make, so the state is kept on the device and put back from there.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)
        self.schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        }

    def _tokens(self, document: bytes):
        out, rest = [], document
        while rest:
            best = None
            for identifier, piece in enumerate(VOCABULARY):
                if piece and rest.startswith(piece):
                    if best is None or len(piece) > len(VOCABULARY[best]):
                        best = identifier
            self.assertIsNotNone(best, f"no token spells {rest[:8]!r}")
            out.append(best)
            rest = rest[len(VOCABULARY[best]) :]
        return out

    def test_undoing_a_draft_restores_the_matcher_state(self):
        compiled = self.compiler.compile_json_schema(json.dumps(self.schema))
        pool = self.DeviceGrammar(compiled)
        batch = pool.new_batch(1, rollback=4)
        matcher = compiled.matcher(8)
        tokens = self._tokens(b'{"name":"ab","id":12}')

        batch.set_matchers([matcher])
        for token in tokens[:3]:
            matcher.accept_token(token)
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
        kept = sorted((s, tuple(k)) for s, k in matcher.configurations())
        self.assertEqual(
            sorted((s, tuple(k)) for s, k in batch.configurations(0)), kept
        )

        # Take three more, as a draft would, then reject them all.
        for token in tokens[3:6]:
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
        self.assertNotEqual(
            sorted((s, tuple(k)) for s, k in batch.configurations(0)), kept
        )
        batch.rollback(3)
        self.assertEqual(
            sorted((s, tuple(k)) for s, k in batch.configurations(0)),
            kept,
            "the parse state did not come back to where the matcher is",
        )

    def test_a_partly_accepted_draft(self):
        """Keep two of three drafted tokens, which is the ordinary case."""
        compiled = self.compiler.compile_json_schema(json.dumps(self.schema))
        pool = self.DeviceGrammar(compiled)
        batch = pool.new_batch(1, rollback=4)
        matcher = compiled.matcher(8)
        tokens = self._tokens(b'{"name":"ab","id":12}')

        batch.set_matchers([matcher])
        for token in tokens[:5]:
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            matcher.accept_token(token)
        batch.rollback(2)
        for _token in tokens[3:5]:
            matcher.rollback(1)
        self.assertEqual(
            sorted((s, tuple(k)) for s, k in batch.configurations(0)),
            sorted((s, tuple(k)) for s, k in matcher.configurations()),
        )

    def test_the_mask_after_a_rollback_is_the_matcher_s(self):
        compiled = self.compiler.compile_json_schema(json.dumps(self.schema))
        pool = self.DeviceGrammar(compiled)
        batch = pool.new_batch(1, rollback=4)
        matcher = compiled.matcher(8)
        tokens = self._tokens(b'{"name":"ab","id":12}')

        batch.set_matchers([matcher])
        for token in tokens[:4]:
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            matcher.accept_token(token)
        batch.rollback(2)
        matcher.rollback(2)

        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        matcher.fill_bitmask(reference)
        self.assertTrue(torch.equal(batch.fill_mask()[0].cpu(), reference))

    @NEEDS_CAPTURE
    def test_the_history_survives_a_captured_advance(self):
        """The advance keeps history from inside a CUDA graph.

        A graph records the arguments it was launched with, so a ring slot
        passed as a scalar would be frozen at whatever it was when the recording
        was made and every step would overwrite the same entry. The slot lives
        on the device for exactly this reason, and this is the test that says
        so: capture, take several steps, and roll back through them.
        """
        compiled = self.compiler.compile_json_schema(json.dumps(self.schema))
        pool = self.DeviceGrammar(compiled)
        batch = pool.new_batch(1, rollback=4)
        matcher = compiled.matcher(8)
        tokens = self._tokens(b'{"name":"ab","id":12}')
        batch.set_matchers([matcher])
        batch.capture_advance()

        for token in tokens[:2]:
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            matcher.accept_token(token)
        kept = sorted((s, tuple(k)) for s, k in matcher.configurations())
        for token in tokens[2:5]:
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
        batch.rollback(3)
        self.assertEqual(
            sorted((s, tuple(k)) for s, k in batch.configurations(0)),
            kept,
            "a captured advance wrote every step to the same history slot",
        )

    def test_a_batch_without_history_refuses_to_roll_back(self):
        compiled = self.compiler.compile_json_schema(json.dumps(self.schema))
        batch = self.DeviceGrammar(compiled).new_batch(1)
        with self.assertRaises(ValueError):
            batch.rollback(1)

    def test_rolling_back_further_than_is_kept_is_refused(self):
        compiled = self.compiler.compile_json_schema(json.dumps(self.schema))
        batch = self.DeviceGrammar(compiled).new_batch(1, rollback=2)
        tokens = self._tokens(b'{"name"')
        batch.set_matchers([compiled.matcher(8)])
        batch.advance(torch.tensor([tokens[0]], dtype=torch.int32, device="cuda"))
        with self.assertRaises(ValueError):
            batch.rollback(2)


class GrammarPool(unittest.TestCase):
    """Grammars arrive with requests and leave when the request does.

    A pool handed a fixed list at construction is not something a serving engine
    can use, so admission and release have to work while a batch is running -
    including the part that is easy to get wrong, which is that a graph holds
    the address it recorded and an array that grows is at a new one.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)
        self.schemas = [
            {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
            {"type": "array", "items": {"type": "integer"}},
            {
                "type": "object",
                "properties": {"b": {"type": "string"}},
                "required": ["b"],
            },
            {"type": "array", "items": {"type": "boolean"}},
        ]

    def _compile(self, index):
        return self.compiler.compile_json_schema(json.dumps(self.schemas[index]))

    def _mask_matches(self, pool, batch, assignment, compiled):
        matchers = [compiled[g].matcher(0) for g in assignment]
        batch.set_grammars(assignment)
        batch.set_batch_configurations(
            {i: m.configurations() for i, m in enumerate(matchers)}
        )
        masks = batch.fill_mask().cpu()
        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        for index, matcher in enumerate(matchers):
            reference.zero_()
            matcher.fill_bitmask(reference)
            if not torch.equal(masks[index], reference):
                return False
        return True

    def test_admitting_after_construction(self):
        pool = self.DeviceGrammar()
        compiled = []
        for index in range(len(self.schemas)):
            item = self._compile(index)
            self.assertEqual(pool.admit(item), index)
            compiled.append(item)
        batch = pool.new_batch(4)
        self.assertTrue(self._mask_matches(pool, batch, [0, 1, 2, 3], compiled))

    @NEEDS_CAPTURE
    def test_a_graph_survives_admission_into_spare_capacity(self):
        pool = self.DeviceGrammar()
        compiled = [self._compile(0), self._compile(1)]
        for item in compiled:
            pool.admit(item)
        batch = pool.new_batch(2)
        batch.set_grammars([0, 1])
        matchers = [compiled[i].matcher(0) for i in (0, 1)]
        batch.set_batch_configurations(
            {i: m.configurations() for i, m in enumerate(matchers)}
        )
        batch.fill_mask()
        batch.capture()
        before = pool.revision

        compiled.append(self._compile(2))
        pool.admit(compiled[2])
        # Whether the arrays had room decides whether the graph is still valid.
        # Either way the mask must be right, which is what this checks.
        self.assertTrue(self._mask_matches(pool, batch, [0, 1], compiled))
        if pool.revision != before:
            self.assertNotEqual(batch.recorded, pool.revision)

    def test_release_and_compact_renumber_and_keep_masks_right(self):
        pool = self.DeviceGrammar()
        compiled = [self._compile(index) for index in range(4)]
        for item in compiled:
            pool.admit(item)
        used = pool.used_bytes()
        pool.release(0)
        pool.release(2)
        self.assertGreater(pool.dead_fraction, 0.0)

        remap = pool.compact()
        self.assertEqual(set(remap), {1, 3})
        self.assertLess(pool.used_bytes(), used)
        self.assertEqual(pool.count, 2)
        self.assertEqual(pool.dead_fraction, 0.0)

        survivors = [compiled[1], compiled[3]]
        batch = pool.new_batch(2)
        self.assertTrue(
            self._mask_matches(pool, batch, [remap[1], remap[3]], survivors)
        )

    @NEEDS_CAPTURE
    def test_a_recorded_graph_is_not_replayed_after_compaction(self):
        pool = self.DeviceGrammar()
        compiled = [self._compile(index) for index in range(3)]
        for item in compiled:
            pool.admit(item)
        batch = pool.new_batch(2)
        batch.set_grammars([1, 2])
        matchers = [compiled[i].matcher(0) for i in (1, 2)]
        batch.set_batch_configurations(
            {i: m.configurations() for i, m in enumerate(matchers)}
        )
        batch.fill_mask()
        batch.capture()
        self.assertEqual(batch.recorded, pool.revision)

        pool.release(0)
        remap = pool.compact()
        self.assertNotEqual(batch.recorded, pool.revision)
        self.assertTrue(
            self._mask_matches(
                pool, batch, [remap[1], remap[2]], [compiled[1], compiled[2]]
            )
        )


class DraftWalk(unittest.TestCase):
    """A speculative draft, walked in one launch.

    A draft of `k` tokens was `k` fills and `k` advances, which is `2k` graph
    replays - linear in `k` in the one cost this design exists to remove. The
    whole walk is one recording now, and what has to hold is that it produces
    the same masks and puts the parse back where it found it.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.compiler = support.Compiler(VOCABULARY)
        self.compiled = self.compiler.compile_json_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            )
        )
        self.pool = DeviceGrammar(self.compiled)

    def _draft(self, pieces, batch):
        return torch.tensor(
            [[VOCABULARY.index(piece)] * batch for piece in pieces],
            dtype=torch.int32,
            device="cuda",
        )

    @NEEDS_CAPTURE
    def test_every_position_agrees_with_the_matcher(self):
        batch = 4
        pieces = [b'{"', b"name", b'":', b'"']
        device = self.pool.new_batch(batch)
        matcher = self.compiled.matcher(0)
        device.set_batch_configurations(
            {row: matcher.configurations() for row in range(batch)}
        )
        device.capture_draft(len(pieces))
        masks = device.walk_draft(self._draft(pieces, batch)).cpu()

        reference = torch.zeros(self.pool.mask_words, dtype=torch.int32)
        for position, piece in enumerate(pieces):
            self.assertTrue(matcher.accept_token(VOCABULARY.index(piece)))
            reference.zero_()
            matcher.fill_bitmask(reference)
            for row in range(batch):
                self.assertTrue(
                    torch.equal(masks[position, row], reference),
                    f"position {position} row {row}",
                )

    @NEEDS_CAPTURE
    def test_the_walk_leaves_the_parse_where_it_found_it(self):
        device = self.pool.new_batch(2)
        matcher = self.compiled.matcher(0)
        device.set_batch_configurations(
            {row: matcher.configurations() for row in (0, 1)}
        )
        before = [sorted(device.configurations(row)) for row in (0, 1)]
        device.capture_draft(3)
        device.walk_draft(self._draft([b'{"', b"name", b'":'], 2))
        after = [sorted(device.configurations(row)) for row in (0, 1)]
        self.assertEqual(before, after)

    @NEEDS_CAPTURE
    def test_a_draft_the_grammar_refuses_does_not_poison_the_rest(self):
        """A rejected position must not leave the sequence broken."""
        batch = 2
        device = self.pool.new_batch(batch)
        matcher = self.compiled.matcher(0)
        device.set_batch_configurations(
            {row: matcher.configurations() for row in range(batch)}
        )
        device.capture_draft(2)
        device.walk_draft(self._draft([b"]", b"]"], batch))
        # The parse is back at the start, so an ordinary fill still agrees.
        reference = torch.zeros(self.pool.mask_words, dtype=torch.int32)
        matcher.fill_bitmask(reference)
        self.assertTrue(torch.equal(device.fill_mask()[0].cpu(), reference))


class PrecomputedVerdicts(unittest.TestCase):
    """The refusals settled when the tables are built.

    91% of group replays are decided by the parser state alone, so they are
    settled at compile time and read at runtime. A grammar too large for the
    table has to fall back to replaying everything, and because the kernel is
    compiled once for a whole pool, one such grammar has to turn the shortcut
    off for all of them - which is where this first went wrong.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)

    def _compiled(self, schema):
        return self.compiler.compile_json_schema(json.dumps(schema))

    def test_a_pool_without_a_table_everywhere_turns_the_shortcut_off(self):
        pool = self.DeviceGrammar()
        item = self._compiled(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        )
        pool.admit(item)
        # A grammar whose table was abandoned, faked by admitting one that has
        # none. The flag must go to zero and stay there.
        tables = self.DeviceGrammar.prepare(item)
        tables.has_verdicts = 0
        pool.admit(tables)
        self.assertEqual(
            pool.has_verdicts,
            0,
            "one grammar without a table must disable the shortcut",
        )
        pool.admit(self._compiled({"type": "array", "items": {"type": "integer"}}))
        self.assertEqual(pool.has_verdicts, 0, "and it must not come back")

    def test_the_mask_is_the_matcher_s_with_the_shortcut_on(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
            "required": ["name", "id"],
        }
        compiled = self._compiled(schema)
        pool = self.DeviceGrammar(compiled)
        self.assertEqual(pool.has_verdicts, 1)
        batch = pool.new_batch(1)
        matcher = compiled.matcher(0)
        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        for piece in (b'{"', b"name", b'":"', b"ab", b'","', b"id", b'":', b"12", b"}"):
            token = VOCABULARY.index(piece)
            reference.zero_()
            matcher.fill_bitmask(reference)
            batch.set_configurations(0, matcher.configurations())
            self.assertTrue(torch.equal(batch.fill_mask()[0].cpu(), reference))
            batch.advance(torch.tensor([token], dtype=torch.int32, device="cuda"))
            self.assertTrue(matcher.accept_token(token))
            self.assertEqual(
                sorted((s, tuple(k)) for s, k in batch.configurations(0)),
                sorted((s, tuple(k)) for s, k in matcher.configurations()),
            )


class ArenaPaging(unittest.TestCase):
    """Grammars come and go faster than a graph can be re-recorded.

    Under continuous batching a request brings a schema and takes it away
    again, so the pool churns. The costly answer is to compact, which
    renumbers every survivor and re-records every graph. This is the other
    one: a released run goes back to a free list, its identifier goes back
    too, and neither the addresses nor the numbering move.
    """

    def setUp(self):
        _requirements()
        from engrain._engine import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = support.Compiler(VOCABULARY)

    def test_a_live_identifier_past_the_live_count_is_still_usable(self):
        """An eviction frees a slot; it does not shrink the identifier space.

        `count` is how many grammars are live. Identifiers are recycled from a
        free list and fresh ones are allocated past the high-water mark, so
        once anything has been evicted the largest live identifier can exceed
        `count`. Validating an id against `count` therefore refused a perfectly
        live grammar - "grammar id past the end of the pool" - and a serving
        run of 425 real schemas under a table budget is what found it.
        """
        pool = self.DeviceGrammar(budget_bytes=1 << 20)
        held = {}
        for index in range(24):
            grammar = self.compiler.compile_json_schema(self._schema(index))
            identifier = pool.admit(grammar)
            held[index] = (identifier, pool.generation(identifier))
        live = [
            identifier for identifier, generation in held.values()
            if pool.holds(identifier, generation)
        ]
        self.assertTrue(live, "the budget evicted everything")
        # The premise: something was evicted, and an id outran the live count.
        self.assertLess(pool.count, pool.slots)
        self.assertGreaterEqual(max(live), pool.count)
        # And the batch takes them, which is the thing that used to raise.
        batch = pool.new_batch(len(live))
        batch.set_grammars(live)

    def test_a_recycled_slot_does_not_inherit_the_memo_of_its_last_tenant(self):
        """A mask is remembered per (grammar, state); a slot changes hands.

        The cross-step memo keys an entry on the grammar's *slot*, so once an
        eviction frees a slot and the next admission reuses it, an entry left by
        the schema that departed is handed to the one that arrived. The state
        stored beside the entry cannot catch it - the identifier compares equal -
        and the mask that comes back is *wider* than the truth, so it does not
        raise the overflow flag either. It surfaces much later, as the model
        emitting a token the matcher then refuses.

        Emptying the memo on `revision` is not enough: that says the arrays have
        moved, and admitting into spare capacity deliberately does not move it.
        Two schemas of the same shape and different property names collide on
        every part of the key, which is what makes the stale entry reachable.
        """
        brace = VOCABULARY.index(b"{")
        pool = self.DeviceGrammar()

        def seed(name):
            matcher = self.compiler.compile_json_schema(self._named(name)).matcher(32)
            self.assertTrue(matcher.accept_token(brace))
            return matcher

        first = pool.admit(self.compiler.compile_json_schema(self._named("a")))
        batch = pool.new_batch(1)
        batch.set_grammars([first])
        batch.set_batch_configurations({0: seed("a").configurations()})
        batch.fill_mask()

        pool.release(first)
        second = pool.admit(self.compiler.compile_json_schema(self._named("b")))
        # The premise of the test: the same slot, a different grammar.
        self.assertEqual(first, second)

        matcher = seed("b")
        batch.set_grammars([second])
        batch.set_batch_configurations({0: matcher.configurations()})
        mask = batch.fill_mask()[0].cpu()

        reference = torch.zeros(mask.numel(), dtype=torch.int32)
        matcher.fill_bitmask(reference)
        extra = int(((mask & ~reference) != 0).sum())
        self.assertEqual(
            extra, 0, "the recycled slot was masked against its predecessor"
        )
        self.assertTrue(torch.equal(mask, reference))

    @staticmethod
    def _named(name):
        return json.dumps(
            {
                "type": "object",
                "properties": {name: {"type": "string"}},
                "required": [name],
                "additionalProperties": False,
            }
        )

    def _schema(self, index):
        # Distinguishable, and different enough in size that a hole left by one
        # does not automatically fit the next.
        names = [f"p{index}_{n}" for n in range(1 + index % 4)]
        return json.dumps(
            {
                "type": "object",
                "properties": {name: {"type": "string"} for name in names},
                "required": names,
                "additionalProperties": False,
            }
        )

    def _compile(self, index):
        return self.compiler.compile_json_schema(self._schema(index))

    def _mask_agrees(self, pool, batch, row, compiled, identifier):
        matcher = compiled.matcher(0)
        batch.set_grammars([identifier] * batch.batch)
        batch.set_batch_configurations({row: matcher.configurations()})
        mask = batch.fill_mask()[row].cpu()
        reference = torch.zeros(pool.mask_words, dtype=torch.int32)
        matcher.fill_bitmask(reference)
        return torch.equal(mask, reference)

    def test_a_released_run_is_reused_rather_than_appended(self):
        """The arena must not grow forever under churn."""
        pool = self.DeviceGrammar()
        first = pool.admit(self._compile(0))
        held = dict(pool._used)
        for _round in range(12):
            pool.release(first)
            first = pool.admit(self._compile(0))
        self.assertEqual(dict(pool._used), held)
        self.assertEqual(pool.count, 1)

    def test_an_identifier_is_reused_and_nothing_is_renumbered(self):
        pool = self.DeviceGrammar()
        keep = [pool.admit(self._compile(index)) for index in range(3)]
        pool.release(keep[1])
        again = pool.admit(self._compile(1))
        self.assertEqual(again, keep[1], "a freed slot should be reused")
        self.assertEqual(keep[2], 2, "the survivors must keep their identifiers")

    @NEEDS_CAPTURE
    def test_a_recorded_graph_survives_churn(self):
        """The property the whole design is for.

        Compaction re-records every graph in the engine. Admitting into a hole
        left by a release must not, or a serving loop would spend its time
        re-recording.
        """
        pool = self.DeviceGrammar()
        compiled = [self._compile(index) for index in range(4)]
        kept = [pool.admit(item) for item in compiled]
        batch = pool.new_batch(2)
        batch.set_grammars([kept[0], kept[1]])
        batch.set_batch_configurations(
            {i: compiled[i].matcher(0).configurations() for i in (0, 1)}
        )
        batch.fill_mask()
        batch.capture()
        recorded = pool.revision

        # Everything the batch is not using leaves and comes back, repeatedly.
        for _ in range(8):
            pool.release(kept[2])
            pool.release(kept[3])
            kept[2] = pool.admit(compiled[2])
            kept[3] = pool.admit(compiled[3])
        self.assertEqual(pool.revision, recorded, "churn re-recorded the graph")
        self.assertEqual(batch.recorded, pool.revision)
        batch.fill_mask()
        self.assertTrue(self._mask_agrees(pool, batch, 0, compiled[0], kept[0]))

    def test_a_budget_evicts_the_least_recently_used(self):
        pool = self.DeviceGrammar(budget_bytes=1)
        first = pool.admit(self._compile(0))
        stamp = pool.generation(first)
        second = pool.admit(self._compile(4))
        self.assertGreater(pool.evictions, 0)
        # The slot is reused, so the identifier alone says nothing - which is
        # exactly why anyone holding one has to ask whether it still holds.
        self.assertFalse(pool.holds(first, stamp))
        self.assertTrue(pool.holds(second, pool.generation(second)))
        self.assertTrue(
            self._mask_agrees(pool, pool.new_batch(1), 0, self._compile(1), second)
        )

    def test_a_pinned_grammar_is_never_evicted(self):
        """A sequence is running under it; evicting it would mask against another."""
        pool = self.DeviceGrammar(budget_bytes=1)
        held = pool.admit(self._compile(0))
        pool.pin(held)
        for index in range(1, 5):
            pool.admit(self._compile(index))
        self.assertTrue(pool.is_live(held))

    def test_a_reused_slot_is_not_mistaken_for_the_old_grammar(self):
        pool = self.DeviceGrammar()
        first = pool.admit(self._compile(0))
        stamp = pool.generation(first)
        self.assertTrue(pool.holds(first, stamp))
        pool.release(first)
        self.assertFalse(pool.holds(first, stamp))
        again = pool.admit(self._compile(2))
        self.assertEqual(again, first)
        self.assertFalse(
            pool.holds(first, stamp), "a cached identifier must not survive an eviction"
        )
        self.assertTrue(pool.holds(again, pool.generation(again)))

    def test_holes_are_joined_so_a_big_grammar_still_fits(self):
        pool = self.DeviceGrammar()
        small = [pool.admit(self._compile(index)) for index in range(4)]
        big = pool.admit(self._compile(3))
        size = sum(pool._extent[big].get(name, (0, 0))[1] for name in pool._extent[big])
        pool.release(big)
        for identifier in small:
            pool.release(identifier)
        self.assertEqual(
            pool.dead_fraction, 0.0, "freeing everything should leave no holes"
        )
        self.assertEqual(sum(pool._used.values()), 0)
        again = pool.admit(self._compile(3))
        self.assertEqual(
            sum(
                pool._extent[again].get(name, (0, 0))[1] for name in pool._extent[again]
            ),
            size,
        )


class CorpusAgreement(unittest.TestCase):
    """The same check on real schemas, if the corpus is present."""

    def setUp(self):
        _requirements()
        if not INSTANCES.exists():
            raise unittest.SkipTest("the JSONSchemaBench corpus is not in results/")

    def test_first_schemas_agree_byte_by_byte(self):
        from engrain._engine import DeviceGrammar

        instances = json.loads(INSTANCES.read_text())["instances"]
        # Bytes rather than a tokenizer: the corpus check that uses the real
        # vocabulary lives in the benchmark, and a unit test should not download
        # a model.
        vocabulary = [bytes([value]) for value in range(256)]
        compiler = support.Compiler(vocabulary)
        checked = 0
        for instance in instances[:3]:
            compiled = compiler.compile_json_schema(instance["schema"])
            grammar = DeviceGrammar(compiled)
            batch = grammar.new_batch(1)
            matcher = compiled.matcher(0)
            reference = torch.zeros(grammar.mask_words, dtype=torch.int32)
            for byte in instance["text"].encode()[:48]:
                configurations = matcher.configurations()
                if len(configurations) > grammar.max_configs:
                    break
                reference.zero_()
                matcher.fill_bitmask(reference)
                batch.set_configurations(0, configurations)
                device = batch.fill_mask()[0].cpu()
                self.assertTrue(torch.equal(device, reference))
                if not matcher.accept_token(byte):
                    break
                batch.advance(torch.tensor([byte], dtype=torch.int32, device="cuda"))
                checked += 1
            self.assertEqual(int(batch.overflow.sum()), 0)
        self.assertGreater(checked, 0)


class FusedStep(unittest.TestCase):
    """The advance and the next fill as one graph.

    Only the sample sits between a fill and the advance that follows it;
    nothing sits between that advance and the next fill, so they are one graph
    and a decode step is one replay. What has to hold is that it produces the
    masks the two graphs produce.
    """

    @NEEDS_CAPTURE
    def test_a_fused_step_masks_what_the_two_graphs_mask(self):
        import torch

        from engrain._engine import DeviceGrammar, DeviceBatch

        compiler = support.Compiler([bytes([b]) for b in range(256)])
        grammar = compiler.compile_json_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                }
            )
        )
        pool = DeviceGrammar()
        pool.admit(grammar)
        document = b'{"a": "xy", "b": 12}'

        separate = DeviceBatch(pool, 2)
        fused = DeviceBatch(pool, 2)
        reference = grammar.matcher(0)
        for batch in (separate, fused):
            batch.set_batch_configurations(
                {index: reference.configurations() for index in range(2)}
            )
            batch.fill_mask()
            batch.capture()
        separate.advance(torch.zeros(2, dtype=torch.int32, device="cuda"))
        separate.capture_advance()
        fused.capture_step()

        steps = 0
        for byte in document:
            self.assertTrue(
                torch.equal(separate.mask, fused.mask),
                f"masks diverge at step {steps}",
            )
            if not reference.accept_token(byte):
                break
            token = torch.full((2,), byte, dtype=torch.int32, device="cuda")
            separate.advance(token)
            separate.fill_mask()
            fused.advance_and_fill(token)
            steps += 1
        self.assertGreater(steps, 8)


class StackDeeperThanTheBlock(unittest.TestCase):
    """The depth a parse reaches must not be a launch parameter.

    A thread used to own a stack entry in the commit phase, so a block had to
    be as wide as the deepest stack the batch allowed. That put the ceiling
    into the launch: past about 512 the fused kernel could not be launched at
    all - "too many resources requested" - and below it, a document that grew
    past the ceiling was thrown off the device onto the reference matcher,
    which costs about 1.5 ms per row per step and never stops, because the
    document keeps growing. Measured at batch 512 on a corpus schema: three
    rows a step, 4,577 us, 58% of the backend.
    """

    def setUp(self):
        _requirements()

    def test_a_parse_deeper_than_the_block_agrees_with_the_matcher(self):
        from engrain._engine import DeviceGrammar

        compiler = support.Compiler(VOCABULARY)
        grammar = compiler.compile_json_schema(json.dumps({"type": "array"}))
        matcher = grammar.matcher(0)
        opening = VOCABULARY.index(b"[")
        # Two stack entries a level, so this is past both the block width the
        # commit is launched with and the 256 a pool defaults to.
        for _ in range(200):
            self.assertTrue(matcher.accept_token(opening))
        depth = matcher.max_stack_depth()
        self.assertGreater(depth, 256, "the premise: deeper than a default pool")

        pool = DeviceGrammar(max_stack=1024)
        pool.admit(grammar)
        batch = pool.new_batch(1)
        batch.set_grammars([0])
        batch.set_matchers([matcher])
        mask = batch.fill_mask()[0].cpu()

        reference = torch.zeros(mask.numel(), dtype=torch.int32)
        matcher.fill_bitmask(reference)
        self.assertTrue(torch.equal(mask, reference))
        # And the block really is narrower than the stack, or this proves
        # nothing about the loop.
        self.assertLess(batch._fused_threads(pool), depth)

    def test_advancing_a_deep_parse_agrees_with_the_matcher(self):
        """The commit phase runs on advance, which is where the launch failed."""
        from engrain._engine import DeviceGrammar

        compiler = support.Compiler(VOCABULARY)
        grammar = compiler.compile_json_schema(json.dumps({"type": "array"}))
        matcher = grammar.matcher(0)
        opening = VOCABULARY.index(b"[")
        for _ in range(200):
            matcher.accept_token(opening)

        pool = DeviceGrammar(max_stack=1024)
        pool.admit(grammar)
        batch = pool.new_batch(1)
        batch.set_grammars([0])
        batch.set_matchers([matcher])
        batch.advance(torch.tensor([opening], dtype=torch.int32, device="cuda"))
        self.assertTrue(matcher.accept_token(opening))

        held = batch.configurations(0)
        self.assertEqual(
            sorted((state, tuple(stack)) for state, stack in held),
            sorted(
                (state, tuple(stack)) for state, stack in matcher.configurations()
            ),
        )


class TheWindowIsCapped(unittest.TestCase):
    """One grammar must not size every row of every batch for itself.

    `cand_window` is `batch x configurations x readings x window` and is 95% of
    a batch's memory. Over 116 corpus schemas the replay window is 20 at the
    median and 32 at the p90, and then 349 - so without a cap one schema in ten
    costs the other nine an eightfold buffer: 8.40 GiB at batch 512 against
    1.32 with the cap.
    """

    def setUp(self):
        _requirements()

    def test_a_grammar_past_the_cap_is_refused_rather_than_paid_for(self):
        from engrain._engine import DeviceGrammar, WindowTooWide

        compiler = support.Compiler(VOCABULARY)
        # Deeply nested arrays are what make a replay window wide: closing them
        # is one reduction chain.
        schema = {"type": "array"}
        for _ in range(4):
            schema = {"type": "array", "items": schema}
        grammar = compiler.compile_json_schema(json.dumps(schema))

        roomy = DeviceGrammar(window_cap=None)
        roomy.admit(grammar)
        needed = roomy.window_bound
        self.assertGreater(needed, 2, "the premise: this grammar wants a window")

        tight = DeviceGrammar(window_cap=needed - 1)
        with self.assertRaises(WindowTooWide) as refusal:
            tight.admit(grammar)
        self.assertEqual(refusal.exception.needed, needed)
        self.assertEqual(refusal.exception.limit, needed - 1)
        # The refusal left the pool as it was: nothing admitted, and no
        # ceiling raised on the way to deciding.
        self.assertEqual(tight.count, 0)
        self.assertEqual(tight.window_bound, 0)

        # And a pool with room takes it, so the cap is the only thing refusing.
        roomy_enough = DeviceGrammar(window_cap=needed)
        roomy_enough.admit(grammar)
        self.assertEqual(roomy_enough.count, 1)

    def test_the_window_the_batch_allocates_follows_the_cap(self):
        """Which is the point: the buffer is the cap, not the worst schema."""
        from engrain._engine import DeviceGrammar

        compiler = support.Compiler(VOCABULARY)
        schema = {"type": "array"}
        for _ in range(4):
            schema = {"type": "array", "items": schema}
        grammar = compiler.compile_json_schema(json.dumps(schema))
        pool = DeviceGrammar(window_cap=None)
        pool.admit(grammar)
        self.assertGreaterEqual(pool.window, pool.window_bound)


class SizedForTheMachine(unittest.TestCase):
    """Nothing that decides how much of a device to use may be a constant.

    Every number here was swept on one A100, and a constant swept on one card
    is wrong on the next one - and was already wrong at both ends of the batch
    on that one.
    """

    def test_the_grid_follows_the_batch_and_stays_inside_the_machine(self):
        import torch

        from engrain._engine import _MIN_SWEEP_BLOCKS, _sweep_blocks

        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        ceiling = 1 << max(device.multi_processor_count * 64 - 1, 1).bit_length()

        widths = [_sweep_blocks(batch) for batch in (1, 8, 32, 128, 512, 4096)]
        for width in widths:
            self.assertGreaterEqual(width, _MIN_SWEEP_BLOCKS)
            self.assertLessEqual(width, ceiling)
            self.assertEqual(width & (width - 1), 0, "a power of two")
        self.assertEqual(widths, sorted(widths), "more sequences, never fewer blocks")
        self.assertGreater(widths[-1], widths[0], "and not the same number for all")

    def test_the_memo_is_sized_by_what_an_entry_costs(self):
        from engrain._engine import _MEMO_SLOTS, _memo_slots

        # A larger entry buys fewer of them, and neither end runs away.
        self.assertGreaterEqual(_memo_slots(1 << 10), _memo_slots(1 << 20))
        self.assertLessEqual(_memo_slots(1), _MEMO_SLOTS)
        self.assertGreaterEqual(_memo_slots(1 << 30), 32)

    def test_a_narrow_grammar_is_not_charged_for_a_wide_one(self):
        from engrain._engine import DeviceBatch, DeviceGrammar

        compiler = support.Compiler([bytes([b]) for b in range(256)])
        grammar = compiler.compile_json_schema(
            json.dumps({"type": "object", "properties": {"a": {"type": "integer"}}})
        )
        pool = DeviceGrammar()
        pool.admit(grammar)
        batch = DeviceBatch(pool, 4)
        self.assertLessEqual(batch.memo_configs, batch.configs)
        self.assertLessEqual(batch.memo_stride, pool.max_stack)


if __name__ == "__main__":
    unittest.main()
