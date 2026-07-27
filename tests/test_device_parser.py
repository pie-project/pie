"""The device parser against the reference matcher, as a test rather than a script.

Every claim this engine makes reduces to one thing: the mask on the device is
the mask the CPU matcher would have produced, bit for bit, at every step of a
real document. The optimisations that shrank the scratch by three orders of
magnitude are all of the form "compute less, differently", and each one is a
chance to compute less than the grammar allows - which does not crash, it
quietly forbids a legal token. So this checks bits, not shapes.

Skipped where there is no GPU, no Triton, or no compiled `gpugrammar`, since the
rest of the suite runs without them.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAS_CUDA = False

try:
    import gpugrammar

    HAS_GPUGRAMMAR = True
except Exception:  # noqa: BLE001
    HAS_GPUGRAMMAR = False

INSTANCES = Path(__file__).resolve().parents[1] / "results" / "jsonschemabench-instances.json"

# Enough vocabulary to make the token-to-terminal mapping non-trivial without
# pulling a tokenizer in: the pieces a JSON document is actually made of, plus
# multi-character tokens that span several terminals and one that can be read
# two ways.
VOCABULARY = [
    b"{", b"}", b"[", b"]", b":", b",", b'"', b" ", b"  ", b"\\n",
    b'{"', b'":', b'","', b'":"', b'"}', b"true", b"false", b"null",
    b"0", b"1", b"2", b"12", b"123", b"-", b".", b"e",
    b"a", b"b", b"c", b"name", b"id", b"value", b"items",
    b'"name"', b'"id"', b'"a"', b'"b"', b"ab", b"abc",
]


def _requirements():
    if not HAS_CUDA:
        raise unittest.SkipTest("no CUDA device")
    if not HAS_GPUGRAMMAR:
        raise unittest.SkipTest("gpugrammar is not built")


class DeviceParserAgreement(unittest.TestCase):
    """The device mask equals the CPU matcher's, step by step."""

    def setUp(self):
        _requirements()
        from gpu_lr1.device_parser import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = gpugrammar.Compiler(VOCABULARY)

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
            self.assertTrue(accepted, f"the matcher refused its own document at {steps}")
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
        from gpu_lr1.device_parser import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = gpugrammar.Compiler(VOCABULARY)
        self.schemas = [
            {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            {"type": "array", "items": {"type": "integer"}},
            {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
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
        batch.set_batch_configurations({i: matchers[i].configurations() for i in range(6)})
        batch.fill_mask()
        batch.capture()

        second = [2, 2, 1, 1, 0, 0]
        batch.set_grammars(second)
        matchers = [self.pool[second[i]].matcher(0) for i in range(6)]
        batch.set_batch_configurations({i: matchers[i].configurations() for i in range(6)})
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
        from gpu_lr1.device_parser import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = gpugrammar.Compiler(VOCABULARY)

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
        from gpu_lr1.device_parser import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = gpugrammar.Compiler(VOCABULARY)
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
        for token in tokens[3:5]:
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
        from gpu_lr1.device_parser import DeviceGrammar

        self.DeviceGrammar = DeviceGrammar
        self.compiler = gpugrammar.Compiler(VOCABULARY)
        self.schemas = [
            {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
            {"type": "array", "items": {"type": "integer"}},
            {"type": "object", "properties": {"b": {"type": "string"}}, "required": ["b"]},
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

    def test_a_graph_survives_admission_into_spare_capacity(self):
        pool = self.DeviceGrammar()
        compiled = [self._compile(0), self._compile(1)]
        for item in compiled:
            pool.admit(item)
        batch = pool.new_batch(2)
        batch.set_grammars([0, 1])
        matchers = [compiled[i].matcher(0) for i in (0, 1)]
        batch.set_batch_configurations({i: m.configurations() for i, m in enumerate(matchers)})
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

    def test_a_recorded_graph_is_not_replayed_after_compaction(self):
        pool = self.DeviceGrammar()
        compiled = [self._compile(index) for index in range(3)]
        for item in compiled:
            pool.admit(item)
        batch = pool.new_batch(2)
        batch.set_grammars([1, 2])
        matchers = [compiled[i].matcher(0) for i in (1, 2)]
        batch.set_batch_configurations({i: m.configurations() for i, m in enumerate(matchers)})
        batch.fill_mask()
        batch.capture()
        self.assertEqual(batch.recorded, pool.revision)

        pool.release(0)
        remap = pool.compact()
        self.assertNotEqual(batch.recorded, pool.revision)
        self.assertTrue(
            self._mask_matches(pool, batch, [remap[1], remap[2]], [compiled[1], compiled[2]])
        )


class CorpusAgreement(unittest.TestCase):
    """The same check on real schemas, if the corpus is present."""

    def setUp(self):
        _requirements()
        if not INSTANCES.exists():
            raise unittest.SkipTest("the JSONSchemaBench corpus is not in results/")

    def test_first_schemas_agree_byte_by_byte(self):
        from gpu_lr1.device_parser import DeviceGrammar

        instances = json.loads(INSTANCES.read_text())["instances"]
        # Bytes rather than a tokenizer: the corpus check that uses the real
        # vocabulary lives in the benchmark, and a unit test should not download
        # a model.
        vocabulary = [bytes([value]) for value in range(256)]
        compiler = gpugrammar.Compiler(vocabulary)
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


if __name__ == "__main__":
    unittest.main()
