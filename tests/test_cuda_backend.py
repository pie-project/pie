"""The CUDA backend's build path, which is the thing that silently breaks.

Not a test of any kernel's logic - there is no logic yet. These check that a
kernel written in `.cu` reaches the GPU: nvcc compiled it, `build.rs` put the
fatbin in the shared object, the driver loaded it from memory, and a launch
made on PyTorch's own stream during graph capture was *recorded* rather than
run. That last one is the property the whole architecture rests on, and it is
worth a test before any parser code depends on it.
"""

from __future__ import annotations

import unittest

from engrain import _engrain

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAVE_CUDA = False


class TheBuildProducedKernels(unittest.TestCase):
    """Fails when nvcc was missing at build time, which is otherwise silent."""

    def test_a_fatbin_is_embedded(self):
        self.assertTrue(
            _engrain.cuda_available(),
            "no CUDA kernels in this build; was nvcc found when it was compiled?",
        )
        # Five architectures of SASS plus PTX. A few hundred bytes would mean
        # an empty or truncated fatbin that still technically exists.
        self.assertGreater(_engrain.cuda_fatbin_bytes(), 1024)


class AKernelReachesTheDevice(unittest.TestCase):
    def setUp(self):
        if not HAVE_CUDA:
            raise unittest.SkipTest("no CUDA device")
        if not _engrain.cuda_available():
            raise unittest.SkipTest("this build has no CUDA kernels")
        self.count = 1024
        self.out = torch.zeros(self.count, dtype=torch.int32, device="cuda")

    def _launch(self, name, *scalars):
        _engrain.cuda_launch(
            name,
            (self.count + 255) // 256,
            256,
            torch.cuda.current_stream().cuda_stream,
            [self.out.data_ptr()],
            list(scalars),
        )

    def test_an_eager_launch_computes_what_it_says(self):
        self._launch("en_probe_identity", self.count, 7)
        torch.cuda.synchronize()
        want = torch.arange(self.count, dtype=torch.int32, device="cuda") + 7
        self.assertTrue(bool((self.out == want).all()))

    def test_a_launch_during_capture_is_recorded_and_not_run(self):
        # The property the engine exists for. If a launch ran during capture
        # instead of being recorded, the graph would be empty and every replay
        # would be a silent no-op - which looks exactly like a fast engine.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._launch("en_probe_accumulate", self.count, 1)
        torch.cuda.synchronize()
        self.assertEqual(int(self.out[0]), 0, "the launch ran during capture")

        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(self.out[0]), 5, "the recorded graph did not replay")

    def test_the_whole_row_is_written_not_just_the_first(self):
        self._launch("en_probe_identity", self.count, 0)
        torch.cuda.synchronize()
        self.assertEqual(int(self.out[self.count - 1]), self.count - 1)

    def test_a_kernel_that_does_not_exist_is_reported(self):
        with self.assertRaises(RuntimeError):
            self._launch("en_no_such_kernel", self.count, 0)


class TheDifferentialHarnessCanFail(unittest.TestCase):
    """A comparison that has never failed is a comparison nobody has tested.

    `ENGRAIN_BACKEND=differential` runs both backends on the same input and
    refuses to continue if they disagree. It is the only check that can catch a
    CUDA-only difference - the verifications compare a backend against the
    reference matcher, which finds a wrong answer but not one both backends
    would share. So it has to be shown to notice, by making it notice.
    """

    def setUp(self):
        if not HAVE_CUDA:
            raise unittest.SkipTest("no CUDA device")
        import json

        import support

        vocabulary = [bytes([i]) for i in range(256)]
        self.engine = support.Engine(vocabulary)
        self.grammar = self.engine.compile_json_schema(
            json.dumps(
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}},
                    "required": ["a"],
                }
            )
        )

    def _batch_that_disagrees(self, path, field):
        from engrain import _engine

        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammar] * 4)
        raw = batch.raw
        name = "_fill_cuda" if path == "fill" else "_advance_cuda"
        original = getattr(raw, name)

        def wrong():
            result = original()
            getattr(raw, field)[0] += 1  # one entry, of one tensor
            return result

        setattr(raw, name, wrong)
        raw.backend = _engine._DIFFERENTIAL
        return raw

    def test_a_difference_in_the_mask_is_caught(self):
        raw = self._batch_that_disagrees("fill", "mask")
        with self.assertRaises(AssertionError) as caught:
            raw._fill()
        self.assertIn("mask", str(caught.exception))

    def test_a_difference_in_the_parse_state_is_caught(self):
        raw = self._batch_that_disagrees("advance", "lexer_state")
        with self.assertRaises(AssertionError) as caught:
            raw.advance(torch.zeros(4, dtype=torch.int32, device="cuda"))
        self.assertIn("lexer_state", str(caught.exception))

    def test_agreeing_backends_do_not_raise(self):
        from engrain import _engine

        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammar] * 4)
        batch.raw.backend = _engine._DIFFERENTIAL
        # Both sides are Triton until a kernel is ported, so this asserts the
        # snapshot and restore are complete rather than that the port is right.
        # It is still the load-bearing half: an incomplete restore would make
        # every later comparison meaningless, and it did - the rollback history
        # was missing from the first version and the rollback checks caught it.
        batch.fill_mask()
        batch.advance(torch.zeros(4, dtype=torch.int32, device="cuda"))
        batch.fill_mask()

    def test_differential_refuses_to_be_captured(self):
        from engrain import _engine

        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammar] * 4)
        batch.raw.backend = _engine._DIFFERENTIAL
        # Recording would capture whichever backend ran last and drop the
        # comparison, which is a graph that silently checks nothing.
        with self.assertRaises(RuntimeError):
            batch.capture()


class TheBackendIsSelectable(unittest.TestCase):
    def test_only_the_three_names_are_accepted(self):
        import os

        from engrain import _engine

        held = os.environ.get("ENGRAIN_BACKEND")
        try:
            for name in ("triton", "cuda", "differential"):
                os.environ["ENGRAIN_BACKEND"] = name
                self.assertEqual(_engine._chosen_backend(), name)
            os.environ["ENGRAIN_BACKEND"] = "cudaa"
            with self.assertRaises(ValueError):
                _engine._chosen_backend()
        finally:
            if held is None:
                os.environ.pop("ENGRAIN_BACKEND", None)
            else:
                os.environ["ENGRAIN_BACKEND"] = held

    def test_what_is_ported_is_reported(self):
        from engrain import _engine

        # Empty while the port is starting. This exists so that a claim about
        # which paths are CUDA is checkable rather than a comment.
        self.assertIsInstance(_engine.ported(), frozenset)


class TheArenaStructDescribesThePool(unittest.TestCase):
    """Twenty-six pointers, packed by Python, read by a kernel.

    Every later kernel takes `const en::Arena*` instead of twenty-six
    arguments, which is the fix for the thing the Triton launches do worst -
    246 argument slots across eleven launches, all of them `int32*`, none of
    them checkable. The cost of that fix is this: if Python packs the fields in
    a different order from the struct, a table read through the wrong field is
    still a valid pointer and still returns numbers. So the packing is checked
    against a kernel that reads back values the host can derive on its own.
    """

    SLOTS = 10

    def setUp(self):
        if not HAVE_CUDA or not _engrain.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import support

        self.engine = support.Engine([bytes([i]) for i in range(256)])
        # Three different shapes, so a base that happened to be zero for one
        # grammar cannot hide a wrong field.
        self.grammars = [
            self.engine.compile_json_schema(json.dumps(schema))
            for schema in (
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
                {"type": "array", "items": {"type": "boolean"}},
            )
        ]
        self.pool = self.engine.pool

    def _readback(self, sequences):
        batch = self.pool.new_batch(sequences)
        batch.set_grammars(
            [self.engine.admit(self.grammars[i % 3]) for i in range(sequences)]
        )
        out = torch.zeros(sequences * self.SLOTS, dtype=torch.int32, device="cuda")
        _engrain.cuda_launch(
            "en_arena_readback",
            (sequences + 31) // 32,
            32,
            torch.cuda.current_stream().cuda_stream,
            [
                self.pool.arena_struct().data_ptr(),
                batch.grammar_of.data_ptr(),
                out.data_ptr(),
            ],
            [sequences, self.SLOTS],
        )
        torch.cuda.synchronize()
        return batch, out.cpu().reshape(sequences, self.SLOTS)

    def test_the_two_sides_agree_on_the_struct_size(self):
        _, seen = self._readback(2)
        self.assertEqual(int(seen[0][8]), self.pool.arena_slots)
        self.assertEqual(int(seen[0][9]), 20)  # NBASES

    def test_every_table_is_read_through_the_base_the_host_uses(self):
        sequences = 6
        batch, seen = self._readback(sequences)
        bases = self.pool.bases.cpu()
        nbases = 20
        wrong = []
        for index in range(sequences):
            grammar = int(batch.grammar_of[index])
            at = grammar * nbases
            group, action, goto = (
                int(bases[at + 0]),
                int(bases[at + 8]),
                int(bases[at + 10]),
            )
            want = {
                0: group,
                1: action,
                2: goto,
                3: int(self.pool.group_offsets[group]),
                4: int(self.pool.group_offsets[group + 1]),
                5: int(self.pool.action_offsets[action]),
                6: int(self.pool.goto_offsets[goto]),
                7: int(self.pool.reading_offsets[int(bases[at + 3])]),
            }
            for slot, expected in want.items():
                if int(seen[index][slot]) != expected:
                    wrong.append(
                        f"sequence {index} grammar {grammar} slot {slot}: "
                        f"kernel {int(seen[index][slot])} != host {expected}"
                    )
        self.assertEqual(wrong, [])

    def test_the_struct_is_rebuilt_when_the_pool_moves(self):
        import json

        first = self.pool.arena_struct()
        held = first.clone()
        revision = self.pool.revision
        # Admitting until an array has to grow is what moves the addresses -
        # the same event that invalidates a recorded graph.
        for size in range(4, 40):
            self.engine.compile_json_schema(
                json.dumps(
                    {
                        "type": "object",
                        "properties": {
                            f"p{n}": {"type": "string"} for n in range(size)
                        },
                    }
                )
            )
            if self.pool.revision != revision:
                break
        self.assertNotEqual(self.pool.revision, revision, "the pool never moved")
        self.assertFalse(
            bool(torch.equal(self.pool.arena_struct(), held)),
            "the struct still holds the addresses from before the pool moved",
        )


class TheCudaLocateAgreesWithTriton(unittest.TestCase):
    """`en_locate` against `_locate_kernel`, entry for entry.

    The first ported kernel that does real work: arena lookups through a
    grammar's bases and CSR traversal, but not the reduction chain, which is
    why it is first. Four outputs, and all four matter - `found` is the answer,
    and the three `old_*` are the pre-advance state the candidate pass reads
    while the advance is overwriting the live one.
    """

    def setUp(self):
        if not HAVE_CUDA or not _engrain.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import support

        self.engine = support.Engine([bytes([i]) for i in range(256)])
        self.grammars = [
            self.engine.compile_json_schema(json.dumps(schema))
            for schema in (
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                },
                {"type": "array", "items": {"type": "boolean"}},
            )
        ]

    def _both(self, sequences, token_seed):
        from engrain import _engine

        batch = self.engine.batch(size=sequences)
        batch.set_grammars([self.grammars[i % 3] for i in range(sequences)])
        raw = batch.raw
        grammar = raw.grammar
        rows = raw.batch * raw.configs

        torch.manual_seed(token_seed)
        raw.token.copy_(
            torch.randint(0, 256, (sequences,), dtype=torch.int32, device="cuda")
        )
        raw._count_and_scan(grammar, rows, raw.counts, raw.live_offsets, skip=0, unit=1)
        torch.cuda.synchronize()

        raw.found.fill_(_engine._NO_GROUP)
        _engine._locate_kernel[(raw.sweep_blocks,)](
            grammar.group_offsets,
            grammar.group_set_kind,
            grammar.group_set_offset,
            grammar.group_set_length,
            grammar.set_payload,
            grammar.verdict_offsets,
            grammar.verdicts,
            grammar.verdict_stride,
            raw.lexer_state,
            raw.stack,
            raw.depth,
            raw.config_count,
            raw.widest,
            raw.token,
            raw.grammar_of,
            grammar.bases,
            raw.live_offsets,
            raw.found,
            raw.old_lexer,
            raw.old_count,
            raw.old_stack,
            ROWS=rows,
            CONFIGS=raw.configs,
            GROUP_BLOCK=_engine._GROUP_BLOCK,
            SEARCH_STEPS=grammar.search_steps,
            STACK_STRIDE=grammar.max_stack,
            HAS_VERDICTS=grammar.has_verdicts,
            NO_GROUP=_engine._NO_GROUP,
            VOCAB=grammar.vocab_size,
        )
        torch.cuda.synchronize()
        theirs = {
            name: getattr(raw, name).clone()
            for name in ("found", "old_lexer", "old_count", "old_stack")
        }

        raw.found.fill_(_engine._NO_GROUP)
        raw.old_lexer.zero_()
        raw.old_count.zero_()
        raw.old_stack.zero_()
        raw._locate_cuda(grammar, rows)
        torch.cuda.synchronize()
        return raw, theirs

    def test_the_two_kernels_produce_the_same_four_arrays(self):
        for seed in range(8):
            raw, theirs = self._both(16, seed)
            for name, expected in theirs.items():
                mine = getattr(raw, name)
                self.assertTrue(
                    bool(torch.equal(mine, expected)),
                    f"seed {seed}: {name} differs in "
                    f"{int((mine != expected).sum())} of {mine.numel()}",
                )

    def test_it_holds_at_a_serving_batch_size(self):
        raw, theirs = self._both(256, 99)
        for name, expected in theirs.items():
            self.assertTrue(bool(torch.equal(getattr(raw, name), expected)), name)

    def test_a_token_no_group_holds_is_reported_as_no_group(self):
        from engrain import _engine

        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammars[0]] * 4)
        raw = batch.raw
        rows = raw.batch * raw.configs
        # Past the vocabulary, so nothing can hold it.
        raw.token.fill_(255)
        raw._count_and_scan(
            raw.grammar, rows, raw.counts, raw.live_offsets, skip=0, unit=1
        )
        raw.found.zero_()
        raw._locate_cuda(raw.grammar, rows)
        torch.cuda.synchronize()
        live = int(raw.live_offsets[rows])
        self.assertGreater(live, 0, "nothing was live, so nothing was tested")
        self.assertEqual(int(raw.found[0]), _engine._NO_GROUP)


class TheCudaCommitAgreesWithTriton(unittest.TestCase):
    """`en_commit` against `_commit_kernel`, on candidates a real advance made.

    The collection is serial on purpose: the reference matcher deduplicates in
    a particular order and stops at its ceiling, so a parallel collection
    producing the same *set* could still produce a different *prefix*. Equality
    with the matcher is the verification strategy, so the order is part of the
    contract and not an implementation detail.
    """

    def setUp(self):
        if not HAVE_CUDA or not _engrain.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import support

        self.engine = support.Engine([bytes([i]) for i in range(256)])
        self.grammars = [
            self.engine.compile_json_schema(json.dumps(schema))
            for schema in (
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                },
                {"type": "array", "items": {"type": "boolean"}},
            )
        ]

    OUTPUTS = (
        "lexer_state",
        "stack",
        "depth",
        "config_count",
        "terminated",
        "overflow",
        "widest",
    )

    def test_both_commits_produce_the_same_configuration_set(self):
        sequences = 16
        batch = self.engine.batch(size=sequences)
        batch.set_grammars([self.grammars[i % 3] for i in range(sequences)])
        raw = batch.raw
        grammar = raw.grammar
        candidates = 0
        for step in range(6):
            # A token each grammar admits, so there are candidates to collect.
            tokens = []
            for index in range(sequences):
                allowed = self.grammars[index % 3].matcher(0).allowed_tokens()
                tokens.append(allowed[step % len(allowed)] if allowed else 0)
            raw.token.copy_(torch.tensor(tokens, dtype=torch.int32, device="cuda"))
            raw._advance_prepare_triton(grammar, raw.batch * raw.configs)
            torch.cuda.synchronize()
            candidates += int(raw.cand_count.sum())

            before = {name: getattr(raw, name).clone() for name in self.OUTPUTS}
            raw._commit_triton(grammar)
            torch.cuda.synchronize()
            theirs = {name: getattr(raw, name).clone() for name in self.OUTPUTS}

            for name, value in before.items():
                getattr(raw, name).copy_(value)
            raw._commit_cuda(grammar)
            torch.cuda.synchronize()
            for name in self.OUTPUTS:
                mine = getattr(raw, name)
                self.assertTrue(
                    bool(torch.equal(mine, theirs[name])),
                    f"step {step}: {name} differs in "
                    f"{int((mine != theirs[name]).sum())} of {mine.numel()}",
                )
            for name, value in theirs.items():
                getattr(raw, name).copy_(value)
        self.assertGreater(candidates, 0, "no candidate was ever collected")

    def test_a_refused_token_terminates_rather_than_emptying_the_set(self):
        batch = self.engine.batch(size=4)
        batch.set_grammars([self.grammars[0]] * 4)
        raw = batch.raw
        grammar = raw.grammar
        # 255 is not in any group of a JSON string's start state.
        raw.token.fill_(255)
        raw._advance_prepare_triton(grammar, raw.batch * raw.configs)
        raw._commit_cuda(grammar)
        torch.cuda.synchronize()
        # A mask filled from an empty set would allow everything, so the set is
        # left alone and the sequence is marked instead.
        self.assertEqual(raw.terminated[0].item(), 1)
        self.assertGreater(int(raw.config_count[0]), 0)


class TheCudaCandidateAgreesWithTriton(unittest.TestCase):
    """`en_candidate` against `_candidate_kernel`.

    The largest kernel of the port and the one it is for: a thread replays a
    configuration rather than a block, which is the 3.42x lever, and the
    reduction chain is a device function called from both the reading walk and
    the pending probe rather than written out twice.

    Only the *defined* part of each array is compared. `cand_lexer` and its
    friends hold a row's candidates in `[0, cand_count[row])`; past that they
    are whatever the previous step left, and the two kernels leave different
    rubbish there.
    """

    def setUp(self):
        if not HAVE_CUDA or not _engrain.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import support

        self.engine = support.Engine([bytes([i]) for i in range(256)])
        self.grammars = [
            self.engine.compile_json_schema(json.dumps(schema))
            for schema in (
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                },
                {"type": "array", "items": {"type": "boolean"}},
            )
        ]

    SNAPSHOT = (
        "cand_count",
        "cand_lexer",
        "cand_depth",
        "cand_floor",
        "cand_window",
        "overflow",
    )

    def _defined_differences(self, raw, theirs, max_readings, window):
        counts = theirs["cand_count"]
        wrong = 0
        for row in range(counts.numel()):
            made = int(counts[row])
            if made == 0:
                continue
            base = row * max_readings
            for field in ("cand_lexer", "cand_depth", "cand_floor"):
                mine = getattr(raw, field)[base : base + made]
                wrong += int((mine != theirs[field][base : base + made]).sum())
            for index in range(made):
                depth = int(theirs["cand_depth"][base + index])
                floor = int(theirs["cand_floor"][base + index])
                live = max(0, depth - floor)
                at = (base + index) * window
                mine = raw.cand_window[at : at + live]
                wrong += int((mine != theirs["cand_window"][at : at + live]).sum())
        return wrong

    def test_both_kernels_produce_the_same_candidates(self):
        made = 0
        for sequences in (4, 16):
            batch = self.engine.batch(size=sequences)
            batch.set_grammars([self.grammars[i % 3] for i in range(sequences)])
            raw = batch.raw
            grammar = raw.grammar
            rows = raw.batch * raw.configs
            matchers = [self.grammars[i % 3].matcher(0) for i in range(sequences)]
            for step in range(6):
                tokens = []
                for index in range(sequences):
                    allowed = matchers[index].allowed_tokens()
                    tokens.append(allowed[step % len(allowed)] if allowed else 0)
                raw.token.copy_(torch.tensor(tokens, dtype=torch.int32, device="cuda"))
                raw._count_and_scan(
                    grammar, rows, raw.live_counts, raw.live_offsets, skip=0, unit=1
                )
                raw._locate_triton(grammar, rows)
                torch.cuda.synchronize()
                before = {n: getattr(raw, n).clone() for n in self.SNAPSHOT}

                raw._candidate_triton(grammar, rows)
                torch.cuda.synchronize()
                theirs = {n: getattr(raw, n).clone() for n in self.SNAPSHOT}
                made += int(theirs["cand_count"].sum())

                for name, value in before.items():
                    getattr(raw, name).copy_(value)
                raw._candidate_cuda(grammar, rows)
                torch.cuda.synchronize()

                self.assertTrue(
                    bool(torch.equal(raw.cand_count, theirs["cand_count"])),
                    f"batch {sequences} step {step}: "
                    f"{int((raw.cand_count != theirs['cand_count']).sum())} rows differ",
                )
                self.assertTrue(bool(torch.equal(raw.overflow, theirs["overflow"])))
                self.assertEqual(
                    self._defined_differences(
                        raw, theirs, raw.max_readings, grammar.window
                    ),
                    0,
                )

                for name, value in theirs.items():
                    getattr(raw, name).copy_(value)
                raw._commit_triton(grammar)
                torch.cuda.synchronize()
                for index in range(sequences):
                    matchers[index].accept_token(tokens[index])
        self.assertGreater(made, 0, "no candidate was ever produced")

    def test_the_mask_and_the_landing_ask_different_questions(self):
        # The one real semantic difference between the two copies of the chain
        # in the Triton engine, and the bug this port made: for the mask a
        # shift and an accept both mean "readable" and neither is recorded; for
        # a candidate a shift has to be pushed and an accept means the parse is
        # finished and cannot read on. Porting the mask's answer into the
        # candidate path produced zero candidates where there should have been
        # two, on a grammar with no conflicts at all.
        sequences = 8
        batch = self.engine.batch(size=sequences)
        batch.set_grammars([self.grammars[1]] * sequences)
        raw = batch.raw
        grammar = raw.grammar
        rows = raw.batch * raw.configs
        raw.token.fill_(self.grammars[1].matcher(0).allowed_tokens()[0])
        raw._count_and_scan(
            grammar, rows, raw.live_counts, raw.live_offsets, skip=0, unit=1
        )
        raw._locate_triton(grammar, rows)
        raw._candidate_cuda(grammar, rows)
        torch.cuda.synchronize()
        self.assertGreater(
            int(raw.cand_count.sum()), 0, "a readable token produced no candidate"
        )


class TheCudaMaskSweepAgreesWithTriton(unittest.TestCase):
    """`en_mask` against `_mask_kernel`, over real parse states.

    The sweep is the other caller of `replay_chain`, so with the candidate
    already ported this kernel is mostly the group enumeration and the verdict
    shortcut. Three outputs, and `row_floor` is the subtle one: it is the
    lowest stack entry any reading looked at, which is what the cross-step memo
    keys on, so getting it too shallow would silently reuse a mask.
    """

    def setUp(self):
        if not HAVE_CUDA or not _engrain.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import support

        self.engine = support.Engine([bytes([i]) for i in range(256)])
        self.grammars = [
            self.engine.compile_json_schema(json.dumps(schema))
            for schema in (
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
                    "required": ["a"],
                },
                {"type": "array", "items": {"type": "boolean"}},
            )
        ]

    def test_both_sweeps_admit_the_same_groups_and_read_as_deep(self):
        fields = ("admitted", "high_water", "row_floor", "overflow")
        items = 0
        for sequences in (4, 16):
            batch = self.engine.batch(size=sequences)
            batch.set_grammars([self.grammars[i % 3] for i in range(sequences)])
            raw = batch.raw
            grammar = raw.grammar
            rows = raw.batch * raw.configs
            matchers = [self.grammars[i % 3].matcher(0) for i in range(sequences)]
            for step in range(5):
                raw._count_and_scan(
                    grammar, rows, raw.counts, raw.work_offsets, skip=1, unit=0
                )
                raw.row_floor.fill_(2**30)
                raw.high_water.zero_()
                torch.cuda.synchronize()
                before = {name: getattr(raw, name).clone() for name in fields}

                raw._mask_triton(grammar, rows)
                torch.cuda.synchronize()
                theirs = {name: getattr(raw, name).clone() for name in fields}
                live = int(raw.work_offsets[rows])
                items += live

                for name, value in before.items():
                    getattr(raw, name).copy_(value)
                raw._mask_cuda(grammar, rows)
                torch.cuda.synchronize()
                for name in fields:
                    mine = getattr(raw, name)
                    expected = theirs[name]
                    if name == "admitted":
                        # Only the live prefix is written; past it is whatever
                        # the previous step left, deliberately - clearing the
                        # ceiling was 13 MB a step at batch 512.
                        mine, expected = mine[:live], expected[:live]
                    self.assertTrue(
                        bool(torch.equal(mine, expected)),
                        f"batch {sequences} step {step}: {name} differs in "
                        f"{int((mine != expected).sum())} of {mine.numel()}",
                    )

                for name, value in theirs.items():
                    getattr(raw, name).copy_(value)
                tokens = []
                for index in range(sequences):
                    allowed = matchers[index].allowed_tokens()
                    tokens.append(allowed[step % len(allowed)] if allowed else 0)
                raw.token.copy_(torch.tensor(tokens, dtype=torch.int32, device="cuda"))
                raw._advance_triton()
                torch.cuda.synchronize()
                for index in range(sequences):
                    matchers[index].accept_token(tokens[index])
        self.assertGreater(items, 0, "the sweep never had anything to do")

    def test_the_two_backends_fill_the_same_mask(self):
        batch = self.engine.batch(size=16)
        batch.set_grammars([self.grammars[i % 3] for i in range(16)])
        raw = batch.raw
        theirs = raw._fill_triton().clone()
        raw.memo_hash.fill_(-1)  # or the second fill answers from the first
        mine = raw._fill_cuda().clone()
        self.assertTrue(bool(torch.equal(mine, theirs)))


if __name__ == "__main__":
    unittest.main()
