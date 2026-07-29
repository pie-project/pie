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

from gpugrammar import _gpugrammar

try:
    import torch

    HAVE_CUDA = torch.cuda.is_available()
except Exception:  # noqa: BLE001
    HAVE_CUDA = False


class TheBuildProducedKernels(unittest.TestCase):
    """Fails when nvcc was missing at build time, which is otherwise silent."""

    def test_a_fatbin_is_embedded(self):
        self.assertTrue(
            _gpugrammar.cuda_available(),
            "no CUDA kernels in this build; was nvcc found when it was compiled?",
        )
        # Five architectures of SASS plus PTX. A few hundred bytes would mean
        # an empty or truncated fatbin that still technically exists.
        self.assertGreater(_gpugrammar.cuda_fatbin_bytes(), 1024)


class AKernelReachesTheDevice(unittest.TestCase):
    def setUp(self):
        if not HAVE_CUDA:
            raise unittest.SkipTest("no CUDA device")
        if not _gpugrammar.cuda_available():
            raise unittest.SkipTest("this build has no CUDA kernels")
        self.count = 1024
        self.out = torch.zeros(self.count, dtype=torch.int32, device="cuda")

    def _launch(self, name, *scalars):
        _gpugrammar.cuda_launch(
            name,
            (self.count + 255) // 256,
            256,
            torch.cuda.current_stream().cuda_stream,
            [self.out.data_ptr()],
            list(scalars),
        )

    def test_an_eager_launch_computes_what_it_says(self):
        self._launch("gg_probe_identity", self.count, 7)
        torch.cuda.synchronize()
        want = torch.arange(self.count, dtype=torch.int32, device="cuda") + 7
        self.assertTrue(bool((self.out == want).all()))

    def test_a_launch_during_capture_is_recorded_and_not_run(self):
        # The property the engine exists for. If a launch ran during capture
        # instead of being recorded, the graph would be empty and every replay
        # would be a silent no-op - which looks exactly like a fast engine.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._launch("gg_probe_accumulate", self.count, 1)
        torch.cuda.synchronize()
        self.assertEqual(int(self.out[0]), 0, "the launch ran during capture")

        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(int(self.out[0]), 5, "the recorded graph did not replay")

    def test_the_whole_row_is_written_not_just_the_first(self):
        self._launch("gg_probe_identity", self.count, 0)
        torch.cuda.synchronize()
        self.assertEqual(int(self.out[self.count - 1]), self.count - 1)

    def test_a_kernel_that_does_not_exist_is_reported(self):
        with self.assertRaises(RuntimeError):
            self._launch("gg_no_such_kernel", self.count, 0)



class TheDifferentialHarnessCanFail(unittest.TestCase):
    """A comparison that has never failed is a comparison nobody has tested.

    `GPUGRAMMAR_BACKEND=differential` runs both backends on the same input and
    refuses to continue if they disagree. It is the only check that can catch a
    CUDA-only difference - the verifications compare a backend against the
    reference matcher, which finds a wrong answer but not one both backends
    would share. So it has to be shown to notice, by making it notice.
    """

    def setUp(self):
        if not HAVE_CUDA:
            raise unittest.SkipTest("no CUDA device")
        import json

        import gpugrammar

        vocabulary = [bytes([i]) for i in range(256)]
        self.engine = gpugrammar.Engine(vocabulary)
        self.grammar = self.engine.compile_json_schema(
            json.dumps({
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            })
        )

    def _batch_that_disagrees(self, path, field):
        from gpugrammar import _engine

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
        from gpugrammar import _engine

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
        from gpugrammar import _engine

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

        from gpugrammar import _engine

        held = os.environ.get("GPUGRAMMAR_BACKEND")
        try:
            for name in ("triton", "cuda", "differential"):
                os.environ["GPUGRAMMAR_BACKEND"] = name
                self.assertEqual(_engine._chosen_backend(), name)
            os.environ["GPUGRAMMAR_BACKEND"] = "cudaa"
            with self.assertRaises(ValueError):
                _engine._chosen_backend()
        finally:
            if held is None:
                os.environ.pop("GPUGRAMMAR_BACKEND", None)
            else:
                os.environ["GPUGRAMMAR_BACKEND"] = held

    def test_what_is_ported_is_reported(self):
        from gpugrammar import _engine

        # Empty while the port is starting. This exists so that a claim about
        # which paths are CUDA is checkable rather than a comment.
        self.assertIsInstance(_engine.ported(), frozenset)



class TheArenaStructDescribesThePool(unittest.TestCase):
    """Twenty-six pointers, packed by Python, read by a kernel.

    Every later kernel takes `const gg::Arena*` instead of twenty-six
    arguments, which is the fix for the thing the Triton launches do worst -
    246 argument slots across eleven launches, all of them `int32*`, none of
    them checkable. The cost of that fix is this: if Python packs the fields in
    a different order from the struct, a table read through the wrong field is
    still a valid pointer and still returns numbers. So the packing is checked
    against a kernel that reads back values the host can derive on its own.
    """

    SLOTS = 10

    def setUp(self):
        if not HAVE_CUDA or not _gpugrammar.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import gpugrammar

        self.engine = gpugrammar.Engine([bytes([i]) for i in range(256)])
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
        _gpugrammar.cuda_launch(
            "gg_arena_readback",
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
                json.dumps({
                    "type": "object",
                    "properties": {
                        f"p{n}": {"type": "string"} for n in range(size)
                    },
                })
            )
            if self.pool.revision != revision:
                break
        self.assertNotEqual(self.pool.revision, revision, "the pool never moved")
        self.assertFalse(
            bool(torch.equal(self.pool.arena_struct(), held)),
            "the struct still holds the addresses from before the pool moved",
        )



class TheCudaLocateAgreesWithTriton(unittest.TestCase):
    """`gg_locate` against `_locate_kernel`, entry for entry.

    The first ported kernel that does real work: arena lookups through a
    grammar's bases and CSR traversal, but not the reduction chain, which is
    why it is first. Four outputs, and all four matter - `found` is the answer,
    and the three `old_*` are the pre-advance state the candidate pass reads
    while the advance is overwriting the live one.
    """

    def setUp(self):
        if not HAVE_CUDA or not _gpugrammar.cuda_available():
            raise unittest.SkipTest("no CUDA device or no kernels in this build")
        import json

        import gpugrammar

        self.engine = gpugrammar.Engine([bytes([i]) for i in range(256)])
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
        from gpugrammar import _engine

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
            grammar.group_offsets, grammar.group_set_kind, grammar.group_set_offset,
            grammar.group_set_length, grammar.set_payload, grammar.verdict_offsets,
            grammar.verdicts, grammar.verdict_stride, raw.lexer_state, raw.stack,
            raw.depth, raw.config_count, raw.widest, raw.token, raw.grammar_of,
            grammar.bases, raw.live_offsets, raw.found, raw.old_lexer,
            raw.old_count, raw.old_stack,
            ROWS=rows, CONFIGS=raw.configs, GROUP_BLOCK=_engine._GROUP_BLOCK,
            SEARCH_STEPS=grammar.search_steps, STACK_STRIDE=grammar.max_stack,
            HAS_VERDICTS=grammar.has_verdicts, NO_GROUP=_engine._NO_GROUP,
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
        from gpugrammar import _engine

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


if __name__ == "__main__":
    unittest.main()
