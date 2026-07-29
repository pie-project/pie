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


if __name__ == "__main__":
    unittest.main()
