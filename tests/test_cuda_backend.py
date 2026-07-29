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


if __name__ == "__main__":
    unittest.main()
