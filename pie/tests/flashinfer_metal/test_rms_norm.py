"""Tests for custom Metal RMS norm kernels."""

import time

import pytest
import torch
import torch.nn.functional as fun

from flashinfer_metal._compiler import MetalCompiler


@pytest.fixture
def compiler():
    return MetalCompiler()


def reference_rms_norm(input, weight, eps):
    """Reference: PyTorch functional rms_norm."""
    return fun.rms_norm(input, normalized_shape=[input.shape[-1]], weight=weight, eps=eps)


class TestRmsNorm:
    """Test standalone RMS norm kernel."""

    @pytest.mark.parametrize("H", [2880, 3072, 4096])
    def test_correctness_bf16(self, compiler, H):
        torch.manual_seed(42)
        eps = 1e-5
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        compiler.run_rms_norm(x, w, out, eps)
        ref = reference_rms_norm(x, w, eps)

        # Compare against bf16-rounded reference (kernel writes bf16)
        torch.testing.assert_close(
            out.float().cpu(), ref.float().cpu(), atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.parametrize("H", [2880, 3072, 4096])
    def test_correctness_f16(self, compiler, H):
        torch.manual_seed(42)
        eps = 1e-5
        x = torch.randn(1, H, dtype=torch.float16, device="mps")
        w = torch.randn(H, dtype=torch.float16, device="mps")
        out = torch.empty(1, H, dtype=torch.float16, device="mps")

        compiler.run_rms_norm(x, w, out, eps)
        ref = reference_rms_norm(x, w, eps)

        torch.testing.assert_close(
            out.float().cpu(), ref.float().cpu(), atol=1e-2, rtol=1e-2,
        )

    def test_writes_to_preallocated(self, compiler):
        """Verify kernel writes to provided output buffer (no allocation)."""
        H = 2880
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        ptr = out.data_ptr()

        compiler.run_rms_norm(x, w, out, 1e-5)

        assert out.data_ptr() == ptr, "Kernel should write to pre-allocated buffer"

    @pytest.mark.parametrize("seed", [0, 1, 42, 123])
    def test_deterministic(self, compiler, seed):
        torch.manual_seed(seed)
        H = 2880
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        out1 = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        out2 = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        compiler.run_rms_norm(x, w, out1, 1e-5)
        compiler.run_rms_norm(x, w, out2, 1e-5)

        assert torch.equal(out1.cpu(), out2.cpu())


class TestResidualRmsNorm:
    """Test fused residual-add + RMS norm kernel."""

    @pytest.mark.parametrize("H", [2880, 3072, 4096])
    def test_correctness_bf16(self, compiler, H):
        torch.manual_seed(42)
        eps = 1e-5
        a = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        b = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        res_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        norm_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        compiler.run_residual_rms_norm(a, b, w, res_out, norm_out, eps)

        # Reference
        ref_res = a + b
        ref_norm = reference_rms_norm(ref_res, w, eps)

        torch.testing.assert_close(
            res_out.float().cpu(), ref_res.float().cpu(), atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            norm_out.float().cpu(), ref_norm.float().cpu(), atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.parametrize("H", [2880, 3072, 4096])
    def test_correctness_f16(self, compiler, H):
        torch.manual_seed(42)
        eps = 1e-5
        a = torch.randn(1, H, dtype=torch.float16, device="mps")
        b = torch.randn(1, H, dtype=torch.float16, device="mps")
        w = torch.randn(H, dtype=torch.float16, device="mps")
        res_out = torch.empty(1, H, dtype=torch.float16, device="mps")
        norm_out = torch.empty(1, H, dtype=torch.float16, device="mps")

        compiler.run_residual_rms_norm(a, b, w, res_out, norm_out, eps)

        ref_res = a + b
        ref_norm = reference_rms_norm(ref_res, w, eps)

        torch.testing.assert_close(
            res_out.float().cpu(), ref_res.float().cpu(), atol=1e-2, rtol=1e-2,
        )
        torch.testing.assert_close(
            norm_out.float().cpu(), ref_norm.float().cpu(), atol=1e-2, rtol=1e-2,
        )

    def test_writes_to_preallocated(self, compiler):
        """Verify kernel writes to provided output buffers (no allocation)."""
        H = 2880
        a = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        b = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        res_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        norm_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        res_ptr = res_out.data_ptr()
        norm_ptr = norm_out.data_ptr()

        compiler.run_residual_rms_norm(a, b, w, res_out, norm_out, 1e-5)

        assert res_out.data_ptr() == res_ptr
        assert norm_out.data_ptr() == norm_ptr

    def test_residual_equals_a_plus_b(self, compiler):
        """Verify the residual output is exactly a + b."""
        torch.manual_seed(99)
        H = 2880
        a = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        b = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        res_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        norm_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        compiler.run_residual_rms_norm(a, b, w, res_out, norm_out, 1e-5)

        ref_res = (a.float() + b.float()).bfloat16()
        torch.testing.assert_close(res_out.cpu(), ref_res.cpu(), atol=0, rtol=0)


class TestRmsNormMicrobench:
    """Microbenchmarks: custom Metal kernel vs fun.rms_norm (MPSGraph)."""

    def _bench(self, fn, warmup=50, iters=200):
        for _ in range(warmup):
            fn()
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6  # microseconds

    def test_bench_standalone_rms_norm(self, compiler):
        """Benchmark: custom kernel vs fun.rms_norm for H=2880, n=1."""
        H = 2880
        eps = 1e-5
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        us_custom = self._bench(lambda: compiler.run_rms_norm(x, w, out, eps))
        us_mpsgraph = self._bench(lambda: fun.rms_norm(x, [H], w, eps))

        print(f"\n  Standalone RMS norm H={H}:")
        print(f"    Custom Metal:  {us_custom:.1f} μs")
        print(f"    fun.rms_norm:  {us_mpsgraph:.1f} μs")
        print(f"    Speedup:       {us_mpsgraph / us_custom:.2f}×")

    def test_bench_fused_residual_rms_norm(self, compiler):
        """Benchmark: fused residual+norm vs separate (add + rms_norm)."""
        H = 2880
        eps = 1e-5
        a = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        b = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        res_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        norm_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        us_fused = self._bench(
            lambda: compiler.run_residual_rms_norm(a, b, w, res_out, norm_out, eps)
        )

        def separate():
            r = a + b
            return fun.rms_norm(r, [H], w, eps)

        us_separate = self._bench(separate)

        print(f"\n  Fused residual + RMS norm H={H}:")
        print(f"    Fused Metal:   {us_fused:.1f} μs")
        print(f"    Separate:      {us_separate:.1f} μs (add + rms_norm)")
        print(f"    Speedup:       {us_separate / us_fused:.2f}×")

    def test_bench_per_layer_savings(self, compiler):
        """Estimate total savings per decode step (24 layers × 2 fused ops)."""
        H = 2880
        eps = 1e-5
        a = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        b = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        res_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        norm_out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        us_fused = self._bench(
            lambda: compiler.run_residual_rms_norm(a, b, w, res_out, norm_out, eps)
        )

        def separate():
            r = a + b
            return fun.rms_norm(r, [H], w, eps)

        us_separate = self._bench(separate)

        saved_per_layer_us = (us_separate - us_fused) * 2  # 2 fused ops per layer
        total_saved_us = saved_per_layer_us * 24  # 24 layers
        total_saved_ms = total_saved_us / 1000

        print(f"\n  Projected decode step savings:")
        print(f"    Saved per layer: {saved_per_layer_us:.1f} μs (2 × fused ops)")
        print(f"    Total (24 layers): {total_saved_ms:.2f} ms")

    def test_bench_prealloc_add(self, compiler):
        """Benchmark: torch.add(out=) vs regular add to measure allocation savings."""
        H = 2880
        a = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        b = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        us_alloc = self._bench(lambda: a + b)
        us_prealloc = self._bench(lambda: torch.add(a, b, out=out))

        # Measure a no-op cast (same dtype)
        us_noop_cast = self._bench(lambda: a.to(torch.bfloat16))
        # Measure actual cast (bf16 → f32)
        us_cast = self._bench(lambda: a.to(torch.float32))

        print(f"\n  Pre-allocation savings (H={H}):")
        print(f"    a + b (allocates):      {us_alloc:.1f} μs")
        print(f"    torch.add(out=):        {us_prealloc:.1f} μs")
        print(f"    Savings per add:        {us_alloc - us_prealloc:.1f} μs")
        print(f"    .to(same dtype, noop):  {us_noop_cast:.1f} μs")
        print(f"    .to(float32):           {us_cast:.1f} μs")
        print(f"  ---")
        saves = (us_alloc - us_prealloc)
        # 2 residual adds per layer × 24 layers
        print(f"    Est. savings (48 adds): {saves * 48 / 1000:.2f} ms")

    def test_bench_python_overhead_breakdown(self, compiler):
        """Pinpoint which Python operations in the wrapper are expensive."""
        H = 2880
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")
        params = torch.empty(2, dtype=torch.float32, device="mps")

        # 1. Measure tensor[i] = value (GPU scalar write)
        us_scalar_write = self._bench(lambda: params.__setitem__(0, float(H)))
        # 2. Measure .contiguous() on already-contiguous tensor
        us_contig = self._bench(lambda: x.contiguous())
        # 3. Measure .view(-1)
        us_view = self._bench(lambda: x.view(-1))
        # 4. Measure .contiguous().view(-1) combined
        us_contig_view = self._bench(lambda: x.contiguous().view(-1))
        # 5. Measure getattr(lib, "rms_norm_bf16")
        compiler.run_rms_norm(x, w, out, 1e-5)  # ensure compiled
        lib = compiler._libs["rms_norm"]
        us_getattr = self._bench(lambda: getattr(lib, "rms_norm_bf16"))
        # 6. Measure dict lookup
        us_dict = self._bench(lambda: compiler._libs.get("rms_norm"))
        # 7. Measure setting 2 params values
        us_two_params = self._bench(lambda: (
            params.__setitem__(0, float(H)),
            params.__setitem__(1, 1e-5),
        ))

        print(f"\n  Python wrapper overhead breakdown:")
        print(f"    tensor[i] = float (1 write):   {us_scalar_write:.1f} μs")
        print(f"    2 param writes:                {us_two_params:.1f} μs")
        print(f"    .contiguous():                 {us_contig:.1f} μs")
        print(f"    .view(-1):                     {us_view:.1f} μs")
        print(f"    .contiguous().view(-1):        {us_contig_view:.1f} μs")
        print(f"    getattr(lib, 'kernel_name'):   {us_getattr:.1f} μs")
        print(f"    dict.get('key'):               {us_dict:.1f} μs")

        # Total: rms_norm wrapper does: 1 dict lookup + 2 param writes +
        #   3 contiguous().view(-1) + 1 getattr + kernel dispatch
        estimated = us_dict + us_two_params + 3 * us_contig_view + us_getattr
        print(f"  ---")
        print(f"    Estimated wrapper overhead:     {estimated:.1f} μs")
        print(f"    × 6 dispatches/layer × 24:     {estimated * 6 * 24 / 1000:.1f} ms")

    def test_bench_compile_shader_python_overhead(self, compiler):
        """Measure raw compile_shader dispatch cost vs Python wrapper overhead.

        Tests how much of the ~110 μs per custom Metal dispatch is:
        (a) inherent to compile_shader encoding, vs
        (b) Python wrapper overhead (.contiguous().view(-1), params setting, etc.)
        """
        H = 2880
        eps = 1e-5
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        # Ensure kernel is compiled
        compiler.run_rms_norm(x, w, out, eps)

        # Pre-flatten all args (eliminate .contiguous().view(-1))
        x_flat = x.contiguous().view(-1)
        w_flat = w.contiguous().view(-1)
        out_flat = out.view(-1)
        params = compiler._params_cache[("rms_norm", H, eps)]

        # Get direct reference to kernel function (eliminate getattr)
        lib = compiler._libs["rms_norm"]
        kernel_fn = compiler._kernel_fn_cache.get("rms_norm_bf16") or lib.rms_norm_bf16

        # Measure just the kernel dispatch with pre-prepared args
        us_raw = self._bench(
            lambda: kernel_fn(
                x_flat, w_flat, out_flat, params,
                threads=(256, 1, 1), group_size=(256, 1, 1),
            )
        )

        # Measure full wrapper path
        us_wrapper = self._bench(lambda: compiler.run_rms_norm(x, w, out, eps))

        print(f"\n  compile_shader Python overhead breakdown:")
        print(f"    Raw kernel dispatch:     {us_raw:.1f} μs")
        print(f"    Full wrapper:            {us_wrapper:.1f} μs")
        print(f"    Python overhead:         {us_wrapper - us_raw:.1f} μs")
        print(f"  ---")
        if us_raw > 50:
            print(f"    Bottleneck: compile_shader encoding itself ({us_raw:.0f} μs)")
            print(f"    Reducing Python overhead won't help much")
        else:
            print(f"    Bottleneck: Python wrapper ({us_wrapper - us_raw:.0f} μs)")
            print(f"    Optimizing wrapper code would help significantly")

    def test_bench_moe_route_raw(self, compiler):
        """Measure moe_route with pre-prepared args to check remaining overhead."""
        logits = torch.randn(1, 32, dtype=torch.bfloat16, device="mps")
        expert_ids = torch.empty(4, dtype=torch.int32, device="mps")
        fused_scales = torch.empty(4, dtype=torch.float32, device="mps")

        # Warm up and populate caches
        compiler.run_moe_route_topk(logits, expert_ids, fused_scales, 32, 4, 1.0)

        # Full wrapper path
        us_wrapper = self._bench(
            lambda: compiler.run_moe_route_topk(logits, expert_ids, fused_scales, 32, 4, 1.0)
        )

        # Raw dispatch with pre-prepared args
        lib = compiler._libs["moe_routing"]
        kernel_fn = compiler._kernel_fn_cache.get("moe_route_topk_bf16") or lib.moe_route_topk_bf16
        params = compiler._params_cache[("moe_routing", 32, 4, 1.0, 0)]
        logits_flat = logits.contiguous().view(-1)

        us_raw = self._bench(
            lambda: kernel_fn(
                logits_flat, expert_ids, fused_scales, params,
                threads=(1, 1, 1), group_size=(1, 1, 1),
            )
        )

        print(f"\n  MoE route overhead (E=32, K=4):")
        print(f"    Raw dispatch:   {us_raw:.1f} μs")
        print(f"    Full wrapper:   {us_wrapper:.1f} μs")
        print(f"    Wrapper delta:  {us_wrapper - us_raw:.1f} μs")

    def test_bench_zero_and_cast(self, compiler):
        """Measure output.zero_() and .to(dtype) costs for MoE GEMM2 output."""
        H = 2880
        f32_buf = torch.zeros(1, H, dtype=torch.float32, device="mps")
        bf16_buf = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        us_zero = self._bench(lambda: f32_buf.zero_())
        us_cast_alloc = self._bench(lambda: f32_buf.to(torch.bfloat16))
        us_copy_cast = self._bench(lambda: bf16_buf.copy_(f32_buf))

        print(f"\n  GEMM2 output costs (H={H}):")
        print(f"    output.zero_():           {us_zero:.1f} μs")
        print(f"    .to(bf16) (allocates):    {us_cast_alloc:.1f} μs")
        print(f"    copy_(f32→bf16, in-place):{us_copy_cast:.1f} μs")
        saves = us_cast_alloc - us_copy_cast
        print(f"    Savings per layer:        {saves:.1f} μs")
        print(f"    × 24 layers:              {saves * 24 / 1000:.2f} ms")

    def test_bench_dispatch_overhead_breakdown(self, compiler):
        """Break down dispatch overhead: PyTorch add, fun.rms_norm, fun.linear,
        custom Metal kernel. Helps understand where CPU submission time goes."""
        H = 2880
        eps = 1e-5
        x = torch.randn(1, H, dtype=torch.bfloat16, device="mps")
        w = torch.randn(H, dtype=torch.bfloat16, device="mps")
        wmat = torch.randn(H, H, dtype=torch.bfloat16, device="mps")
        out = torch.empty(1, H, dtype=torch.bfloat16, device="mps")

        us_add = self._bench(lambda: x + x)
        us_rms = self._bench(lambda: fun.rms_norm(x, [H], w, eps))
        us_linear = self._bench(lambda: fun.linear(x, wmat))
        us_custom_rms = self._bench(lambda: compiler.run_rms_norm(x, w, out, eps))

        # Also test existing custom Metal kernels for reference
        # MoE routing is a tiny op like rms_norm
        expert_ids = torch.empty(4, dtype=torch.int32, device="mps")
        fused_scales = torch.empty(4, dtype=torch.float32, device="mps")
        logits = torch.randn(1, 32, dtype=torch.bfloat16, device="mps")
        us_moe_route = self._bench(
            lambda: compiler.run_moe_route_topk(logits, expert_ids, fused_scales, 32, 4, 1.0)
        )

        print(f"\n  Dispatch overhead breakdown (H={H}, n=1):")
        print(f"    PyTorch add:           {us_add:.1f} μs")
        print(f"    fun.rms_norm:          {us_rms:.1f} μs")
        print(f"    fun.linear:            {us_linear:.1f} μs")
        print(f"    Custom Metal rms_norm: {us_custom_rms:.1f} μs")
        print(f"    Custom Metal moe_route:{us_moe_route:.1f} μs")
        print(f"  ---")
        print(f"  Per-layer estimates (7 attn + 7 moe dispatches):")
        # Attn: rms_norm + linear + rope + kv_append + attn_decode + linear + add
        attn_est = us_rms + us_linear + us_moe_route * 3 + us_linear + us_add
        # MoE: rms_norm + linear + cast + pad + route + gemm1 + gemm2 + cast + add
        moe_est = us_rms + us_linear + us_add + us_add + us_moe_route * 3 + us_add + us_add
        print(f"    Attn (3 custom + 2 MPSGraph + 2 PyTorch): {attn_est:.0f} μs")
        print(f"    MoE  (3 custom + 2 MPSGraph + 4 PyTorch): {moe_est:.0f} μs")
        print(f"    Total 24 layers: {(attn_est + moe_est) * 24 / 1000:.1f} ms")
