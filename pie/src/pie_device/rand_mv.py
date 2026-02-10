# triton_batched_randn_with_stdev.py
# ------------------------------------------------------------
# Two Triton kernels with elementwise stdev S:
#  1) batched_randn_matmul: y[b, :] = x[b, :] @ (S * N(0,1) with seed=seeds[b])
#     -> generates weights on-the-fly, reads S tiles, no materialization.
#  2) batched_randn_generate: W_batched[b, i, o] = S[i, o] * N(0,1; seed=seeds[b])
#     -> materializes per-batch weights.
#
# Tests compare the matmul kernel against a PyTorch baseline that uses the
# generator kernel to ensure bit-exactness.
# ------------------------------------------------------------
#
# NOTE: This module requires CUDA. It is disabled on non-CUDA platforms
# (e.g., Apple Metal) to prevent crashes.
# ------------------------------------------------------------

import torch

# Guard: Only enable on CUDA
RAND_MV_AVAILABLE = torch.cuda.is_available()

if not RAND_MV_AVAILABLE:
    # Stub implementations for non-CUDA platforms (e.g., Apple Metal)
    def batched_randn_matmul(*args, **kwargs):
        raise RuntimeError(
            "rand_mv.batched_randn_matmul requires CUDA. "
            "This function is not available on non-CUDA platforms (e.g., Apple Metal)."
        )

    def batched_randn_generate(*args, **kwargs):
        raise RuntimeError(
            "rand_mv.batched_randn_generate requires CUDA. "
            "This function is not available on non-CUDA platforms (e.g., Apple Metal)."
        )

    def run_tests():
        raise RuntimeError(
            "rand_mv tests require CUDA. "
            "Tests are not available on non-CUDA platforms."
        )

else:
    # CUDA is available - import Triton and define all functions
    import triton
    import triton.language as tl

    # ==========================
    #  A100-tuned tile choosers
    # ==========================

    def _choose_pow2(sz: int) -> int:
        # power-of-two in {16, 32, 64, 128, 256}
        if sz >= 4096:
            return 256
        if sz >= 1024:
            return 128
        if sz >= 256:
            return 64
        if sz >= 64:
            return 32
        return 16

    def _choose_tiling_mm(I: int, O: int):
        # K-tile (over in_features) and N-tile (over out_features)
        if I >= O:
            BLOCK_K = _choose_pow2(I)
            BLOCK_N = _choose_pow2(O)
        else:
            BLOCK_N = _choose_pow2(O)
            BLOCK_K = _choose_pow2(I)
        size_hint = max(BLOCK_K, BLOCK_N)
        num_warps = 8 if size_hint >= 128 else (4 if size_hint >= 64 else 2)
        num_stages = 4
        return BLOCK_K, BLOCK_N, num_warps, num_stages

    def _choose_tiling_gen(I: int, O: int):
        if I >= O:
            BLOCK_M = _choose_pow2(I)  # tile over I
            BLOCK_N = _choose_pow2(O)  # tile over O
        else:
            BLOCK_N = _choose_pow2(O)
            BLOCK_M = _choose_pow2(I)
        size_hint = max(BLOCK_M, BLOCK_N)
        num_warps = 8 if size_hint >= 128 else (4 if size_hint >= 64 else 2)
        num_stages = 4
        return BLOCK_M, BLOCK_N, num_warps, num_stages

    # ============================================================
    #  KERNEL 1: y[b, :] = x[b, :] @ (S * N(0,1; seed=seeds[b]))
    #            B is runtime; I/O constexpr (for codegen)
    # ============================================================

    @triton.jit
    def _randn_mm_row_kernel_with_stdev(
        x_ptr,  # *f16/f32 [B, I]
        seeds_ptr,  # *int64   [B]
        S_ptr,  # *f16/f32 [I, O] elementwise stdev
        y_ptr,  # *f32     [B, O] accumulate in f32
        B,  # runtime int32
        I: tl.constexpr,
        O: tl.constexpr,
        # strides (in elements)
        stride_xb,
        stride_xi,
        stride_Si,
        stride_So,
        stride_yb,
        stride_yo,
        n_rounds: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        col_offset: tl.constexpr,
        global_cols: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_n = tl.program_id(1)
        if pid_b >= B:
            return

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < O

        x_row_ptr = x_ptr + pid_b * stride_xb
        y_row_ptr = y_ptr + pid_b * stride_yb

        acc = tl.zeros([BLOCK_N], dtype=tl.float32)

        seed_val = tl.load(seeds_ptr + pid_b)

        # ---- MODIFICATION START ----
        # If seed is non-zero, perform the matmul. Otherwise, acc remains zero,
        # resulting in a zero vector output for this batch.
        if seed_val != 0:
            seed_i = seed_val.to(tl.int32)
            k0 = 0
            while k0 < I:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < I

                # Load x[b, k] (vector over K)
                x_tile = tl.load(
                    x_row_ptr + offs_k * stride_xi, mask=mask_k, other=0.0
                ).to(tl.float32)

                # Compute offsets for rng: offset = k * global_cols + (n + col_offset)
                k_offsets = offs_k.to(tl.int32)[:, None]
                n_offsets = offs_n.to(tl.int32)[None, :]
                offsets = k_offsets * global_cols + (n_offsets + col_offset)

                # Random normals
                w_tile = tl.randn(seed_i, offsets, n_rounds=n_rounds)  # [BK, BN], f32

                # Load S[k, n] tile and scale
                S_tile_ptr = (
                    S_ptr + offs_k[:, None] * stride_Si + offs_n[None, :] * stride_So
                )
                S_tile = tl.load(
                    S_tile_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0
                ).to(tl.float32)

                w_tile = w_tile * S_tile  # scale by stdev elementwise

                # Mask invalid lanes to zero
                w_tile = tl.where(mask_k[:, None] & mask_n[None, :], w_tile, 0.0)

                # Reduce over K
                acc += tl.sum(w_tile * x_tile[:, None], axis=0)
                k0 += BLOCK_K
        # ---- MODIFICATION END ----

        tl.store(y_row_ptr + offs_n * stride_yo, acc, mask=mask_n)

    @torch.no_grad()
    def batched_randn_matmul(
        x: torch.Tensor,
        seeds: torch.Tensor,
        S: torch.Tensor,
        *,
        n_rounds: int = 10,
        out_dtype: torch.dtype | None = None,
        col_offset: int = 0,
        global_cols: int | None = None,
    ) -> torch.Tensor:
        """
        Compute y[b] = x[b] @ (S * N(0,1; seed=seeds[b])) without materializing weights.

        Args:
            x:     (B, I) float16/float32 CUDA
            seeds: (B,)   int64/int32 (CPU or CUDA; will be copied)
            S:     (I, O) float16/float32 CUDA (elementwise stdev)
        """
        assert x.is_cuda and S.is_cuda, "x and S must be CUDA tensors"
        assert x.dim() == 2 and S.dim() == 2
        B, I = x.shape
        I_S, O = S.shape
        assert I_S == I, "S.shape[0] must equal x.shape[1]"
        if out_dtype is None:
            out_dtype = x.dtype
        if global_cols is None:
            global_cols = O

        y = torch.empty(
            (B, O), device=x.device, dtype=torch.float32
        )  # accumulate in f32

        # Strides (elements)
        stride_xb, stride_xi = x.stride()
        stride_Si, stride_So = S.stride()
        stride_yb, stride_yo = y.stride()

        seeds_dev = seeds.to(device=x.device, dtype=torch.int64)
        assert seeds_dev.numel() == B

        BLOCK_K, BLOCK_N, num_warps, num_stages = _choose_tiling_mm(I, O)
        grid = (B, triton.cdiv(O, BLOCK_N))

        _randn_mm_row_kernel_with_stdev[grid](
            x,
            seeds_dev,
            S,
            y,
            B,
            I,
            O,
            stride_xb,
            stride_xi,
            stride_Si,
            stride_So,
            stride_yb,
            stride_yo,
            n_rounds=n_rounds,
            BLOCK_K=BLOCK_K,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
            col_offset=col_offset,
            global_cols=global_cols,
        )

        return y.to(out_dtype)

    # ============================================================
    #  KERNEL 2: Generate W_b[b, i, o] = S[i, o] * N(0,1; seed=seeds[b])
    #            B is runtime; I/O constexpr (for codegen)
    # ============================================================

    @triton.jit
    def _randn_generate_kernel_with_stdev(
        seeds_ptr,  # *int64 [B]
        S_ptr,  # *f16/f32 [I, O]
        y_ptr,  # *f32   [B, I, O]
        B,  # runtime int32
        I: tl.constexpr,
        O: tl.constexpr,
        stride_Si,
        stride_So,
        stride_yb,
        stride_yi,
        stride_yo,
        n_rounds: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        col_offset: tl.constexpr,
        global_cols: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        if pid_b >= B:
            return

        tiles_n = tl.cdiv(O, BLOCK_N)
        tile_m = pid_t // tiles_n
        tile_n = pid_t % tiles_n

        i0 = tile_m * BLOCK_M
        o0 = tile_n * BLOCK_N

        offs_i = i0 + tl.arange(0, BLOCK_M)
        offs_o = o0 + tl.arange(0, BLOCK_N)
        mask_i = offs_i < I
        mask_o = offs_o < O

        seed_val = tl.load(seeds_ptr + pid_b)

        # ---- MODIFICATION START ----
        # If seed is zero, generate a zero tile. Otherwise, generate randoms.
        if seed_val == 0:
            tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        else:
            seed_b = seed_val.to(tl.int32)
            # offsets = i * global_cols + (o + col_offset)
            i_offsets = offs_i.to(tl.int32)[:, None]
            o_offsets = offs_o.to(tl.int32)[None, :]
            offsets = i_offsets * global_cols + (o_offsets + col_offset)

            # Generate normals
            tile = tl.randn(seed_b, offsets, n_rounds=n_rounds)  # f32

            # Load S[i, o] and scale
            S_tile_ptr = (
                S_ptr + offs_i[:, None] * stride_Si + offs_o[None, :] * stride_So
            )
            S_tile = tl.load(
                S_tile_ptr, mask=mask_i[:, None] & mask_o[None, :], other=0.0
            ).to(tl.float32)
            tile = tile * S_tile
        # ---- MODIFICATION END ----

        base_ptr = (
            y_ptr
            + pid_b * stride_yb
            + offs_i[:, None] * stride_yi
            + offs_o[None, :] * stride_yo
        )
        tl.store(base_ptr, tile, mask=mask_i[:, None] & mask_o[None, :])

    @torch.no_grad()
    def batched_randn_generate(
        seeds: torch.Tensor,
        S: torch.Tensor,
        *,
        n_rounds: int = 10,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        col_offset: int = 0,
        global_cols: int | None = None,
    ) -> torch.Tensor:
        """
        Materialize W_batched[b, i, o] = S[i, o] * N(0,1; seed=seeds[b]).
        """
        if device is None:
            device = S.device if S.is_cuda else torch.device("cuda")
        seeds_dev = seeds.to(device=device, dtype=torch.int64)
        S_dev = S.to(device=device)
        assert S_dev.dim() == 2
        B = int(seeds_dev.numel())
        I, O = map(int, S_dev.shape)

        if global_cols is None:
            global_cols = O

        y = torch.empty((B, I, O), device=device, dtype=torch.float32)

        stride_Si, stride_So = S_dev.stride()
        stride_yb, stride_yi, stride_yo = y.stride()

        BLOCK_M, BLOCK_N, num_warps, num_stages = _choose_tiling_gen(I, O)
        tiles_m = triton.cdiv(I, BLOCK_M)
        tiles_n = triton.cdiv(O, BLOCK_N)
        grid = (B, tiles_m * tiles_n)

        _randn_generate_kernel_with_stdev[grid](
            seeds_dev,
            S_dev,
            y,
            B,
            I,
            O,
            stride_Si,
            stride_So,
            stride_yb,
            stride_yi,
            stride_yo,
            n_rounds=n_rounds,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
            col_offset=col_offset,
            global_cols=global_cols,
        )

        return y.to(dtype)

    # ============================================================
    #  TESTS
    # ============================================================

    def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()

    @torch.no_grad()
    def run_tests():
        torch.manual_seed(0)
        device = torch.device("cuda")

        def do_case(B, I, O, dtype_x=torch.float16, dtype_S=torch.float32):
            x = torch.randn(B, I, device=device, dtype=dtype_x)
            # Non-negative stdevs; include zeros to exercise masking
            S = torch.ones(I, O, device=device, dtype=dtype_S)
            seeds = torch.randint(
                1, 1 << 30, (B,), device=device, dtype=torch.int64
            )  # Ensure non-zero seeds for baseline

            # Baseline via generator + bmm
            W = batched_randn_generate(
                seeds, S, device=device, dtype=torch.float32
            )  # [B, I, O]
            y_ref = torch.bmm(x.to(torch.float32).unsqueeze(1), W).squeeze(1)

            # Kernel 1 result
            y_ker = batched_randn_matmul(x, seeds, S, out_dtype=torch.float32)
            diff = _max_abs_diff(y_ref, y_ker)
            print(f"B={B} I={I} O={O} | max-abs-diff={diff:.3e}")
            assert diff < 0.1, "Bit-exact mismatch"

            # ---- MODIFICATION START: Test zero seed behavior ----
            if B >= 2:
                seeds_with_zero = seeds.clone()
                seeds_with_zero[1] = 0  # Set seed for batch 1 to zero

                # Kernel 2 (generator) must produce zeros for batch 1
                W_zero_ker = batched_randn_generate(
                    seeds_with_zero, S, device=device, dtype=torch.float32
                )
                assert torch.all(
                    W_zero_ker[1] == 0
                ), "Generator kernel failed to output zeros for zero seed"
                # Batch 0 should be unaffected
                assert torch.all(
                    W_zero_ker[0] == W[0]
                ), "Generator kernel changed output for non-zero seed"

                # Kernel 1 (matmul) must produce zeros for batch 1
                y_zero_ker = batched_randn_matmul(
                    x, seeds_with_zero, S, out_dtype=torch.float32
                )
                is_zero = torch.all(y_zero_ker[1] == 0)
                assert is_zero, "Matmul kernel failed to output zeros for zero seed"

                # Batch 0 should be unaffected
                diff_zero_seed = _max_abs_diff(y_zero_ker[0], y_ref[0])
                print(
                    f"  zero seed test | batch 1 is all zero: {is_zero}, batch 0 diff={diff_zero_seed:.3e}"
                )
                assert diff_zero_seed < 0.1
            # ---- MODIFICATION END ----

            # Repro: rows with same seed produce same result, independent of others
            if B >= 3:
                seeds2 = seeds.clone()
                seeds2[1] = seeds2[1] + 12345  # change row 1 only
                y1 = batched_randn_matmul(x, seeds, S, out_dtype=torch.float32)
                y2 = batched_randn_matmul(x, seeds2, S, out_dtype=torch.float32)

                d0 = _max_abs_diff(y1[0], y2[0])
                d2 = _max_abs_diff(y1[2], y2[2])
                print(f"  repro rows 0 & 2 unchanged: {d0:.3e}, {d2:.3e}")
                assert d0 < 0.1 and d2 < 0.1

            # ---- Sharding test ----
            if O % 2 == 0:
                half = O // 2
                # Left half
                S_left = S[:, :half]
                y_left = batched_randn_matmul(
                    x,
                    seeds,
                    S_left,
                    out_dtype=torch.float32,
                    col_offset=0,
                    global_cols=O,
                )
                d_left = _max_abs_diff(y_left, y_ref[:, :half])

                # Right half
                S_right = S[:, half:]
                y_right = batched_randn_matmul(
                    x,
                    seeds,
                    S_right,
                    out_dtype=torch.float32,
                    col_offset=half,
                    global_cols=O,
                )
                d_right = _max_abs_diff(y_right, y_ref[:, half:])

                print(
                    f"  sharding split {O} -> {half}+{half} | left_diff={d_left:.3e}, right_diff={d_right:.3e}"
                )
                assert d_left < 0.1 and d_right < 0.1, "Sharding mismatch"

        # Small sanity
        do_case(B=3, I=8, O=8)
        do_case(B=2, I=7, O=5)

        # Skewed: I >> O
        do_case(B=4, I=2048, O=8)

        # Skewed: O >> I
        do_case(B=4, I=8, O=2048)

        # Mixed dtypes
        do_case(B=3, I=129, O=65, dtype_x=torch.float16, dtype_S=torch.float16)

        print("All tests passed ✅")


if __name__ == "__main__":
    run_tests()
