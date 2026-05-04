# pie_kernels/metal/rand_mv.py
# ─────────────────────────────────────────────────────────────
# Metal GPU counterparts for the two Triton kernels in
# pie_kernels/cuda/rand_mv.py:
#
#   batched_randn_matmul   — y[b] = x[b] @ (S * N(0,1; seed=seeds[b]))
#   batched_randn_generate — W[b,i,o] = S[i,o] * N(0,1; seed=seeds[b])
#
# On MPS  → native Metal kernels (metal_rand_mv.metal)
# On CPU  → pure-PyTorch Philox fallback (for testing only)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import torch

__all__ = ["batched_randn_matmul", "batched_randn_generate"]

_HAS_MPS = torch.backends.mps.is_available()

if _HAS_MPS:
    from ._compiler import MetalCompiler

    def _get_compiler() -> MetalCompiler:
        return MetalCompiler()


# ─────────────────────────────────────────────────────────────
# CPU reference implementation (pure-PyTorch Philox-4x32)
#
# Kept for two purposes:
#   1. Testing on non-MPS machines
#   2. Cross-checking Metal kernel outputs
#
# All arithmetic uses int64 tensors holding uint32 values
# to avoid Python/PyTorch signed-integer overflow issues.
# ─────────────────────────────────────────────────────────────

_PHILOX_KEY_A   = 0x9E3779B9
_PHILOX_KEY_B   = 0xBB67AE85
_PHILOX_ROUND_A = 0xD2511F53
_PHILOX_ROUND_B = 0xCD9E8D57
_MASK32 = 0xFFFFFFFF
_MASK16 = 0xFFFF


def _umulhi(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Upper 32 bits of unsigned 32×32→64 multiply (16-bit split)."""
    a_lo, a_hi = a & _MASK16, a >> 16
    b_lo, b_hi = b & _MASK16, b >> 16
    mid = a_lo * b_hi + a_hi * b_lo + (a_lo * b_lo >> 16)
    return (a_hi * b_hi + (mid >> 16)) & _MASK32


def _umul_lo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Lower 32 bits of unsigned 32×32 multiply."""
    a_lo, a_hi = a & _MASK16, a >> 16
    b_lo, b_hi = b & _MASK16, b >> 16
    return (a_lo * b_lo + ((a_lo * b_hi + a_hi * b_lo) << 16)) & _MASK32


def _philox_4x32(
    c0: torch.Tensor, c1: torch.Tensor,
    c2: torch.Tensor, c3: torch.Tensor,
    k0: torch.Tensor, k1: torch.Tensor,
    n_rounds: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Philox-4x32 core.  All values are int64 in [0, 2^32)."""
    A, B = _PHILOX_ROUND_A, _PHILOX_ROUND_B
    for _ in range(n_rounds):
        _c0, _c2 = c0, c2
        c0 = _umulhi(B, _c2) ^ c1 ^ k0
        c2 = _umulhi(A, _c0) ^ c3 ^ k1
        c1 = _umul_lo(B, _c2)
        c3 = _umul_lo(A, _c0)
        k0 = (k0 + _PHILOX_KEY_A) & _MASK32
        k1 = (k1 + _PHILOX_KEY_B) & _MASK32
    return c0, c1, c2, c3


def _triton_randn(seed: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Reproduce ``tl.randn(seed, offsets)`` exactly (CPU tensors)."""
    seed, offsets = seed.to(torch.int64), offsets.to(torch.int64)
    k0, k1 = seed & _MASK32, (seed >> 32) & _MASK32
    z = torch.zeros_like(offsets)
    i1, i2, _, _ = _philox_4x32(offsets & _MASK32, z, z, z, k0, k1)
    # uint_to_uniform_float
    SCALE = 4.6566127342e-10
    for t in (i1, i2):
        neg = t >= (1 << 31)
        t[neg] = _MASK32 - t[neg]
    u1 = torch.clamp(i1.float() * SCALE, min=1e-7)
    u2 = i2.float() * SCALE
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(6.283185307179586 * u2)


def _cpu_generate(
    seeds: torch.Tensor, S: torch.Tensor, *,
    col_offset: int, global_cols: int,
) -> torch.Tensor:
    B, (I, O) = seeds.numel(), S.shape
    y = torch.empty(B, I, O, dtype=torch.float32)
    i_idx = torch.arange(I, dtype=torch.int64)
    o_idx = torch.arange(O, dtype=torch.int64)
    offsets = i_idx[:, None] * global_cols + o_idx[None, :] + col_offset
    for b in range(B):
        sv = seeds[b].item()
        if sv == 0:
            y[b].zero_()
        else:
            seed_t = torch.full_like(offsets, sv & _MASK32)
            y[b] = _triton_randn(seed_t, offsets) * S.float()
    return y


def _cpu_matmul(
    x: torch.Tensor, seeds: torch.Tensor, S: torch.Tensor, *,
    col_offset: int, global_cols: int,
) -> torch.Tensor:
    B, I = x.shape
    O = S.shape[1]
    y = torch.empty(B, O, dtype=torch.float32)
    i_idx = torch.arange(I, dtype=torch.int64)
    o_idx = torch.arange(O, dtype=torch.int64)
    offsets = i_idx[:, None] * global_cols + o_idx[None, :] + col_offset
    for b in range(B):
        sv = seeds[b].item()
        if sv == 0:
            y[b].zero_()
        else:
            seed_t = torch.full_like(offsets, sv & _MASK32)
            W = _triton_randn(seed_t, offsets) * S.float()
            y[b] = x[b].float() @ W
    return y


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

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
    """Materialize ``W[b, i, o] = S[i, o] * N(0,1; seed=seeds[b])``.

    Bit-exact with the Triton kernel ``_randn_generate_kernel_with_stdev``.
    """
    if device is None:
        device = S.device
    S_f32 = S.to(device=device, dtype=torch.float32)
    assert S_f32.dim() == 2
    B = seeds.numel()
    I, O = S_f32.shape
    if global_cols is None:
        global_cols = O

    # MPS → Metal kernel
    if _HAS_MPS and str(device) in ("mps", "mps:0"):
        assert n_rounds == 10, f"Metal kernel requires n_rounds=10, got {n_rounds}"
        out = torch.empty(B, I, O, device="mps", dtype=torch.float32)
        _get_compiler().run_randn_generate(
            seeds.to(device="mps", dtype=torch.int32),
            S_f32.to("mps"), out, col_offset, global_cols,
        )
        return out.to(dtype)

    # CPU fallback
    return _cpu_generate(
        seeds, S_f32, col_offset=col_offset, global_cols=global_cols,
    ).to(dtype)


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
    """Compute ``y[b] = x[b] @ (S * N(0,1; seed=seeds[b]))``.

    Bit-exact with the Triton kernel ``_randn_mm_row_kernel_with_stdev``.
    Fused Metal kernel — never materializes the weight matrix.
    """
    assert x.dim() == 2 and S.dim() == 2
    B, I = x.shape
    assert S.shape[0] == I, "S.shape[0] must equal x.shape[1]"
    O = S.shape[1]
    if out_dtype is None:
        out_dtype = x.dtype
    if global_cols is None:
        global_cols = O

    # MPS → Metal kernel
    if _HAS_MPS and x.device.type == "mps":
        assert n_rounds == 10, f"Metal kernel requires n_rounds=10, got {n_rounds}"
        out = torch.empty(B, O, device="mps", dtype=torch.float32)
        _get_compiler().run_randn_matmul(
            x, seeds.to(device="mps", dtype=torch.int32),
            S.to(device="mps", dtype=torch.float32),
            out, col_offset, global_cols,
        )
        return out.to(out_dtype)

    # CPU fallback
    return _cpu_matmul(
        x, seeds, S, col_offset=col_offset, global_cols=global_cols,
    ).to(out_dtype)
