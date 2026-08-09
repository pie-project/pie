//! `quant/quant_bf16_to_fp8.cu`'s four launchers, in Rust.
//!
//! The W8A8 and block-FP8 quantiser pair that `bind::quant_gemm` runs around
//! a cuBLAS call: quantise the activation, GEMM in the narrow type, widen the
//! accumulator back to bf16. Every `__global__` here is NVRTC's, out of
//! `kernels-cuda-new/csrc/src/quant/quant_bf16_to_fp8.cuh`, and this module
//! is the `<<<>>>` half.
//!
//! # Why this file exists rather than three more `ffi::pie_k_quant_*` calls
//!
//! `table/driver_internal.rs` carried three rows whose only purpose was to
//! give `bind::quant_gemm` a shim entry to call across the ABI, and it wrote
//! down its own retirement:
//!
//! > Give those three grids rules and all three rows leave this table for
//! > `families::quant`, `bind::quant_gemm` fires them through
//! > `bind::jit::fire` instead of through `ffi::pie_k_*`, and
//! > `quant/quant_bf16_to_fp8.cu` loses its last consumer.
//!
//! Two of the three did not get rules, and that is the part worth reading.
//! `new-horizon.md` §10.5 refuses vocabulary grown for one kernel, and each
//! of these grids is one kernel:
//!
//! ```text
//!   w8a8_dequant            block(32, 8)                 a 2-D BLOCK
//!   quant_act_fp8_per_group grid(ceil(k / gs), m)        x = k / an operand
//! ```
//!
//! So they are [`kernels::LaunchRule::Unstated`] rows of `families::quant`
//! and the rectangles are stated HERE, beside the `<<<>>>` line each came
//! from. That is exactly what `fire/attn_score.rs` does with `FOLD_GRID_Y`
//! and what `fire::hand` exists to serve: a driver-owned [`Launch`] is the
//! escape hatch for a geometry no rule states, so "no rule" never has to mean
//! "invent one".
//!
//! The third launcher needed no rule at all. `quantize_bf16_to_int8_per_token`
//! was a C++ forwarder — its whole body was a call to
//! `quantize_bf16_to_int8_per_channel` with the arguments renamed — and that
//! one HAS had a `LaunchRule::Rms` row since `families::quant` landed. It is
//! fired through `bind::jit::fire` here, geometry and all, which is the
//! prescription's own words.
//!
//! # The fourth launcher had no consumer and is not ported
//!
//! `launch_dequant_int8_to_bf16_per_channel` fired
//! `dequant_int8_per_channel<bf16>` over `ceil(rows * cols / 256)` blocks —
//! `LaunchRule::Elementwise`, and `families::quant`'s
//! `quant::dequant_int8_to_bf16_per_channel` row states it. Nothing called
//! the launcher: no `.cu`, `.cpp`, `.cuh` or `.hpp` in any archive, no
//! `table::quant` row (so `emit_c_shim` emitted no entry for it), and no
//! hand-written arm in `driver-cuda/src`. The `.hpp` called it a "correctness
//! fallback for runtime INT8 weights when cuBLAS cannot run W8A8 for a
//! shape"; the fallback is `bind::quant_gemm`'s own, which dequantises
//! through the routed `quant::dequant_fp8_e4m3_to_bf16*` rows. A launcher
//! with an empty consumer set is deleted, not ported — porting it would give
//! a dead entry point a second implementation.
//!
//! # Every refusal is the C++'s, and none is a fallback
//!
//! The guards below are transcribed from the launcher, including the ones
//! that look redundant. They are not: `m <= 0 || k <= 0 || group_size <= 0`
//! guards a DIVISION as well as a grid, and a zero `group_size` would make
//! `n_groups` a divide-by-zero before any launch happened.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::fire;

/// `quant_bf16_to_fp8.cu:23` — `constexpr int BLOCK = 256;`.
///
/// Load-bearing for the per-channel reduction and not merely its width:
/// `quant_per_channel` sizes its `extern __shared__ float warp_max[]` at
/// `BLOCK / 32` from the LAUNCH and folds by reading `tid < kBlock / 32` with
/// `kBlock` a file-scope constant of the `.cuh`. A block wider than 256 reads
/// past the array; narrower, and lanes that hold partials are never folded.
/// `LaunchRule::Rms` is the same 256 and the same `(256 / 32) * 4` bytes,
/// which is why the per-channel quantiser is fired through the rule rather
/// than by hand here.
const BLOCK: u32 = 256;

/// `quant_bf16_to_fp8.cu:109` — `constexpr int BX = 32, BY = 8;`, the W8A8
/// dequant's block.
///
/// A 2-D block, which is the entire reason `quant::dequant_int32_w8a8_to_bf16`
/// is `LaunchRule::Unstated`: the kernel recovers `n` from
/// `blockIdx.x * blockDim.x + threadIdx.x` and `m` from the `y` pair, so the
/// two axes are the two extents of the output rectangle and neither can be
/// folded into the other.
const W8A8_BX: u32 = 32;
/// The other half of the pair above.
const W8A8_BY: u32 = 8;

/// `quant_bf16_to_fp8.cu:131` — the blockwise FP8 quantiser's `128`.
///
/// One block per (row, group) and 128 threads striding the group, so this
/// width is an occupancy choice and not a contract: the kernel's loop is
/// bounded by `count = min(gs, k - base)` and strides by `blockDim.x`. It is
/// transcribed rather than rounded because the launcher chose it and nothing
/// in this port measured a different one.
const GROUP_QUANT_BLOCK: u32 = 128;

/// Per-row symmetric INT8 quantisation of a `[rows, cols]` bf16 rectangle.
///
/// One scale per row, `scale_inv_row = absmax / 127`, outliers clamped —
/// the same op whether the rows are output channels of a weight or tokens of
/// an activation, which is why `quantize_bf16_to_int8_per_token` was a
/// forwarder onto this and why there is one function here.
///
/// `quant_bf16_to_fp8.cu:67-76`:
///
/// ```text
/// if (rows == 0 || cols == 0) return;
/// device::quant_per_channel<device::int8_sym>
///     <<<rows, BLOCK, ROW_REDUCE_SHMEM, stream>>>(
///         static_cast<const device::bf16*>(W_bf16), W_int8, scale_inv_dev, cols);
/// ```
///
/// Fired through [`crate::bind::jit::fire`] and not by hand: that grid IS
/// `LaunchRule::Rms` — one block per row, 256 wide, `(256 / 32) * 4` bytes of
/// dynamic shared memory — and `families::quant`'s
/// `quant::quantize_bf16_to_int8_per_channel` row has stated it all along.
/// The rule reads `rows` from [`kernels_cuda_new::Dims::rows`] and the row's
/// `cols` operand is `Source::InWidth(0)`, so `width` must carry the same
/// number the kernel is handed.
///
/// # Safety
///
/// `w_bf16` is `[rows, cols]` bf16, `out_int8` `[rows, cols]` **signed** i8,
/// `scale_inv` `[rows]` f32, all device memory live for the launch, and
/// `stream` a live `cudaStream_t`.
pub unsafe fn quantize_bf16_to_int8_per_channel(
    w_bf16: *const std::ffi::c_void,
    out_int8: *mut i8,
    scale_inv: *mut f32,
    rows: i32,
    cols: i32,
    stream: *mut std::ffi::c_void,
) {
    // `quant_bf16_to_fp8.cu:71` — `if (rows == 0 || cols == 0) return;`
    // transcribed with `<=` for the same reason every ported guard is: the
    // C++ took `int` and a negative would have produced an enormous unsigned
    // grid, which the original only avoided because no caller passed one.
    if rows <= 0 || cols <= 0 {
        return;
    }
    // SAFETY: the caller's assertion, forwarded unchanged. The operand list
    // is the DEVICE row's, in the row's order, and `Args::bind` checks it.
    unsafe {
        crate::bind::jit::fire(
            "quant::quantize_bf16_to_int8_per_channel",
            kernels_cuda_new::Dims {
                rows: rows.unsigned_abs(),
                width: cols.unsigned_abs(),
                in_width: cols.unsigned_abs(),
                ..kernels_cuda_new::Dims::default()
            },
            &[
                crate::bind::device::ArgValue::Ptr(w_bf16.cast_mut()),
                crate::bind::device::ArgValue::Ptr(out_int8.cast()),
                crate::bind::device::ArgValue::Ptr(scale_inv.cast()),
                crate::bind::device::ArgValue::I32(cols),
            ],
            stream,
        );
    }
}

/// The W8A8 epilogue: an `[M, N]` int32 accumulator widened to bf16 through
/// a per-row activation scale and a per-column weight scale.
///
/// `quant_bf16_to_fp8.cu:103-115`:
///
/// ```text
/// if (M == 0 || N == 0) return;
/// constexpr int BX = 32, BY = 8;
/// const dim3 block(BX, BY);
/// const dim3 grid((N + BX - 1) / BX, (M + BY - 1) / BY);
/// device::w8a8_dequant<<<grid, block, 0, stream>>>(
///     acc_int32, act_scale_inv, w_scale_inv,
///     static_cast<device::bf16*>(out_bf16), M, N);
/// ```
///
/// Transcribed digit for digit. `M` and `N` cross as operands as well as
/// sizing the grid because the kernel's `if (n >= N || m >= M) return;` is
/// what stops the last block of each axis, exactly as the launcher left it.
///
/// # Safety
///
/// `acc_int32` is `[M, N]` i32, `act_scale_inv` `[M]` f32, `w_scale_inv`
/// `[N]` f32, `out_bf16` `[M, N]` writable bf16, and `stream` live.
pub unsafe fn dequant_int32_w8a8_to_bf16(
    acc_int32: *const i32,
    act_scale_inv: *const f32,
    w_scale_inv: *const f32,
    out_bf16: *mut std::ffi::c_void,
    m: i32,
    n: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:108` — `if (M == 0 || N == 0) return;`
    if m <= 0 || n <= 0 {
        return;
    }
    let launch = Launch {
        grid: [n.unsigned_abs().div_ceil(W8A8_BX), m.unsigned_abs().div_ceil(W8A8_BY), 1],
        block: [W8A8_BX, W8A8_BY, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(acc_int32.cast_mut().cast()),
        ArgValue::Ptr(act_scale_inv.cast_mut().cast()),
        ArgValue::Ptr(w_scale_inv.cast_mut().cast()),
        ArgValue::Ptr(out_bf16),
        ArgValue::I32(m),
        ArgValue::I32(n),
    ];
    fire("quant::dequant_int32_w8a8_to_bf16", launch, &values, stream);
}

/// Blockwise (per-token-group) FP8 E4M3 activation quantisation — the
/// activation half of DeepSeek-style block FP8.
///
/// One f32 scale per contiguous `group_size` run along K, emitted row-major
/// `[m, ceil(k / group_size)]`, which is bit-identical to the column-major
/// `[ceil(k / gs), m]` tensor cuBLASLt wants for
/// `CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F` on operand B. That equality is
/// why nothing transposes between this and the GEMM, and it is the reason the
/// layout is not free to change.
///
/// `quant_bf16_to_fp8.cu:119-135`:
///
/// ```text
/// if (m <= 0 || k <= 0 || group_size <= 0) return;
/// const int n_groups = (k + group_size - 1) / group_size;
/// const dim3 grid(static_cast<unsigned>(n_groups), static_cast<unsigned>(m));
/// device::quant_act_fp8_per_group<<<grid, 128, 0, stream>>>(
///     static_cast<const device::bf16*>(act_bf16),
///     act_fp8, act_scale, m, k, group_size, n_groups);
/// CUDA_CHECK(cudaGetLastError());
/// ```
///
/// `n_groups` is computed ONCE and used twice — it is `grid.x` and it is the
/// operand the kernel bounds `blockIdx.x` against at
/// `quant_bf16_to_fp8.cuh:340`. Two derivations of one quotient is how a grid
/// and a guard come to disagree, so the port keeps the single binding.
///
/// The launcher's trailing `CUDA_CHECK(cudaGetLastError())` has no
/// transcription and does not need one: `fire::hand::fire` panics with the
/// symbol named on any launch error, which is the same claim made earlier and
/// with more information in it.
///
/// # Safety
///
/// `act_bf16` is `[m, k]` bf16, `act_fp8` `[m, k]` writable u8, `act_scale`
/// `[m, ceil(k / group_size)]` writable f32, and `stream` live.
pub unsafe fn quantize_bf16_to_fp8_e4m3_per_token_group(
    act_bf16: *const std::ffi::c_void,
    act_fp8: *mut u8,
    act_scale: *mut f32,
    m: i32,
    k: i32,
    group_size: i32,
    stream: *mut std::ffi::c_void,
) {
    // `:128` — and the third term guards the DIVISION below, not a grid.
    if m <= 0 || k <= 0 || group_size <= 0 {
        return;
    }
    // `:129` — `(k + group_size - 1) / group_size`, in i32 as the C++ had it,
    // so a `k` near `i32::MAX` overflows here exactly where it overflowed
    // there rather than silently differing.
    let n_groups = (k + group_size - 1) / group_size;
    let launch = Launch {
        grid: [n_groups.unsigned_abs(), m.unsigned_abs(), 1],
        block: [GROUP_QUANT_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(act_bf16.cast_mut()),
        ArgValue::Ptr(act_fp8.cast()),
        ArgValue::Ptr(act_scale.cast()),
        ArgValue::I32(m),
        ArgValue::I32(k),
        ArgValue::I32(group_size),
        ArgValue::I32(n_groups),
    ];
    fire("quant::quantize_bf16_to_fp8_e4m3_per_token_group", launch, &values, stream);
}

/// The unused `BLOCK` above is the per-channel reduction's, and it is
/// referenced here so that the constant and its paragraph cannot be deleted
/// as dead while the launch it describes is fired through a rule that
/// hard-codes the same number.
const _: () = assert!(BLOCK == 256);
