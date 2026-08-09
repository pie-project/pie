//! `norm/rmsnorm.cu`'s host program, in Rust.
//!
//! # What this file is, and what the archive file was
//!
//! `crates/kernels-cuda/csrc/src/norm/rmsnorm.cu` held five launchers, no
//! `__global__`, and one host predicate — `rmsnorm_vec8_ok` — that decides
//! between a scalar kernel and its vectorised twin by reading three POINTER
//! ADDRESSES. No `LaunchRule` can see an address and no `Source` can state
//! one, which is why the file survived every routing pass: the predicate is
//! host code, so the host code had to move languages rather than tables.
//!
//! Every kernel fired here lives in `kernels-cuda-new/csrc/src/norm/rmsnorm.cuh`
//! and is compiled by NVRTC. Nothing in this file is a fallback for anything.
//!
//! # THE THREE ROWS THIS FILE FREES
//!
//! `new-horizon.md` §49.1 counted seven rows routable *today* — unit exists,
//! every operand sourced — and blocked only because C++ still called their
//! launcher. Three of the seven were held by `rmsnorm.cu`, and by §10.10's
//! rule (*a launcher goes when its whole consumer set has gone, and the shim
//! is only one consumer*) they could not move while the file composed with
//! its own siblings:
//!
//! | row | what held it |
//! |---|---|
//! | `norm::rmsnorm_bf16` | `rmsnorm.cu:42`, called from `:59` and `:63` |
//! | `norm::rmsnorm_strided_bf16` | `rmsnorm.cu:42` — `rmsnorm_bf16` **is** a call to it |
//! | `quant::bf16_to_fp16` | `rmsnorm.cu:64`, its last C++ caller in the tree |
//!
//! C++ calling C++ is a call no Rust dispatch can intercept. Those three
//! calls are [`strided`], [`strided`] again and [`cast_to_fp16`] below, and
//! all three symbols are named in `device::JIT_DISPATCHED` with this change.
//!
//! # `quant::bf16_to_fp16` was NOT held by `gemm.cpp`
//!
//! `device.rs` recorded it as held there. That sentence was stale:
//! `gemm/gemm.cpp` includes `quant/dequant_fp4.hpp` and
//! `quant/dequant_fp8.hpp` and never `quant/dequant_wna16.hpp`, which is the
//! header the symbol is declared in. `rmsnorm.cu:64` was its only C++ call
//! site, swept over `.cu`, `.cuh`, `.cpp` and `.hpp` in both archives. Its
//! launcher stays where it is (`quant/dequant_wna16.cu`, another agent's
//! file); routing the row drops the shim entry and nothing else.
//!
//! (`gemm/gemm.cpp` has since been deleted whole, so that sweep can no
//! longer be re-run against it. The finding does not depend on re-running:
//! the file is gone, which is the strongest form of "does not include it".)

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

use super::hand::{aligned16, fire};

/// `rmsnorm.cu:26` — `rmsnorm_vec8_ok`.
///
/// True when every row of a `[num_rows, hidden]` bf16 view starts on a
/// 16-byte boundary and is a whole number of 8-element vectors.
///
/// The order is the C++'s own so the two read as one list, and — as
/// `families::norm`'s `RMSNORM_STRIDED_VEC8` says of the `Term` list that
/// mirrors it — order is not semantic here: `&&` short-circuits, but every
/// clause is a test on a value the caller already holds, so nothing is
/// deferred by an earlier `false`.
#[must_use]
fn vec8_ok(
    x: *const c_void,
    y: *const c_void,
    weight: *const c_void,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
) -> bool {
    hidden % 8 == 0
        && x_row_stride % 8 == 0
        && y_row_stride % 8 == 0
        && aligned16(x)
        && aligned16(y)
        && aligned16(weight)
}

/// `rmsnorm.cu:80` — `norm::rmsnorm_strided_bf16`, both arms.
///
/// **This is not the routed row's dispatch path and must not be confused
/// with it.** `norm::rmsnorm_strided_bf16` is named in
/// `device::JIT_DISPATCHED` with this change, and a fire that reaches it
/// through the generated arm gets `LaunchRule::Rms` — 256 threads — with
/// `RMSNORM_STRIDED_VEC8` choosing between the scalar row and the
/// `#vec8` row at 256. That trade is documented, measured and timed in
/// `families/norm.rs`, and it is deliberately NOT this.
///
/// This is the LAUNCHER, reproduced for the two callers that were inside the
/// file — `rmsnorm.cu:59` and `:63`, both through `rmsnorm_bf16` at `:42` —
/// and it fires the launcher's own instantiations at the launcher's own
/// widths. `BLOCK` sizes the `__shared__ float[BLOCK]` that
/// `block_reduce_sum_exact` folds through, so firing a 256-instantiation at
/// 512 threads folds 256 floats no thread wrote: finite, plausible and wrong.
/// The 512 arm therefore names a row of its own
/// (`norm::rmsnorm_strided_bf16#vec8_512`) rather than borrowing the 256 one.
///
/// # Safety
///
/// `x`, `weight` and `y` must address live device memory of the extents the
/// strides describe, and `stream` must be a live `cudaStream_t` — for the
/// duration of the launch, which is asynchronous, so that ends at the next
/// synchronisation and not at this call's return.
#[allow(clippy::too_many_arguments)]
pub unsafe fn strided(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    x_row_stride: i32,
    y_row_stride: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // `rmsnorm.cu:86` — `dim3 grid(num_rows)`. The C++ launched a zero grid
    // here and cudaGetLastError reported it at the next synchronisation on
    // whatever call happened to be next; `module.fire` refuses an empty
    // rectangle by name (`Ungeometric::Empty`), so an empty batch must not
    // reach it. Nothing to do is not a refusal.
    if num_rows <= 0 {
        return;
    }
    let grid = [num_rows.unsigned_abs(), 1, 1];
    if vec8_ok(x, y.cast_const(), weight, hidden, x_row_stride, y_row_stride) {
        // `rmsnorm.cu:88-94` — `constexpr int VBLOCK = 512;`
        // `device::rmsnorm_vec8<VBLOCK, false><<<grid, VBLOCK, 0, stream>>>`.
        // `y_fp16` is `nullptr`: the row spells it `BufMut | null` because
        // `EMIT_FP16=false` reads it only inside a dead `if constexpr`.
        fire(
            "norm::rmsnorm_strided_bf16#vec8_512",
            Launch { grid, block: [512, 1, 1], smem: 0 },
            &[
                ArgValue::Ptr(x.cast_mut()),
                ArgValue::Ptr(weight.cast_mut()),
                ArgValue::Ptr(y),
                ArgValue::Ptr(std::ptr::null_mut()),
                ArgValue::I32(hidden),
                ArgValue::I32(x_row_stride),
                ArgValue::I32(y_row_stride),
                ArgValue::F32(eps),
            ],
            stream,
        );
        return;
    }
    // `rmsnorm.cu:85,97-102` — `constexpr int BLOCK = 256;`
    // `device::rmsnorm<device::bf16, BLOCK><<<grid, block, 0, stream>>>`.
    // This one already had a row at its own width: `RMSNORM_SIGS[0]`,
    // `norm::device::rmsnorm<device::bf16, 256>`.
    fire(
        "norm::rmsnorm_strided_bf16",
        Launch { grid, block: [256, 1, 1], smem: 0 },
        &[
            ArgValue::Ptr(x.cast_mut()),
            ArgValue::Ptr(weight.cast_mut()),
            ArgValue::Ptr(y),
            ArgValue::I32(hidden),
            ArgValue::I32(x_row_stride),
            ArgValue::I32(y_row_stride),
            ArgValue::F32(eps),
        ],
        stream,
    );
}

/// `rmsnorm.cu:38` — `norm::rmsnorm_bf16`, which is one call and nothing else.
///
/// The unstrided view of [`strided`]: `hidden` is both strides. Kept as a
/// function rather than inlined at its two call sites because that is what
/// the archive did, and because the identity `rmsnorm_bf16(...) ==
/// rmsnorm_strided_bf16(..., hidden, hidden, ...)` is the whole content of
/// the symbol.
///
/// # Safety
///
/// [`strided`]'s, unchanged.
unsafe fn unstrided(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // SAFETY: the caller's obligation, forwarded verbatim.
    unsafe { strided(x, weight, y, num_rows, hidden, hidden, hidden, eps, stream) }
}

/// `rmsnorm.cu:64` — `kernels::quant::bf16_to_fp16(y, y_fp16, num_rows * hidden, stream)`.
///
/// The second launch that made `norm::rmsnorm_bf16_with_fp16` unrowable: *a
/// row is one kernel, and this one is two whenever the rows are unaligned*
/// (`rmsnorm.cu:51`). It is still two; what changed is that the host program
/// sequencing them is Rust.
///
/// The geometry is `quant/dequant_wna16.cu:72`'s, which
/// `LaunchRule::Slab` also reproduces exactly (`SLAB_VEC` 8, `BLOCK` 256,
/// `SLAB_GRID_MAX` 1024) — stated here rather than taken from the rule
/// because this call site is a port of a launcher and not a fire of a row.
///
/// # Safety
///
/// `src` must address `count` live bf16 elements, `dst` `count` live fp16
/// elements, and `stream` must be live across the launch.
unsafe fn cast_to_fp16(src: *const c_void, dst: *mut c_void, count: i64, stream: *mut c_void) {
    // `dequant_wna16.cu:65` — `if (count == 0) return;`
    if count <= 0 {
        return;
    }
    // `dequant_wna16.cu:66-71`:
    //   constexpr int BS = 256;
    //   const long long n_vec8 = n / 8;
    //   const long long units  = n_vec8 > 0 ? n_vec8 : n;
    //   const int blocks = min((units + BS - 1) / BS, 1024);
    const BS: i64 = 256;
    let n_vec8 = count / 8;
    let units = if n_vec8 > 0 { n_vec8 } else { count };
    let blocks = ((units + BS - 1) / BS).min(1024).max(1);
    // `dequant_wna16.cu:72` —
    // `device::bf16_to_narrow<__half><<<max(blocks,1), BS, 0, stream>>>(in, out, n)`.
    fire(
        "quant::bf16_to_fp16",
        Launch {
            grid: [u32::try_from(blocks).unwrap_or(1024), 1, 1],
            block: [256, 1, 1],
            smem: 0,
        },
        &[
            ArgValue::Ptr(src.cast_mut()),
            ArgValue::Ptr(dst),
            ArgValue::I64(count),
        ],
        stream,
    );
}

/// `rmsnorm.cu:54` — `norm::rmsnorm_bf16_with_fp16`, all three arms.
///
/// RMSNorm that also writes an fp16 copy of its output, for a consumer that
/// wants fp16 — the MXFP4 decode GEMV. The archive's comment for why no row
/// names this entry point is the reason it is here: *that fallback is a
/// SECOND launch (`quant::bf16_to_fp16`), which is why no row names this
/// entry point: a row is one kernel, and this one is two whenever the rows
/// are unaligned.*
///
/// The three arms, in the order the C++ tests them:
///
/// 1. `:58` — `y_fp16 == nullptr`: no fp16 copy was asked for, so this is
///    [`unstrided`] and nothing else.
/// 2. `:62` — the rows do not vectorise: [`unstrided`] writes the bf16
///    result, then [`cast_to_fp16`] reads it back and narrows it. Two
///    launches over the same buffer, ordered by the stream.
/// 3. `:69` — the fused arm, one launch, `EMIT_FP16=true`.
///
/// # The known defect this port does NOT fix
///
/// `families/norm.rs` records that `rmsnorm_vec8` with `EMIT_FP16=true` is
/// wrong above one row — `rmsnorm.cuh:318` writes the fp16 copy without a row
/// offset — and `kernels-cuda-new/tests/launch_rules.rs`'s
/// `the_emit_fp16_kernel_is_wrong_above_one_row` pins the signature so the
/// defect cannot be renamed away. A port that quietly corrected it would make
/// this symbol compute something the archive did not, on a path a golden was
/// recorded over. The defect is the kernel's and stays the kernel's.
///
/// # Safety
///
/// `x`, `weight` and `y` must address `num_rows * hidden` live bf16 elements;
/// `y_fp16`, when non-null, `num_rows * hidden` live fp16 elements. `stream`
/// must be live across every launch this makes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn with_fp16(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    y_fp16: *mut c_void,
    num_rows: i32,
    hidden: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // `rmsnorm.cu:58-61`.
    if y_fp16.is_null() {
        // SAFETY: the caller's obligation, forwarded.
        unsafe { unstrided(x, weight, y, num_rows, hidden, eps, stream) };
        return;
    }
    // `rmsnorm.cu:62-68`. The predicate reads `hidden` for BOTH strides,
    // which is what makes this the unstrided view.
    if !vec8_ok(x, y.cast_const(), weight, hidden, hidden, hidden) {
        // SAFETY: as above, and the second launch reads what the first wrote
        // on the same stream, which orders them.
        unsafe {
            unstrided(x, weight, y, num_rows, hidden, eps, stream);
            cast_to_fp16(
                y.cast_const(),
                y_fp16,
                i64::from(num_rows) * i64::from(hidden),
                stream,
            );
        }
        return;
    }
    if num_rows <= 0 {
        return;
    }
    // `rmsnorm.cu:69-77`:
    //   constexpr int VBLOCK = 512;
    //   dim3 grid(num_rows);
    //   device::rmsnorm_vec8<VBLOCK, false, true><<<grid, VBLOCK, 0, stream>>>(
    //       x, weight, y, y_fp16, hidden, hidden, hidden, eps);
    //
    // `norm::rmsnorm_bf16_with_fp16#vec8` already existed at BLOCK 256 and is
    // NOT what this fires. That row's own doc says why it is 256 — it states
    // `LaunchRule::Rms`, which launches 256 threads, and an instantiation at
    // 512 launched at 256 folds a half-written `__shared__ float[512]`. This
    // call site builds the launch itself, so it can have the launcher's width
    // and does: `#vec8_512`.
    fire(
        "norm::rmsnorm_bf16_with_fp16#vec8_512",
        Launch { grid: [num_rows.unsigned_abs(), 1, 1], block: [512, 1, 1], smem: 0 },
        &[
            ArgValue::Ptr(x.cast_mut()),
            ArgValue::Ptr(weight.cast_mut()),
            ArgValue::Ptr(y),
            ArgValue::Ptr(y_fp16),
            ArgValue::I32(hidden),
            ArgValue::I32(hidden),
            ArgValue::I32(hidden),
            ArgValue::F32(eps),
        ],
        stream,
    );
}

/// The width above which the vectorised fused norm prefers a 512-thread block.
///
/// `rmsnorm.cu:160` — `if (hidden_size >= 2560)`. A comparison, which is why
/// the launcher could not become a row: every `Term` in the `LaunchRule`
/// vocabulary is unary (`new-horizon.md` §44.6).
const RASR_VEC512_ABOVE: i32 = 2560;

/// `rmsnorm.cu:119` — `norm::rmsnorm_residual_add_scale_rmsnorm_bf16`, all
/// three arms.
///
/// gemma-4 fires this: four statements and 221 golden lines. It is the one
/// symbol in this file whose ahead-of-time row is FULLY SOURCED, so it had a
/// generated arm calling `pie_k_norm_rmsnorm_residual_add_scale_rmsnorm_bf16`
/// before this change and has a generated arm calling this after it. The
/// operand list, the guard and the staging are the same text either way —
/// `abi::emit_dispatch` builds them from the row and then chooses a callee.
///
/// # The measurement, which the port carries rather than consumes
///
/// The scalar form walks the row three times, once per pass, and measured
/// **10.79 us/call in gemma-4-26B's decode — 8% of the step** — against 2.51
/// for the vectorized plain norm. Swept under graph replay at the shapes
/// these models use, in us:
///
/// ```text
///   hidden   scalar256  scalar512   vec256  vec512  vec1024
///     2048        4.38       3.68     2.72    2.93     3.31
///     2816        6.17       4.83     3.46    3.12     3.51
///     5376        8.48       6.55     4.44    4.07     4.02
/// ```
///
/// Against the shipping scalar/256 that is −38%, −49% and −53%. The
/// vectorized twin is **bit-identical** to the scalar form at all three sizes
/// (0 of 2048/2816/5376 bf16 values differ) — only the two sum reductions
/// reassociate, and at these lengths that rounds to the same bf16.
///
/// vec512 is chosen above hidden 2560 and vec256 below: it is best at 2816,
/// within 1.5% of best at 5376, and the 2048 case prefers the narrower block.
/// Scalar keeps hidden < 2560's old width only when the rows are unaligned —
/// which is the `BLOCK = 512` at `:179`, and it is 512 for every width
/// because the unaligned path is the one the sweep could not improve.
///
/// # Safety
///
/// `x`, `weight`, `hidden`, `next_weight` and `norm_out` must address live
/// device memory of `num_rows * hidden_size` (the two weights, `hidden_size`)
/// bf16 elements, and `stream` must be live across the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn residual_add_scale_rmsnorm(
    x: *const c_void,
    weight: *const c_void,
    hidden: *mut c_void,
    scale: f32,
    next_weight: *const c_void,
    norm_out: *mut c_void,
    num_rows: i32,
    hidden_size: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // `rmsnorm.cu:152` — `dim3 grid(num_rows)`. See `strided` for why an
    // empty batch returns instead of launching one.
    if num_rows <= 0 {
        return;
    }
    let grid = [num_rows.unsigned_abs(), 1, 1];
    let values = [
        ArgValue::Ptr(x.cast_mut()),
        ArgValue::Ptr(weight.cast_mut()),
        ArgValue::Ptr(hidden),
        ArgValue::F32(scale),
        ArgValue::Ptr(next_weight.cast_mut()),
        ArgValue::Ptr(norm_out),
        ArgValue::I32(hidden_size),
        ArgValue::F32(eps),
    ];
    // `rmsnorm.cu:153-158` — `vec_ok`. FIVE addresses and one width, and note
    // that it is NOT `rmsnorm_vec8_ok`: there are no strides here (the kernel
    // is packed by construction) and there are two more buffers.
    let vec_ok = hidden_size % 8 == 0
        && aligned16(x)
        && aligned16(hidden.cast_const())
        && aligned16(norm_out.cast_const())
        && aligned16(weight)
        && aligned16(next_weight);
    if vec_ok {
        if hidden_size >= RASR_VEC512_ABOVE {
            // `rmsnorm.cu:160-168` — `constexpr int kB = 512;`
            // `device::rmsnorm_rasr_vec8<kB><<<grid, kB, 0, stream>>>(...)`.
            fire(
                "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_512",
                Launch { grid, block: [512, 1, 1], smem: 0 },
                &values,
                stream,
            );
            return;
        }
        // `rmsnorm.cu:170-177` — `constexpr int kB = 256;`
        // `device::rmsnorm_rasr_vec8<kB><<<grid, kB, 0, stream>>>(...)`.
        fire(
            "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#vec8_256",
            Launch { grid, block: [256, 1, 1], smem: 0 },
            &values,
            stream,
        );
        return;
    }
    // `rmsnorm.cu:179-189` — `constexpr int BLOCK = 512;`
    // `device::rmsnorm_residual_add_scale_rmsnorm<device::bf16, BLOCK>
    //     <<<grid, block, 0, stream>>>(...)`.
    fire(
        "norm::rmsnorm_residual_add_scale_rmsnorm_bf16#scalar_512",
        Launch { grid, block: [512, 1, 1], smem: 0 },
        &values,
        stream,
    );
}

// # `norm::rmsnorm_gated_fp32_in_bf16` is not here, and that is the whole of
// its migration
//
// `rmsnorm.cu:199` launched `device::rmsnorm_gated_f32_in<device::bf16, 256>`
// at `<<<num_rows, 256>>>`. The symbol is already named in
// `device::JIT_DISPATCHED` (`RMSNORM_SIGS[8]`, `LaunchRule::RowsPerHead`), so
// `emit_c_shim` emitted no entry for it and nothing in the tree reached that
// launcher — it was dead C++ waiting for its file to go. It went with the
// file. A port would have been a second, unreachable copy of a routed row.
