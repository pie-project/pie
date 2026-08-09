//! `norm/dsv4_hc.cu`'s host program, in Rust.
//!
//! # What the archive file was
//!
//! Four host launchers for deepseek-v4's hyper-connection residual, and no
//! `__global__`: every kernel is in
//! `kernels-cuda-new/csrc/src/norm/dsv4_hc.cuh` and is compiled by NVRTC.
//! §43.9 had already deleted three siblings — `hc_expand_bf16`,
//! `attn_sink_correction_bf16` and `per_head_rmsnorm_bf16` — by naming them
//! in `device::JIT_DISPATCHED`; these four could not follow, because their
//! ahead-of-time rows state **no `Source` on any operand**.
//!
//! # Why the rows are unsourced, and why that is not fixed here
//!
//! HC's mixing matrices are not values a statement names. `mixes`, `scale`
//! and `base` are three `float` slabs the layer carries, `post_mix` and
//! `comb_mix` are scratch the launcher hands from one kernel to the next, and
//! `sinkhorn_iters` and `hc_post_alpha` are model constants. A `Source` for
//! any of them would be a guess about where a lowering puts a buffer, and
//! **a half-bound row is worse than an unbound one**: `emit_dispatch` skips a
//! row with one unbound operand whole, so a row that sourced five of thirteen
//! generates exactly as much as a row that sources none, while claiming four
//! bindings nobody checked.
//!
//! So the four rows stay unsourced and honest. What this file changes is the
//! OTHER half: `execution::RUST_SERVED` names them, `emit_c_shim` stops
//! emitting `pie_k_norm_hc_*`, and `norm/dsv4_hc.cu` goes. The Rust below is
//! the program those entries forwarded to, and the day a `Source` can state
//! an HC slab the generated arm has something to call.
//!
//! # `MAX_HC_MULT` is a precondition, and it is a refusal
//!
//! `dsv4_hc.cuh:91` — `constexpr int MAX_HC_MULT = 8`. `hc_post` keeps its
//! `M` residual values in registers (`float r[MAX_HC_MULT]`), so a larger
//! multiplier is not a slower launch, it is an out-of-bounds write into a
//! register array. The launcher refused it at `dsv4_hc.cu:59` and this
//! refuses it in the same place, loudly — see [`hc_post_bf16`].

use std::ffi::c_void;

use kernels_cuda_new::runtime::{ArgValue, Launch};

use super::hand::fire;

/// `dsv4_hc.cu:18` — `constexpr int BLOCK = 256;`
///
/// One constant for all four launchers, which is what the file's anonymous
/// namespace made it.
const BLOCK: u32 = 256;

/// `dsv4_hc.cuh:91` — `constexpr int MAX_HC_MULT = 8;`
///
/// The width of `hc_post`'s register array, and therefore the largest
/// multiplier the kernel can be launched with. Stated here as well as in the
/// device text because the check that reads it is on this side.
const MAX_HC_MULT: i32 = 8;

/// `dsv4_hc.cu:22` — `norm::hc_pre_postprocess_bf16`.
///
/// The per-token mixing matrix: reads the three `float` slabs, runs
/// `sinkhorn_iters` normalisation passes over the `hc_mult × hc_mult` matrix
/// in shared memory, writes `post_mix` and `comb_mix` for the layer's
/// [`hc_post_bf16`] to read, and collapses the `hc_mult` residual streams into
/// the layer's bf16 input.
///
/// One block per token, striding the hidden axis.
///
/// # Safety
///
/// `residual` and `layer_input` must address `n * hc_mult * hidden_size` and
/// `n * hidden_size` live bf16 elements; `mixes`, `scale` and `base` the
/// slabs the layer carries; `post_mix` and `comb_mix` scratch of `n *
/// hc_mult` and `n * hc_mult * hc_mult` floats. `stream` must be live across
/// the launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_pre_postprocess_bf16(
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const c_void,
    post_mix: *mut f32,
    comb_mix: *mut f32,
    layer_input: *mut c_void,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    hc_eps: f32,
    hc_post_alpha: f32,
    sinkhorn_iters: i32,
    stream: *mut c_void,
) {
    // `dsv4_hc.cu:38` — `if (N <= 0) return;`
    if n <= 0 {
        return;
    }
    // `dsv4_hc.cu:39-44` —
    // `device::hc_pre_postprocess<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(...)`.
    fire(
        "norm::hc_pre_postprocess_rows_bf16",
        Launch { grid: [n.unsigned_abs(), 1, 1], block: [BLOCK, 1, 1], smem: 0 },
        &[
            ArgValue::Ptr(mixes.cast_mut().cast()),
            ArgValue::Ptr(scale.cast_mut().cast()),
            ArgValue::Ptr(base.cast_mut().cast()),
            ArgValue::Ptr(residual.cast_mut()),
            ArgValue::Ptr(post_mix.cast()),
            ArgValue::Ptr(comb_mix.cast()),
            ArgValue::Ptr(layer_input),
            ArgValue::I32(hc_mult),
            ArgValue::I32(hidden_size),
            ArgValue::F32(hc_eps),
            ArgValue::F32(hc_post_alpha),
            ArgValue::I32(sinkhorn_iters),
        ],
        stream,
    );
}

/// `dsv4_hc.cu:47` — `norm::hc_post_bf16`.
///
/// The write-back half: takes the layer's output and the `hc_mult` residual
/// streams it was collapsed from, and re-expands with the mixing weights
/// [`hc_pre_postprocess_bf16`] wrote. Elementwise over `n * hidden_size`, so
/// the grid is a slab and not a row count.
///
/// # THE REFUSAL
///
/// `dsv4_hc.cu:59` — `if (total <= 0 || hc_mult > device::MAX_HC_MULT) return;`
///
/// The C++ returned silently on both. The empty extent stays silent, because
/// nothing to do is not a refusal. **`hc_mult > MAX_HC_MULT` does not**: it
/// is a model whose hyper-connection width the compiled kernel cannot hold,
/// and a silent return there is a layer that reads its own uninitialised
/// residual and produces plausible tokens. The device header says so at
/// `dsv4_hc.cuh:228` — *"the `M > MAX_HC_MULT` refusal moved here from the
/// launcher, and it had to"* — the kernel now checks it as well, so this
/// panic is the diagnosis rather than the safety.
///
/// # Panics
///
/// If `hc_mult` exceeds `MAX_HC_MULT`.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out_residual` addressing `n * hc_mult
/// * hidden_size` live bf16 elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_post_bf16(
    x: *const c_void,
    residual: *const c_void,
    post_mix: *const f32,
    comb_mix: *const f32,
    out_residual: *mut c_void,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
) {
    assert!(
        hc_mult <= MAX_HC_MULT,
        "norm::hc_post_bf16: hc_mult={hc_mult} exceeds MAX_HC_MULT={MAX_HC_MULT} -- \
         `hc_post` holds its residual streams in a register array of that width \
         (`norm/dsv4_hc.cuh:91`), so this configuration has no kernel"
    );
    // `dsv4_hc.cu:58-60`:
    //   const long long total = (long long)N * hidden_size;
    //   if (total <= 0 || hc_mult > MAX_HC_MULT) return;
    //   const int grid = (total + BLOCK - 1) / BLOCK;
    let total = i64::from(n) * i64::from(hidden_size);
    if total <= 0 {
        return;
    }
    let grid = (total + i64::from(BLOCK) - 1) / i64::from(BLOCK);
    // `dsv4_hc.cu:61-66` —
    // `device::hc_post<device::bf16><<<grid, BLOCK, 0, stream>>>(...)`.
    fire(
        "norm::hc_post_elems_bf16",
        Launch {
            grid: [u32::try_from(grid).unwrap_or(u32::MAX), 1, 1],
            block: [BLOCK, 1, 1],
            smem: 0,
        },
        &[
            ArgValue::Ptr(x.cast_mut()),
            ArgValue::Ptr(residual.cast_mut()),
            ArgValue::Ptr(post_mix.cast_mut().cast()),
            ArgValue::Ptr(comb_mix.cast_mut().cast()),
            ArgValue::Ptr(out_residual),
            ArgValue::I32(n),
            ArgValue::I32(hc_mult),
            ArgValue::I32(hidden_size),
        ],
        stream,
    );
}

/// `dsv4_hc.cu:69` — `norm::hc_head_postprocess_bf16`.
///
/// The final collapse: the same mixing arithmetic as
/// [`hc_pre_postprocess_bf16`] but writing one bf16 stream rather than
/// scratch, for the LM head to read. No `post_mix`/`comb_mix` outputs,
/// because nothing after it re-expands.
///
/// Note the argument order: `hc_eps` comes **after** `stream` in the
/// launcher's C++ signature and therefore in the ahead-of-time row, and
/// before it in the kernel's. The row is the launcher's and the fire is the
/// kernel's, which is exactly the difference `execution::sig_of` documents.
///
/// # Safety
///
/// [`hc_pre_postprocess_bf16`]'s, with `out` addressing `n * hidden_size`
/// live bf16 elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hc_head_postprocess_bf16(
    mixes: *const f32,
    scale: *const f32,
    base: *const f32,
    residual: *const c_void,
    out: *mut c_void,
    n: i32,
    hc_mult: i32,
    hidden_size: i32,
    stream: *mut c_void,
    hc_eps: f32,
) {
    // `dsv4_hc.cu:81` — `if (N <= 0) return;`
    if n <= 0 {
        return;
    }
    // `dsv4_hc.cu:82-86` —
    // `device::hc_head_postprocess<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(...)`.
    fire(
        "norm::hc_head_postprocess_rows_bf16",
        Launch { grid: [n.unsigned_abs(), 1, 1], block: [BLOCK, 1, 1], smem: 0 },
        &[
            ArgValue::Ptr(mixes.cast_mut().cast()),
            ArgValue::Ptr(scale.cast_mut().cast()),
            ArgValue::Ptr(base.cast_mut().cast()),
            ArgValue::Ptr(residual.cast_mut()),
            ArgValue::Ptr(out),
            ArgValue::I32(hc_mult),
            ArgValue::I32(hidden_size),
            ArgValue::F32(hc_eps),
        ],
        stream,
    );
}

/// `dsv4_hc.cu:89` — `norm::hc_rmsnorm_to_f32`.
///
/// RMSNorm from bf16 into `float`. The `float` result is what the mixing
/// matrices are computed in, which is why this exists as its own symbol
/// rather than as a `norm::rmsnorm_*` with a wider output: the consumer is
/// [`hc_pre_postprocess_bf16`], not a GEMM.
///
/// One block per row.
///
/// # Safety
///
/// `input` must address `n * dim` live bf16 elements, `output` `n * dim` live
/// floats, and `stream` must be live across the launch.
pub unsafe fn hc_rmsnorm_to_f32(
    input: *const c_void,
    output: *mut f32,
    n: i32,
    dim: i32,
    eps: f32,
    stream: *mut c_void,
) {
    // `dsv4_hc.cu:97` — `if (N <= 0) return;`
    if n <= 0 {
        return;
    }
    // `dsv4_hc.cu:98-100` —
    // `device::hc_rmsnorm_to_f32<device::bf16, BLOCK><<<N, BLOCK, 0, stream>>>(...)`.
    fire(
        "norm::hc_rmsnorm_to_f32_rows",
        Launch { grid: [n.unsigned_abs(), 1, 1], block: [BLOCK, 1, 1], smem: 0 },
        &[
            ArgValue::Ptr(input.cast_mut()),
            ArgValue::Ptr(output.cast()),
            ArgValue::I32(dim),
            ArgValue::F32(eps),
        ],
        stream,
    );
}
