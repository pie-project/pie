//! The GEMM family: every matmul this backend can launch.
//!
//! Five host programs — [`dense`], [`quant`], [`gemv`], [`absorb`] and
//! [`lora`] — behind one flat set of feature-free entry points, each handing
//! `ctx.cublas()?` to a `_cuda`-gated half so a CUDA-less build refuses as a
//! value. Extents are `i32` — cuBLAS's `m`/`n`/`k` and leading dimensions —
//! so a zero refuses as `Empty`.

#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Routine};
use crate::{driver_bound, routine};
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::routine::{In, Out, Weight, Unbound};

use core::ffi::c_void;

/// MLA's absorb pair: two `cublasGemmStridedBatchedEx` over the head axis,
/// feature-free like [`gemv`] since both are in [`ROUTINES`].
pub mod absorb;
/// The dense matmul's host program: the autotuner, the plan cache and the
/// cuBLASLt recipes. `_cuda`-gated: its struct fields name `cudarc` types.
#[cfg(feature = "_cuda")]
pub mod dense;
/// The GEMV's host program: selection among four instantiations, launched
/// through `Ctx` like every other family.
pub mod gemv;
/// The LoRA adapter correction's launch half: three matmul passes over a
/// staged lane set. Feature-free — it names no `cudarc` type at all.
pub mod lora;
/// The quantised matmuls' host program: `gemm.cpp`'s router, cuBLASLt
/// recipes and caches. `_cuda`-gated for [`dense`]'s reason, though its
/// three entry points are feature-free.
#[cfg(feature = "_cuda")]
pub mod quant;

/// `gemm::act_x_wt_bf16` — the dense matmul, tactic-selected.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle. An empty
/// rectangle is not refused: `M == 0` reaches cuBLAS and reports success.
///
/// # Safety
///
/// `act`, `w` and `y` must address `M*K`, `N*K` and `M*N` live bf16 elements
/// and outlive the launch (asynchronous on the handle's stream).
#[kernels_macros::routine]
pub fn act_x_wt_bf16(
    ctx: &Ctx,
    act: In<0, c_void>,
    // Bound as `cx.weight_named(0)?`: a weight has no region, so `N` comes
    // off `y` rather than off this bank.
    w: Weight<0, *const c_void>,
    y: Out<0, c_void>,
    // Unsourced: 0.0 or 1.0 depends on which symbol the trace states, not on
    // an operand.
    beta: f32,
) -> Result<(), Refusal> {
    // `m` is `y`'s rows; `n` and `k` are `y`'s and `act`'s strides — cuBLAS's
    // `ldb`/`lda`, pitches not extents. A swap compiles; cuBLAS executes it.
    let dst = crate::layout::stated(y.all("n or k"))?;
    let src = crate::layout::stated(act.all("n or k"))?;
    let m = dst.rows;
    let n = dst.stride;
    let k = src.stride;
    // SAFETY: the three matrices address live device memory of the extents
    // `m`, `n` and `k` describe.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(ctx.cublas()?, act.ptr, w.ptr, y.ptr, m, n.0, k.0, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.ptr, y.ptr, m, n.0, k.0, beta);
    Ok(())
}

/// `gemm::act_x_w` — the spelling the lowering emits, `beta = 0`.
///
/// # Safety
///
/// [`act_x_wt_bf16`]'s.
#[kernels_macros::routine]
pub fn act_x_w(
    ctx: &Ctx,
    act: In<0, c_void>,
    w: Weight<0, *const c_void>,
    y: Out<0, c_void>,
    // `OpKind::Matmul::beta_one` picks the symbol.
    #[lit(0.0)] beta: f32,
) -> Result<(), Refusal> {
    act_x_wt_bf16(ctx, act, w, y, beta)
}

/// `gemm::act_x_w_acc` — the same call with `beta = 1`.
///
/// # Safety
///
/// [`act_x_wt_bf16`]'s, and `residual` aliases `y`.
#[kernels_macros::routine]
pub fn act_x_w_acc(
    ctx: &Ctx,
    act: In<0, c_void>,
    w: Weight<0, *const c_void>,
    y: Out<0, c_void>,
    // Aliases `y`, one buffer named twice; taken and never read so the
    // allocator doesn't hand this launch a buffer nothing wrote.
    residual: In<1, c_void>,
    #[lit(1.0)] beta: f32,
) -> Result<(), Refusal> {
    let _ = residual.all("the residual")?;
    act_x_wt_bf16(ctx, act, w, y, beta)
}

/// `gemm::act_x_wt_bf16_out_fp32` — one `cublasGemmEx`, bf16 in, fp32 out.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle — as
/// [`act_x_wt_bf16`], including the empty rectangle.
///
/// # Safety
///
/// `act` and `w` must address `M*K` and `N*K` live bf16 elements, `y` must
/// address `M*N` live floats, and all three must outlive the launch.
#[kernels_macros::routine]
pub fn act_x_wt_bf16_out_fp32(
    ctx: &Ctx,
    act: In<0, c_void>,
    // The named weight, as [`act_x_wt_bf16`]'s.
    w: Weight<0, *const c_void>,
    y: Out<0, f32>,
) -> Result<(), Refusal> {
    // [`act_x_wt_bf16`]'s `m`/`n`/`k` hazard; the one `gemm` symbol a binder
    // resolves (`bind/arms/gemm.rs`).
    let dst = crate::layout::stated(y.all("n or k"))?;
    let src = crate::layout::stated(act.all("n or k"))?;
    let m = dst.rows;
    let n = dst.stride;
    let k = src.stride;
    // SAFETY: as [`act_x_wt_bf16`]'s, with `y` addressing `m * n` floats.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16_out_fp32(ctx.cublas()?, act.ptr, w.ptr, y.ptr, m, n.0, k.0);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.ptr, y.ptr, m, n.0, k.0);
    Ok(())
}

/// `gemm::grouped_act_x_wt_bf16` — one `cublasGemmGroupedBatchedEx`.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle. An empty
/// group list is a legal no-op, reported as success (callers use `?`).
///
/// # Safety
///
/// The three pointer arrays must be device arrays of `group_count`
/// addresses; `m_array_host` is the one host array. Host addresses for
/// the pointer arrays fault at the next synchronize.
#[kernels_macros::routine]
pub fn grouped_act_x_wt_bf16(
    ctx: &Ctx,
    // `Unbound`, not `In`: `lora.rs` passes pointer arithmetic into its own
    // device pointer slab here, not a statement operand. `m_array_host` is
    // the one host array of the four.
    act_ptrs_dev: Unbound<*const *const c_void>,
    w_ptrs_dev: Unbound<*const *const c_void>,
    y_ptrs_dev: Unbound<*const *mut c_void>,
    m_array_host: Unbound<*const i32>,
    // Unmarked: fire-global (`routine!` below spells this launcher `whole`),
    // not any fire's row count or region width.
    group_count: i32,
    // Shared across the group, marked by width since no region is built.
    // Adjacent `i32`s: a swap is a GEMM against the wrong shared extent,
    // executed rather than rejected.
    #[source(OutWidth(0))] n: i32,
    #[source(InWidth(0))] k: i32,
    // [`act_x_wt_bf16`]'s `beta`, unmarked for its reason.
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the three pointer arrays are device arrays of `group_count`
    // addresses, per the doc above.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::grouped_act_x_wt_bf16(
            handle,
            act_ptrs_dev.ptr,
            w_ptrs_dev.ptr,
            y_ptrs_dev.ptr,
            m_array_host.ptr,
            group_count,
            n,
            k,
            beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ =
        (handle, act_ptrs_dev, w_ptrs_dev, y_ptrs_dev, m_array_host, group_count, n, k, beta);
    Ok(())
}

/// `gemm::act_x_wt_bias_bf16` — the dense matmul plus a fused bias add.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle, plus
/// whatever the bias add refuses — not an empty rectangle; see the guard
/// below.
///
/// # Safety
///
/// `act`, `w`, `bias` and `y` must address live device memory of the extents
/// `M`, `N` and `K` describe, and `y` must be writable.
#[kernels_macros::routine]
pub fn act_x_wt_bias_bf16(
    ctx: &Ctx,
    act: In<0, c_void>,
    // Two named weights in the statement's order (`w`, then `bias`);
    // swapping them compiles and binds the wrong tensor.
    w: Weight<0, *const c_void>,
    bias: Weight<1, *const c_void>,
    y: Out<0, c_void>,
    // [`act_x_wt_bf16`]'s `beta`: unsourced since an accumulating bias
    // projection is not stated today; a `Lit` would be a guess.
    beta: f32,
) -> Result<(), Refusal> {
    // [`act_x_wt_bf16`]'s `m`/`n`/`k` hazard again. `y`'s own extents are
    // forwarded to `norm::add_bias` below rather than restated.
    let dst = crate::layout::stated(y.all("n or k"))?;
    let src = crate::layout::stated(act.all("n or k"))?;
    let m = dst.rows;
    let n = dst.stride;
    let k = src.stride;
    // SAFETY: as [`act_x_wt_bf16`]'s, plus `bias` addressing `n` bf16
    // elements when it is not null.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::act_x_wt_bf16(ctx.cublas()?, act.ptr, w.ptr, y.ptr, m, n.0, k.0, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.ptr, y.ptr, m, n.0, k.0, beta);
    if bias.ptr.is_null() {
        return Ok(());
    }
    // Guards `m` only: `norm::add_bias` routes through `Ctx::launch`, which
    // refuses a zero-row grid, so this keeps parity with the GEMM's own
    // empty-rectangle success above. `n` was already refused by the view.
    if m <= 0 {
        return Ok(());
    }
    // Reads what the GEMM above wrote, on the same stream — this file's one
    // routine that fires another family's. `bias` is rewrapped rather than
    // forwarded: `add_bias`'s bias is its own bank zero, so a `Weight<1, _>`
    // would claim the wrong bank on the callee's side.
    crate::norm::add_bias::<bf16>(
        ctx,
        Out { ptr: dst.ptr.cast::<bf16>(), rows: dst.rows, width: dst.width },
        Weight { ptr: bias.ptr.cast::<bf16>() },
    )
}

// The quantised three: `quant`'s bodies, [`act_x_wt_bf16`]'s shape, but
// `driver_bound!` below and not `routine!` — a statement states values, not
// a weight's dtype, byte count, or scale representation, so none of these
// parameters carry a `#[source(..)]`, `m`/`n`/`k` included.

/// `gemm::act_x_wt_channel_scaled` — one scale per output channel of `W`.
///
/// # Errors
///
/// [`Refusal::Absent`] if this context carries no cuBLAS handle. Everything
/// else the router refuses, it panics on — see `quant`'s header.
///
/// # Safety
///
/// Every pointer must address live device memory of the extents `m`, `n` and
/// `k` describe, `w` holding at least `n * k` elements of `w_dtype` and
/// `scale` at least `n` values.
#[kernels_macros::routine]
pub fn act_x_wt_channel_scaled(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    zero_point: *const c_void,
    channel_axis: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the caller's obligation, forwarded.
    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_channel_scaled(
            handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point,
            channel_axis, y, m, n, k, beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point,
        channel_axis, y, m, n, k, beta,
    );
    Ok(())
}

/// `gemm::act_x_wt_grouped_scaled` — one scale per group along `K`, and for
/// FP8 per 2-D block.
///
/// # Errors
///
/// As [`act_x_wt_channel_scaled`].
///
/// # Safety
///
/// As [`act_x_wt_channel_scaled`], with the scale count `group_size` implies.
#[kernels_macros::routine]
pub fn act_x_wt_grouped_scaled(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    w_dtype: i32,
    w_nbytes: usize,
    scale: *const c_void,
    scale_dtype: i32,
    scale_numel: usize,
    zero_point: *const c_void,
    group_size: i32,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the caller's obligation, forwarded.
    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_grouped_scaled(
            handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point,
            group_size, y, m, n, k, beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (
        handle, act, w, w_dtype, w_nbytes, scale, scale_dtype, scale_numel, zero_point, group_size,
        y, m, n, k, beta,
    );
    Ok(())
}

/// `gemm::act_x_wt_mxfp4_marlin` — nibble-packed MXFP4 with E8M0 block
/// scales, dequanted and run through the classic GEMM.
///
/// # Errors
///
/// As [`act_x_wt_channel_scaled`].
///
/// # Safety
///
/// `w` must hold at least `ceil(n * k / 2)` bytes and `scale` at least
/// `n * ceil(k / 32)` bytes.
#[kernels_macros::routine]
pub fn act_x_wt_mxfp4_marlin(
    ctx: &Ctx,
    act: *const c_void,
    w: *const c_void,
    w_nbytes: usize,
    scale: *const c_void,
    scale_numel: usize,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32,
) -> Result<(), Refusal> {
    let handle = ctx.cublas()?;
    // SAFETY: the caller's obligation, forwarded.
    #[cfg(feature = "_cuda")]
    unsafe {
        quant::act_x_wt_mxfp4_marlin(
            handle, act, w, w_nbytes, scale, scale_numel, y, m, n, k, beta,
        );
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (handle, act, w, w_nbytes, scale, scale_numel, y, m, n, k, beta);
    Ok(())
}

/// The GEMV, re-exported at family level so `routine!` can name it.
pub use gemv::gemv_bf16;

/// MLA's absorb pair, re-exported at family level for [`gemv_bf16`]'s reason.
pub use absorb::{mla_absorb_latent_to_v_bf16, mla_absorb_q_to_latent_bf16};

/// The adapter correction, re-exported for the same reason as the two above.
pub use lora::lora_qkv_correction;

// The derived column: `#[kernels_macros::routine]` derives a
// `&[kernels::Derived]` from each signature alone. Only
// [`act_x_wt_bf16_out_fp32`] resolves it — `bind/arms/gemm.rs` names it. The
// rest decline: `beta` (on [`act_x_wt_bf16`], [`act_x_wt_bias_bf16`],
// `gemv_bf16`) depends on which symbol fired, not on an operand;
// [`grouped_act_x_wt_bf16`]'s pointer parameters mix host and device arrays
// with no `Source` to tell them apart; the two `mla_absorb_*` extents are
// `Route::Driver`, whose `bind/mod.rs` call site still spells them as bare
// `i32`s; and the quantised three plus `lora_qkv_correction` are excluded by
// `driver_bound!`'s `args: &[]` construction.

const _: () = {
    // Pins `w` as the named weight, not a positional input.
    assert!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[3].source.is_none());

    // Pins the two slot indices as stated by the signature, not arrived at
    // by counting.
    assert!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[2].stated);

    // The one column in this file a fire actually resolves (`bind/arms/gemm.rs`).
    assert!(<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED[2].stated);

    // Pins the two banks' order: swapping `w` and `bias` still compiles and
    // binds the wrong tensor.
    assert!(<act_x_wt_bias_bf16 as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(kernels::source_is_named(&<act_x_wt_bias_bf16 as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<act_x_wt_bias_bf16 as ::kernels::Derivation>::DERIVED[2].source, <kernels::keys::NamedWeight2 as kernels::keys::Fact>::KEY));
    assert!(matches!(<act_x_wt_bias_bf16 as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `beta`, the entry this row is blocked on.
    assert!(<act_x_wt_bias_bf16 as ::kernels::Derivation>::DERIVED[4].source.is_none());

    // Entries 0..=3 are all `None`: the column claims no operands, even for
    // `y_ptrs_dev`'s `*mut` pointee.
    assert!(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED.len() == 8);
    assert!(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[2].source.is_none());
    // Its group count: `whole` is the reason it carries no mark.
    assert!(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[4].source.is_none());
    // Name pins: 0..=3 are four `Unbound`s, and the first two share type
    // `*const *const c_void` — swapping them compiles and passes every
    // `source` pin above. Only the name catches it.
    assert!(crate::layout::derived_name_is(
        <grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[0].name,
        "act_ptrs_dev"
    ));
    assert!(crate::layout::derived_name_is(
        <grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[1].name,
        "w_ptrs_dev"
    ));
    assert!(crate::layout::derived_name_is(
        <grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[2].name,
        "y_ptrs_dev"
    ));
    assert!(crate::layout::derived_name_is(
        <grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[3].name,
        "m_array_host"
    ));
    assert!(crate::layout::derived_name_is(
        <grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[4].name,
        "group_count"
    ));
    // [5]/[6] are `n`/`k`, adjacent `i32`s the name pins rather than the
    // source: a swap is a GEMM against the wrong shared extent, and cuBLAS
    // executes it.
    assert!(crate::layout::derived_name_is(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[5].name, "n"));
    assert!(crate::layout::derived_name_is(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[6].name, "k"));
    assert!(crate::layout::derived_name_is(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[7].name, "beta"));

    // The leg, pinned whole: `dense.rs`'s tuner calls it with typed Rust,
    // which checks argument order but not meaning. It keeps `InRow`/`OutRow`
    // rather than its parent's regions because the tuner holds raw pointers,
    // not statement operands.
    assert!(<gemv::gemv_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(kernels::source_is_named(&<gemv::gemv_bf16 as ::kernels::Derivation>::DERIVED[0].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(matches!(<gemv::gemv_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<gemv::gemv_bf16 as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `beta` again: the leg is blocked in its own right, since
    // `run_dense_tactic` also refuses any value but 0.0 or 1.0.
    assert!(<gemv::gemv_bf16 as ::kernels::Derivation>::DERIVED[3].source.is_none());

    // `kv_b_proj`'s positional bank, pinned against [`act_x_wt_bf16`]'s named
    // one — swapping `Weight(0)` for a named source binds a plausible
    // pointer from the wrong side.
    assert!(matches!(<absorb::mla_absorb_q_to_latent_bf16 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Weight, 0))));
    // The four extents this row is blocked on: refused by design in
    // `bind/mod.rs`'s `mla_absorb`, not by omission.
    assert!(<absorb::mla_absorb_q_to_latent_bf16 as ::kernels::Derivation>::DERIVED[4].source.is_none());

    // The full inversion: an aux slab read as `In(0)`, and two statement
    // inputs read as outputs because they are spelled `*mut`.
    assert!(<lora::lora_qkv_correction as ::kernels::Derivation>::DERIVED.len() == 9);
    assert!(matches!(<lora::lora_qkv_correction as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<lora::lora_qkv_correction as ::kernels::Derivation>::DERIVED[6].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(<lora::lora_qkv_correction as ::kernels::Derivation>::DERIVED[7].source, Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
};

/// This family's routines: the four dense entry points, the GEMV, MLA's
/// absorb pair, and the four forms the driver fires by path. The absorbs are
/// here rather than in `attn` because a trace states them as `gemm::`.
pub static ROUTINES: &[Routine] = &[
    routine!(act_x_wt_bf16, ),
    routine!(act_x_w, ),
    routine!(act_x_w_acc, in_place = &[(1, 0)]),
    routine!(act_x_wt_bf16_out_fp32, ),
    routine!(grouped_act_x_wt_bf16, whole, ),
    routine!(act_x_wt_bias_bf16, ),
    // A leg's column is checked against its parent: `dense` forwards
    // `act_x_wt_bf16`'s own marks and drops only `m`; not `driver_bound!`,
    // since every parameter here is a plain `Arg`.
    routine!(gemv_bf16),
    // Neither crosses, but the marks make `bind/mod.rs`'s hand binding
    // legible: the four `None`s say the rest rides `spec.params`, refused by
    // design.
    routine!(mla_absorb_q_to_latent_bf16),
    routine!(mla_absorb_latent_to_v_bf16),
    // What the driver fires by path: `driver_bound!`, since no statement
    // supplies a weight's representation for a `routine!` row to recover.
    driver_bound!(act_x_wt_channel_scaled),
    driver_bound!(act_x_wt_grouped_scaled),
    driver_bound!(act_x_wt_mxfp4_marlin),
    // Declines for a different reason: its arguments are a `Staged` borrow
    // an arena `Ctx` cannot offer, not unstatable values.
    driver_bound!(lora_qkv_correction),
];

/// `gemm`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);
