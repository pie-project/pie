//! The GEMM family: every matmul this backend can launch.
//!
//! Five host programs — [`dense`], [`quant`], [`gemv`], [`absorb`] and
//! [`lora`] — behind one flat set of feature-free entry points, each handing
//! `ctx.cublas()?` to a `_cuda`-gated half so a CUDA-less build refuses as a
//! value. Extents are `i32` — cuBLAS's `m`/`n`/`k` and leading dimensions —
//! so a zero refuses as `Empty`.

use kernels::keys;
use kernels_macros::routine;
use crate::jit::Ctx;
use crate::jit::abi::bf16;
use crate::jit::abi::Tensor;
use kernels::Refusal;
use kernels::routine::{Asks, Const, In, InOut, Out};

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
#[routine]
pub fn act_x_wt_bf16(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    // Bound as `cx.weight_named(0)?`: a weight has no region, so `N` comes
    // off `y` rather than off this bank.
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>) -> Result<(), Refusal> {
    // `beta = 0`: this symbol is the plain `x · Wᵀ`, no accumulate. Like
    // `BetaZero`/`BetaOne` (migration.md §3.9) the SYMBOL decides the
    // number, so it is a body literal here, never a statement operand or an
    // environment fact.
    let beta = 0.0f32;
    act_x_wt_bf16_beta(ctx, act, w, y, beta)
}

/// The cuBLAS call [`act_x_wt_bf16`], [`act_x_w`] and [`act_x_w_acc`] share,
/// parameterised over `beta` — the one thing that tells the three apart.
///
/// Not itself a `#[routine]`: a routine's every parameter must be a mark,
/// and `beta` is a plain number each of the three callers bakes in as a
/// literal rather than reading from a statement or the environment, so it
/// stays an ordinary `fn` argument on the shared inner call.
fn act_x_wt_bf16_beta(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>,
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
        dense::act_x_wt_bf16(ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    Ok(())
}

/// `gemm::act_x_w` — the spelling the lowering emits, `beta = 0`.
///
/// # Safety
///
/// [`act_x_wt_bf16`]'s.
#[routine]
pub fn act_x_w(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>) -> Result<(), Refusal> {
    let beta = 0.0f32;
    act_x_wt_bf16_beta(ctx, act, w, y, beta)
}

/// `gemm::act_x_w_acc` — the same call with `beta = 1`.
///
/// # Safety
///
/// [`act_x_wt_bf16`]'s, and `residual` aliases `y`.
#[routine]
pub fn act_x_w_acc(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    w: Const<Tensor<c_void>>,
    // ONE BUFFER THAT USED TO BE NAMED TWICE. `y: Out<0, _>` and `residual:
    // In<1, _>` were one address, and the row said so beside them as
    // `in_place`; `InOut` is that sentence at the parameter, and its position
    // is what makes it operand 1 -- which is the half `in_place = &[(1, 0)]`
    // got backwards, against its own `(output, input)` order.
    //
    // `beta = 1` is why it is read as well as written: the accumulate reads
    // the residual out of the destination it then overwrites.
    y: InOut<Tensor<c_void>>) -> Result<(), Refusal> {
    let beta = 1.0f32;
    // The same address under the mark the shared call takes: `InOut` and
    // `Out` differ in what the STATEMENT did, not in what cuBLAS writes.
    let dst = Out { ptr: y.ptr, rows: y.rows, width: y.width };
    act_x_wt_bf16_beta(ctx, act, w, dst, beta)
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
#[routine]
pub fn act_x_wt_bf16_out_fp32(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    // The named weight, as [`act_x_wt_bf16`]'s.
    w: Const<Tensor<c_void>>,
    y: Out<Tensor<f32>>) -> Result<(), Refusal> {
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
        dense::act_x_wt_bf16_out_fp32(ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0);
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
#[routine(whole)]
pub fn grouped_act_x_wt_bf16(
    ctx: &Ctx<'_>,
    // `Unbound`, not `In`: `lora.rs` passes pointer arithmetic into its own
    // device pointer slab here, not a statement operand. `m_array_host` is
    // the one host array of the four.
    act_ptrs_dev: *const *const c_void,
    w_ptrs_dev: *const *const c_void,
    y_ptrs_dev: *const *mut c_void,
    m_array_host: *const i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<i32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    group_count: i32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    beta: f32) -> Result<(), Refusal> {
    let n = ctx.ask::<i32, keys::OutWidth0>()?;
    let k = ctx.ask::<i32, keys::InWidth0>()?;
    let handle = ctx.cublas()?;
    // SAFETY: the three pointer arrays are device arrays of `group_count`
    // addresses, per the doc above.
    #[cfg(feature = "_cuda")]
    unsafe {
        dense::grouped_act_x_wt_bf16(
            handle,
            act_ptrs_dev,
            w_ptrs_dev,
            y_ptrs_dev,
            m_array_host,
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
#[routine]
pub fn act_x_wt_bias_bf16(
    ctx: &Ctx<'_>,
    act: In<Tensor<c_void>>,
    // Two named weights in the statement's order (`w`, then `bias`);
    // swapping them compiles and binds the wrong tensor.
    w: Const<Tensor<c_void>>,
    bias: Const<Tensor<c_void>>,
    y: Out<Tensor<c_void>>) -> Result<(), Refusal> {
    // ZERO, AS `act_x_w`'s IS, and for the same reason stated the same way:
    // this statement PRODUCES `y`, so there is nothing in it to accumulate
    // into. `beta` was `#[unbound]` and that was the one entry keeping the
    // row out of the binder -- "the entry this row is blocked on", says the
    // pin below -- so gpt-oss's biased projections had a routine, a column,
    // a golden and no way to fire. The number that separates the twins is
    // stated by the SYMBOL here exactly as it is there.
    let beta = 0.0f32;
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
        dense::act_x_wt_bf16(ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = (ctx.cublas()?, act.ptr, w.v, y.ptr, m, n.0, k.0, beta);
    if bias.v.is_null() {
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
        InOut { ptr: dst.ptr.cast::<bf16>(), rows: dst.rows, width: dst.width },
        Const { v: bias.v.cast::<bf16>() },
    )
}

// The quantised three: `quant`'s bodies, [`act_x_wt_bf16`]'s shape, but
// `untraced!` below and not `routine!` — a statement states values, not
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
#[routine(untraced)]
pub fn act_x_wt_channel_scaled(
    ctx: &Ctx<'_>,
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
    beta: f32) -> Result<(), Refusal> {
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
#[routine(untraced)]
pub fn act_x_wt_grouped_scaled(
    ctx: &Ctx<'_>,
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
    beta: f32) -> Result<(), Refusal> {
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
#[routine(untraced)]
pub fn act_x_wt_mxfp4_marlin(
    ctx: &Ctx<'_>,
    act: *const c_void,
    w: *const c_void,
    w_nbytes: usize,
    scale: *const c_void,
    scale_numel: usize,
    y: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    beta: f32) -> Result<(), Refusal> {
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

// The derived column: `#[routine]` derives a
// `&[kernels::Derived]` from each signature alone. Only
// [`act_x_wt_bf16_out_fp32`] resolves it — `bind/arms/gemm.rs` names it. The
// rest decline: `beta` (on [`act_x_wt_bf16`], [`act_x_wt_bias_bf16`],
// `gemv_bf16`) depends on which symbol fired, not on an operand;
// [`grouped_act_x_wt_bf16`]'s pointer parameters mix host and device arrays
// with no `Source` to tell them apart; the two `mla_absorb_*` extents are
// `Route::Driver`, whose `bind/mod.rs` call site still spells them as bare
// `i32`s; and the quantised three plus `lora_qkv_correction` are excluded by
// `untraced!`'s `args: &[]` construction.

const _: () = {
    // Pins `w` as the named weight, not a positional input.
    assert!(<act_x_wt_bf16 as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bf16)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bf16)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bf16)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

    // Pins the two slot indices as stated by the signature, not arrived at
    // by counting.

    // The one column in this file a fire actually resolves (`bind/arms/gemm.rs`).
    assert!(<act_x_wt_bf16_out_fp32 as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bf16_out_fp32)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bf16_out_fp32)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bf16_out_fp32)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // Pins the two banks' order: swapping `w` and `bias` still compiles and
    // binds the wrong tensor.
    assert!(<act_x_wt_bias_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bias_bf16)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bias_bf16)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(act_x_wt_bias_bf16)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `beta` LEFT THE COLUMN: the symbol states it, so there is no fifth
    // entry and nothing left for this row to be blocked on.

    // Entries 0..=3 are all `None`: the column claims no operands, even for
    // `y_ptrs_dev`'s `*mut` pointee.
    assert!(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED.len() == 6);
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(grouped_act_x_wt_bf16)[2].is_none());
    // Its group count: `whole` is the reason it carries no mark.
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(grouped_act_x_wt_bf16)[4].is_none());
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
    // `n` AND `k` LEFT THE PARAMETER LIST. They were `Env<i32, keys::
    // OutWidth0>` and `Env<i32, keys::InWidth0>` -- the rectangle's own
    // extents -- and the body asks its context for them now, so there is no
    // parameter left to hold apart by name. What still needs holding apart is
    // the pair that stayed: `group_count` and `beta`, adjacent and both the
    // arm's own.
    assert!(crate::layout::derived_name_is(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[4].name, "group_count"));
    assert!(crate::layout::derived_name_is(<grouped_act_x_wt_bf16 as ::kernels::Derivation>::DERIVED[5].name, "beta"));

    // The leg, pinned whole: `dense.rs`'s tuner calls it with typed Rust,
    // which checks argument order but not meaning. It keeps `InRow`/`OutRow`
    // rather than its parent's regions because the tuner holds raw pointers,
    // not statement operands.
    assert!(<gemv::gemv_bf16 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gemv::gemv_bf16)[0], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gemv::gemv_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gemv::gemv_bf16)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // `beta` again: the leg is blocked in its own right, since
    // `run_dense_tactic` also refuses any value but 0.0 or 1.0.
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gemv::gemv_bf16)[3].is_none());

    // `kv_b_proj` IS THE CHAIN, not the bare bank, and the change is the
    // whole of §3.8. `Weight<E>` derived `Slot(Kind::Weight, 0)` where the row
    // spelled the positional form and `Named("weight")` where it spelled the
    // named one -- two marks for one operand, so a routine had to know which
    // SHAPE of op would reach it, and reading only the name is what made
    // gemma-4 refuse at its PLE prologue. `Const<Tensor<E>>` inherits one
    // chain for both: *"the named bank first and the positional one after"*,
    // which is what the binder already tried in that order.
    assert!(matches!(
        kernels::routine::sources::<crate::jit::Cuda, _, _>(absorb::mla_absorb_q_to_latent_bf16)[1],
        Some(kernels::Source::Or(
            kernels::Source::Named(_),
            kernels::Source::Slot(kernels::Kind::Weight, 0)
        ))
    ));
    // The four extents this row is blocked on: refused by design in
    // `bind/mod.rs`'s `mla_absorb`, not by omission.
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(absorb::mla_absorb_q_to_latent_bf16)[4].is_none());

    // `lora_qkv_correction` IS PINNED THE OTHER WAY NOW: it carries NO column,
    // and that is the claim.
    //
    // It used to be pinned as *"the full inversion -- an aux slab read as
    // `In(0)`, and two statement inputs read as outputs because they are
    // spelled `*mut`"*, which is a description of a column that was wrong in
    // every entry. The row is `#[routine(untraced)]`: the driver fires it
    // through a typed call, and it takes a `Staged<'_>` -- an aggregate the
    // arm builds out of this fire's staging, which no `ArgValue` can carry and
    // no `Source` can name. So the honest column is the empty one, and a row
    // that grew one would mean somebody had started binding it from a
    // statement.
    // The NAMES survive -- they are read off the syntax and are what hold nine
    // adjacent arguments apart -- and the SOURCE column is the one that is
    // empty, because nothing binds them.
    assert!(<lora::lora_qkv_correction as ::kernels::Derivation>::DERIVED.len() == 9);
    assert!(<lora::lora_qkv_correction as ::kernels::Derivation>::SOURCES.is_empty());
};

