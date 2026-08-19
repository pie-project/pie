//! The MLP family: the gated activations between the two dense projections.
//!
//! [`swiglu`], [`swiglu_clamp`], [`situ`], [`geglu_tanh`] and [`gpt_oss_glu`]
//! take `gate` and `up` as two regions; the `chunked_*` forms take one packed
//! `rows * 2 * width` region and halve it themselves. Extents come off the
//! bound `In`/`Out` regions' `rows`/`width`, never a separate parameter.

use kernels::{Bind, Fire};
use kernels::routine::{Asks, Const, In, InOut, Out};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch};
use crate::jit::abi::Tensor;
use crate::jit::abi::{bf16, f16};
use kernels::Refusal;
use kernels::keys;

/// Threads per block, everywhere in this family.
const BLOCK: u32 = 256;

/// Threads per warp — the unit the reductions' shared scratch is counted in.
const WARP: u32 = 32;

// Widths reaching only `Ctx::launch`'s grid need no guard, since it refuses a
// zero grid itself. `sigmoid_dot_scalar_gate_add`'s `h` and `gaussian_topk`'s
// `dim` reach a kernel argument instead, so those two guard by hand.

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: i32) -> Launch {
    Launch::flat(n.unsigned_abs(), BLOCK)
}

/// `LaunchRule::ElementwiseRows`, as the expression it evaluates to.
#[must_use]
const fn elementwise_rows(rows: i32, width: i32) -> Launch {
    Launch::grid([rows.unsigned_abs(), width.unsigned_abs().div_ceil(BLOCK), 1], [BLOCK, 1, 1])
}

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: i32) -> Launch {
    /// The dynamic shared memory the two reducing kernels fold through.
    const RMS_SMEM: u32 = (BLOCK / WARP) * 4;

    Launch::per_row(rows.unsigned_abs(), BLOCK).smem(RMS_SMEM)
}

/// gpt-oss's GLU alpha, used as a fixed constant since no deployment fact
/// carries it yet.
pub const GPT_OSS_GLU_ALPHA: f32 = 1.702;

/// `y[i] = silu(gate[i]) * up[i]` over the result — `mlp::swiglu_bf16`.
///
/// # Safety
///
/// `gate` and `up` must address `y.rows * y.width` live bf16 elements and
/// `y` that many writable ones.
#[routine(bf16)]
pub fn swiglu<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);
    // SAFETY: `call()`'s contract — every pointer addresses live device
    // memory of the extent the kernel reads it as.
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::swiglu<{}>", T::CPP))).apply(elementwise(n)), &[gate.arg(), up.arg(), y.arg(), n.arg()])
}

/// The same with the gate clamped to `±limit` — `mlp::swiglu_clamp_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s.
#[routine(bf16)]
pub fn swiglu_clamp<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let limit = ctx.ask::<f32, keys::GluLimit>()?;

    let n = y.rows.saturating_mul(y.width);
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::swiglu_clamp<{}>", T::CPP))).apply(elementwise(n)), &[gate.arg(), up.arg(), y.arg(), n.arg(), limit.arg()])
}

/// SiTU — `mlp::situ_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s.
#[routine(bf16)]
pub fn situ<T>(
    ctx: &Ctx<'_>,
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    beta: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    linear_beta: f32) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::situ<{}>", T::CPP))).apply(elementwise(n)), &[gate.arg(), up.arg(), y.arg(), n.arg(), beta.arg(), linear_beta.arg()])
}

/// GeGLU-tanh — `mlp::geglu_tanh_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s. `y` may alias `gate`.
#[routine(bf16)]
pub fn geglu_tanh<T>(
    ctx: &Ctx<'_>,
    // TWO MARKS AND NOT ONE, and the row's old `in_place = &[(0, 0)]` is why
    // that has to be said. The pair was a MAY-ALIAS: through a statement the
    // allocator gave result 0 operand 0's offset, and the kernel takes the
    // gate and the destination as two pointers because it does not require
    // that -- `tower::gemma4_vision` calls this directly with three distinct
    // buffers. `InOut` claims ONE ADDRESS, which would be false there.
    //
    // So the hint is dropped rather than mis-stated. The four marks have no
    // word for *"the statement may place one buffer here"*; §11.5 records the
    // three routines that had one.
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::geglu_tanh<{}>", T::CPP))).apply(elementwise(n)), &[gate.arg(), up.arg(), y.arg(), n.arg()])
}

/// `y = max(x, 0)^2` — `mlp::relu2_bf16`.
///
/// # Safety
///
/// `x` must address `y.rows * y.width` live bf16 elements and `y` that many
/// writable ones.
#[routine(bf16)]
pub fn relu2<T>(ctx: &Ctx<'_>, x: In<Tensor<T>>, y: Out<Tensor<T>>) -> Result<(), Refusal> {
    let n = y.rows.saturating_mul(y.width);
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::relu2<{}>", T::CPP))).apply(elementwise(n)), &[x.arg(), y.arg(), n.arg()])
}

/// gpt-oss's clamped GLU — `mlp::gpt_oss_glu_bf16`.
///
/// # Safety
///
/// [`swiglu`]'s, plus: when `y_fp16` is `Some`, it must address
/// `y.rows * y.width` writable fp16 elements. `y` may alias `gate`.
#[routine(bf16)]
pub fn gpt_oss_glu<T>(
    ctx: &Ctx<'_>,
    // As [`geglu_tanh`]'s pair, and a may-alias for the same reason.
    gate: In<Tensor<T>>,
    up: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let limit = ctx.ask::<f32, keys::GluLimit>()?;
    let alpha = ctx.ask::<f32, keys::GluAlpha>()?;

    let n = y.rows.saturating_mul(y.width);
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::gpt_oss_glu<{}>", T::CPP))).apply(elementwise(n)), &[
                gate.arg(),
                up.arg(),
                y.arg(),
                // Always null: this fp16 side-output slot is never filled.
                ::core::ptr::null_mut::<f16>().arg(),
                n.arg(),
                limit.arg(),
                alpha.arg(),
            ])
}

/// SwiGLU over a packed gate‖up bank — `mlp::chunked_swiglu_bf16`.
///
/// # Safety
///
/// `packed` must address `y.rows * 2 * y.width` live bf16 elements and `y`
/// `y.rows * y.width` writable ones.
#[routine(bf16)]
pub fn chunked_swiglu<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    // Extent comes off `y` (`rows * width`), not `packed` (`rows * 2 * width`).
    //
    // `Out`, NOT `InOut`, and §11.6's rule is the one that decides it: *a
    // routine that fires its mark TWICE requires the alias, one that fires
    // two marks does not*. This fires `packed` and `y` separately, so the
    // kernel takes a source and a destination and does not need them to be
    // one address. HEAD said the same with `Out<0, T>`; its row's
    // `in_place = &[(0, 1)]` was an ALLOCATOR instruction beside the mark,
    // not a claim about the parameter, and `InOut` conflates the two.
    //
    // The dense MLP states one operand here, so an `InOut` — which derives
    // `Source::Alias(1, 0)` and reads INPUT 1 — refused every llama, qwen3
    // and gemma fire at its tenth launch with "the fire does not carry an
    // input operand". The statement shape that really does place a
    // destination is [`chunked_swiglu_into`] below.
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu<{}>", T::CPP))).apply(elementwise_rows(y.rows, y.width)), &[packed.arg(), y.arg(), y.width.arg()])
}

/// [`chunked_swiglu`] over a destination the STATEMENT places —
/// `mlp::chunked_swiglu_into_bf16`.
///
/// One kernel, two statement shapes, so two rows — the split
/// `attn::write_kv_to_pages_bf16`/`_quantised` already makes for the same
/// reason. The dense MLP hands this activation a fresh result and lets the
/// arena place it; qwen3.5's ALIGNED MoE leg cannot, because
/// `build_moe_ptrs_aligned` has already baked that buffer's base address into
/// the device pointer arrays the batched-cuBLAS fallback dereferences. There
/// the destination is an operand and the result must BE it, which is what
/// `InOut` says and what HEAD's `in_place = &[(0, 1)]` asked the allocator
/// for.
///
/// One mark could not serve both: `Out` alone leaves the aligned leg writing
/// a buffer the pointer arrays do not name, and `InOut` alone reaches for an
/// input the dense MLP does not place.
///
/// # Safety
///
/// [`chunked_swiglu`]'s, and `y` must be the buffer the statement placed.
#[routine(bf16)]
pub fn chunked_swiglu_into<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: InOut<Tensor<T>>) -> Result<(), Refusal> {
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu<{}>", T::CPP))).apply(elementwise_rows(y.rows, y.width)), &[packed.arg(), y.arg(), y.width.arg()])
}

/// The packed form with the gate clamped —
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
#[routine(bf16)]
pub fn chunked_swiglu_clamp<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let limit = ctx.ask::<f32, keys::GluLimit>()?;

    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::chunked_swiglu_clamp<{}>", T::CPP))).apply(elementwise_rows(y.rows, y.width)), &[packed.arg(), y.arg(), y.width.arg(), limit.arg()])
}

/// SiTU over a packed bank — `mlp::chunked_situ_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
#[routine(bf16)]
pub fn chunked_situ<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    beta: f32,
    // NOTHING SUPPLIES THIS AND THE SIGNATURE SAYS SO. It was
    // `Env<f32, keys::Unstated>`, a mark that claimed no source at
    // all; `#[unbound]` is that sentence without the fake key.
    #[unbound]
    linear_beta: f32) -> Result<(), Refusal> {
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::chunked_situ<{}>", T::CPP))).apply(elementwise_rows(y.rows, y.width)), &[packed.arg(), y.arg(), y.width.arg(), beta.arg(), linear_beta.arg()])
}

/// GeGLU-tanh over a packed bank — `mlp::chunked_geglu_tanh_bf16`.
///
/// # Safety
///
/// [`chunked_swiglu`]'s.
#[routine(bf16)]
pub fn chunked_geglu_tanh<T>(
    ctx: &Ctx<'_>,
    packed: In<Tensor<T>>,
    y: Out<Tensor<T>>) -> Result<(), Refusal> {
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::chunked_geglu_tanh<{}>", T::CPP))).apply(elementwise_rows(y.rows, y.width)), &[packed.arg(), y.arg(), y.width.arg()])
}

/// `out += y * sigmoid(x · gate_w)` — `mlp::sigmoid_dot_scalar_gate_add_bf16`.
///
/// # Safety
///
/// `x`, `y` and `out` must each address `out.rows * out.width` live bf16
/// elements — `out` writable, and it is the residual stream the statement
/// takes as its second operand, which is what `in_place = &[(0, 1)]`
/// declares. `gate_w` must address `out.width` live bf16 elements.
#[routine(bf16)]
pub fn sigmoid_dot_scalar_gate_add<T>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    // The statement's only weight, so `Weight<0, *const T>` and a name lookup agree.
    gate_w: Const<Tensor<T>>,
    out: InOut<Tensor<T>>,
    // Stated as `In<2, *const _>`, not derived: `in_place = &[(0, 1)]` claims operand
    // 1 for the aliased `out`, so the true next statement operand is 2.
    y: In<Tensor<T>>) -> Result<(), Refusal> {
    // `h` is a pitch, not an extent: `swiglu.cuh` strides `x`, `out` and `y`
    // by the same number. `layout::stated` turns `all()`'s `Absent` into this
    // family's own `Empty { what: "the row width" }`.
    let row = crate::layout::stated(out.all("the row width"))?;
    let h = row.stride;
    ctx.fire(Fire::at("mlp/swiglu.cuh", crate::jit::symbol(&format!("::pie::mlp::sigmoid_dot_scalar_gate_add<{}>", T::CPP))).apply(rms(row.rows)), &[x.arg(), gate_w.arg(), out.arg(), y.arg(), h.arg()])
}

/// AltUp's activation sparsity, in place — `mlp::gaussian_topk_bf16`.
///
/// # Safety
///
/// `x` must address `x.rows * x.width` live and writable bf16 elements.
#[routine(bf16)]
pub fn gaussian_topk<T>(
    ctx: &Ctx<'_>,
    // In place (`in_place = &[(0, 0)]`): also `In(0)`, but `Out<0, *mut T>` matches
    // the launcher's own `*mut`.
    x: InOut<Tensor<T>>) -> Result<(), Refusal> {
    let std_multiplier = ctx.ask::<f32, keys::ParamF32_0>()?;
    // Same reason as `sigmoid_dot_scalar_gate_add`'s `h`: `dim` is both the
    // loop bound and the row stride in `gaussian_topk.cuh`, so one number
    // serves as both extent and pitch.
    let row = crate::layout::stated(x.all("the row width"))?;
    let dim = row.stride;
    ctx.fire(Fire::at("mlp/gaussian_topk.cuh", crate::jit::symbol(&format!("::pie::mlp::gaussian_topk<{}>", T::CPP))).apply(rms(row.rows)), &[x.arg(), dim.arg(), std_multiplier.arg()])
}

// Pins each launcher's derived operand-binding metadata, so a change to
// `#[source(...)]` that alters a slot's binding is caught here, not at first use.
const _: () = {
    assert!(<sigmoid_dot_scalar_gate_add as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(sigmoid_dot_scalar_gate_add::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(sigmoid_dot_scalar_gate_add::<bf16>)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(sigmoid_dot_scalar_gate_add::<bf16>)[2], Some(kernels::Source::Alias(1, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(sigmoid_dot_scalar_gate_add::<bf16>)[3], Some(kernels::Source::Slot(kernels::Kind::In, 2))));

    assert!(<gpt_oss_glu as ::kernels::Derivation>::DERIVED.len() == 3);

    assert!(<chunked_swiglu as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunked_swiglu::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    // `Out(0)`, matching `chunked_geglu_tanh` directly below — the twin that
    // kept it. An `Alias(1, 0)` here claims a second input the statement does
    // not place, and a result half the operand's width cannot be that buffer.
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunked_swiglu::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    assert!(<chunked_geglu_tanh as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunked_geglu_tanh::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunked_geglu_tanh::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // Index 2 (the beta pair's first half) is deliberately `None`: this
    // row stays refused rather than bound to a guessed source.
    assert!(kernels::routine::sources::<crate::jit::Cuda, _, _>(chunked_situ::<bf16>)[2].is_none());

    // The counterpart: this one is bound, and off the statement rather
    // than off a fact, which is why it needs a hand arm.
    assert!(<gaussian_topk as ::kernels::Derivation>::DERIVED.len() == 1);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gaussian_topk::<bf16>)[0], Some(kernels::Source::Alias(0, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.

    assert!(<swiglu as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(swiglu::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(swiglu::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(swiglu::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    assert!(<relu2 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(relu2::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(relu2::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    assert!(<geglu_tanh as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(geglu_tanh::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(geglu_tanh::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(geglu_tanh::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};


// Pins `std_multiplier`'s `#[source(ParamF32(0))]` binding, so "simplifying"
// the attribute away and silently returning the row to unbound is caught here.
const _: () = {
    let d = kernels::routine::sources::<crate::jit::Cuda, _, _>(gaussian_topk::<bf16>);
    assert!(d.len() == 1);
    assert!(matches!(d[0], Some(kernels::Source::Alias(0, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
};
