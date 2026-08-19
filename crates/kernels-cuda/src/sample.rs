//! The CUDA `sample` family: greedy decode, and the fused LM head in front
//! of it.
//!
//! Two plain argmaxes over a logit rectangle (bf16 and f32),
//! [`lm_head_gemv_argmax_int8`] fusing the int8 LM-head GEMV in front of one,
//! and [`argmax_compact_scatter`] scattering ids through a row selector.

use kernels::routine::Asks;
use kernels::{Bind, Fire, keys};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch};
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::routine::{Const, In, Out};
// `keys` is imported as the module, not the facts inside it: `stated_source`
// only emits a source when the path's second-to-last segment is `keys`, so
// `use kernels::keys::Vocab;` would silently derive `None`.

/// `LaunchRule::Rms`, as the expression it evaluates to.
#[must_use]
const fn rms(rows: i32) -> Launch {
    const ARGMAX_BLOCK: u32 = 256;

    Launch::per_row(rows.unsigned_abs(), ARGMAX_BLOCK).smem((ARGMAX_BLOCK / 32) * 4)
}
/// Greedy decode straight off an int8 LM head.
///
/// Two launches ordered by the stream: the GEMV writes one partial pair per
/// (block, row), and the selector folds them. `call()`'s contract: every
/// pointer is live for the extents `num_rows`, `hidden` and `vocab` describe,
/// and `token_ids` is writable for `num_rows` i32.
#[routine]
pub fn lm_head_gemv_argmax_int8(
    ctx: &Ctx<'_>,
    hidden_states: In<Tensor<bf16>>,
    lm_head_weight: Const<Tensor<i8>>,
    scale_inv: Const<Tensor<f32>>,
    token_ids: Out<Tensor<i32>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: HEAD spelled each of these `Env<keys::_>` and no
    // builder ever began stating them. A `Const` mark PROMISES the statement
    // carries the number at its slot in the params run; where nothing states
    // one the promise breaks at the fire, not at the type. §11.20.
    let vocab = ctx.ask::<i32, keys::Vocab>()?;

    let num_rows = token_ids.rows;
    // Output zero's width, not input zero's; the two guards are independent.
    if token_ids.width <= 0 {
        return Err(Refusal::Empty { what: "the token id column's width" });
    }
    // `smem` below is `hidden * 4`; zero would launch over a row of nothing.
    if hidden_states.width <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    // Guard first, view second: `all()` refuses `Absent` and the guard above
    // refuses `Empty`, and callers match on the latter.
    let states = hidden_states.all("hidden")?;
    /// Vocab rows a block covers per grid step.
    const GEMV_WARPS: u32 = 8;

    /// `grid.x` for the fused GEMV.
    ///
    /// # Errors
    ///
    /// [`Refusal::Device`] if the SM count cannot be read. No default: the
    /// number is also the operand the kernel strides the vocab by.
    fn blocks_x(ctx: &Ctx<'_>, vocab: i32) -> Result<u32, Refusal> {
    const BLOCKS_PER_SM: u32 = 2;

    let max_blocks_x = ctx.multiprocessors()? * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    Ok(max_blocks_x.min(min_blocks_x).max(1))
    }

            const SELECT_BLOCK: u32 = 128;

            const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

    // `*vocab` because `blocks_x` takes a plain `i32`; `vocab.arg()` below
    // needs no star, a method call walks the deref chain.
    let num_blocks_x = blocks_x(ctx, vocab)?;
    let rows = num_rows.unsigned_abs();
    let tiles = i32::try_from(num_blocks_x).unwrap_or(i32::MAX);
    let partial_pairs = ctx.scratch(
        "sample::lm_head_argmax_pairs",
        num_blocks_x as usize * rows as usize * core::mem::size_of::<u64>(),
    )?;
    let smem = states.width.unsigned_abs().saturating_mul(4);

    // SAFETY: `call()`'s contract — every pointer addresses live device
    // memory of the extent the kernel reads it as; the stream orders the
    // second launch after the first.
    ctx.fire(Fire::at("sample/argmax.cuh", "::pie::sample::lm_head_gemv_argmax_int8<::pie::bf16>").apply(Launch::grid([num_blocks_x, rows, 1], [GEMV_BLOCK_DIM, 1, 1]).smem(smem)), &[
                states.ptr.arg(),
                lm_head_weight.arg(),
                scale_inv.arg(),
                partial_pairs.arg(),
                num_rows.arg(),
                // A row pitch, not an extent: the kernel advances both
                // `hidden_states` and `lm_head_weight` by it. Equal to
                // `states.width` only because `all()` builds a packed view.
                states.stride.arg(),
                vocab.arg(),
                tiles.arg(),
            ])?;
        ctx.fire(Fire::at("sample/argmax.cuh", "::pie::sample::select_lm_head_argmax_pairs").apply(Launch::flat(rows, SELECT_BLOCK)), &[partial_pairs.cast_const().arg(), token_ids.arg(), num_rows.arg(), tiles.arg()])
}

/// `sample::argmax_bf16` — the greedy decode over a bf16 logit row.
///
/// `call()`'s contract: `logits` addresses `rows * vocab` live bf16 elements
/// and `out` `rows` writable `i32`s.
#[routine(bf16)]
pub fn argmax<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    out: Out<Tensor<i32>>) -> Result<(), Refusal> {
    // `vocab` here is the logits operand's own row width, not
    // [`lm_head_gemv_argmax_int8`]'s `keys::Vocab` fact.
    let rows = logits.rows;
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // Guard then view, for [`lm_head_gemv_argmax_int8`]'s reason. The kernel
    // argument is the extent rather than the stride: the returned token id is
    // an index into it.
    let src = logits.all("vocab")?;
    // SAFETY: `call()`'s contract — every pointer is live for the extent the
    // kernel reads it as.
    ctx.fire(Fire::at("sample/argmax.cuh", crate::jit::symbol(&format!("::pie::sample::argmax<{}>", T::CPP))).apply(rms(rows)), &[src.ptr.arg(), out.arg(), src.width.arg()])
}

/// `sample::argmax_f32` — the fp32 twin.
///
/// [`argmax`]'s obligation, with `f32` for `bf16`.
#[routine]
pub fn argmax_f32(ctx: &Ctx<'_>, logits: In<Tensor<f32>>, out: Out<Tensor<i32>>) -> Result<(), Refusal> {
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // [`argmax`]'s guard-then-view and [`argmax`]'s width, for its reasons.
    let src = logits.all("vocab")?;
    ctx.fire(Fire::at("sample/argmax.cuh", "::pie::sample::argmax<\
                                             ::pie::sample::f32>").apply(rms(src.rows)), &[src.ptr.arg(), out.arg(), src.width.arg()])
}

/// `sample::argmax_compact_scatter_bf16` — the compact form.
///
/// [`argmax`]'s contract, plus `row_indices` addressing `rows` live `i32`s
/// and `out` writable at every index it names.
#[routine(bf16)]
pub fn argmax_compact_scatter<T>(
    ctx: &Ctx<'_>,
    logits: In<Tensor<T>>,
    // A row selector, so its width is neither a pitch nor an element count
    // and no region is built from it.
    row_indices: In<Tensor<i32>>,
    out: Out<Tensor<i32>>) -> Result<(), Refusal> {
    // Off the logits: this scatters, so the rows counted are the rows read.
    let rows = logits.rows;
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // [`argmax`]'s guard-then-view and [`argmax`]'s width; the scatter
    // changes which rows are written, not this number.
    let src = logits.all("vocab")?;
    // SAFETY: as [`argmax`]'s, plus `out` writable at every index
    // `row_indices` names.
    ctx.fire(Fire::at("sample/argmax.cuh", crate::jit::symbol(&format!("::pie::sample::argmax_compact_scatter<{}>", T::CPP))).apply(rms(rows)), &[src.ptr.arg(), row_indices.arg(), out.arg(), src.width.arg()])
}

// `vocab` is a fact rather than a `Lit` because it is constant per checkpoint,
// not per kernel: a `Lit` would bind every fire correctly until the second
// checkpoint. The three plain argmaxes are unstated on purpose — the greedy
// readout is `tensor-ir`'s `Op::ReduceArgmax`, lowered by `tensor-compiler`
// into the region's own kernel, which never enters this crate.
const _: () = {
    // The pinned entries are the ones this family got wrong once: two named
    // banks read as inputs, and a result read as one only because `token_ids`
    // is spelled `*mut`.
    assert!(<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED.len() == 4);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(lm_head_gemv_argmax_int8)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(lm_head_gemv_argmax_int8)[1], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(lm_head_gemv_argmax_int8)[2], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 1)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(lm_head_gemv_argmax_int8)[3], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // Counted `In(0)` and stated `In(0)` are the same `Source` and a
    // different claim, so `stated` is pinned separately.

    // `row_indices`'s `In(1)` is derived by counting off a counter that
    // `In<0, *const _>` set; the pin says that was intended.
    assert!(<argmax_compact_scatter as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax_compact_scatter::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax_compact_scatter::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax_compact_scatter::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The two plain argmaxes are unstated, so these entries are the whole of
    // what is known about them.
    assert!(<argmax as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax::<bf16>)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<argmax_f32 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax_f32)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(argmax_f32)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};

// `Stride` is `#[repr(transparent)]`, which is what lets `states.stride.0` be
// the whole of it where the kernel wants one 32-bit int. If it ever gains a
// field, nothing else here would say so.
const _: () = {
    assert!(core::mem::size_of::<kernels::Stride>() == core::mem::size_of::<i32>());
    assert!(core::mem::align_of::<kernels::Stride>() == core::mem::align_of::<i32>());
};

