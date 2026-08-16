//! The CUDA `sample` family: greedy decode, and the fused LM head in front
//! of it.
//!
//! Two plain argmaxes over a logit rectangle (bf16 and f32),
//! [`lm_head_gemv_argmax_int8`] fusing the int8 LM-head GEMV in front of one,
//! and [`argmax_compact_scatter`] scattering ids through a row selector.
#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::jit::Abi;
use crate::jit::abi::Inst;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::keys;
use kernels::{Env, In, Out, Weight};
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
#[kernels_macros::routine]
pub fn lm_head_gemv_argmax_int8(
    ctx: &Ctx,
    hidden_states: In<0, bf16>,
    lm_head_weight: Weight<0, *const i8>,
    scale_inv: Weight<1, *const f32>,
    token_ids: Out<0, i32>,
    // The int8 LM head is a named weight, so its vocabulary extent is no
    // operand's shape.
    vocab: Env<keys::Vocab>,
) -> Result<(), Refusal> {
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
    fn blocks_x(ctx: &Ctx, vocab: i32) -> Result<u32, Refusal> {
    const BLOCKS_PER_SM: u32 = 2;

    let max_blocks_x = ctx.multiprocessors()? * BLOCKS_PER_SM;
    let min_blocks_x = vocab.unsigned_abs().div_ceil(GEMV_WARPS);
    Ok(max_blocks_x.min(min_blocks_x).max(1))
    }

            const SELECT_BLOCK: u32 = 128;

            const GEMV_BLOCK_DIM: u32 = GEMV_WARPS * 32;

    // `**vocab` because `blocks_x` takes a plain `i32`; `vocab.arg()` below
    // needs no star, a method call walks the deref chain.
    let num_blocks_x = blocks_x(ctx, **vocab)?;
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
    unsafe {
        ctx.launch(
            "sample/argmax.cuh",
            "::pie::sample::lm_head_gemv_argmax_int8<::pie::bf16>",
            Launch::grid([num_blocks_x, rows, 1], [GEMV_BLOCK_DIM, 1, 1]).smem(smem),
            &[
                states.ptr.arg(),
                lm_head_weight.ptr.arg(),
                scale_inv.ptr.arg(),
                partial_pairs.arg(),
                num_rows.arg(),
                // A row pitch, not an extent: the kernel advances both
                // `hidden_states` and `lm_head_weight` by it. Equal to
                // `states.width` only because `all()` builds a packed view.
                states.stride.0.arg(),
                vocab.arg(),
                tiles.arg(),
            ],
        )?;
        ctx.launch(
            "sample/argmax.cuh",
            "::pie::sample::select_lm_head_argmax_pairs",
            Launch::flat(rows, SELECT_BLOCK),
            &[partial_pairs.cast_const().arg(), token_ids.ptr.arg(), num_rows.arg(), tiles.arg()],
        )
    }
}

/// `sample::argmax_bf16` — the greedy decode over a bf16 logit row.
///
/// `call()`'s contract: `logits` addresses `rows * vocab` live bf16 elements
/// and `out` `rows` writable `i32`s.
#[kernels_macros::routine]
pub fn argmax<T>(
    ctx: &Ctx,
    logits: In<0, T>,
    out: Out<0, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
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
    unsafe {
        ctx.launch("sample/argmax.cuh", &format!("::pie::sample::argmax<{}>", T::CPP), rms(rows), &[src.ptr.arg(), out.ptr.arg(), src.width.arg()])
    }
}

/// `sample::argmax_f32` — the fp32 twin.
///
/// [`argmax`]'s obligation, with `f32` for `bf16`.
#[kernels_macros::routine]
pub fn argmax_f32(ctx: &Ctx, logits: In<0, f32>, out: Out<0, i32>) -> Result<(), Refusal> {
    if logits.width <= 0 {
        return Err(Refusal::Empty { what: "vocab" });
    }
    // [`argmax`]'s guard-then-view and [`argmax`]'s width, for its reasons.
    let src = logits.all("vocab")?;
    // SAFETY: as [`argmax`]'s.
    unsafe {
        ctx.launch("sample/argmax.cuh", "::pie::sample::argmax<\
                                             ::pie::sample::f32>", rms(src.rows), &[src.ptr.arg(), out.ptr.arg(), src.width.arg()])
    }
}

/// `sample::argmax_compact_scatter_bf16` — the compact form.
///
/// [`argmax`]'s contract, plus `row_indices` addressing `rows` live `i32`s
/// and `out` writable at every index it names.
#[kernels_macros::routine]
pub fn argmax_compact_scatter<T>(
    ctx: &Ctx,
    logits: In<0, T>,
    // A row selector, so its width is neither a pitch nor an element count
    // and no region is built from it.
    row_indices: In<1, i32>,
    out: Out<0, i32>,
) -> Result<(), Refusal>
where
    T: Inst + kernels::Elem,
    *const T: Abi,
{
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
    unsafe {
        ctx.launch(
            "sample/argmax.cuh",
            &format!("::pie::sample::argmax_compact_scatter<{}>", T::CPP),
            rms(rows),
            &[src.ptr.arg(), row_indices.ptr.arg(), out.ptr.arg(), src.width.arg()],
        )
    }
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
    assert!(<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(matches!(<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(kernels::source_is_named(&<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[1].source, <kernels::keys::NamedWeight as kernels::keys::Fact>::KEY));
    assert!(kernels::source_is_named(&<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[2].source, <kernels::keys::NamedWeight2 as kernels::keys::Fact>::KEY));
    assert!(matches!(<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[3].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(kernels::source_is_named(&<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[4].source, <kernels::keys::Vocab as kernels::keys::Fact>::KEY));
    // Counted `In(0)` and stated `In(0)` are the same `Source` and a
    // different claim, so `stated` is pinned separately.
    assert!(<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[0].stated);
    assert!(<lm_head_gemv_argmax_int8 as ::kernels::Derivation>::DERIVED[3].stated);

    // `row_indices`'s `In(1)` is derived by counting off a counter that
    // `In<0, _>` set; the pin says that was intended.
    assert!(<argmax_compact_scatter as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(<argmax_compact_scatter as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<argmax_compact_scatter as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(<argmax_compact_scatter as ::kernels::Derivation>::DERIVED[1].stated);
    assert!(matches!(<argmax_compact_scatter as ::kernels::Derivation>::DERIVED[2].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // The two plain argmaxes are unstated, so these entries are the whole of
    // what is known about them.
    assert!(<argmax as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<argmax as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<argmax as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(<argmax_f32 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(<argmax_f32 as ::kernels::Derivation>::DERIVED[0].source, Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(<argmax_f32 as ::kernels::Derivation>::DERIVED[1].source, Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};

// `Stride` is `#[repr(transparent)]`, which is what lets `states.stride.0` be
// the whole of it where the kernel wants one 32-bit int. If it ever gains a
// field, nothing else here would say so.
const _: () = {
    assert!(core::mem::size_of::<kernels::Stride>() == core::mem::size_of::<i32>());
    assert!(core::mem::align_of::<kernels::Stride>() == core::mem::align_of::<i32>());
};

/// This family's routines, and what a trace may say about each.
pub static ROUTINES: &[Routine] = &[
    routine!(lm_head_gemv_argmax_int8, ),
    routine!(argmax_bf16 = argmax::<bf16>, ),
    routine!(argmax_f32, ),
    routine!(argmax_compact_scatter_bf16 = argmax_compact_scatter::<bf16>, ),
];

/// `sample`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);
