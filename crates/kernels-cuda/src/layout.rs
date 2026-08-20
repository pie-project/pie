//! The `layout` family: the reshaping kernels a fire needs around its matmuls
//! — row splits, gathers, embeds, transposes and the KV envelope updates.
//!
//! Most entries are `routine!`-declared, so a statement places their operands
//! and the derived column binds them. The four envelope roots are the
//! exception: the driver composes their operands while building a wave, so
//! their host programs live in `crate::driver_internal`.

use kernels::{Bind, Fire};
use kernels_macros::routine;
use crate::jit::{Ctx, Launch, aligned16};
// The one host program declared elsewhere still needs a bare identifier here.
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use kernels::Refusal;
use kernels::Region;
// `keys` is imported as the module, not the facts inside it: `stated_source`
// only emits a source when the path's second-to-last segment is `keys`, so
// `use kernels::keys::Vocab;` would silently derive `None`.
use kernels::keys;
use kernels::routine::{Asks, Const, In, Out};

// The four roots below get no `routine!` line — the driver composes their
// operands while building a wave, not a statement. They still exist so
// `every_carried_file_is_reachable` and `every_instantiation_compiles` find
// the kernel text and hand it to NVRTC.

const BLOCK: u32 = 256;

const WARP: u32 = 32;

/// `LaunchRule::RouteRows`, as the expression it evaluates to.
#[must_use]
fn route_rows(rows: i32, width: i32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows.unsigned_abs(),
        width.unsigned_abs().div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK),
    )
}

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// A whole-region view, refused in this file's variant.
///
/// [`In::all`] and [`Out::all`] refuse a zero-width region as
/// [`Refusal::Absent`]; every guard in this file instead wants
/// [`Refusal::Empty`], since a region's zero width means a statement declared
/// none, not that the fire dropped an argument.
///
/// # Errors
///
/// Whatever `view` holds, with [`Refusal::Absent`] rewritten to
/// [`Refusal::Empty`] over the same word.
pub(crate) fn stated<P>(view: Result<Region<P>, Refusal>) -> Result<Region<P>, Refusal> {
    match view {
        Err(refusal) => Err(this_family(refusal)),
        ok => ok,
    }
}

/// [`Refusal::Absent`] said in this file's variant; everything else verbatim.
#[must_use]
const fn this_family(refusal: Refusal) -> Refusal {
    match refusal {
        Refusal::Absent { what } => Refusal::Empty { what },
        said => said,
    }
}

/// `layout::split_bf16_rows` — one packed row out to two.
///
/// # Safety
///
/// `src` addresses `n * (left_dim + right_dim)` live bf16 elements; `left`
/// and `right` `n * left_dim`/`n * right_dim` writable ones.
#[routine]
pub fn split_bf16_rows(
    ctx: &Ctx<'_>,
    src: In<Tensor<bf16>>,
    left: Out<Tensor<bf16>>,
    right: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // Views come from the results, not `src`; `all()` also catches a
    // zero-width half that `route_rows`'s `.max(1)` would otherwise hide.
    let left_half = stated(left.all("left_dim or right_dim"))?;
    let right_half = stated(right.all("left_dim or right_dim"))?;
    let n = left_half.rows;
    // Row pitches, not element counts; `.0` reads one back as a plain `i32`.
    let left_dim = left_half.stride;
    let right_dim = right_half.stride;
    ctx.fire(Fire::at("layout/deinterleave.cuh", "::pie::layout::split_rows<::pie::bf16>").apply(route_rows(n, left_dim.0)), &[src.arg(), left.arg(), right.arg(), left_dim.arg(), right_dim.arg()])
}

/// `layout::split_qwen_gdn_ba_bf16` — Qwen's GDN bank, split by halves.
///
/// # Safety
///
/// `ba` addresses `n * 2 * v_h` live bf16 elements; `b_out`/`a_out`
/// `n * v_h` writable ones each.
#[routine(bf16)]
pub fn split_qwen_gdn_ba<T>(
    ctx: &Ctx<'_>,
    ba: In<Tensor<T>>,
    b_out: Out<Tensor<T>>,
    a_out: Out<Tensor<T>>) -> Result<(), Refusal> {
    // One view, off result 0, serves both halves: the kernel strides each by
    // the same `v_h` pitch.
    let half = stated(b_out.all("v_h"))?;
    let n = half.rows;
    let v_h = half.stride;
    ctx.fire(Fire::at("layout/deinterleave.cuh", crate::jit::symbol(&format!("::pie::layout::split_qwen_gdn_ba<{}>", T::CPP))).apply(route_rows(n, v_h.0)), &[ba.arg(), b_out.arg(), a_out.arg(), v_h.arg()])
}

/// `layout::deinterleave_rows_bf16` — gpt-oss's parity split, row-shaped.
///
/// # Safety
///
/// `fused` addresses `2 * rows * h` live bf16 elements; `gate_out`/`up_out`
/// `rows * h` writable ones each.
#[routine(bf16, internal)]
pub fn deinterleave_rows<T>(
    ctx: &Ctx<'_>,
    fused: In<Tensor<T>>,
    gate_out: Out<Tensor<T>>,
    up_out: Out<Tensor<T>>) -> Result<(), Refusal> {
    // Unstated: shapes come off `gate_out` itself — `h` its row pitch,
    // `rows` its row count.
    let gate = stated(gate_out.all("h"))?;
    let rows = gate.rows;
    let h = gate.stride;
    ctx.fire(Fire::at("layout/deinterleave.cuh", crate::jit::symbol(&format!("::pie::layout::deinterleave_rows<{}>", T::CPP))).apply(route_rows(rows, h.0)), &[fused.arg(), gate_out.arg(), up_out.arg(), h.arg()])
}

/// `layout::deinterleave_vec_bf16` — the same split, one thread per element.
///
/// # Safety
///
/// `fused` addresses `2 * i` live bf16 elements; `gate_out`/`up_out` `i`
/// writable ones each.
#[routine(bf16, internal)]
pub fn deinterleave_vec<T>(
    ctx: &Ctx<'_>,
    fused: In<Tensor<T>>,
    gate_out: Out<Tensor<T>>,
    up_out: Out<Tensor<T>>) -> Result<(), Refusal> {
    // `i` is `rows * width`, saturating (`Region::elements`). `all()` catches
    // a zero width; this catches a zero row count.
    let gate = stated(gate_out.all("i"))?;
    let i = gate.elements();
    if i <= 0 {
        return Err(Refusal::Empty { what: "i" });
    }
    ctx.fire(Fire::at("layout/deinterleave.cuh", crate::jit::symbol(&format!("::pie::layout::deinterleave_vec<{}>", T::CPP))).apply(elementwise(i.unsigned_abs())), &[fused.arg(), gate_out.arg(), up_out.arg(), i.arg()])
}

/// `layout::concat_bf16_rows` — `[N, left] ++ [N, right]`.
///
/// # Safety
///
/// `left`/`right` address `rows * left_dim`/`rows * right_dim` live bf16
/// elements; `out` `rows * (left_dim + right_dim)` writable ones.
#[routine(internal)]
pub fn concat_bf16_rows(
    ctx: &Ctx<'_>,
    left: In<Tensor<bf16>>,
    right: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    // Each operand needs its own view: one guard on the summed width would
    // miss a zero-width half (`0 + 512` is still a nonzero 512).
    let left_half = stated(left.all("left_dim or right_dim"))?;
    let right_half = stated(right.all("left_dim or right_dim"))?;
    let rows = out.rows;
    let left_dim = left_half.stride;
    let right_dim = right_half.stride;
    ctx.fire(Fire::at("layout/deinterleave.cuh", "::pie::layout::concat_rows<::pie::bf16>").apply(route_rows(rows, left_dim.0)), &[left.arg(), right.arg(), out.arg(), left_dim.arg(), right_dim.arg()])
}

/// `layout::gather_bf16_rows` — the epilogue's gather.
///
/// # Safety
///
/// `src` addresses the rows `row_indices` names at `width` u16 elements
/// each; `row_indices` `num_dst_rows` live `i32`s; `dst` `num_dst_rows *
/// width` writable u16 elements.
#[routine]
pub fn gather_bf16_rows(
    ctx: &Ctx<'_>,
    src: In<Tensor<u16>>,
    dst: Out<Tensor<u16>>) -> Result<(), Refusal> {
    let row_indices = ctx.ask::<*const i32, keys::SamplingIndices>()?;
    // `dst` is the dense side, so its view supplies rows and width; `src`
    // has neither since `row_indices` indexes it. The width is a pitch
    // shared by both, advancing a source row and a destination slot alike.
    let dense = stated(dst.all("width"))?;
    let num_dst_rows = dense.rows;
    let width = dense.stride;
    ctx.fire(Fire::at("layout/gather_rows.cuh", "::pie::layout::gather_rows<::pie::u16>").apply(route_rows(num_dst_rows, width.0)), &[src.arg(), row_indices.arg(), dst.arg(), width.arg()])
}

/// `layout::transpose_bf16_nld_to_lnd` — the PLE relay.
///
/// A row that is not a whole number of PLE planes is refused rather than
/// truncated, which would silently transpose fewer planes than placed.
///
/// # Errors
///
/// [`Refusal::Empty`] if `dim` is zero or negative, and [`Refusal::Narrow`]
/// if `width` is not a whole number of `dim`-wide planes.
///
/// # Safety
///
/// `src`/`dst` address `n * width` live u16 elements, `dst` writable.
#[routine]
pub fn transpose_bf16_nld_to_lnd(
    ctx: &Ctx<'_>,
    src: In<Tensor<u16>>,
    dst: Out<Tensor<u16>>) -> Result<(), Refusal> {
    // ASKED, NOT `Const`: every one of these was `Env<keys::_>` before the
    // four marks, and no builder ever began stating them. A `Const` mark
    // PROMISES the statement carries the number at its slot in the params
    // run; where nothing states one the promise is broken at the fire, not
    // at the type. See `.wiki/migration.md` §11.20.
    let dim = ctx.ask::<i32, keys::PleDim>()?;

    // Off `src`, not `dst`: this transposes, so `dst` is the same elements
    // reordered. `.width`, not `.stride` — no pitch reaches this kernel,
    // which rebuilds addresses from `n`, `layers` and `dim`.
    let source = stated(src.all("width"))?;
    let n = source.rows;
    let width = source.width;
    // `*dim`: comparison, `%`, `/` and `try_from` don't deref through the
    // wrapper the way a method call does.
    if dim <= 0 {
        return Err(Refusal::Empty { what: "ple_dim" });
    }
    if width % dim != 0 {
        return Err(Refusal::Narrow {
            what: "the row is not a whole number of PLE planes",
            at: i64::from(width),
        });
    }
    let layers = width / dim;
    let total = usize::try_from(n).unwrap_or(0)
        * usize::try_from(layers).unwrap_or(0)
        * usize::try_from(dim).unwrap_or(0);
    ctx.fire(Fire::at("layout/gather_rows.cuh", "::pie::layout::transpose_nld_to_lnd<::pie::u16>").apply(elementwise(u32::try_from(total).unwrap_or(u32::MAX))), &[src.arg(), dst.arg(), n.arg(), layers.arg(), dim.arg(), total.arg()])
}

/// `layout::copy_if_valid_slot` — copy a slot's bytes if the slot is valid.
///
/// # Safety
///
/// `src`/`dst` address `bytes` live bytes, `dst` writable, `slot_ids`
/// indexable at `request`.
#[routine(internal)]
pub fn copy_if_valid_slot(
    ctx: &Ctx<'_>,
    // The pointers state their slots; the scalars below don't.
    src: In<Tensor<u8>>,
    dst: Out<Tensor<u8>>,
    // `bytes`/`request` are both checkpoint constants, not facts the
    // environment answers — a statement stating this row states them too, so
    // each claims the next params-run slot rather than `ctx.ask`ing for one.
    // Swapping them compiles cleanly; `bytes` is a byte count, not an
    // element count.
    bytes: Const<usize>,
    // The index is stated (`In<1, *const _>`), not left to counting position.
    slot_ids: In<Tensor<i32>>,
    request: Const<usize>) -> Result<(), Refusal> {
    ctx.fire(Fire::at("layout/slot_ops.cuh", "::pie::layout::copy_if_valid_slot").apply(Launch::grid([1, 1, 1], [256, 1, 1])), &[src.arg(), dst.arg(), bytes.arg(), slot_ids.arg(), request.arg()])
}

const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 { head_dim.unsigned_abs() } else { 256 }
}

/// `layout::envelope_merge_written_bf16` — fold explicitly-written KV rows
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live, with `ctx`'s
/// stream live across the launch.
#[routine(untraced, internal)]
pub fn envelope_merge_written(
    ctx: &Ctx<'_>,
    // Fired by path (`attn/mod.rs`), not a statement, so every parameter
    // below is a plain value the caller already has in hand rather than a
    // mark — a path-fired launch has no `Facts`, even for the pointers
    // matching a `keys::` entry.
    k_curr: *const bf16,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: crate::jit::abi::MaybeConst<u8>,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32) -> Result<(), Refusal> {
    const FUSE_MAX_TOKENS: i32 = 128;

    let launch = Launch::grid(
        [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        [threads_for(head_dim), 1, 1],
    );

    if num_tokens <= FUSE_MAX_TOKENS {
        // SAFETY: every pointer is live for the extent the kernel reads it as.
        return ctx.fire(Fire::at("layout/envelope.cuh", "::pie::layout::merge_written_fused<::pie::i32(0)>").apply(launch), &[
                    k_curr.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    row_valid.arg(),
                    env_min.arg(),
                    env_max.arg(),
                    num_tokens.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                ]);
    }

    // Two launches, in order: the reset seeds every page this batch starts,
    // or a merge running first would have its folds overwritten.
    // SAFETY: every pointer is live for the extent the kernel reads it as.
    ctx.fire(Fire::at("layout/envelope.cuh", "::pie::layout::reset_started_pages<::pie::i32(0)>").apply(launch), &[
                w_page.arg(),
                w_off.arg(),
                row_valid.arg(),
                env_min.arg(),
                env_max.arg(),
                num_tokens.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ])?;
        ctx.fire(Fire::at("layout/envelope.cuh", "::pie::layout::merge_written<::pie::i32(0)>").apply(launch), &[
                k_curr.arg(),
                w_page.arg(),
                row_valid.arg(),
                env_min.arg(),
                env_max.arg(),
                num_tokens.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ])
}

/// `layout::envelope_seed_empty_bf16` — write the `+inf`/`-inf` identity
///
/// # Safety
///
/// Both planes are device addresses the caller keeps live, with `ctx`'s
/// stream live across the launch.
#[routine(untraced, internal)]
pub fn envelope_seed_empty(
    ctx: &Ctx<'_>,
    // The pool's planes — see `envelope_merge_written`'s standing note. This
    // fires once at pool construction, before any fire exists.
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32) -> Result<(), Refusal> {
    const SEED_BLOCK: u32 = 256;

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    // SAFETY: every pointer is live device memory of the extent the kernel
    // reads it as.
    ctx.fire(Fire::at("layout/envelope.cuh", "::pie::layout::seed_empty<::pie::i32(0)>").apply(Launch::grid([u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1], [SEED_BLOCK, 1, 1])), &[env_min.arg(), env_max.arg(), n.arg()])
}

/// `layout::envelope_update_appended_bf16` — fold the pages an append touched
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `ctx`'s stream is held live for the same window.
#[routine(untraced, internal)]
pub fn envelope_update_appended(
    ctx: &Ctx<'_>,
    // Same standing note as `envelope_merge_written`. `k_pages` is `*const
    // bf16` where `keys::KvKeys` is `*mut u8` — same key, but this parameter
    // has already picked an element type the fact deliberately leaves open.
    k_pages: *const bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_requests: i32,
    max_touched: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32) -> Result<(), Refusal> {
    // SAFETY: every pointer is live device memory of the extent the kernel
    // reads it as.
    ctx.fire(Fire::at("layout/envelope.cuh", "::pie::layout::update_appended<::pie::bf16>").apply(Launch::grid(
                [max_touched.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
                [threads_for(head_dim), 1, 1],
            )), &[
                k_pages.arg(),
                qo_indptr.arg(),
                kv_page_indices.arg(),
                kv_page_indptr.arg(),
                kv_last_page_lens.arg(),
                env_min.arg(),
                env_max.arg(),
                num_requests.arg(),
                page_size.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ])
}

const VEC_WIDTH: i32 = 8;

/// Host-side mirror of `embed.cu`'s vectorised-path test.
#[must_use]
pub fn vectorisable(hidden: i32, weight: *const bf16, y: *const bf16) -> bool {
    hidden % VEC_WIDTH == 0 && aligned16(weight.cast()) && aligned16(y.cast())
}

/// `layout::embed_bf16` — the first launch of every fire.
///
/// # Safety
///
/// `token_ids` must address `num_tokens` live `i32`s, `weight` `vocab *
/// hidden` live bf16 elements, `y` `num_tokens * hidden` writable ones, and
/// `ctx`'s stream must be live across the launch.
#[routine]
pub fn embed_bf16(
    ctx: &Ctx<'_>,
    weight: Const<Tensor<bf16>>,
    y: Out<Tensor<bf16>>) -> Result<(), Refusal> {
    let token_ids = ctx.ask::<*const i32, keys::TokenIds>()?;
    // `vocab` IS the named weight's vocabulary and no operand carries it —
    // and it cannot be a `Const` either, because the statement that fires
    // this has no params run to put it in. `OpKind::Embed` is a SEMANTIC op:
    // it carries a weight name and nothing else, and `lower::walk` builds a
    // params run only for `OpKind::Launch`. So a `Const<i32>` here promised a
    // number no trace text can pass, and the promise was broken at the FIRST
    // launch of every fire — `layout::embed_bf16: the fire does not carry a
    // statement parameter`, which is where the CUDA e2e path died.
    //
    // HEAD spelled it `Env<keys::Vocab>` and `driver-cuda`'s hand arm still
    // binds `ArgValue::I32(f.vocab()?)`. The rule on `Asks` says a fact the
    // checkpoint fixes is a constant and constants belong in the statement;
    // the rule that OVERRIDES it is that a `Const` mark promises the
    // statement carries the number, and where no trace text can keep the
    // promise the mark is wrong however the first rule reads.
    let vocab = ctx.ask::<i32, keys::Vocab>()?;
    const EMBED_BLOCK: u32 = 256;

    let dst = stated(y.all("hidden"))?;
    let num_tokens = dst.rows;
    // A pitch, not a plain extent: the kernel strides both the weight and
    // `y` by it, and the vectorised path also divides by it via `.0`.
    let hidden = dst.stride;

    let vec = vectorisable(hidden.0, weight.v, dst.ptr.cast_const());
    let per_row = if vec { hidden.0 / VEC_WIDTH } else { hidden.0 };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
    let instantiation = if vec { "::pie::layout::embed<\
                                      ::pie::true_type::value>" } else { "::pie::layout::embed<::pie::false_type::value>" };
    // SAFETY: every pointer is live device memory of the extent the kernel
    // reads it as.
    ctx.fire(Fire::at("layout/embed.cuh", instantiation).apply(Launch::grid([blocks, 1, 1], [EMBED_BLOCK, 1, 1])), &[
                token_ids.arg(),
                weight.arg(),
                dst.ptr.arg(),
                hidden.arg(),
                vocab.arg(),
                num_tokens.arg(),
                per_row.arg(),
            ])
}

// `#[routine]` derives the argument column from the signature alone, and a
// weight and an input are both `const T*` — indistinguishable by counting.
// The wrappers state what counting can't: `Env<keys::_>` for a fact,
// `Weight<n, _>` for a named tensor, `In`/`Out`/`InSlot`/`OutSlot` for an
// operand, `Unbound` for a parameter no statement places.
//
// A source pin alone is blind to a permutation: swapping two same-variant
// parameters passes every indexed assert below. `derived_name_is` closes
// that gap with a byte loop, since `str` can't yet be compared in a const on
// stable (rust-lang/rust#143874).
pub(crate) const fn derived_name_is(actual: &str, expected: &str) -> bool {
    let (a, b) = (actual.as_bytes(), expected.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = {
    // Entries 0 and 1 are what the macro gets wrong from the signature
    // alone.
    const _T0: [(); 1] = [(); matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(embed_bf16)[0], Some(kernels::Source::Or(_, _))) as usize];
    const _T1: [(); 1] = [(); matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(embed_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))) as usize];
    // TWO ENTRIES, NOT THREE. `vocab` left the column when it stopped being a
    // `Const`: the statement that fires this is an `OpKind::Embed`, which has
    // no params run for a scalar to be read out of, so the fact is asked in
    // the body. A third entry here again would mean a scalar came back to a
    // signature nothing can state one for.
    assert!(<embed_bf16 as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(embed_bf16)[0], Some(kernels::Source::Or(kernels::Source::Named(_), kernels::Source::Slot(kernels::Kind::Weight, 0)))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(embed_bf16)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // Stated, so `alias()` must leave it alone — it only ever corrects a
    // derived guess, not a stated index.

    assert!(<gather_bf16_rows as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gather_bf16_rows)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(gather_bf16_rows)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    // The entry this line pinned is gone from the column: the
    // parameter it named left the signature when its fact stopped
    // being asked for as a parameter. See the routine.
    // Result zero answers both itself: its `rows` is the row count, its
    // `width` the out width.

    // Pinned for the second result: no scalar is left for a subtraction to
    // answer wrongly.
    assert!(<split_bf16_rows as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(split_bf16_rows)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(split_bf16_rows)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(split_bf16_rows)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(<split_qwen_gdn_ba as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(split_qwen_gdn_ba::<crate::jit::abi::bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));

    // Pinned against entry 2 flipping back if the PLE width word is ever
    // removed.
    assert!(<transpose_bf16_nld_to_lnd as ::kernels::Derivation>::DERIVED.len() == 2);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(transpose_bf16_nld_to_lnd)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(transpose_bf16_nld_to_lnd)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));

    // `bytes`/`request` stay unbound — see the note at the signature above.
    // Entries 0, 1 and 3 state which operand, nothing about shape.
    assert!(<copy_if_valid_slot as ::kernels::Derivation>::DERIVED.len() == 5);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(copy_if_valid_slot)[2], Some(kernels::Source::Slot(kernels::Kind::Param, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(copy_if_valid_slot)[4], Some(kernels::Source::Slot(kernels::Kind::Param, 1))));
    // Non-adjacent but still interchangeable: both `usize`, both `None` — a
    // source pin can't tell them apart, so name pins hold them apart.
    assert!(derived_name_is(<copy_if_valid_slot as ::kernels::Derivation>::DERIVED[2].name, "bytes"));
    assert!(derived_name_is(<copy_if_valid_slot as ::kernels::Derivation>::DERIVED[4].name, "request"));

    // Unbound because a path-fired launch has no stated `Facts`: an
    // `In<0, *const _>` here would force every caller to invent a rows/width for
    // the driver's own arena.
    const _TSRC: [(); 999] = [(); {
        let s = kernels::routine::sources::<crate::jit::Cuda, _, _>(envelope_update_appended);
        let mut all_none = true;
        let mut i = 0;
        while i < s.len() { if s[i].is_some() { all_none = false; } i += 1; }
        if all_none { 999 } else { 0 }
    }];
    // A `untraced` ROW CARRIES NO COLUMN, and that is the claim: the
    // driver fires it through a typed call, so nothing binds it from a
    // statement and a row that grew a column would mean something had
    // started to.
    assert!(<envelope_update_appended as ::kernels::Derivation>::DERIVED.len() == 12);
    // AND NO SOURCE COLUMN AT ALL. The five entries this used to pin as `None`
    // -- `num_requests`, `max_touched`, `page_size`, `num_kv_heads`,
    // `head_dim` -- were the shape of a column that existed and answered
    // nothing. A `untraced` row has none, which is the stronger claim and
    // the true one; the NAMES below are what still hold the order.
    assert!(<envelope_update_appended as ::kernels::Derivation>::SOURCES.is_empty());
    // `None` five times asserts nothing about order; the names below fix the
    // launch order — permuting the last three (a page geometry) is a wrong
    // stride, not a type error.
    assert!(derived_name_is(<envelope_update_appended as ::kernels::Derivation>::DERIVED[7].name, "num_requests"));
    assert!(derived_name_is(<envelope_update_appended as ::kernels::Derivation>::DERIVED[8].name, "max_touched"));
    assert!(derived_name_is(<envelope_update_appended as ::kernels::Derivation>::DERIVED[9].name, "page_size"));
    assert!(derived_name_is(<envelope_update_appended as ::kernels::Derivation>::DERIVED[10].name, "num_kv_heads"));
    assert!(derived_name_is(<envelope_update_appended as ::kernels::Derivation>::DERIVED[11].name, "head_dim"));

    // The sharpest case in the file: `env_min`/`env_max` are both `*mut
    // bf16`, both `Unbound`. Swapping them compiles, launches, and seeds the
    // running minimum into the maximum plane — no source pin catches it.
    assert!(derived_name_is(<envelope_seed_empty as ::kernels::Derivation>::DERIVED[0].name, "env_min"));
    assert!(derived_name_is(<envelope_seed_empty as ::kernels::Derivation>::DERIVED[1].name, "env_max"));
    assert!(derived_name_is(<envelope_seed_empty as ::kernels::Derivation>::DERIVED[2].name, "num_pages"));
    assert!(derived_name_is(<envelope_seed_empty as ::kernels::Derivation>::DERIVED[3].name, "num_kv_heads"));
    assert!(derived_name_is(<envelope_seed_empty as ::kernels::Derivation>::DERIVED[4].name, "head_dim"));

    // Unstated launchers, pinned as the baseline: if `model-dsl` ever states
    // one of them, this is what it is checked against.
    assert!(<deinterleave_rows as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(deinterleave_rows::<bf16>)[1], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(deinterleave_rows::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(<deinterleave_vec as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(deinterleave_vec::<bf16>)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 1))));
    assert!(<concat_bf16_rows as ::kernels::Derivation>::DERIVED.len() == 3);
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(concat_bf16_rows)[0], Some(kernels::Source::Slot(kernels::Kind::In, 0))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(concat_bf16_rows)[1], Some(kernels::Source::Slot(kernels::Kind::In, 1))));
    assert!(matches!(kernels::routine::sources::<crate::jit::Cuda, _, _>(concat_bf16_rows)[2], Some(kernels::Source::Slot(kernels::Kind::Out, 0))));
};


// Three facts every view above relies on, none checkable at run time:
// `Stride` is layout-identical to `i32` (every `.0` depends on it, and `Ty`
// checks by name only); `Region::elements` saturates, which `deinterleave_vec`
// relies on for a refusal-sized rather than negative product; and
// `this_family` rewrites only `Absent` into `Empty`, leaving `Narrow` and a
// fact's own `Empty` untouched.
const _: () = {
    assert!(core::mem::size_of::<kernels::Stride>() == core::mem::size_of::<i32>());
    assert!(core::mem::align_of::<kernels::Stride>() == core::mem::align_of::<i32>());

    let r = kernels::Region {
        ptr: core::ptr::null::<u16>(),
        rows: 7,
        width: 5,
        stride: kernels::Stride(5),
    };
    assert!(r.elements() == 35);

    let wide = kernels::Region {
        ptr: core::ptr::null::<u16>(),
        rows: i32::MAX,
        width: i32::MAX,
        stride: kernels::Stride(i32::MAX),
    };
    assert!(wide.elements() == i32::MAX);

    let Refusal::Empty { what } = this_family(Refusal::Absent { what: "v_h" }) else {
        panic!("Absent must come back said in this family's variant")
    };
    assert!(derived_name_is(what, "v_h"));

    let Refusal::Empty { what } = this_family(Refusal::Empty { what: "ple_dim" }) else {
        panic!("a fact's own Empty must pass through")
    };
    assert!(derived_name_is(what, "ple_dim"));

    let Refusal::Narrow { what, at } = this_family(Refusal::Narrow { what: "i", at: 3 }) else {
        panic!("only Absent is rewritten")
    };
    assert!(derived_name_is(what, "i"));
    assert!(at == 3);
};
