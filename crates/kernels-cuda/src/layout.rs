#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine, aligned16};
// `driver_bound!` names its `fn` by IDENTIFIER, exactly as `routine!` does, so
// the one host program declared here that does not live in this file has to be
// nameable without its path.
use crate::driver_internal::split_q_gate_bf16;
use crate::{driver_bound, routine};
use crate::jit::Abi;
use crate::jit::abi::Elem;
use crate::jit::abi::bf16;
use kernels::Refusal;

// ===========================================================================
// The four roots whose launchers are the DRIVER's
//
// Each of the four `.cuh` files below was orphaned in the same way: the `.cu`
// beside it in the archive crate held the only `<<<>>>` that named its
// kernels, that file was deleted, and nothing in THIS crate had ever named
// its text. `src/source.rs` lists every file under `kernels/` regardless, so the six
// kernels stayed in the binary as text no compile could arrive at, which is
// what `every_carried_file_is_reachable` reported. A root is what answers it,
// and `every_instantiation_compiles` is what then hands each of the six to
// NVRTC.
//
// None of the four gets a `routine!` line, and that is not an omission. A
// routine is a symbol a TRACE may name, and each of these files says in its own
// header why no statement can name its kernels: the operands are composed by
// the driver while it builds a wave, not produced by a `Source`. So the host
// programs are `crate::driver_internal`'s, which is the module for exactly this
// -- and `deinterleave::inst`'s `split_q_gate` was already the shape, one
// kernel of a family root fired only from there.
// ===========================================================================

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `LaunchRule::RouteRows`, as the expression it evaluates to.
#[must_use]
fn route_rows(rows: i32, width: i32) -> Launch {
    /// `runtime/launch.rs:581` — `const MAX_BLOCK: u32 = 1024;`, the cap
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

/// `layout::split_bf16_rows` — one packed row out to two.
///
/// # Safety
///
/// `src` must address `n * (left_dim + right_dim)` live bf16 elements, `left`
/// and `right` `n * left_dim` and `n * right_dim` writable ones, and `ctx`'s
/// stream must be live across the launch.
pub fn split_bf16_rows(
    ctx: &Ctx,
    src: *const bf16,
    left: *mut bf16,
    right: *mut bf16,
    n: i32,
    left_dim: i32,
    right_dim: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            "::pie::layout::split_rows<::pie::bf16>",
            route_rows(n, left_dim),
            &[src.arg(), left.arg(), right.arg(), left_dim.arg(), right_dim.arg()],
        )
    }
}

/// `layout::split_qwen_gdn_ba_bf16` — Qwen's GDN bank, split by halves.
///
/// # Safety
///
/// `ba` must address `n * 2 * v_h` live bf16 elements, `b_out` and `a_out`
/// `n * v_h` writable ones each, and `ctx`'s stream must be live across the
/// launch.
pub fn split_qwen_gdn_ba<T>(
    ctx: &Ctx,
    ba: *const T,
    b_out: *mut T,
    a_out: *mut T,
    n: i32,
    v_h: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            &format!("::pie::layout::split_qwen_gdn_ba<{}>", T::CPP),
            route_rows(n, v_h),
            &[ba.arg(), b_out.arg(), a_out.arg(), v_h.arg()],
        )
    }
}

/// `layout::deinterleave_rows_bf16` — gpt-oss's parity split, row-shaped.
///
/// # Safety
///
/// `fused` must address `2 * rows * h` live bf16 elements, `gate_out` and
/// `up_out` `rows * h` writable ones each, and `ctx`'s stream must be live
/// across the launch.
pub fn deinterleave_rows<T>(
    ctx: &Ctx,
    fused: *const T,
    gate_out: *mut T,
    up_out: *mut T,
    rows: i32,
    h: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            &format!("::pie::layout::deinterleave_rows<{}>", T::CPP),
            route_rows(rows, h),
            &[fused.arg(), gate_out.arg(), up_out.arg(), h.arg()],
        )
    }
}

/// `layout::deinterleave_vec_bf16` — the same split, one thread per element.
///
/// # Safety
///
/// `fused` must address `2 * i` live bf16 elements and `gate_out`/`up_out`
/// `i` writable ones each; `ctx`'s stream must be live across the launch.
pub fn deinterleave_vec<T>(
    ctx: &Ctx,
    fused: *const T,
    gate_out: *mut T,
    up_out: *mut T,
    i: i32,
) -> Result<(), Refusal>
where
    T: Elem,
    *const T: Abi,
    *mut T: Abi,
{
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            &format!("::pie::layout::deinterleave_vec<{}>", T::CPP),
            elementwise(i.unsigned_abs()),
            &[fused.arg(), gate_out.arg(), up_out.arg(), i.arg()],
        )
    }
}

/// `layout::concat_bf16_rows` — `[N, left] ++ [N, right]`.
///
/// # Safety
///
/// `left` and `right` must address `rows * left_dim` and `rows * right_dim`
/// live bf16 elements, `out` `rows * (left_dim + right_dim)` writable ones,
/// and `ctx`'s stream must be live across the launch.
pub fn concat_bf16_rows(
    ctx: &Ctx,
    left: *const bf16,
    right: *const bf16,
    out: *mut bf16,
    rows: i32,
    left_dim: i32,
    right_dim: i32,
) -> Result<(), Refusal> {
    if left_dim + right_dim <= 0 {
        return Err(Refusal::Empty { what: "left_dim + right_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/deinterleave.cuh",
            "::pie::layout::concat_rows<::pie::bf16>",
            route_rows(rows, left_dim),
            &[left.arg(), right.arg(), out.arg(), left_dim.arg(), right_dim.arg()],
        )
    }
}

/// `layout::gather_bf16_rows` — the epilogue's gather.
///
/// # Safety
///
/// `src` must address the rows `row_indices` names at `width` u16 elements
/// each, `row_indices` `num_dst_rows` live `i32`s, `dst` `num_dst_rows *
/// width` writable u16 elements, and `ctx`'s stream must be live across the
/// launch.
pub fn gather_bf16_rows(
    ctx: &Ctx,
    src: *const u16,
    row_indices: *const i32,
    dst: *mut u16,
    num_dst_rows: i32,
    width: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/gather_rows.cuh",
            "::pie::layout::gather_rows<::pie::u16>",
            route_rows(num_dst_rows, width),
            &[src.arg(), row_indices.arg(), dst.arg(), width.arg()],
        )
    }
}

/// `layout::transpose_bf16_nld_to_lnd` — the PLE relay.
///
/// # Safety
///
/// `src` and `dst` must address `n * layers * dim` live u16 elements, `dst`
/// writable, and `ctx`'s stream must be live across the launch.
pub fn transpose_bf16_nld_to_lnd(
    ctx: &Ctx,
    src: *const u16,
    dst: *mut u16,
    n: i32,
    layers: i32,
    dim: i32,
) -> Result<(), Refusal> {
    let total = usize::try_from(n).unwrap_or(0)
        * usize::try_from(layers).unwrap_or(0)
        * usize::try_from(dim).unwrap_or(0);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/gather_rows.cuh",
            "::pie::layout::transpose_nld_to_lnd<::pie::u16>",
            elementwise(u32::try_from(total).unwrap_or(u32::MAX)),
            &[src.arg(), dst.arg(), n.arg(), layers.arg(), dim.arg(), total.arg()],
        )
    }
}

/// `layout::copy_if_valid_slot` — copy a slot's bytes if the slot is valid.
///
/// # Safety
///
/// `src` and `dst` must address `bytes` live bytes, `dst` writable,
/// `slot_ids` must be indexable at `request`, and `ctx`'s stream must be live
/// across the launch.
pub fn copy_if_valid_slot(
    ctx: &Ctx,
    src: *const u8,
    dst: *mut u8,
    bytes: usize,
    slot_ids: *const i32,
    request: usize,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/slot_ops.cuh",
            "::pie::layout::copy_if_valid_slot",
            Launch::grid([1, 1, 1], [256, 1, 1]),
            &[src.arg(), dst.arg(), bytes.arg(), slot_ids.arg(), request.arg()],
        )
    }
}

/// `envelope.cu:37` and `:134` — `head_dim < 256 ? head_dim : 256`.
const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 { head_dim.unsigned_abs() } else { 256 }
}

/// `layout::envelope_merge_written_bf16` — fold explicitly-written KV rows
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `ctx`'s stream is held live for the same window.
pub fn envelope_merge_written(
    ctx: &Ctx,
    k_curr: *const bf16,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: crate::jit::abi::MaybeConst<u8>,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    /// `envelope.cuh:374`, `kEnvelopeFuseMaxTokens`.
    const FUSE_MAX_TOKENS: i32 = 128;

    let launch = Launch::grid(
        [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        [threads_for(head_dim), 1, 1],
    );

    if num_tokens <= FUSE_MAX_TOKENS {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                "layout/envelope.cuh",
                "::pie::layout::merge_written_fused<::pie::i32(0)>",
                launch,
                &[
                    k_curr.arg(),
                    w_page.arg(),
                    w_off.arg(),
                    row_valid.arg(),
                    env_min.arg(),
                    env_max.arg(),
                    num_tokens.arg(),
                    num_kv_heads.arg(),
                    head_dim.arg(),
                ],
            )
        };
    }

    // Past the fuse threshold the step is TWO launches, in this order: the
    // reset writes the identity into every page this batch STARTS, and a
    // merge that ran first would have its folds overwritten.
    //
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/envelope.cuh",
            "::pie::layout::reset_started_pages<::pie::i32(0)>",
            launch,
            &[
                w_page.arg(),
                w_off.arg(),
                row_valid.arg(),
                env_min.arg(),
                env_max.arg(),
                num_tokens.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        )?;
        ctx.launch(
            "layout/envelope.cuh",
            "::pie::layout::merge_written<::pie::i32(0)>",
            launch,
            &[
                k_curr.arg(),
                w_page.arg(),
                row_valid.arg(),
                env_min.arg(),
                env_max.arg(),
                num_tokens.arg(),
                num_kv_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

/// `layout::envelope_seed_empty_bf16` — write the `+inf`/`-inf` identity
///
/// # Safety
///
/// Both planes are device addresses the caller keeps live across the launch,
/// and `ctx`'s stream is held live for the same window.
pub fn envelope_seed_empty(
    ctx: &Ctx,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    /// `envelope.cu:71` — the seed's own block, which is fixed rather than
    const SEED_BLOCK: u32 = 256;

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/envelope.cuh",
            "::pie::layout::seed_empty<::pie::i32(0)>",
            Launch::grid([u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1], [SEED_BLOCK, 1, 1]),
            &[env_min.arg(), env_max.arg(), n.arg()],
        )
    }
}

/// `layout::envelope_update_appended_bf16` — fold the pages an append touched
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `ctx`'s stream is held live for the same window.
pub fn envelope_update_appended(
    ctx: &Ctx,
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
    head_dim: i32,
) -> Result<(), Refusal> {
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/envelope.cuh",
            "::pie::layout::update_appended<::pie::bf16>",
            Launch::grid(
                [max_touched.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
                [threads_for(head_dim), 1, 1],
            ),
            &[
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
            ],
        )
    }
}

/// `embed.cu:35` — the vector width, in `bf16` elements.
const VEC_WIDTH: i32 = 8;

/// `embed.cu:33-35` — the host test that picks `VEC`.
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
pub fn embed_bf16(
    ctx: &Ctx,
    token_ids: *const i32,
    weight: *const bf16,
    y: *mut bf16,
    num_tokens: i32,
    hidden: i32,
    vocab: i32,
) -> Result<(), Refusal> {
    /// `embed.cu:31` — `constexpr int BLOCK = 256;`.
    const EMBED_BLOCK: u32 = 256;

    let vec = vectorisable(hidden, weight, y.cast_const());
    let per_row = if vec { hidden / VEC_WIDTH } else { hidden };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
    let instantiation = if vec { "::pie::layout::embed<\
                                      ::pie::true_type::value>" } else { "::pie::layout::embed<::pie::false_type::value>" };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            "layout/embed.cuh",
            instantiation,
            Launch::grid([blocks, 1, 1], [EMBED_BLOCK, 1, 1]),
            &[
                token_ids.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                vocab.arg(),
                num_tokens.arg(),
                per_row.arg(),
            ],
        )
    }
}

/// This family's routines, and what a trace may say about each.
///
/// The argument lists are DERIVED from the `fn`s above -- `routine!` sees only
/// the identifier. What is stated beside it is what no signature carries, and
/// for `layout` that is NOTHING: not one contract here claims `whole`, an
/// in-place pair or a depth-prefix plan, because every one of these kernels
/// reads one buffer and writes another.
///
/// The last line is `driver_bound!` and its host program is not above it —
/// see [`crate::driver_internal`]'s header, which is where that `fn` lives
/// and why it stays there.
pub static ROUTINES: &[Routine] = &[
    routine!(split_bf16_rows),
    routine!(split_qwen_gdn_ba_bf16 = split_qwen_gdn_ba::<bf16>),
    routine!(deinterleave_rows_bf16 = deinterleave_rows::<bf16>),
    routine!(deinterleave_vec_bf16 = deinterleave_vec::<bf16>),
    routine!(concat_bf16_rows),
    routine!(gather_bf16_rows),
    routine!(transpose_bf16_nld_to_lnd),
    routine!(copy_if_valid_slot),
    routine!(envelope_merge_written),
    routine!(envelope_seed_empty),
    routine!(envelope_update_appended),
    routine!(embed_bf16),
    // The qwen3.5 hybrid lowers `OpKind::SplitQGate` to this symbol, and
    // nothing declared it. The host program is
    // `driver_internal::split_q_gate_bf16` and stays there; the declaration
    // has to be here, because `Family::symbol` is the module path's first
    // segment plus the routine's name and no `Family` in `driver_internal`
    // could offer a `layout::` symbol at all.
    //
    // **This declares the symbol and does not arm it.** A fire naming it
    // still refuses with `NoArm` -- `bind/arms/layout.rs` has no entry for
    // it, and writing one is fire-path work this declaration does not do.
    driver_bound!(split_q_gate_bf16),
];

/// `layout`, as a trace names it.
pub static FAMILY: Family = crate::family!(ROUTINES);
