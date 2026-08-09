#![allow(clippy::too_many_arguments)]

use crate::jit::{Ctx, Family, Launch, Routine};
use crate::routine;
use crate::x::Abi;
use crate::x::abi::bf16;
use kernels::Refusal;

use core::ffi::c_void;

/// `layout/deinterleave.cuh` — the packed-bank splits and the row concat.
pub mod deinterleave {

    use crate::jit::Root;

    /// `layout/deinterleave.cuh` — the root these routines compile a symbol
    /// out of.
    pub static ROOT: Root = Root::new(
        "layout/deinterleave",
        include_str!("../../csrc/src/layout/deinterleave.cuh"),
        "layout/deinterleave.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Absolute, because a routine body names the instantiation itself rather
    /// than a label some other table maps to one. The `<...>` argument is what
    /// used to be a row's `elem`.
    ///
    /// `pub(in crate::x)` and not `pub(super)`: `split_q_gate` is a driver op
    /// with no routine in this family, so the body that names its
    /// instantiation is `x::driver_internal`'s.
    pub(in crate::x) mod inst {
        /// `deinterleave.cuh:85` — gpt-oss's parity split, row-shaped.
        pub const DEINTERLEAVE_ROWS: &str = "::pie_cuda_driver::kernels::layout::device::deinterleave_rows\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `deinterleave.cuh:109` — the same split, one thread per element.
        pub const DEINTERLEAVE_VEC: &str = "::pie_cuda_driver::kernels::layout::device::deinterleave_vec\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `deinterleave.cuh:152` — `[N, left] ++ [N, right]`.
        pub const CONCAT_ROWS: &str = "::pie_cuda_driver::kernels::layout::device::concat_rows\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `deinterleave.cuh:188` — one packed row out to two.
        pub const SPLIT_ROWS: &str = "::pie_cuda_driver::kernels::layout::device::split_rows\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `deinterleave.cuh:170` — Qwen's GDN bank, split by halves.
        pub const SPLIT_QWEN_GDN_BA: &str = "::pie_cuda_driver::kernels::layout::device::split_qwen_gdn_ba\
             <::pie_cuda_driver::kernels::device::bf16>";
        /// `deinterleave.cuh:130` — full attention's per-head query/gate cut.
        pub const SPLIT_Q_GATE: &str = "::pie_cuda_driver::kernels::layout::device::split_q_gate\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `layout/gather_rows.cuh` — the epilogue's gather and the PLE relay.
pub mod gather_rows {
    use crate::jit::Root;

    /// `layout/gather_rows.cuh` — the root these routines compile a symbol
    /// out of.
    pub static ROOT: Root = Root::new(
        "layout/gather_rows",
        include_str!("../../csrc/src/layout/gather_rows.cuh"),
        "layout/gather_rows.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Both are instantiated at `device::u16` and not at `device::bf16`: both
    /// are pure copies, neither ever converts to float, and a tag type that
    /// promises arithmetic nobody performs is a tag type that invites it.
    pub(super) mod inst {
        /// `gather_rows.cuh:78` — the epilogue's gather.
        pub const GATHER_ROWS: &str = "::pie_cuda_driver::kernels::layout::device::gather_rows\
             <::pie_cuda_driver::kernels::device::u16>";
        /// `gather_rows.cuh:132` — the PLE relay, `[N, L, D] -> [L, N, D]`.
        pub const TRANSPOSE_NLD_TO_LND: &str = "::pie_cuda_driver::kernels::layout::device::transpose_nld_to_lnd\
             <::pie_cuda_driver::kernels::device::u16>";
    }
}

/// `layout/slot_ops.cuh` — the slot-conditional byte copy.
pub mod slot_ops {
    use crate::jit::Root;

    /// `layout/slot_ops.cuh` — the root this routine compiles a symbol out of.
    pub static ROOT: Root = Root::new(
        "layout/slot_ops",
        include_str!("../../csrc/src/layout/slot_ops.cuh"),
        "layout/slot_ops.cuh",
    );

    /// The template-id NVRTC is handed, spelled as it is handed it.
    ///
    /// No `<...>`: the `__global__` has no template parameter list, which is
    /// the fact a row spelled `DeviceKernel::PLAIN`.
    pub(super) mod inst {
        /// `slot_ops.cuh:64` — copy a slot's bytes if the slot is valid.
        pub const COPY_IF_VALID_SLOT: &str =
            "::pie_cuda_driver::kernels::layout::device::copy_if_valid_slot";
    }
}

/// `layout/envelope.cuh` — the quest per-page key envelope tier.
pub mod envelope {

    use crate::jit::Root;

    /// `layout/envelope.cuh` — the root these routines compile a symbol out
    /// of.
    pub static ROOT: Root = Root::new(
        "layout/envelope",
        include_str!("../../csrc/src/layout/envelope.cuh"),
        "layout/envelope.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// Four of the five take a NON-TYPE template argument, `device::i32(0)`,
    /// which is what a row's `elem` carried and why the spelling is written
    /// out here rather than assembled from an element type.
    pub(super) mod inst {
        /// `envelope.cuh:377` — the whole maintenance step, fused.
        pub const MERGE_WRITTEN_FUSED: &str = "::pie_cuda_driver::kernels::layout::device::merge_written_fused\
             <::pie_cuda_driver::kernels::device::i32(0)>";
        /// `envelope.cuh:492` — the FIRST of the two launches taken above the
        /// fuse threshold.
        pub const RESET_STARTED_PAGES: &str = "::pie_cuda_driver::kernels::layout::device::reset_started_pages\
             <::pie_cuda_driver::kernels::device::i32(0)>";
        /// `envelope.cuh:535` — the SECOND of the two.
        pub const MERGE_WRITTEN: &str = "::pie_cuda_driver::kernels::layout::device::merge_written\
             <::pie_cuda_driver::kernels::device::i32(0)>";
        /// `envelope.cuh:337` — the `+inf`/`-inf` identity across a plane.
        pub const SEED_EMPTY: &str = "::pie_cuda_driver::kernels::layout::device::seed_empty\
             <::pie_cuda_driver::kernels::device::i32(0)>";
        /// `envelope.cuh:238` — the incremental fold of the appended pages.
        pub const UPDATE_APPENDED: &str = "::pie_cuda_driver::kernels::layout::device::update_appended\
             <::pie_cuda_driver::kernels::device::bf16>";
    }
}

/// `layout/embed.cuh` — the flat embedding gather.
pub mod embed {

    use crate::jit::Root;

    /// `layout/embed.cuh` — the root this routine compiles a symbol out of.
    pub static ROOT: Root = Root::new(
        "layout/embed",
        include_str!("../../csrc/src/layout/embed.cuh"),
        "layout/embed.cuh",
    );

    /// The template-ids NVRTC is handed, spelled as it is handed them.
    ///
    /// One template, two instantiations, and which one fires is a host
    /// predicate over two addresses — [`vectorisable`](super::vectorisable).
    pub(super) mod inst {
        /// `embed.cuh:60` — the vectorised gather, eight bf16 per thread.
        pub const VEC: &str = "::pie_cuda_driver::kernels::layout::device::embed\
             <::pie_cuda_driver::kernels::device::true_type::value>";
        /// The same gather, one element per thread.
        pub const SCALAR: &str = "::pie_cuda_driver::kernels::layout::device::embed\
             <::pie_cuda_driver::kernels::device::false_type::value>";
    }
}

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
const WARP: u32 = 32;

/// `runtime/launch.rs:581` — `const MAX_BLOCK: u32 = 1024;`, the cap
const MAX_BLOCK: u32 = 1024;

/// `LaunchRule::RouteRows`, as the expression it evaluates to.
#[must_use]
fn route_rows(rows: i32, width: i32) -> Launch {
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
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if left_dim <= 0 {
        return Err(Refusal::Empty { what: "left_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &deinterleave::ROOT,
            deinterleave::inst::SPLIT_ROWS,
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
pub fn split_qwen_gdn_ba_bf16(
    ctx: &Ctx,
    ba: *const bf16,
    b_out: *mut bf16,
    a_out: *mut bf16,
    n: i32,
    v_h: i32,
) -> Result<(), Refusal> {
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if v_h <= 0 {
        return Err(Refusal::Empty { what: "v_h" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &deinterleave::ROOT,
            deinterleave::inst::SPLIT_QWEN_GDN_BA,
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
pub fn deinterleave_rows_bf16(
    ctx: &Ctx,
    fused: *const bf16,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    rows: i32,
    h: i32,
) -> Result<(), Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if h <= 0 {
        return Err(Refusal::Empty { what: "h" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &deinterleave::ROOT,
            deinterleave::inst::DEINTERLEAVE_ROWS,
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
pub fn deinterleave_vec_bf16(
    ctx: &Ctx,
    fused: *const bf16,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    i: i32,
) -> Result<(), Refusal> {
    if i <= 0 {
        return Err(Refusal::Empty { what: "num_elements" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &deinterleave::ROOT,
            deinterleave::inst::DEINTERLEAVE_VEC,
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
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if left_dim + right_dim <= 0 {
        return Err(Refusal::Empty { what: "left_dim + right_dim" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &deinterleave::ROOT,
            deinterleave::inst::CONCAT_ROWS,
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
    if num_dst_rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gather_rows::ROOT,
            gather_rows::inst::GATHER_ROWS,
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
    if n <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if layers <= 0 {
        return Err(Refusal::Empty { what: "layers" });
    }
    if dim <= 0 {
        return Err(Refusal::Empty { what: "ple_dim" });
    }
    let total = usize::try_from(n).unwrap_or(0)
        * usize::try_from(layers).unwrap_or(0)
        * usize::try_from(dim).unwrap_or(0);
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &gather_rows::ROOT,
            gather_rows::inst::TRANSPOSE_NLD_TO_LND,
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
    if bytes == 0 {
        return Err(Refusal::Empty { what: "bytes" });
    }
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &slot_ops::ROOT,
            slot_ops::inst::COPY_IF_VALID_SLOT,
            Launch::grid([1, 1, 1], [256, 1, 1]),
            &[src.arg(), dst.arg(), bytes.arg(), slot_ids.arg(), request.arg()],
        )
    }
}

/// `envelope.cu:37` and `:134` — `head_dim < 256 ? head_dim : 256`.
const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 { head_dim.unsigned_abs() } else { 256 }
}

/// `envelope.cu:71` — the seed's own block, which is fixed rather than
const SEED_BLOCK: u32 = 256;

/// `envelope.cuh:374`, `kEnvelopeFuseMaxTokens`.
const FUSE_MAX_TOKENS: i32 = 128;

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
    row_valid: crate::x::abi::MaybeConst<u8>,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), Refusal> {
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Err(Refusal::Empty { what: "the layer's kv heads or head_dim" });
    }

    let launch = Launch::grid(
        [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        [threads_for(head_dim), 1, 1],
    );

    if num_tokens <= FUSE_MAX_TOKENS {
        // SAFETY: `call()`'s contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                &envelope::ROOT,
                envelope::inst::MERGE_WRITTEN_FUSED,
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
            &envelope::ROOT,
            envelope::inst::RESET_STARTED_PAGES,
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
            &envelope::ROOT,
            envelope::inst::MERGE_WRITTEN,
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
    if num_pages <= 0 {
        return Err(Refusal::Empty { what: "num_pages" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Err(Refusal::Empty { what: "the layer's kv heads or head_dim" });
    }

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &envelope::ROOT,
            envelope::inst::SEED_EMPTY,
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
    if num_requests <= 0 {
        return Err(Refusal::Empty { what: "num_requests" });
    }
    if max_touched <= 0 {
        return Err(Refusal::Empty { what: "the touched-page bound" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0 {
        return Err(Refusal::Empty { what: "the layer's kv heads, head_dim or page_size" });
    }

    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &envelope::ROOT,
            envelope::inst::UPDATE_APPENDED,
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

/// `embed.cu:31` — `constexpr int BLOCK = 256;`.
const EMBED_BLOCK: u32 = 256;

/// `embed.cu:35` — the vector width, in `bf16` elements.
const VEC_WIDTH: i32 = 8;

/// `(uintptr_t)p % 16 == 0`, which is what `fire::hand::aligned16` was.
#[must_use]
fn aligned16(p: *const c_void) -> bool {
    (p as usize) % 16 == 0
}

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
    if num_tokens <= 0 {
        return Err(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Err(Refusal::Empty { what: "hidden" });
    }
    let vec = vectorisable(hidden, weight, y.cast_const());
    let per_row = if vec { hidden / VEC_WIDTH } else { hidden };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
    let instantiation = if vec { embed::inst::VEC } else { embed::inst::SCALAR };
    // SAFETY: `call()`'s contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &embed::ROOT,
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
pub static ROUTINES: &[Routine] = &[
    routine!(split_bf16_rows),
    routine!(split_qwen_gdn_ba_bf16),
    routine!(deinterleave_rows_bf16),
    routine!(deinterleave_vec_bf16),
    routine!(concat_bf16_rows),
    routine!(gather_bf16_rows),
    routine!(transpose_bf16_nld_to_lnd),
    routine!(copy_if_valid_slot),
    routine!(envelope_merge_written),
    routine!(envelope_seed_empty),
    routine!(envelope_update_appended),
    routine!(embed_bf16),
];

/// `layout`, as a trace names it.
pub static FAMILY: Family = Family { namespace: "layout", routines: ROUTINES };
