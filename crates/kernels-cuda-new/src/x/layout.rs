#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::bf16;
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

/// `layout/deinterleave.cuh` — the packed-bank splits and the row concat.
pub mod deinterleave {
    use super::bf16;

    unit! {
        /// gpt-oss's parity deinterleave, Qwen's GDN halves, the concat/split
        unit DEINTERLEAVE = "layout/deinterleave",
            text = include_str!("../../csrc/src/layout/deinterleave.cuh"),
            file = "layout/deinterleave.cuh";

        /// `deinterleave.cuh:85` — gpt-oss packs gate and up ROW BY ROW, so
        fn deinterleave_rows = "layout::device::deinterleave_rows" <T> (
            fused: *const T,
            gate_out: *mut T,
            up_out: *mut T,
            h: i32,
        ) where *const T, *mut T {
            "layout::deinterleave_rows_bf16" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:109` — the flat form of the same split, one
        fn deinterleave_vec = "layout::device::deinterleave_vec" <T> (
            fused: *const T,
            gate_out: *mut T,
            up_out: *mut T,
            i: i32,
        ) where *const T, *mut T {
            "layout::deinterleave_vec_bf16" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:152` — `[N, left] ++ [N, right] -> [N,
        fn concat_rows = "layout::device::concat_rows" <T> (
            left: *const T,
            right: *const T,
            out: *mut T,
            left_dim: i32,
            right_dim: i32,
        ) where *const T, *mut T {
            "layout::concat_bf16_rows" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:188` — the inverse: one packed row out to two.
        fn split_rows = "layout::device::split_rows" <T> (
            src: *const T,
            left: *mut T,
            right: *mut T,
            left_dim: i32,
            right_dim: i32,
        ) where *const T, *mut T {
            "layout::split_bf16_rows" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:170` — Qwen's GDN bank, split by HALVES where
        fn split_qwen_gdn_ba = "layout::device::split_qwen_gdn_ba" <T> (
            ba: *const T,
            b_out: *mut T,
            a_out: *mut T,
            v_h: i32,
        ) where *const T, *mut T {
            "layout::split_qwen_gdn_ba_bf16" => where [T = bf16] "device::bf16",
        }

        /// `deinterleave.cuh:130` — full attention's per-head query/gate cut.
        fn split_q_gate = "layout::device::split_q_gate" <T> (
            packed: *const T,
            q_out: *mut T,
            gate_out: *mut T,
            n: i32,
            num_heads: i32,
            head_dim: i32,
        ) where *const T, *mut T {
            "layout::split_q_gate_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `layout/gather_rows.cuh` — the epilogue's gather and the PLE relay.
pub mod gather_rows {
    unit! {
        /// Two of the header's four kernels, both at `device::u16`.
        unit GATHER_ROWS = "layout/gather_rows",
            text = include_str!("../../csrc/src/layout/gather_rows.cuh"),
            file = "layout/gather_rows.cuh";

        /// `gather_rows.cuh:78` — THE EPILOGUE'S GATHER.
        fn gather_rows = "layout::device::gather_rows" <T> (
            src: *const T,
            row_indices: *const i32,
            dst: *mut T,
            width: i32,
        ) where *const T, *mut T {
            "layout::gather_bf16_rows" => where [T = u16] "device::u16",
        }

        /// `gather_rows.cuh:132` — the PLE relay: `[N, L, D] -> [L, N, D]`,
        fn transpose_nld_to_lnd = "layout::device::transpose_nld_to_lnd" <T> (
            src: *const T,
            dst: *mut T,
            n: i32,
            layers: i32,
            dim: i32,
            total: usize,
        ) where *const T, *mut T {
            "layout::transpose_bf16_nld_to_lnd" => where [T = u16] "device::u16",
        }
    }
}

/// `layout/slot_ops.cuh` — the slot-conditional byte copy.
pub mod slot_ops {
    unit! {
        /// One of the header's two kernels.
        unit SLOT_OPS = "layout/slot_ops",
            text = include_str!("../../csrc/src/layout/slot_ops.cuh"),
            file = "layout/slot_ops.cuh";

        /// `slot_ops.cuh:64` — copy a slot's bytes if the slot is valid.
        fn copy_if_valid_slot = "layout::device::copy_if_valid_slot" (
            src: *const u8,
            dst: *mut u8,
            bytes: usize,
            slot_ids: *const i32,
            request: usize,
        ) {
            "layout::copy_if_valid_slot" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `layout/envelope.cuh` — the quest per-page key envelope tier.
pub mod envelope {
    use super::bf16;
    use crate::x::abi::MaybeConst;

    unit! {
        /// Five of the header's seven kernels.
        unit ENVELOPE = "layout/envelope",
            text = include_str!("../../csrc/src/layout/envelope.cuh"),
            file = "layout/envelope.cuh";

        /// `envelope.cuh:377` — the whole maintenance step for a SHORT
        fn merge_written_fused = "layout::device::merge_written_fused" (
            k_curr: *const bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_tokens: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) {
            "layout::envelope_merge_written_fused_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:492` — the FIRST of the two launches taken when
        fn reset_started_pages = "layout::device::reset_started_pages" (
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_tokens: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) {
            "layout::envelope_reset_started_pages_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:535` — the SECOND of the two.
        fn merge_written = "layout::device::merge_written" (
            k_curr: *const bf16,
            w_page: *const u32,
            row_valid: MaybeConst<u8>,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_tokens: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) {
            "layout::envelope_merge_written_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:337` — the `+inf`/`-inf` identity across a whole
        fn seed_empty = "layout::device::seed_empty" (
            env_min: *mut bf16,
            env_max: *mut bf16,
            n: usize,
        ) {
            "layout::envelope_seed_empty_bf16" => "device::i32(0)",
        }

        /// `envelope.cuh:238` — the incremental fold of the pages an append
        fn update_appended = "layout::device::update_appended" <T> (
            k_pages: *const T,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            env_min: *mut bf16,
            env_max: *mut bf16,
            num_requests: i32,
            page_size: i32,
            num_kv_heads: i32,
            head_dim: i32,
        ) where *const T {
            "layout::envelope_update_appended_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `layout/embed.cuh` — the flat embedding gather.
pub mod embed {
    use super::bf16;

    unit! {
        /// One template, two instantiations — the first launch of every fire.
        unit EMBED = "layout/embed",
            text = include_str!("../../csrc/src/layout/embed.cuh"),
            file = "layout/embed.cuh";

        /// `embed.cuh:60` — gather one row of the vocabulary table per token.
        fn embed = "layout::device::embed" (
            token_ids: *const i32,
            weight: *const bf16,
            y: *mut bf16,
            hidden: i32,
            vocab: i32,
            num_tokens: i32,
            per_row: i32,
        ) {
            "layout::embed#vec" => "device::true_type::value",
            "layout::embed#scalar" => "device::false_type::value",
        }
    }
}

/// The units `layout` compiles.
pub static UNITS: &[Unit] = &[
    deinterleave::DEINTERLEAVE,
    embed::EMBED,
    envelope::ENVELOPE,
    gather_rows::GATHER_ROWS,
    slot_ops::SLOT_OPS,
];

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
        width
            .unsigned_abs()
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
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
/// and `right` `n * left_dim` and `n * right_dim` writable ones, and `stream`
/// must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn split_bf16_rows(
    src: *const bf16,
    left: *mut bf16,
    right: *mut bf16,
    n: i32,
    left_dim: i32,
    right_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if left_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "left_dim" });
    }
    unsafe {
        deinterleave::raw::split_rows(
            "layout::split_bf16_rows",
            route_rows(n, left_dim),
            src,
            left,
            right,
            left_dim,
            right_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::split_qwen_gdn_ba_bf16` — Qwen's GDN bank, split by halves.
///
/// # Safety
///
/// `ba` must address `n * 2 * v_h` live bf16 elements, `b_out` and `a_out`
/// `n * v_h` writable ones each, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn split_qwen_gdn_ba_bf16(
    ba: *const bf16,
    b_out: *mut bf16,
    a_out: *mut bf16,
    n: i32,
    v_h: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if v_h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "v_h" });
    }
    unsafe {
        deinterleave::raw::split_qwen_gdn_ba(
            "layout::split_qwen_gdn_ba_bf16",
            route_rows(n, v_h),
            ba,
            b_out,
            a_out,
            v_h,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::deinterleave_rows_bf16` — gpt-oss's parity split, row-shaped.
///
/// # Safety
///
/// `fused` must address `2 * rows * h` live bf16 elements, `gate_out` and
/// `up_out` `rows * h` writable ones each, and `stream` must be live across
/// the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn deinterleave_rows_bf16(
    fused: *const bf16,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    rows: i32,
    h: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "h" });
    }
    unsafe {
        deinterleave::raw::deinterleave_rows(
            "layout::deinterleave_rows_bf16",
            route_rows(rows, h),
            fused,
            gate_out,
            up_out,
            h,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::deinterleave_vec_bf16` — the same split, one thread per element.
///
/// # Safety
///
/// `fused` must address `2 * i` live bf16 elements and `gate_out`/`up_out`
/// `i` writable ones each; `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn deinterleave_vec_bf16(
    fused: *const bf16,
    gate_out: *mut bf16,
    up_out: *mut bf16,
    i: i32,
    stream: *mut c_void,
) -> Fired {
    if i <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_elements" });
    }
    unsafe {
        deinterleave::raw::deinterleave_vec(
            "layout::deinterleave_vec_bf16",
            elementwise(i.unsigned_abs()),
            fused,
            gate_out,
            up_out,
            i,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::concat_bf16_rows` — `[N, left] ++ [N, right]`.
///
/// # Safety
///
/// `left` and `right` must address `rows * left_dim` and `rows * right_dim`
/// live bf16 elements, `out` `rows * (left_dim + right_dim)` writable ones,
/// and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn concat_bf16_rows(
    left: *const bf16,
    right: *const bf16,
    out: *mut bf16,
    rows: i32,
    left_dim: i32,
    right_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if left_dim + right_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "left_dim + right_dim" });
    }
    unsafe {
        deinterleave::raw::concat_rows(
            "layout::concat_bf16_rows",
            route_rows(rows, left_dim),
            left,
            right,
            out,
            left_dim,
            right_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::gather_bf16_rows` — the epilogue's gather.
///
/// # Safety
///
/// `src` must address the rows `row_indices` names at `width` u16 elements
/// each, `row_indices` `num_dst_rows` live `i32`s, `dst` `num_dst_rows *
/// width` writable u16 elements, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn gather_bf16_rows(
    src: *const u16,
    row_indices: *const i32,
    dst: *mut u16,
    num_dst_rows: i32,
    width: i32,
    stream: *mut c_void,
) -> Fired {
    if num_dst_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if width <= 0 {
        return Fired::Declined(Refusal::Empty { what: "width" });
    }
    unsafe {
        gather_rows::raw::gather_rows(
            "layout::gather_bf16_rows",
            route_rows(num_dst_rows, width),
            src,
            row_indices,
            dst,
            width,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::transpose_bf16_nld_to_lnd` — the PLE relay.
///
/// # Safety
///
/// `src` and `dst` must address `n * layers * dim` live u16 elements, `dst`
/// writable, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn transpose_bf16_nld_to_lnd(
    src: *const u16,
    dst: *mut u16,
    n: i32,
    layers: i32,
    dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if layers <= 0 {
        return Fired::Declined(Refusal::Empty { what: "layers" });
    }
    if dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "ple_dim" });
    }
    let total = usize::try_from(n).unwrap_or(0)
        * usize::try_from(layers).unwrap_or(0)
        * usize::try_from(dim).unwrap_or(0);
    unsafe {
        gather_rows::raw::transpose_nld_to_lnd(
            "layout::transpose_bf16_nld_to_lnd",
            elementwise(u32::try_from(total).unwrap_or(u32::MAX)),
            src,
            dst,
            n,
            layers,
            dim,
            total,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::copy_if_valid_slot` — copy a slot's bytes if the slot is valid.
///
/// # Safety
///
/// `src` and `dst` must address `bytes` live bytes, `dst` writable,
/// `slot_ids` must be indexable at `request`, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn copy_if_valid_slot(
    src: *const u8,
    dst: *mut u8,
    bytes: usize,
    slot_ids: *const i32,
    request: usize,
    stream: *mut c_void,
) -> Fired {
    if bytes == 0 {
        return Fired::Declined(Refusal::Empty { what: "bytes" });
    }
    unsafe {
        slot_ops::raw::copy_if_valid_slot(
            "layout::copy_if_valid_slot",
            Launch { grid: [1, 1, 1], block: [256, 1, 1], smem: 0, smem_opt_in: false },
            src,
            dst,
            bytes,
            slot_ids,
            request,
            stream,
        );
    }
    Fired::Launched
}

/// `envelope.cu:37` and `:134` — `head_dim < 256 ? head_dim : 256`.
const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 {
        head_dim.unsigned_abs()
    } else {
        256
    }
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
/// and `stream` is the caller's stream, held live for the same window.
#[cfg(feature = "_cuda")]
pub unsafe fn envelope_merge_written(
    k_curr: *const bf16,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: crate::x::abi::MaybeConst<u8>,
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the layer's kv heads or head_dim" });
    }

    let launch = Launch {
        grid: [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        block: [threads_for(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    };

    if num_tokens <= FUSE_MAX_TOKENS {
        unsafe {
            envelope::raw::merge_written_fused(
                "layout::envelope_merge_written_fused_bf16",
                launch,
                k_curr,
                w_page,
                w_off,
                row_valid,
                env_min,
                env_max,
                num_tokens,
                num_kv_heads,
                head_dim,
                stream,
            );
        }
        return Fired::Launched;
    }

    unsafe {
        envelope::raw::reset_started_pages(
            "layout::envelope_reset_started_pages_bf16",
            launch,
            w_page,
            w_off,
            row_valid,
            env_min,
            env_max,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
        );
        envelope::raw::merge_written(
            "layout::envelope_merge_written_bf16",
            launch,
            k_curr,
            w_page,
            row_valid,
            env_min,
            env_max,
            num_tokens,
            num_kv_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::envelope_seed_empty_bf16` — write the `+inf`/`-inf` identity
///
/// # Safety
///
/// Both planes are device addresses the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[cfg(feature = "_cuda")]
pub unsafe fn envelope_seed_empty(
    env_min: *mut bf16,
    env_max: *mut bf16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if num_pages <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_pages" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the layer's kv heads or head_dim" });
    }

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    unsafe {
        envelope::raw::seed_empty(
            "layout::envelope_seed_empty_bf16",
            Launch {
                grid: [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
                block: [SEED_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            env_min,
            env_max,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `layout::envelope_update_appended_bf16` — fold the pages an append touched
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[cfg(feature = "_cuda")]
pub unsafe fn envelope_update_appended(
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
    stream: *mut c_void,
) -> Fired {
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_requests" });
    }
    if max_touched <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the touched-page bound" });
    }
    if num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0 {
        return Fired::Declined(Refusal::Empty {
            what: "the layer's kv heads, head_dim or page_size",
        });
    }

    unsafe {
        envelope::raw::update_appended(
            "layout::envelope_update_appended_bf16",
            Launch {
                grid: [max_touched.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
                block: [threads_for(head_dim), 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            k_pages,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            env_min,
            env_max,
            num_requests,
            page_size,
            num_kv_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `embed.cu:31` — `constexpr int BLOCK = 256;`.
const EMBED_BLOCK: u32 = 256;

/// `embed.cu:35` — the vector width, in `bf16` elements.
const VEC_WIDTH: i32 = 8;

/// `(uintptr_t)p % 16 == 0`, which is what `fire::hand::aligned16` was.
#[cfg(feature = "_cuda")]
#[must_use]
fn aligned16(p: *const c_void) -> bool {
    (p as usize) % 16 == 0
}

/// `embed.cu:33-35` — the host test that picks `VEC`.
#[cfg(feature = "_cuda")]
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
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn embed_bf16(
    token_ids: *const i32,
    weight: *const bf16,
    y: *mut bf16,
    num_tokens: i32,
    hidden: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    let vec = vectorisable(hidden, weight, y.cast_const());
    let per_row = if vec { hidden / VEC_WIDTH } else { hidden };
    let total = i64::from(num_tokens) * i64::from(per_row);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let blocks = ((total + i64::from(EMBED_BLOCK) - 1) / i64::from(EMBED_BLOCK)) as u32;
    let launch = Launch {
        grid: [blocks, 1, 1],
        block: [EMBED_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    };
    let symbol = if vec { "layout::embed#vec" } else { "layout::embed#scalar" };
    unsafe {
        embed::raw::embed(
            symbol, launch, token_ids, weight, y, hidden, vocab, num_tokens, per_row, stream,
        );
    }
    Fired::Launched
}

contract! {
    /// One packed row out to two — `[N, left+right] -> [N, left], [N, right]`.
    SPLIT_ROWS = "layout::split_bf16_rows" as split_rows

    /// Qwen's GDN bank, split by HALVES where the parity deinterleave splits
    SPLIT_QWEN_GDN_BA = "layout::split_qwen_gdn_ba_bf16" as split_qwen_gdn_ba

    /// The embedding gather — the first launch of every fire.
    EMBED = "layout::embed_bf16" as embed

    /// The epilogue's gather: collect the rows a prefill actually samples,
    GATHER_ROWS = "layout::gather_bf16_rows" as gather_rows

    /// The PLE relay: `[N, L, D] -> [L, N, D]`, so a layer reads a contiguous
    TRANSPOSE_NLD_TO_LND = "layout::transpose_bf16_nld_to_lnd" as transpose_nld_to_lnd

    /// PSEUDO-SYMBOL, deliberately unstated — and one of the three
    VERIFY_STASH_STORE = "qwen35_verify_stash_store" as verify_stash_store

    /// [`VERIFY_STASH_STORE`]'s other half, on the same terms.
    VERIFY_STASH_LOAD = "qwen35_verify_stash_load" as verify_stash_load
}

#[cfg(feature = "_cuda")]
bind! {
    SPLIT_ROWS => { cx, stream => {
        unsafe {
            split_bf16_rows(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.out_width(1)?,
                stream,
            )
        }
        .ok()
    }},

    SPLIT_QWEN_GDN_BA => { cx, stream => {
        unsafe {
            split_qwen_gdn_ba_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    EMBED => { cx, stream => {
        unsafe {
            embed_bf16(
                cx.token_ids()?,
                cx.weight_named(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.vocab()?,
                stream,
            )
        }
        .ok()
    }},

    GATHER_ROWS => { cx, stream => {
        unsafe {
            gather_bf16_rows(
                cx.arg_in(0)?.cast_const().cast::<u16>(),
                cx.sampling_indices()?,
                cx.arg_out(0)?.cast::<u16>(),
                cx.rows().count,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    TRANSPOSE_NLD_TO_LND => { cx, stream => {
        let ple_dim = cx.ple_dim()?;
        if ple_dim <= 0 {
            return Err(Refusal::Empty { what: "ple_dim" });
        }
        unsafe {
            transpose_bf16_nld_to_lnd(
                cx.arg_in(0)?.cast_const().cast::<u16>(),
                cx.arg_out(0)?.cast::<u16>(),
                cx.rows().count,
                cx.in_width(0)? / ple_dim,
                ple_dim,
                stream,
            )
        }
        .ok()
    }},
}
