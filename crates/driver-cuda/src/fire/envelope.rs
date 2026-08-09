//! `layout/envelope.cu`'s three launchers, in Rust. **The file is gone.**
//!
//! Quest per-page key envelopes: for every `(page, kv_head)` a min and a max
//! over the page's live keys, `[num_pages, num_kv_heads, head_dim]` bf16 in
//! each of two planes. The tier has three maintenance points and this module
//! is all three —
//!
//! * [`seed_empty`] writes the `+inf`/`-inf` identity into a fresh pool,
//! * [`update_appended`] folds the pages an append touched back in,
//! * [`merge_written`] folds explicitly-written rows in.
//!
//! # Why this is Rust and not four table rows
//!
//! `families/layout.rs`' module header refused a row for every one of these
//! kernels and the refusal was CORRECT about the finding and wrong about the
//! conclusion. No [`kernels::LaunchRule`] states `dim3(num_tokens,
//! num_kv_heads)` at `min(head_dim, 256)`, and none should learn to: §10.5
//! forbids growing that vocabulary for one launcher, and these grids are
//! host arithmetic over cache extents — a page pool's size, a bound on how
//! many pages an append could have touched — that no fire's rectangle
//! contains. So the rows are [`kernels::LaunchRule::Unstated`], the geometry
//! is stated HERE beside the C++ line it came from, and the choice between
//! one kernel and two is a Rust `if` because it always was one.
//!
//! [`merge_written`] is the load-bearing case for that last point. One C++
//! symbol fired one launch or two depending on `num_tokens`, and *"one symbol,
//! two launches"* was the second half of the old refusal. A table row cannot
//! say that. A function can, and this one does, in nine lines.
//!
//! # The `Tu = 0` template parameter
//!
//! Four of the five `__global__`s carry `template <int Tu = 0>` and
//! parameterise nothing with it; it exists because a plain `__global__` in a
//! header caps that header at one includer. The JIT rows spell it
//! `device::i32(0)` — see `families::layout::ENVELOPE` for why a bare `0`
//! does not survive NVRTC's name-map pragma. Nothing in this file has to
//! know that, which is the point of it being on the row.
//!
//! # Who calls this
//!
//! [`seed_empty`] — `bind::abi::seed_envelopes_empty`, which was the tree's
//! one hand-written `ffi::pie_k_layout_launch_envelope_seed_empty_bf16` arm.
//! [`update_appended`] and [`merge_written`] — [`super::kv_paged`], from the
//! two appenders that were `kv_paged.cu:109` and `kv_paged.cu:304`. There is
//! no other caller in any language: the three stub `layout/envelope.hpp`
//! headers under `driver-cuda/tests/oracle/*/stub/` define their own `inline`
//! bodies that log rather than launch, and shadow the real header rather than
//! link against it.

use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `envelope.cu:37` and `:134`, `head_dim < 256 ? head_dim : 256`.
///
/// A block width off a cache extent, capped. Not a `LaunchRule`'s to state —
/// §21.14 refuses a rule whose block comes off a `Dims` field, because a
/// block width is the launcher's property and a fire can make no statement
/// about it true or false.
const fn threads_for(head_dim: i32) -> u32 {
    if head_dim < 256 { head_dim.unsigned_abs() } else { 256 }
}

/// `envelope.cu:71`, the seed's own block, which is fixed rather than derived.
const SEED_BLOCK: u32 = 256;

/// `envelope.cuh:374`, `kEnvelopeFuseMaxTokens`.
///
/// Above this the fused kernel is unsound, not merely slower: it resets a
/// page it is the first writer of and merges into one it is not, deciding per
/// token, and that is only race-free while the launch is small enough that no
/// two blocks reach the same page. **This is a measurement and it survives
/// the port** — the constant is the C++'s, read from the header rather than
/// re-derived here.
const FUSE_MAX_TOKENS: i32 = 128;

/// `layout::envelope_merge_written_fused_bf16`.
const FUSED_SYMBOL: &str = "layout::envelope_merge_written_fused_bf16";
/// `layout::envelope_reset_started_pages_bf16`.
const RESET_SYMBOL: &str = "layout::envelope_reset_started_pages_bf16";
/// `layout::envelope_merge_written_bf16`.
const MERGE_SYMBOL: &str = "layout::envelope_merge_written_bf16";
/// `layout::envelope_seed_empty_bf16`.
const SEED_SYMBOL: &str = "layout::envelope_seed_empty_bf16";
/// `layout::envelope_update_appended_bf16`.
const APPEND_SYMBOL: &str = "layout::envelope_update_appended_bf16";

/// Whether an envelope maintenance step ran.
///
/// `#[must_use]` for `fire/gemv.rs`' reason: *"it declined"* must not be
/// spellable the same way as *"it ran"*. Every launcher here returned `void`
/// and swallowed its own guard, so a Rust caller that ignored the answer
/// would be exactly as blind as the C++ caller was — the difference is that
/// here it has to say so.
#[must_use]
pub enum Envelope {
    /// The launches were issued on the caller's stream.
    Launched,
    /// Nothing was launched, and which extent was empty.
    Declined(Decline),
}

/// Every way a launcher in this module declines.
///
/// Each is a `return` in the C++, reproduced rather than approximated. None
/// is a fallback: an empty extent means there is no envelope work, not that
/// some other path should do it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `envelope.cu:35` — `num_tokens <= 0`.
    NoTokens,
    /// `envelope.cu:68` — `num_pages <= 0`.
    NoPages,
    /// `envelope.cu:131` — `num_requests <= 0`.
    NoRequests,
    /// `envelope.cu:131` — `max_touched <= 0`.
    NoTouchedPages,
    /// `num_kv_heads <= 0`, `head_dim <= 0` or (for the append) `page_size
    /// <= 0` — the cache-shape half of all three guards.
    ///
    /// One variant for the three because they are one condition: a layer
    /// whose heads, channels or page are empty has no envelope to maintain,
    /// and no caller can act differently on which of them was zero.
    EmptyLayer,
}

/// `layout/envelope.cu:24` — `launch_envelope_merge_written_bf16`.
///
/// Folds explicitly-written KV rows into the envelope planes. One launch or
/// two:
///
/// ```text
/// :41   if (num_tokens <= kEnvelopeFuseMaxTokens)
///           device::merge_written_fused<<<grid, threads, 0, stream>>>(...)
/// :49   else
///           device::reset_started_pages<<<grid, threads, 0, stream>>>(...)
/// :54       device::merge_written  <<<grid, threads, 0, stream>>>(...)
/// ```
///
/// with `grid = dim3(num_tokens, num_kv_heads)` (`:36`) and
/// `threads = head_dim < 256 ? head_dim : 256` (`:37`).
///
/// The two-launch arm is ORDERED: the reset writes the identity into pages
/// this batch is the first writer of, and the merge then reads what it wrote.
/// They are two launches on one stream for that reason and not for a
/// register budget, so nothing here may reorder or fuse them.
///
/// `row_valid` may be null — the kernel tests it — so a null caller means
/// *"every row is valid"* and not *"no rows are"*.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see
/// [`super::hand::fire`]. A broken JIT is not a decline.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn merge_written(
    k_curr: *const u16,
    w_page: *const u32,
    w_off: *const u32,
    row_valid: *const u8,
    env_min: *mut u16,
    env_max: *mut u16,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut std::ffi::c_void,
) -> Envelope {
    // `envelope.cu:35`, split so the caller learns which extent was empty.
    if num_tokens <= 0 {
        return Envelope::Declined(Decline::NoTokens);
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Envelope::Declined(Decline::EmptyLayer);
    }

    let launch = Launch {
        grid: [num_tokens.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        block: [threads_for(head_dim), 1, 1],
        smem: 0,
    };

    let k = ArgValue::Ptr(k_curr.cast_mut().cast());
    let page = ArgValue::Ptr(w_page.cast_mut().cast());
    let off = ArgValue::Ptr(w_off.cast_mut().cast());
    let valid = ArgValue::Ptr(row_valid.cast_mut().cast());
    let lo = ArgValue::Ptr(env_min.cast());
    let hi = ArgValue::Ptr(env_max.cast());
    let n = ArgValue::I32(num_tokens);
    let h = ArgValue::I32(num_kv_heads);
    let d = ArgValue::I32(head_dim);

    if num_tokens <= FUSE_MAX_TOKENS {
        let values = [k, page, off, valid, lo, hi, n, h, d];
        super::hand::fire(FUSED_SYMBOL, launch, &values, stream);
        return Envelope::Launched;
    }

    let reset = [page, off, valid, lo, hi, n, h, d];
    super::hand::fire(RESET_SYMBOL, launch, &reset, stream);
    // `merge_written` takes no `w_off`: the reset above consumed it, and this
    // kernel folds every written row unconditionally. Dropping it here is the
    // C++'s argument list at `:54-57`, not an omission.
    let merge = [k, page, valid, lo, hi, n, h, d];
    super::hand::fire(MERGE_SYMBOL, launch, &merge, stream);
    Envelope::Launched
}

/// `layout/envelope.cu:62` — `launch_envelope_seed_empty_bf16`.
///
/// Writes the `+inf`/`-inf` identity across a whole envelope pool, so that a
/// page no one has written yet reduces to *"nothing here"* rather than to
/// whatever the allocation held.
///
/// ```text
/// :76   device::seed_empty<<<blocks, 256, 0, stream>>>(env_min, env_max, n)
/// ```
///
/// with `n = num_pages * num_kv_heads * head_dim` in `usize` (`:69`) and
/// `blocks = (n + 255) / 256` (`:73`).
///
/// **The product is `usize` in both languages and that is load-bearing.** A
/// 64-page pool at 128 heads is nothing; a real one is `num_pages` in the
/// tens of thousands, and `num_pages * num_kv_heads * head_dim` overflows
/// `i32` before it overflows anything the kernel indexes with.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Both planes are device addresses the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
pub unsafe fn seed_empty(
    env_min: *mut u16,
    env_max: *mut u16,
    num_pages: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut std::ffi::c_void,
) -> Envelope {
    // `envelope.cu:68`.
    if num_pages <= 0 {
        return Envelope::Declined(Decline::NoPages);
    }
    if num_kv_heads <= 0 || head_dim <= 0 {
        return Envelope::Declined(Decline::EmptyLayer);
    }

    let n = usize::try_from(num_pages).unwrap_or(0)
        * usize::try_from(num_kv_heads).unwrap_or(0)
        * usize::try_from(head_dim).unwrap_or(0);
    let blocks = n.div_ceil(SEED_BLOCK as usize);

    let launch = Launch {
        grid: [u32::try_from(blocks).unwrap_or(u32::MAX), 1, 1],
        block: [SEED_BLOCK, 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(env_min.cast()),
        ArgValue::Ptr(env_max.cast()),
        ArgValue::Usize(n),
    ];
    super::hand::fire(SEED_SYMBOL, launch, &values, stream);
    Envelope::Launched
}

/// `layout/envelope.cu:115` — `launch_envelope_update_appended_bf16`.
///
/// The incremental form of the whole-cache rebuild: pages are append-only, so
/// re-reducing only the pages this append touched gives the answer a full
/// recompute gives, at the cost of the touched set instead of the cache.
/// (`launch_envelope_recompute_bf16`, the full form, was deleted by §54 for
/// exactly that reason; `device::recompute` is still in the header.)
///
/// ```text
/// :137  device::update_appended<device::bf16><<<grid, threads, 0, stream>>>(
/// ```
///
/// with `grid = dim3(max_touched, num_kv_heads)` (`:133`) and
/// `threads = head_dim < 256 ? head_dim : 256` (`:134`).
///
/// **`max_touched` is a BOUND, not a count.** The caller computes it — see
/// [`super::kv_paged::max_touched_pages`] — and blocks past a request's real
/// page span early out. Nothing measures it, which is precisely why no
/// `LaunchRule` can state this grid.
///
/// Note that `max_touched` is a launch extent and is NOT in the kernel's
/// argument list; the `__global__` takes `num_requests`, `page_size`,
/// `num_kv_heads`, `head_dim` and reads its own `blockIdx.x`.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream, held live for the same window.
#[allow(clippy::too_many_arguments)]
pub unsafe fn update_appended(
    k_pages: *const u16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    env_min: *mut u16,
    env_max: *mut u16,
    num_requests: i32,
    max_touched: i32,
    page_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
    stream: *mut std::ffi::c_void,
) -> Envelope {
    // `envelope.cu:131-134`, split into the three answers a caller can act on.
    if num_requests <= 0 {
        return Envelope::Declined(Decline::NoRequests);
    }
    if max_touched <= 0 {
        return Envelope::Declined(Decline::NoTouchedPages);
    }
    if num_kv_heads <= 0 || head_dim <= 0 || page_size <= 0 {
        return Envelope::Declined(Decline::EmptyLayer);
    }

    let launch = Launch {
        grid: [max_touched.unsigned_abs(), num_kv_heads.unsigned_abs(), 1],
        block: [threads_for(head_dim), 1, 1],
        smem: 0,
    };
    let values = [
        ArgValue::Ptr(k_pages.cast_mut().cast()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indices.cast_mut().cast()),
        ArgValue::Ptr(kv_page_indptr.cast_mut().cast()),
        ArgValue::Ptr(kv_last_page_lens.cast_mut().cast()),
        ArgValue::Ptr(env_min.cast()),
        ArgValue::Ptr(env_max.cast()),
        ArgValue::I32(num_requests),
        ArgValue::I32(page_size),
        ArgValue::I32(num_kv_heads),
        ArgValue::I32(head_dim),
    ];
    super::hand::fire(APPEND_SYMBOL, launch, &values, stream);
    Envelope::Launched
}
