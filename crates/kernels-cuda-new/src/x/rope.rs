//! `rope` — the whole family, as two truths.
//!
//! The device text is `csrc/src/rope/rope.cuh`, unchanged. This file is the
//! other truth: ten `__global__` declarations, twelve host programs, twelve
//! contracts and the binds that connect them. §5 step 3's pilot, and the
//! family that settles the idioms the rest copy.
//!
//! # What this replaces
//!
//! ```text
//!   before                                             lines
//!   driver-cuda/src/fire/rope.rs        9 host programs   783
//!   kernels-cuda-new/src/families/rope.rs  14+14 rows   1,317
//!   kernels-cuda-new/src/table/rope.rs        12 rows      262
//!   driver-cuda/src/bind/service.rs      9 wrappers      ~400
//!                                                      ------
//!                                                       2,762
//!   after
//!   kernels-cuda-new/src/x/rope.rs      12 host programs, 14 device
//!                                        rows, 12 contracts, 12 binds
//! ```
//!
//! Three of the twelve host programs are NEW. `rope_standard_table`,
//! `qk_rmsnorm_rope_bf16` and `rope_partial_bf16` were in
//! `device::JIT_DISPATCHED`: a `LaunchRule` opened their grid and a
//! generated arm bound their operands, so `rope.cu`'s launchers for them
//! went with the file rather than being ported. Their geometry is cited to
//! the rule's own doc and to the `<<<>>>` the rule was ported FROM; nothing
//! here is invented.
//!
//! # The tuning these launchers were written around
//!
//! * **32 KB caps the table at head_dim 8192**; past that the pairs are
//!   recomputed rather than cached, which is what `cache_pairs = 0` means.
//! * **One block per token leaves 147 of the B200's 148 SMs idle during
//!   decode**, where `num_tokens` is 1. Splitting the heads across
//!   `blockIdx.y` is what makes the grid grow with the head count rather
//!   than the batch, and it gives each thread exactly one element — one
//!   load/store round trip rather than a chain of them.
//! * `__sincosf` against a cached `float2` table is the context-length
//!   trade: below `kMaxCachedPairs` the pair is read from shared memory,
//!   above it the transcendental is recomputed per element.
//!
//! # The finding this file closes
//!
//! `families/rope.rs` and `fire/rope.rs` each carried one half of it:
//!
//! > `heads_per_block = half>=256?1:256/half` and `cache_pairs =
//! > half<=4096?half:0` are host conditionals and the `Source` grammar has
//! > no comparison.
//!
//! Four device rows were left entirely unsourced for it, and the near-miss
//! is why: the nearest expression the grammar had,
//! `Div(Lit(256), Div(head_dim, Lit(2)))`, agrees at every `head_dim` up to
//! 512 and returns **0** past it — MLA's 576 gives `256 / 288` — and the
//! kernel reads that value twice, as `head_base` and through
//! `heads_here = min(heads_per_block, total_heads - head_base)`, so a zero
//! makes every block in a full grid fall out of its loop and rotate
//! **nothing, silently**, on the tensor attention reads next.
//!
//! [`heads_per_block`] is that comparison, in a language that has one. The
//! finding is closed by deletion: there is no grammar left to extend.
//!
//! # The four refusals that survive, and the two that did not
//!
//! §2's illustration asked for this family because it holds rows
//! "deliberately left unsourced with a written reason". Under fn-world each
//! of those reasons became a `fn` parameter, and then the question is only
//! whether anything can FILL it:
//!
//! | symbol | was | now |
//! |---|---|---|
//! | `qk_rmsnorm_rope_bf16_devwin` | unsourced: *"a device word the driver writes between replays; no `Source` reads device memory"* | **bound** — `Cx::peel_window` is an ordinary query and `win` an ordinary parameter |
//! | `rope_partial_last_bf16` | unsourced: the host ramp and the head-count `Or` | **bound** — [`ramp_bounds`] runs on the host and `Cx::yarn` names the quartet |
//! | `rope_yarn_bf16` | unsourced: llama-3's frequency factors | still refused, now **at model load** |
//! | `qk_rmsnorm_mrope_bf16` | unsourced: the `(t, h, w)` section split | still refused, at model load |
//! | `rope_partial_bf16_position_delta` | unsourced: a draft/verify fact | still refused, at model load |
//! | `rope_write_kv_bf16` | unsourced | still refused, at model load, for a NEW and smaller reason |
//!
//! A refusal that is true of every fire belongs at load, which is §0's
//! *"every refusal the system can make is made at model load"*. `bind:
//! None` with a written reason is how a contract says so, and the reason is
//! the sentence the row carried.
//!
//! # The eight renames, and why they stay
//!
//! Eight device rows are spelled differently from the symbol a trace
//! states: `rope::rope_bf16` is fired as `rope::rotate_bf16`. The rename
//! was forced — `execution::tests::a_walk_is_only_a_walk` refuses a symbol
//! that is walked AND unit-hosted — and that force is now gone, because no
//! rope symbol is walked any more.
//!
//! They stay anyway, and it is not inertia. **A device row names a
//! `__global__` and a contract names a statement**, and those are different
//! things that happen to have been the same string while `rope.cu` existed.
//! `rotate` really is one kernel behind three statements
//! (`rope_bf16`, `rope_write_kv_bf16` at two page layouts); `rotate_partial`
//! is one kernel behind three more. Collapsing the two namespaces would
//! reintroduce exactly the confusion the rename removed.

#![allow(clippy::too_many_arguments)]

use crate::x::abi::{MaybeConst, bf16, f16};
use crate::x::contract::{Fired, Refusal};
use crate::x::launch::Launch;
use crate::{bind, contract, unit};

use core::ffi::c_void;
use core::ptr::NonNull;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
// ---------------------------------------------------------------------------

unit! {
    /// `rope`'s device text: the table builder, the four fused QK-norm
    /// rotations, the two YaRN forms, and the two partial rotations.
    unit ROPE = "rope/rope",
        text = include_str!("../../csrc/src/rope/rope.cuh"),
        file = "rope/rope.cuh";

    /// `rope.cuh:127` — the cos/sin table `attn`'s fused prepare reads.
    ///
    /// Templated over its POSITION type: `graph_pad` writes its pad lanes'
    /// positions as `u32`, so the day a caller wants the table built
    /// straight off those it costs one instantiation here and no C++
    /// anywhere.
    fn standard_table = "rope::device::standard_table" <P> (
        positions: *const P,
        table: *mut f32,
        head_dim: i32,
        theta: f32,
    ) where *const P {
        "rope::rope_standard_table" => [P = i32] "device::i32",
    }

    /// `rope.cuh:163` — the plain NeoX/GPT-J rotation, optionally fused
    /// with the write into the paged KV cache.
    ///
    /// `template <bool kWriteKv, bool kHnd>`, and **the parameter list is
    /// the same twenty at either value**. The `kWriteKv = false` site
    /// passes `nullptr` eight times and `0` twice rather than taking a
    /// shorter kernel; a declaration that stated the launcher's ten would
    /// bind ten values into a twenty-parameter `cuLaunchKernel`. The eight
    /// nulls are `Option<NonNull<_>>` and `MaybeConst<_>` here, so the
    /// absence is in the type rather than in a comment.
    ///
    /// `num_tokens` is NOT a parameter: the kernel reads `blockIdx.x`.
    fn rotate = "rope::device::rotate" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        interleaved: bool,
        cache_pairs: i32,
        heads_per_block: i32,
        v: MaybeConst<bf16>,
        k_pages: Option<NonNull<bf16>>,
        v_pages: Option<NonNull<bf16>>,
        qo_indptr: MaybeConst<u32>,
        kv_page_indices: MaybeConst<u32>,
        kv_page_indptr: MaybeConst<u32>,
        kv_last_page_lens: MaybeConst<u32>,
        row_valid: MaybeConst<u8>,
        num_requests: i32,
        page_size: i32,
    ) {
        // `device::false_type::value` and not a bare `false`:
        // `DeviceKernel::instantiation` glues
        // `::pie_cuda_driver::kernels::` onto the FIRST token, and a
        // literal cannot carry it. The second argument is a bare literal
        // because the prefix never reaches it.
        "rope::rotate_bf16" => "device::false_type::value, false",
        "rope::rope_write_kv_bf16#nhd" => "device::true_type::value, false",
        "rope::rope_write_kv_bf16#hnd" => "device::true_type::value, true",
    }

    /// `rope.cuh:321` — per-head q/k RMS norms fused with the rotation.
    fn qk_rmsnorm_rotate = "rope::device::qk_rmsnorm_rotate" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
    ) {
        "rope::qk_rmsnorm_rope_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:375` — the same, with the intermediate rounded to bf16
    /// between the norm and the rotation.
    ///
    /// gemma-4 rounds where qwen3_5 does not, and bf16 rounding is which
    /// numbers come out — so the symbol IS the statement.
    fn qk_rmsnorm_rotate_rounded = "rope::device::qk_rmsnorm_rotate_rounded" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
    ) {
        "rope::qk_rmsnorm_rotate_rounded_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:442` — MROPE, over `[num_tokens, 3]` positions.
    ///
    /// **The three section widths are the last three parameters**, and
    /// under the row world they were the reason this row was left entirely
    /// unsourced: a `(t, h, w)` split is a property of a vision checkpoint
    /// that no statement and no context carries. Here they are `s0`, `s1`
    /// and `s2` — ordinary arguments to an ordinary function, which any
    /// caller that knows them can supply.
    fn qk_rmsnorm_rotate_mrope = "rope::device::qk_rmsnorm_rotate_mrope" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
        s0: i32,
        s1: i32,
        s2: i32,
    ) {
        "rope::qk_rmsnorm_rotate_mrope_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:530` — the same fused norm+rotation over a DEVICE-RESIDENT
    /// window.
    ///
    /// **`win` is the parameter the row world could not source.** Its
    /// refusal read: *"a hooked pure-decode fire is graph CAPTURED and
    /// `win_d` is a device word the driver writes between replays; no
    /// `Source` reads device memory."* Every word of that is still true and
    /// none of it matters, because a `fn` parameter is not a `Source`: the
    /// kernel reads `win[0]`/`win[1]` at `rope.cuh:485-487` and early-outs
    /// the lanes outside, and the host's job is only to hand over the
    /// address and span every lane.
    fn qk_rmsnorm_rotate_devwin = "rope::device::qk_rmsnorm_rotate_devwin" (
        q: *mut bf16,
        k: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        win: *const u32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        eps: f32,
    ) {
        "rope::qk_rmsnorm_rotate_devwin_bf16" => "device::i32(128)",
    }

    /// `rope.cuh:610` — llama-3-style YaRN.
    ///
    /// The ramp's two bounds are computed **in the kernel** from
    /// `low_freq_factor` and `high_freq_factor`. That is what distinguishes
    /// it from [`rotate_yarn_original`](raw::rotate_yarn_original), whose
    /// bounds are computed on the host.
    ///
    /// `orig_max_pos` IS A FLOAT, where the trace-facing declaration says
    /// `i32`: the kernel's parameter is `float` (`rope.cuh:598`) because
    /// `yarn_freq` divides by it, and the launcher casts at `rope.cu:249`.
    /// The kernel is authoritative for a `void**`, and an `i32` in this
    /// slot would hand four bytes of integer to a float parameter. In
    /// fn-world the cast is at the one call site and the compiler checks
    /// it.
    ///
    /// A plain `__global__`, so `DeviceKernel::PLAIN`. That used to be half
    /// of this kernel's refusal and is not a limit:
    /// `nvrtcAddNameExpression` takes the bare qualified path and
    /// `cuModuleGetFunction` resolves it.
    fn rotate_yarn = "rope::device::rotate_yarn" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        orig_max_pos: f32,
        heads_per_block: i32,
    ) {
        "rope::rotate_yarn_bf16" => crate::device::PLAIN,
    }

    /// `rope.cuh:656` — YaRN as its paper spells it (OLMo-3, gpt-oss).
    ///
    /// `low_dim`/`high_dim` are the HOST ramp's output and `mscale` is the
    /// attention temperature. The trace-facing declaration names
    /// `beta_fast`, `beta_slow`, `attention_factor` and
    /// `original_max_position`; **two of those four are consumed by
    /// [`ramp_bounds`] and never cross**, and the third crosses under the
    /// kernel's name for it. A declaration derived from the trace-facing
    /// one by dropping the stream would have been wrong in four places at
    /// once, which is the whole reason the KERNEL is read first.
    fn rotate_yarn_original = "rope::device::rotate_yarn_original" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        theta: f32,
        factor: f32,
        low_dim: f32,
        high_dim: f32,
        mscale: f32,
        interleaved: bool,
        heads_per_block: i32,
        cache_pairs: i32,
    ) {
        "rope::rotate_yarn_original_bf16" => crate::device::PLAIN,
    }

    /// `rope.cuh:733` — partial rotary over the FIRST `rotary_dim` lanes.
    ///
    /// Templated over its element type, and the f16 instantiation is the
    /// row the ahead-of-time build could not afford: under nvcc a second
    /// numeric format cost a translation unit's worth of `cicc` for
    /// something no caller had asked for, which is why `rope.cu` named
    /// every kernel `_bf16` and meant "the one instantiation we could pay
    /// for". `rotate_partial` converts through `Elem<T>` and never touches
    /// a `__nv_bfloat16` intrinsic, so fp16 is the same rounding at a
    /// different exponent width — which is why THIS template could be
    /// widened and the six that call `rope_device.cuh`'s `rotate_pair`
    /// could not: that header takes `bf16*`, and it is shared and
    /// read-only.
    ///
    /// `position_delta` sits between `positions` and the extents because
    /// that is where the kernel's signature puts it — exactly the kind of
    /// fact a hand-written binding gets wrong.
    fn rotate_partial = "rope::device::rotate_partial" <T> (
        q: *mut T,
        k: *mut T,
        positions: *const i32,
        position_delta: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
    ) where *mut T {
        "rope::rope_partial_bf16" => [T = bf16] "device::bf16",
        "rope::rotate_partial_position_delta_bf16" => [T = bf16] "device::bf16",
        "rope::rope_partial_f16" => [T = f16] "device::f16",
    }

    /// `rope.cuh:792` — partial rotary over the LAST `rotary_dim` lanes
    /// (deepseek-v4), optionally inverted and optionally YaRN-scaled.
    ///
    /// Which end of the channel axis carries position is a property of the
    /// checkpoint, which is why this is a different kernel and not a flag
    /// on [`rotate_partial`](raw::rotate_partial).
    fn rotate_partial_last = "rope::device::rotate_partial_last" (
        q: *mut bf16,
        k: *mut bf16,
        positions: *const i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        theta: f32,
        inverse: bool,
        interleaved: bool,
        yarn_factor: f32,
        yarn_low_dim: f32,
        yarn_high_dim: f32,
    ) {
        "rope::rotate_partial_last_bf16" => crate::device::PLAIN,
    }
}

// ---------------------------------------------------------------------------
// The measurements. §3.1: the war stories move to the constructors and fns
// that carry them, not to a rule's doc comment.
// ---------------------------------------------------------------------------

/// `rope.cu:82,119,236,276,314,337,382` — `constexpr int BLOCK = 256;`
///
/// The rotation block width, and the divisor in [`heads_per_block`]. Every
/// launcher in this file that splits heads across `blockIdx.y` uses it for
/// both, which is what makes the two uses one constant.
pub const ROTATE_BLOCK: i32 = 256;

/// `rope.cu:45,66,162,189,213` — `constexpr int BLOCK = 128;`
///
/// The fused QK-RMSNorm-plus-rotate width. A different number for a
/// different kernel: these launch one block per (token, head) and stride the
/// head, so the block is the head's width and not the row's.
pub const FUSED_BLOCK: u32 = 128;

/// `rope.cu:84,120,282` — `constexpr int kMaxCachedPairs = 4096;`
///
/// 32 KB of `float2`. Past this the pairs are recomputed per element instead
/// of being staged in shared memory — the `__sincosf`/context-length trade,
/// and the reason a `head_dim` past 8192 costs transcendentals rather than
/// an out-of-bounds shared read.
pub const MAX_CACHED_PAIRS: i32 = 4096;

/// `rope.cu:92,127,240,287` — `half >= BLOCK ? 1 : (BLOCK / half)`.
///
/// **The comparison the `Source` grammar does not have.** Its answer is both
/// a kernel argument and half the grid: `grid.y` is
/// `ceil(total_heads / heads_per_block)`.
///
/// The near-miss that made this a refusal rather than a rule is worth
/// keeping next to the arithmetic. `Div(Lit(256), Div(head_dim, Lit(2)))`
/// agrees with this function at every `head_dim` up to 512 and returns 0
/// past it — MLA's 576 gives `256 / 288` — and a zero here makes every
/// block in a full grid fall out of its loop and rotate nothing, silently.
#[must_use]
pub const fn heads_per_block(half: i32) -> i32 {
    if half >= ROTATE_BLOCK { 1 } else { ROTATE_BLOCK / half }
}

/// `rope.cu:87,123,285` — `half <= kMaxCachedPairs ? half : 0`.
///
/// The second comparison, and the one that also sizes a dynamic shared
/// allocation. **Zero means "recompute", not "none available".**
///
/// `Lit(0)` would have been safe and bit-identical — the kernel recomputes
/// each pair when the cache is empty — but it is not what the launcher
/// computes; `half` would have been an overrun past `head_dim` 8192.
/// Neither spelling was the launcher's, which is why the row refused rather
/// than approximated.
#[must_use]
pub const fn cache_pairs(half: i32) -> i32 {
    if half <= MAX_CACHED_PAIRS { half } else { 0 }
}

/// `rope.cu:93,128,241,288` — the two-axis grid the head split produces.
///
/// `dim3 grid(num_tokens, (total_heads + heads_per_block - 1) / heads_per_block)`.
///
/// A [`Launch`] literal and not a [`Launch::flat`] or [`Launch::per_row`]:
/// this shape fits neither convenience, and §3.1's whole point is that a
/// kernel which fits none writes the literal rather than waiting for a
/// forty-first variant.
#[allow(dead_code)]
#[must_use]
const fn rotate_grid(num_tokens: i32, total_heads: i32, per_block: i32) -> [u32; 3] {
    [
        num_tokens.unsigned_abs(),
        (total_heads + per_block - 1).unsigned_abs() / per_block.unsigned_abs(),
        1,
    ]
}

/// `rope.cu:189-191`, `:45-47`, `:162-164`, `:213-215` — the fused grid.
///
/// `dim3 grid(num_tokens, num_q_heads + num_kv_heads)` at `BLOCK = 128`, on
/// all four fused QK-norm rotations. One block per (token, head), the block
/// striding the head.
#[allow(dead_code)]
#[must_use]
const fn fused_launch(rows: i32, total_heads: i32) -> Launch {
    Launch {
        grid: [rows.unsigned_abs(), total_heads.unsigned_abs(), 1],
        block: [FUSED_BLOCK, 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `rope_device.cuh:112` — `yarn_original_ramp_bounds`, on the host.
///
/// `correction_dim(rot) = span * ln(max_pos / (rot * 2π)) / (2 * ln(theta))`.
///
/// `beta_slow` → "low rotation count" → larger `correction_dim` → the UPPER
/// bound of the ramp (above it, fully interpolated). `beta_fast` → smaller
/// `correction_dim` → the LOWER bound (below it, fully extrapolated). HF
/// clamps to `[0, span/2 - 1]` and so does this.
///
/// The C++ is `__host__ __device__` and shared between its launcher and a
/// fused kernel *"so a fused kernel and `kernels::rope::rope_yarn_original_bf16`
/// cannot disagree about where the ramp starts, which would silently change
/// every position > 0"*. That sharing is now across a language boundary, so
/// the arithmetic is restated here in the order the C++ evaluates it —
/// `logf`, then `floorf`/`ceilf`, then three clamps — and `f32::ln` is the
/// same single-precision operation `logf` is.
///
/// **The span is a parameter for one reason**: [`rope_yarn_original_bf16`]
/// runs it over the full `head_dim` and [`rope_partial_last_bf16`] over
/// `rotary_dim`, the rotated slice, and that difference is the only one
/// between the two ramps. One function with a parameter cannot disagree
/// about anything else; two copies could.
#[must_use]
pub fn ramp_bounds(
    span: i32,
    theta: f32,
    beta_fast: f32,
    beta_slow: f32,
    original_max_position: i32,
) -> (f32, f32) {
    const TWO_PI: f32 = 6.283_185_307_179_586_5_f32;
    let ln_theta = theta.ln();
    #[allow(clippy::cast_precision_loss)]
    let corr_dim = |rot: f32| -> f32 {
        span as f32 * (original_max_position as f32 / (rot * TWO_PI)).ln() / (2.0 * ln_theta)
    };
    let mut low_dim = corr_dim(beta_fast).floor();
    let mut high_dim = corr_dim(beta_slow).ceil();
    if low_dim < 0.0 {
        low_dim = 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let max_pair = (span / 2) as f32 - 1.0;
    if high_dim > max_pair {
        high_dim = max_pair;
    }
    if high_dim < low_dim {
        high_dim = low_dim;
    }
    (low_dim, high_dim)
}

// ---------------------------------------------------------------------------
// Truth two: the host programs. One `fn` per launcher, each returning
// `Fired` so that "it declined" cannot be spelled like "it ran".
// ---------------------------------------------------------------------------

/// `rope::rope_standard_table` — the cos/sin table `attn`'s fused prepare
/// reads.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// This row was in `device::JIT_DISPATCHED` with
/// `LaunchRule::RouteRows`, so its ahead-of-time launcher went with
/// `rope/rope.cu` and there was nothing to port. `families/rope.rs`'s row
/// states the geometry and the two spellings of its block, verbatim:
///
/// > One block per token, the block striding `head_dim/2` pairs —
/// > `RouteRows` sizes it `min(1024, ceil(width/32)*32)` **where the
/// > launcher fixed 256**, so the wider block reaches the same pairs in
/// > fewer iterations and the arithmetic per pair is unchanged.
/// > `powf`/`__sincosf` are the same instructions either way.
///
/// **256 is taken, because it is the launcher's.** The rule's width is the
/// alternative the row measured equal, not a second fact, and a port that
/// picked the rule's would be choosing between two cited numbers on no
/// evidence. Recorded here so that a later reader with a profile can change
/// it knowing both numbers were true.
///
/// # Safety
///
/// `positions` must address `num_tokens` live `i32`s and `table`
/// `num_tokens * head_dim` live floats; `stream` must be live across the
/// launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_standard_table(
    positions: *const i32,
    table: *mut f32,
    num_tokens: i32,
    head_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // The kernel strides `head_dim / 2` pairs and a zero-pair table is no
    // table. `rope.cuh:127`'s own first line.
    if head_dim / 2 <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    unsafe {
        raw::standard_table(
            "rope::rope_standard_table",
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            positions,
            table,
            head_dim,
            theta,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:71` — `rope::rope_bf16`.
///
/// The plain NeoX/GPT-J rotation of q and k where they lie. One launch, and
/// every interesting quantity in it is computed here: [`cache_pairs`] sizes
/// the dynamic shared allocation AND is passed as an operand,
/// [`heads_per_block`] is passed as an operand AND divides the head axis of
/// the grid.
///
/// # Safety
///
/// `q` and `k` must address `num_tokens * num_q_heads * head_dim` and
/// `num_tokens * num_kv_heads * head_dim` live bf16 elements, `positions`
/// `num_tokens` live `i32`s, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    interleaved: bool,
    stream: *mut c_void,
) -> Fired {
    // `rope.cu:85-86` — `const int half = head_dim / 2; if (half <= 0) return;`
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // `rope.cu:87-88`:
    //   const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    //   const usize smem = cache_pairs * 2 * sizeof(float);
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    // `rope.cu:91-94`.
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    // `rope.cu:95-102` —
    // `device::rotate<false, false><<<grid, block, smem, stream>>>(...)`.
    unsafe {
        raw::rotate(
            "rope::rotate_bf16",
            Launch {
                grid: rotate_grid(num_tokens, total_heads, per_block),
                block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
                smem,
                smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
            },
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            interleaved,
            pairs,
            per_block,
            // `rope.cu:101-102` — eight nulls and two zeros: this
            // instantiation is `kWriteKv = false`, so the paged-cache half of
            // the kernel's parameter list is never read. Under the row world
            // these were eight `ArgValue::Ptr(null_mut())` and a reader had
            // to count them against a `void**`; here each absence is the
            // type of the parameter it fills.
            MaybeConst::none(),
            None,
            None,
            MaybeConst::none(),
            MaybeConst::none(),
            MaybeConst::none(),
            MaybeConst::none(),
            MaybeConst::none(),
            0,
            0,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:105` — `rope::rope_write_kv_bf16`.
///
/// The same rotation, fused with the write into the paged KV cache. It is
/// [`rope_bf16`]'s grid, block and shared allocation exactly; what differs
/// is `kWriteKv = true` and eight more operands.
///
/// # Two instantiations behind one symbol, and what that used to cost
///
/// `rope.cu:130-145` closed over a `dim3` and a stream in a generic lambda
/// and called it with `device::true_type{}` or `device::false_type{}`:
///
/// ```text
/// auto launch = [&](auto hnd) {
///     device::rotate<true, decltype(hnd)::value><<<grid, block, smem, stream>>>(...);
/// };
/// if (hnd_layout) launch(device::true_type{}); else launch(device::false_type{});
/// ```
///
/// A run-time `bool` selecting a compile-time template argument is exactly
/// what a `Specialisation` could not state — its `Term`s read operand VALUES
/// and alignments, and this reads a flag saying which of two page layouts
/// the cache was built with. So the two instantiations are two rows,
/// `rope::rope_write_kv_bf16#nhd` and `#hnd`.
///
/// **In fn-world it is an `if`, and that is the whole of the port.** The
/// branch is four lines below. `Specialisation` was a small language for
/// writing `if` in data; a `fn` already has one.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the paged-KV
/// descriptors describe, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_write_kv_bf16(
    q: *mut bf16,
    k: *mut bf16,
    v: *const bf16,
    positions: *const i32,
    k_pages: *mut bf16,
    v_pages: *mut bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    num_tokens: i32,
    num_requests: i32,
    page_size: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    hnd_layout: bool,
    interleaved: bool,
    stream: *mut c_void,
) -> Fired {
    // `rope.cu:121-122` — `if (half <= 0 || num_tokens <= 0) return;`
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // `rope.cu:123-129` — identical to `rope_bf16`'s.
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 2 * 4;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    let launch = Launch {
        grid: rotate_grid(num_tokens, total_heads, per_block),
        block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
        smem,
        smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
    };
    // `rope.cu:144-145` — `if (hnd_layout) launch(true_type{}); else launch(false_type{});`
    let symbol = if hnd_layout {
        "rope::rope_write_kv_bf16#hnd"
    } else {
        "rope::rope_write_kv_bf16#nhd"
    };
    unsafe {
        raw::rotate(
            symbol,
            launch,
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            interleaved,
            pairs,
            per_block,
            MaybeConst::new(v),
            NonNull::new(k_pages),
            NonNull::new(v_pages),
            MaybeConst::new(qo_indptr),
            MaybeConst::new(kv_page_indices),
            MaybeConst::new(kv_page_indptr),
            MaybeConst::new(kv_last_page_lens),
            // `table/rope.rs` marks this operand `U8s | null` and every
            // other one not: a null means every row is live.
            MaybeConst::new(row_valid),
            num_requests,
            page_size,
            stream,
        );
    }
    Fired::Launched
}

/// `rope/rope.cu:189-191` — `rope::qk_rmsnorm_rope_bf16`.
///
/// Per-head q and k RMS norms fused with the rotation.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// This row was in `device::JIT_DISPATCHED` under
/// `LaunchRule::RowsPackedHeadsNarrow`, so its launcher went with
/// `rope/rope.cu`. `families/rope.rs` quotes the three lines it opened:
///
/// ```text
/// constexpr int BLOCK = 128;
/// dim3 grid(num_tokens, num_q_heads + num_kv_heads);
/// device::qk_rmsnorm_rotate<BLOCK><<<grid, BLOCK, 0, stream>>>(
/// ```
///
/// `num_tokens` is not an argument: the kernel reads `blockIdx.x`, and the
/// row's own note says an operand restating the grid is an operand that can
/// disagree with it.
///
/// # Safety
///
/// [`rope_bf16`]'s, plus `q_weight`/`k_weight` addressing `head_dim` live
/// bf16 elements each.
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_rope_bf16(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    unsafe {
        raw::qk_rmsnorm_rotate(
            "rope::qk_rmsnorm_rope_bf16",
            fused_launch(num_tokens, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:148` — `rope::qk_rmsnorm_rope_bf16_devwin`.
///
/// The fused QK-RMSNorm-plus-rotate with a DEVICE-RESIDENT window count:
/// `win` is read by the kernel at `rope.cuh:485-487`, and `n_max` is the
/// host's upper bound on it, which is why the grid is built from `n_max` and
/// the kernel returns early for rows past the real count.
///
/// # The refusal this fn ends
///
/// `families/rope.rs` and `table/rope.rs` both left this row entirely
/// unsourced, in one sentence:
///
/// > A hooked pure-decode fire is graph CAPTURED and `win_d` is a device
/// > word the driver writes between replays; no `Source` reads device
/// > memory, and one invented to make this row look bound would be a guess
/// > in the one place the design has none.
///
/// Every clause of that is still true. It stops MATTERING because `win` is a
/// parameter of a function rather than a cell of a table: a `fn` does not
/// read device memory to pass an address on, and `Cx::peel_window` is the
/// driver handing over the address it already owns. The row could not be
/// bound; the fn is bound below, and `qk_rmsnorm_rope_devwin` becomes
/// fireable for the first time.
///
/// **`n_max` is `Rows::total` and not `Rows::count`**, and that is the whole
/// of the claim: the grid must span every lane whatever the window turns out
/// to be, because the early-out is the kernel's. The ahead-of-time twin is
/// `whole = true` for the same reason read from the other end —
/// `model-compiler`'s `lower.rs:1064` refuses a `whole` statement any window
/// but `[0, rows)`.
///
/// # Safety
///
/// `win` must address two live `u32`s on the device; the rest is
/// [`rope_bf16`]'s obligation with `n_max` for `num_tokens`.
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_rope_bf16_devwin(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    win: *const u32,
    n_max: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    // `rope.cu:161` — `if (n_max <= 0) return;`
    if n_max <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n_max" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // `rope.cu:162-170`:
    //   constexpr int BLOCK = 128;
    //   dim3 grid(n_max, num_q_heads + num_kv_heads);
    //   device::qk_rmsnorm_rotate_devwin<BLOCK><<<grid, BLOCK, 0, stream>>>(...)
    unsafe {
        raw::qk_rmsnorm_rotate_devwin(
            "rope::qk_rmsnorm_rotate_devwin_bf16",
            fused_launch(n_max, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            win,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:29` — `rope::qk_rmsnorm_mrope_bf16`.
///
/// Qwen-VL's multimodal rope: three position sections (temporal, height,
/// width) whose widths are operands, so one position vector drives three
/// different frequency schedules over disjoint slices of the head.
///
/// # The refusal this fn ends, and the one it does not
///
/// The row was unsourced because *"the section split is a property of a
/// vision checkpoint that no statement and no context carries yet, and
/// §10.5 refuses an invented one; the head counts could be sourced as above
/// but a half-bound row is a row whose unbound cells look like an
/// oversight rather than a fact."*
///
/// Half of that dissolves: `section_t`/`section_h`/`section_w` are three
/// ordinary parameters, and a caller that knows them — a vision front-end,
/// a test, a benchmark — can call this today. **The other half does not**:
/// no trace statement carries the split, so [`ENTRIES`] gives this contract
/// no bind and the refusal moves to model load with that sentence attached.
/// A `fn` with no trace-side caller is still a `fn`; §1's ladder says so.
///
/// # Safety
///
/// [`qk_rmsnorm_rope_bf16_devwin`]'s, without `win`, and `positions` must
/// address `num_tokens * 3` live `i32`s rather than `num_tokens`.
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_mrope_bf16(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    mrope_section_t: i32,
    mrope_section_h: i32,
    mrope_section_w: i32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    // `rope.cu:44` — `if (num_tokens <= 0 || num_q_heads + num_kv_heads <= 0) return;`
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // `rope.cu:45-54`:
    //   constexpr int BLOCK = 128;
    //   dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    //   device::qk_rmsnorm_rotate_mrope<BLOCK><<<grid, BLOCK, 0, stream>>>(...)
    unsafe {
        raw::qk_rmsnorm_rotate_mrope(
            "rope::qk_rmsnorm_rotate_mrope_bf16",
            fused_launch(num_tokens, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            mrope_section_t,
            mrope_section_h,
            mrope_section_w,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:200` — `rope::qk_rmsnorm_rope_bf16_rounded`.
///
/// gemma-4's fused QK norm: the same shape as [`qk_rmsnorm_rope_bf16`] with
/// the intermediate rounded to bf16 between the norm and the rotation, which
/// is what the reference implementation does and what the golden was taken
/// over.
///
/// `families/rope.rs` states this row's rule as `LaunchRule::Unstated` and
/// says it is *"launchable only by a caller that builds a `Launch` by hand
/// at a site that cites the `.cu` line beside the numbers"*. This is that
/// site — and in fn-world every launcher is that site, which is why
/// `Unstated` stops being a category.
///
/// # Safety
///
/// [`qk_rmsnorm_mrope_bf16`]'s. `k` and `k_weight` may be null together, and
/// the kernel reads the pair as "there is no k".
#[cfg(feature = "_cuda")]
pub unsafe fn qk_rmsnorm_rope_bf16_rounded(
    q: *mut bf16,
    k: *mut bf16,
    q_weight: *const bf16,
    k_weight: *const bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    let total_heads = num_q_heads + num_kv_heads;
    // `rope.cu:212` — `if (num_tokens <= 0 || num_q_heads + num_kv_heads <= 0) return;`
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if total_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads + num_kv_heads" });
    }
    // `rope.cu:213-221`:
    //   constexpr int BLOCK = 128;
    //   dim3 grid(num_tokens, num_q_heads + num_kv_heads);
    //   device::qk_rmsnorm_rotate_rounded<BLOCK><<<grid, BLOCK, 0, stream>>>(...)
    unsafe {
        raw::qk_rmsnorm_rotate_rounded(
            "rope::qk_rmsnorm_rotate_rounded_bf16",
            fused_launch(num_tokens, total_heads),
            q,
            k,
            q_weight,
            k_weight,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:226` — `rope::rope_yarn_bf16`.
///
/// Llama-3-style YaRN: the ramp is a per-dimension interpolation between the
/// extrapolated and interpolated frequency, and its two bounds are computed
/// **in the kernel** from `low_freq_factor` and `high_freq_factor`. That is
/// what distinguishes it from [`rope_yarn_original_bf16`], whose bounds are
/// computed on the host by [`ramp_bounds`].
///
/// No shared allocation: this kernel recomputes the pair per element. The
/// row world's `LaunchRule::Rope` sized one anyway — see
/// [`crate::x::launch`]'s header for the whole of that story, which is this
/// family's, and for why `smem` being written by the same `fn` that reads it
/// is the structural fix rather than a caught bug.
///
/// # Why this fn has no bind
///
/// `low_freq_factor` and `high_freq_factor` are llama-3's scheme, and
/// nothing carries them. `Cx::yarn` answers the ORIGINAL YaRN quartet
/// (`factor`, `beta_fast`, `beta_slow`, `attention_factor`) and reading two
/// of those into these two would produce a model that runs and is wrong past
/// its training length. Two of a kind is not a carrier.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_yarn_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    factor: f32,
    low_freq_factor: f32,
    high_freq_factor: f32,
    original_max_position: i32,
    stream: *mut c_void,
) -> Fired {
    // `rope.cu:237-238` — `const int half = head_dim / 2; if (half <= 0) return;`
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // `rope.cu:239-242`.
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    // `rope.cu:243-250` — `device::rotate_yarn<<<grid, BLOCK, 0, stream>>>(...)`.
    // `original_max_position` crosses as a `float`: the kernel takes
    // `float orig_max_pos` and `:249` is `static_cast<float>(...)`. Under the
    // row world the row said `I32` and the launcher cast, and only the
    // launcher's cast made it right; here the declaration says `f32` and this
    // is the one cast, checked.
    #[allow(clippy::cast_precision_loss)]
    let orig_max_pos = original_max_position as f32;
    unsafe {
        raw::rotate_yarn(
            "rope::rotate_yarn_bf16",
            Launch {
                grid: rotate_grid(num_tokens, total_heads, per_block),
                block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            factor,
            low_freq_factor,
            high_freq_factor,
            orig_max_pos,
            per_block,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:255` — `rope::rope_yarn_original_bf16` (OLMo-3, gpt-oss).
///
/// The ramp bounds are computed HERE, before the launch, and crossed as two
/// `float` operands — which is the second reason (after [`heads_per_block`])
/// that `families/rope.rs` left this row unsourced: *"`low_dim` and
/// `high_dim`, which `yarn_original_ramp_bounds` computes"*.
///
/// The correction range is over the full `head_dim`;
/// [`rope_partial_last_bf16`] runs the same arithmetic over `rotary_dim`
/// instead, and that difference is the only one between the two ramps.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_yarn_original_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: i32,
    interleaved: bool,
    stream: *mut c_void,
) -> Fired {
    // `rope.cu:272-274` — the ramp, before anything else.
    let (low_dim, high_dim) =
        ramp_bounds(head_dim, theta, beta_fast, beta_slow, original_max_position);
    // `rope.cu:283-284` — `const int half = head_dim / 2; if (half <= 0) return;`
    let half = head_dim / 2;
    if half <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim / 2" });
    }
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    // `rope.cu:285-291`:
    //   const int cache_pairs = half <= kMaxCachedPairs ? half : 0;
    //   const usize shared = cache_pairs * sizeof(float2);
    //
    // `sizeof(float2)` and not `2 * sizeof(float)`: the same eight bytes,
    // written the way the kernel's `extern __shared__ float2` reads them.
    let pairs = cache_pairs(half);
    let smem = pairs.unsigned_abs() * 8;
    let total_heads = num_q_heads + num_kv_heads;
    let per_block = heads_per_block(half);
    // `rope.cu:292-298` —
    // `device::rotate_yarn_original<<<grid, BLOCK, shared, stream>>>(...)`.
    unsafe {
        raw::rotate_yarn_original(
            "rope::rotate_yarn_original_bf16",
            Launch {
                grid: rotate_grid(num_tokens, total_heads, per_block),
                block: [ROTATE_BLOCK.unsigned_abs(), 1, 1],
                smem,
                smem_opt_in: smem > crate::x::launch::OPT_IN_ABOVE,
            },
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            theta,
            factor,
            low_dim,
            high_dim,
            attention_factor,
            interleaved,
            per_block,
            pairs,
            stream,
        );
    }
    Fired::Launched
}

/// `rope::rope_partial_bf16` — partial rotary over the first `rotary_dim`
/// lanes, with a host-supplied `position_delta`.
///
/// # One fn where the archive had two `__global__`s
///
/// `rope.cu` shipped `rope_partial_bf16` and
/// `rope_partial_bf16_position_delta` as two kernels that differed by `+ 0`.
/// `rotate_partial<T>` is one kernel with a delta, which is the same
/// instruction count. Two SYMBOLS survive because two statements do —
/// `rope_partial_q_only` binds the delta to zero and
/// `rope_partial_position_delta` cannot bind it at all — and the symbol is
/// this fn's first parameter.
///
/// # This launcher is NEW for `rope::rope_partial_bf16`
///
/// That symbol was in `device::JIT_DISPATCHED` under
/// `LaunchRule::RouteRows`; only the position-delta form kept a launcher.
/// The geometry is that launcher's, `rope.cu:337-345`, and the two symbols
/// are the same `__global__` at the same extents:
///
/// ```text
/// constexpr int BLOCK = 256;
/// dim3 grid(num_tokens); dim3 block(BLOCK);
/// device::rotate_partial<device::bf16><<<grid, block, 0, stream>>>(
/// ```
///
/// One block per token, no head split: `rotate_partial` strides the whole
/// `[heads, rotary_dim]` slab from a single 256-thread block.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_partial<T>(
    symbol: &'static str,
    q: *mut T,
    k: *mut T,
    positions: *const i32,
    position_delta: i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Fired
where
    *mut T: crate::x::Abi,
{
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if rotary_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rotary_dim" });
    }
    unsafe {
        raw::rotate_partial(
            symbol,
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            q,
            k,
            positions,
            position_delta,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            theta,
            stream,
        );
    }
    Fired::Launched
}

/// `rope.cu:348` — `rope::rope_partial_last_bf16` (deepseek-v4).
///
/// Partial rotary over the LAST `rotary_dim` lanes of the head, optionally
/// inverted (the decode path un-rotates a cached key before re-rotating it
/// at a new position) and optionally YaRN-scaled.
///
/// # The ramp, and why it is [`ramp_bounds`] and not a second copy
///
/// `rope.cu:365-381` runs the same arithmetic as `yarn_original_ramp_bounds`
/// with one substitution — the correction range is over `rotary_dim`, the
/// rotated slice, and not the full `head_dim` — and it runs it only when
/// `yarn_factor > 1 && yarn_original_max_position > 0`, leaving both bounds
/// at zero otherwise. Both of those are stated here; [`ramp_bounds`] takes
/// the span as a parameter precisely so the two callers cannot disagree
/// about anything else.
///
/// # Safety
///
/// [`rope_bf16`]'s.
#[cfg(feature = "_cuda")]
pub unsafe fn rope_partial_last_bf16(
    q: *mut bf16,
    k: *mut bf16,
    positions: *const i32,
    num_tokens: i32,
    num_q_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    rotary_dim: i32,
    theta: f32,
    inverse: bool,
    interleaved: bool,
    yarn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_original_max_position: i32,
    stream: *mut c_void,
) -> Fired {
    // `rope.cu:367-381` — `float low_dim = 0.f, high_dim = 0.f;` and the
    // guarded ramp. Unscaled YaRN leaves both at zero, which the kernel reads
    // as "no ramp".
    let (low_dim, high_dim) = if yarn_factor > 1.0 && yarn_original_max_position > 0 {
        ramp_bounds(
            rotary_dim,
            theta,
            yarn_beta_fast,
            yarn_beta_slow,
            yarn_original_max_position,
        )
    } else {
        (0.0, 0.0)
    };
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if rotary_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rotary_dim" });
    }
    // `rope.cu:382-390`:
    //   constexpr int BLOCK = 256;
    //   dim3 grid(num_tokens); dim3 block(BLOCK);
    //   device::rotate_partial_last<<<grid, block, 0, stream>>>(...)
    unsafe {
        raw::rotate_partial_last(
            "rope::rotate_partial_last_bf16",
            Launch::per_row(num_tokens.unsigned_abs(), ROTATE_BLOCK.unsigned_abs()),
            q,
            k,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            theta,
            inverse,
            interleaved,
            yarn_factor,
            low_dim,
            high_dim,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// The declaration the readers that cannot call read.
//
// Twelve contracts, one per statement, carrying `table/rope.rs`'s twelve rows
// minus everything that described a launcher. `whole`, `in_place` and `sink`
// survive because `model-compiler` reads them; `operands`, `launch` and
// `file` do not, because they are this file's `fn`s.
// ---------------------------------------------------------------------------

contract! {
    /// The cos/sin table `attn`'s fused prepare reads.
    ROPE_STANDARD_TABLE = "rope::rope_standard_table" as rope_standard_table

    /// The plain rotation. `interleaved` is where GLM and the MLA rope dims
    /// differ from Llama/Qwen — a load-time checkpoint fact reaching the
    /// kernel as an argument rather than as a second symbol, which is the one
    /// place this family does it that way.
    ///
    /// Rotates q and k WHERE THEY LIE — `BufMut` on both, and no destination
    /// to give them another. A host that assigns addresses reads the pair
    /// list off this contract.
    ROPE_BF16 = "rope::rope_bf16" as rope {
        in_place: &[(0, 0), (1, 1)],
    }

    /// Norms AND rotates q and k where they lie. `llama_like` states it 84
    /// times per decode text.
    QK_RMSNORM_ROPE_BF16 = "rope::qk_rmsnorm_rope_bf16" as qk_rmsnorm_rope {
        in_place: &[(0, 0), (1, 1)],
    }

    /// The device-window form. A hooked pure-decode fire is graph-CAPTURED
    /// and its hook split rides a DEVICE word, not a host row range — so it
    /// is `whole` for a reason no other `whole` row gives: the window is not
    /// a number the lowering knows, so it cannot be a rectangle at all.
    QK_RMSNORM_ROPE_BF16_DEVWIN = "rope::qk_rmsnorm_rope_bf16_devwin" as qk_rmsnorm_rope_devwin {
        whole: true,
        in_place: &[(0, 0), (1, 1)],
    }

    /// Llama-3-style YaRN. Which of the two YaRN schemes a checkpoint wants
    /// is a load-time fact, so they are two symbols and not one with a flag.
    ROPE_YARN_BF16 = "rope::rope_yarn_bf16" as rope_yarn

    /// MROPE takes `[num_tokens, 3]` positions — a (t, h, w) triple, because
    /// a vision model's tokens sit in a grid. Not the plain
    /// `qk_rmsnorm_rope` with a different theta.
    QK_RMSNORM_MROPE_BF16 = "rope::qk_rmsnorm_mrope_bf16" as qk_rmsnorm_mrope

    /// Ropes the LAST `rotary_dim` channels rather than the first. A
    /// different statement from `rope_partial_q_only`, not a flag on it:
    /// which end of the channel axis carries position is a property of the
    /// checkpoint.
    ROPE_PARTIAL_LAST_BF16 = "rope::rope_partial_last_bf16" as rope_partial_last

    /// Q-only rotation: a KV-shared layer's K was rotated at its source
    /// layer, so one operand is the whole statement. A q-only site states one
    /// result and the second pair falls outside its arity, which `Buffers`
    /// skips.
    ROPE_PARTIAL_BF16 = "rope::rope_partial_bf16" as rope_partial_q_only {
        in_place: &[(0, 0), (1, 1)],
    }

    /// `rope_partial` with `positions` shifted by a host constant, for a
    /// speculative window whose absolute positions are the verify pass's.
    ROPE_PARTIAL_BF16_POSITION_DELTA =
        "rope::rope_partial_bf16_position_delta" as rope_partial_position_delta

    /// gemma-4 rounds where qwen3_5 does not, and bf16 rounding is which
    /// numbers come out — so the symbol IS the statement.
    QK_RMSNORM_ROPE_BF16_ROUNDED = "rope::qk_rmsnorm_rope_bf16_rounded" as qk_rmsnorm_rope_rounded {
        in_place: &[(0, 0), (1, 1)],
    }

    /// YaRN, as its paper spells it. A deployment's scaling is a load-time
    /// config answer, so it picks a symbol here rather than an argument.
    ROPE_YARN_ORIGINAL_BF16 = "rope::rope_yarn_original_bf16" as rope_yarn_original {
        in_place: &[(0, 0), (1, 1)],
    }

    /// The rotation fused with the write into the paged KV cache.
    ROPE_WRITE_KV_BF16 = "rope::rope_write_kv_bf16" as rope_write_kv {
        whole: true,
        sink: Some("kv.pages"),
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Eight binds and four written refusals. Six of these twelve were fireable
// before this file existed; the generated dispatcher had an arm for
// `rope_standard_table`, `rope_bf16`, `qk_rmsnorm_rope_bf16`,
// `rope_partial_bf16`, `qk_rmsnorm_rope_bf16_rounded` and
// `rope_yarn_original_bf16` and for nothing else, because
// `emit_rust_dispatch` skips a row whose operands are `Source::Unbound`. The
// other six were declared, statable by the DSL, and silently unreachable.
//
// After this file: eight fire, and the four that do not each refuse AT MODEL
// LOAD with the sentence its row carried. §0 asks for exactly that — "every
// refusal the system can make is made at model load" — and a `Source::Unbound`
// could not do it, because a row that binds nothing is indistinguishable from
// a row nobody has got to yet.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    ROPE_STANDARD_TABLE => { cx, stream => {
        unsafe {
            rope_standard_table(
                cx.positions()?,
                cx.arg_out(0)?.cast::<f32>(),
                cx.rows().count,
                cx.head_dim()?,
                cx.rope_theta()?,
                stream,
            )
        }
        .ok()
    }},

    ROPE_BF16 => { cx, stream => {
        // `interleaved` WAS `Source::Lit(Lit::Bool(false))`, and the row said
        // why: "no statement and no context carries it: the families that
        // pass `true` (GLM, MLA) are not declared, and a row that pretended
        // otherwise would be guessing on their behalf."
        //
        // `rope_yarn_original`'s row contradicts it in the same table — it
        // sources the same flag from `Source::Ctx("rope_interleaved")` — so
        // one of the two was stale, and it is this one: the context has
        // carried the fact since that row was written. `Cx::rope_interleaved`
        // answers `false` for every deployment the literal was right about,
        // so this is the same behaviour today and the right behaviour the day
        // GLM is declared.
        unsafe {
            rope_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.num_q_heads()?,
                cx.num_kv_heads()?,
                cx.head_dim()?,
                cx.rope_theta()?,
                cx.rope_interleaved(),
                stream,
            )
        }
        .ok()
    }},

    QK_RMSNORM_ROPE_BF16 => { cx, stream => {
        // `Source::CtxNonZero("head_dim")` was the row's way of saying "and
        // this divisor must not be zero". It is a `if` here, and the refusal
        // says which extent — where the row world's answer to a zero divisor
        // was a `Source` variant with the guard folded into its name.
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        unsafe {
            qk_rmsnorm_rope_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                cx.out_width(1)? / head_dim,
                head_dim,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // **The bind the row world could not write.**
    //
    // `win_d` is "a device word the driver writes between replays; no
    // `Source` reads device memory". `Cx::peel_window` is the driver handing
    // over an address it already owns, which is not reading device memory,
    // and this is six lines.
    QK_RMSNORM_ROPE_BF16_DEVWIN => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        // `Rows::total`, not `Rows::count`: the grid spans every lane because
        // the early-out on the window is the KERNEL's. See
        // `qk_rmsnorm_rope_bf16_devwin`'s own doc for the two-ended argument.
        let n_max = cx.rows().total;
        unsafe {
            qk_rmsnorm_rope_bf16_devwin(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                cx.peel_window()?.as_ptr().cast_const(),
                n_max,
                cx.out_width(0)? / head_dim,
                cx.out_width(1)? / head_dim,
                head_dim,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // No bind, and the reason is the row's.
    //
    // Llama-3's `low_freq_factor` and `high_freq_factor` have no carrier.
    // `Cx::yarn` answers the ORIGINAL YaRN quartet, which is a different
    // scheme with the same arity — reading one for the other gives a model
    // that runs and is wrong past its training length, and §10.5 refuses
    // exactly that guess. [`rope_yarn_bf16`] is public and a caller that
    // knows the two factors can call it today.
    ROPE_YARN_BF16 => { none:
        "rope_yarn: llama-3's low_freq_factor/high_freq_factor. No statement \
         and no context carries them, and the YaRN quartet the context does \
         carry is a different scheme with the same arity" },

    // No bind, and the reason is the row's.
    //
    // The `(t, h, w)` section split is a property of a vision checkpoint
    // that no statement and no context carries. Under the row world this
    // left THIRTEEN operands unbound because "a half-bound row is a row
    // whose unbound cells look like an oversight rather than a fact"; here
    // the other ten are bound in [`qk_rmsnorm_mrope_bf16`]'s parameter list
    // and only the three that are genuinely missing are missing.
    QK_RMSNORM_MROPE_BF16 => { none:
        "qk_rmsnorm_mrope: the (t, h, w) section split. A property of a \
         vision checkpoint that no statement and no context carries" },

    // **The second bind the row world could not write.**
    //
    // `families/rope.rs` left this row unsourced for two facts at once: the
    // YaRN ramp bounds, which run on the host, and the `Or` over a second
    // result's width. [`ramp_bounds`] is the first and `Result::unwrap_or`
    // is the second.
    ROPE_PARTIAL_LAST_BF16 => { cx, stream => {
        // The head COUNTS off the CACHE's head dim rather than the ctx's,
        // because a KV-shared layer's q and k disagree — `rope_partial_q_only`
        // states the same and for the same reason.
        let kv = cx.kv_layer()?;
        if kv.head_dim <= 0 {
            return Err(Refusal::Empty { what: "kv head_dim" });
        }
        let q = cx.arg_out(0)?.cast::<bf16>();
        // ZERO when there is no second result, which is the q-only form's
        // whole signal to the kernel.
        let kv_heads = cx.out_width(1).map_or(0, |w| w / kv.head_dim);
        // `rope.cu:367` reads an unscaled deployment as `yarn_factor <= 1`,
        // so a checkpoint with no YaRN block is not a refusal: it is
        // `Yarn::NONE`, whose `factor` is 1 and which the ramp guard in
        // `rope_partial_last_bf16` already handles.
        let yarn = cx.yarn().unwrap_or(crate::x::Yarn::NONE);
        unsafe {
            rope_partial_last_bf16(
                q,
                cx.arg_out(1).unwrap_or(q.cast()).cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / kv.head_dim,
                kv_heads,
                kv.head_dim,
                cx.rotary_width()?,
                cx.theta()?,
                // `inverse` is FALSE and it is not a guess. The inverse
                // rotation un-rotates a cached key before re-rotating it at a
                // new position; it is a step the DRIVER takes between
                // statements, and there is no trace statement whose meaning
                // is "un-rotate". A trace that says `rope_partial_last` says
                // the forward rotation, so the flag is a constant of this
                // BINDING and not of the kernel — which is why it is written
                // here and not in the fn.
                false,
                cx.rope_interleaved(),
                yarn.factor,
                yarn.beta_fast,
                yarn.beta_slow,
                yarn.original_max_position,
                stream,
            )
        }
        .ok()
    }},

    ROPE_PARTIAL_BF16 => { cx, stream => {
        let kv = cx.kv_layer()?;
        if kv.head_dim <= 0 {
            return Err(Refusal::Empty { what: "kv head_dim" });
        }
        let q = cx.arg_out(0)?.cast::<bf16>();
        unsafe {
            rope_partial::<bf16>(
                "rope::rope_partial_bf16",
                q,
                // A Q-ONLY SITE STATES ONE RESULT and the launcher takes q
                // for k with `num_kv_heads = 0`. `Source::Or(&Out(1),
                // &Out(0))` was the row's spelling; this is the same fallback
                // in the language the fallback belongs to.
                cx.arg_out(1).unwrap_or(q.cast()).cast::<bf16>(),
                cx.positions()?,
                // `Source::Lit(Lit::I32(0))`: this statement is the
                // un-shifted form, and the shifted one is a different symbol.
                0,
                cx.rows().count,
                cx.out_width(0)? / kv.head_dim,
                cx.out_width(1).map_or(0, |w| w / kv.head_dim),
                kv.head_dim,
                cx.rotary_width()?,
                cx.theta()?,
                stream,
            )
        }
        .ok()
    }},

    // No bind, and the reason is the row's.
    //
    // The delta is a fact about a CALLER — a draft model re-rotating a
    // rejected suffix at the verify pass's absolute positions — and no
    // statement and no context carries one. [`rope_partial`] is public and
    // generic; a speculative decode path that knows its own offset calls it
    // with this symbol and its delta, which is what the symbol is for.
    ROPE_PARTIAL_BF16_POSITION_DELTA => { none:
        "rope_partial_position_delta: the offset added to every position. A \
         fact about a draft/verify pairing that no statement carries" },

    QK_RMSNORM_ROPE_BF16_ROUNDED => { cx, stream => {
        let kv = cx.kv_layer()?;
        if kv.head_dim <= 0 {
            return Err(Refusal::Empty { what: "kv head_dim" });
        }
        // A Q-ONLY SITE STATES ONE RESULT AND NO K WEIGHT, and the launcher
        // reads the nulls as "there is no k". `Source::Or(&Out(1),
        // &Lit(Lit::Null))` and `Source::Or(&Weight(1), &Lit(Lit::Null))`
        // were the row's two spellings of it; a null pointer is a null
        // pointer in either language, and here the two fall out of the same
        // `unwrap_or` the head count does.
        let k = cx.arg_out(1).unwrap_or(core::ptr::null_mut()).cast::<bf16>();
        let k_weight = cx.weight(1).unwrap_or(core::ptr::null_mut()).cast_const().cast::<bf16>();
        unsafe {
            qk_rmsnorm_rope_bf16_rounded(
                cx.arg_out(0)?.cast::<bf16>(),
                k,
                cx.weight(0)?.cast_const().cast::<bf16>(),
                k_weight,
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / kv.head_dim,
                cx.out_width(1).map_or(0, |w| w / kv.head_dim),
                kv.head_dim,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    ROPE_YARN_ORIGINAL_BF16 => { cx, stream => {
        let head_dim = cx.head_dim()?;
        if head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        // The row read these four as `Ctx("yarn[0]")` .. `Ctx("yarn[3]")` and
        // noted that "`Ctx` names a FIELD PATH, so an index is as nameable as
        // a name". An index is nameable and it is not READABLE: `yarn[2]` is
        // `beta_slow` only if you have the struct open beside you. `Cx::yarn`
        // answers a type with four named fields, and the two-schemes hazard
        // this family carries is exactly the hazard an index cannot warn you
        // about.
        let yarn = cx.yarn()?;
        unsafe {
            rope_yarn_original_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.positions()?,
                cx.rows().count,
                cx.out_width(0)? / head_dim,
                cx.out_width(1)? / head_dim,
                head_dim,
                cx.rope_theta()?,
                yarn.factor,
                yarn.beta_fast,
                yarn.beta_slow,
                yarn.attention_factor,
                yarn.original_max_position,
                cx.rope_interleaved(),
                stream,
            )
        }
        .ok()
    }},

    // No bind, and the reason is NEW and smaller than the row's.
    //
    // Every operand this launcher needs is now reachable: `Cx::kv_layer`
    // carries the pages, the page size and the layout flag, and `Cx::plan`
    // carries the five per-request arrays. What is missing is one line of
    // DECLARATION, not one fact: the contract states no `in_place` pair, so
    // nothing says the rotation happens at the operands' own addresses, and
    // a bind that assumed it would be claiming an aliasing this contract
    // does not make. `families/rope.rs` asks for the staging evidence first.
    //
    // The lift is `in_place: &[(0, 0), (1, 1)]` on the contract above and a
    // bind of the same shape as [`ROPE_BF16`]'s, once a caller's staging is
    // known. [`rope_write_kv_bf16`] is public and complete meanwhile.
    ROPE_WRITE_KV_BF16 => { none:
        "rope_write_kv: the contract states no in_place pair, so which \
         addresses q and k rotate at is not something the declaration \
         determines. Every other operand is reachable" },
}
