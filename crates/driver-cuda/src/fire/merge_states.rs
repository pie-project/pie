//! `cascade.cuh`'s two merge launchers, in Rust — the fold that turns a
//! split-KV attention's partial states back into one answer.
//!
//! This is a row that **left and came back**. `new-horizon.md` §38 deleted
//! `attn::merge_attention_states_bf16` because its whole consumer set was
//! `dsl::cuda::merge_attention_states`, which nothing called; the C++ that
//! actually ran was compiled into `driver-cuda/csrc/attn/attention_flashinfer.cu`
//! and went with that file when the FA2 seams closed. What the deletion did
//! not account for is that the FA2 lattice's own split path needs this fold —
//! not through the DSL, through
//! [`super::flashinfer_fa2_dispatch::Fired::Split`] — so closing the seams
//! set `disable_split_kv: true` and split-KV prefill has been off since.
//! This file is what turns it back on.
//!
//! # THE SPECIFICATION NAMED THE WRONG FUNCTION
//!
//! `kernels-cuda-new/src/table/attn.rs`' *"WHAT THE RUST FORM NEEDS"* block
//! specifies `MergeStates` (`cascade.cuh:637-668`) and its
//! `num_index_sets >= seq_len` arm. That is a real launcher and it is
//! implemented here as [`merge_states`] — but **it is not the one the FA2
//! split path calls.** Both batched dispatches call
//! `VariableLengthMergeStates`:
//!
//! - `prefill.cuh:4350-4352` — `(tmp_v, tmp_s, params.merge_indptr, o, lse,
//!   params.max_total_num_rows, params.total_num_rows, num_qo_heads,
//!   HEAD_DIM_VO, ...)`
//! - `decode.cuh:822-824` — `(tmp_v, tmp_s, params.o_indptr, o, lse,
//!   params.paged_kv.batch_size, nullptr, num_qo_heads, HEAD_DIM, ...)`
//!
//! `MergeStates` is reached only from the SINGLE-request paths
//! (`prefill.cuh:2559`, `decode.cuh:739`), where every row was split into the
//! same number of chunks and one `num_index_sets` describes the batch.
//!
//! The difference is not a performance one. `MergeStatesKernel` folds
//! `num_index_sets` states for **every** row (`cascade.cuh:221`);
//! `PersistentVariableLengthMergeStatesKernel` reads each row's own count as
//! `indptr[pos + 1] - indptr[pos]` (`:401`). A batch whose requests have
//! different KV lengths has different chunk counts per row, so folding it
//! with a uniform count reads another row's partials and produces a wrong
//! answer that no assertion catches. **Implementing only the specified
//! launcher and flipping the flag would have been silent corruption.** Both
//! are here; [`variable_length`] is the one the seam uses.
//!
//! # `VariableLengthAttentionSum` is not here, and that is a measurement
//!
//! Both call sites above are inside `if constexpr (AttentionVariant::use_softmax)`
//! and have an `else` that calls `VariableLengthAttentionSum` instead
//! (`prefill.cuh:4354-4356`, `decode.cuh:825-827`). Every variant this tree
//! instantiates is a `::flashinfer::DefaultAttention` or a
//! `PieScoreCapture`/`PieScoreCaptureWindow` wrapping one
//! (`csrc/src/attn/fa2.cuh:163-197`), and `variants.cuh:33` writes
//! `static constexpr bool use_softmax = true` with no specialisation. The
//! `else` is therefore unreachable for every row in the lattice, and a sum
//! kernel is not rowed. If a variant with `use_softmax = false` is ever
//! added, this paragraph is what has to change first.
//!
//! # A refusal is a refusal, never a fallback
//!
//! `DISPATCH_HEAD_DIM`'s `default:` is `throw std::invalid_argument`. An
//! exception crossing the C ABI is undefined behaviour and in this tree it
//! unwound to `SIGABRT` with no message. Here an uninstantiated head dim is
//! [`Decline::HeadDim`] and the caller decides — and every caller in
//! `bind/service.rs` decides to panic WITH the head dim in the message.
//! Nothing here substitutes a different kernel for one it cannot fire.
//!
//! # Where the geometry comes from
//!
//! Every grid, block and shared-memory figure below cites `cascade.cuh` by
//! line. The arithmetic is `kernels_cuda_new::families::cascade::geometry`
//! and `::smem_bytes`, beside the rows it instantiates, so the numbers this
//! file launches with and the numbers those rows were built from cannot
//! drift.

use std::ffi::c_void;
use std::fmt;

use kernels_cuda_new::families::cascade;
use kernels_cuda_new::runtime::{ArgValue, Launch};

/// `cascade.cuh:645` and `:700` — `constexpr uint32_t num_threads = 128`.
///
/// The block is `(bdx, bdy)` with `bdy = num_threads / bdx`, so the product
/// is this number exactly and the occupancy query's `blockSize` argument and
/// the launch's block extent cannot disagree. `fire/flashinfer_fa2.rs`'
/// `decode_max_grid_size` has to make that distinction because at GQA group 3
/// the FA2 decode block is 120 threads and upstream still queries 128; here
/// there is nothing to get wrong, and this comment is why the reader does not
/// have to check.
const NUM_THREADS: u32 = 128;

/// Whether the fold ran.
///
/// `fire/gemv.rs`' `#[must_use] enum Gemv`, for its reason and one more of
/// this path's own: a declined merge leaves `v_merged` holding whatever the
/// attention kernel did **not** write — the partials went to `tmp_v` — so a
/// caller that ignores this answer reads uninitialised workspace and calls it
/// an attention output. *"It declined"* must not be spellable like *"it
/// ran"*.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum Merged {
    /// Exactly one kernel is on the caller's stream. `v_merged` (and
    /// `s_merged`, if given) hold the answer once it completes.
    Launched,
    /// Nothing was enqueued, and `v_merged` was not written.
    Declined(Decline),
}

impl Merged {
    /// Panic unless the fold ran, naming the caller.
    ///
    /// The seven FA2 call sites all want this and none of them can carry on
    /// without the merge: the alternative is returning an attention output
    /// that is a workspace scratch buffer. `what` is the dispatch's own name
    /// so the message says which of the seven, which a panic from inside this
    /// module could not.
    ///
    /// # Panics
    ///
    /// If the fold declined. See [`Decline`] for the four reasons.
    pub fn expect_launched(self, what: &str) {
        if let Self::Declined(why) = self {
            panic!("{what}: the split-KV merge declined: {why}");
        }
    }

    /// Whether a kernel was enqueued, for a caller that has already decided
    /// what to do about `false`.
    #[must_use]
    pub fn launched(self) -> bool {
        matches!(self, Self::Launched)
    }
}

/// Why a fold launched nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    /// `families::cascade` carries no row at this head dim.
    ///
    /// `DISPATCH_HEAD_DIM`'s `default:` — `throw std::invalid_argument` — as
    /// a value. The four it does instantiate are 64, 128, 256 and 512.
    HeadDim { head_dim: u32 },
    /// The grid would have had a zero extent.
    ///
    /// `dim3(0, n)` is `cudaErrorInvalidConfiguration` at the launch, which
    /// upstream returns as a `cudaError_t` nobody on this path checks. A
    /// batch with no rows or no heads has nothing to fold and saying so is
    /// cheaper than a driver error three frames up.
    ///
    /// **Not the same as `num_index_sets == 0`**, which is a real state the
    /// kernel handles itself: `cascade.cuh:221-229` writes zeros to
    /// `v_merged` and `-inf` to `s_merged`, which is the correct answer for a
    /// row with no partials and is not something to refuse.
    NoWork { seq_len: u32, num_heads: u32 },
    /// [`merge_states`]' small arm wanted a block wider than CUDA allows.
    ///
    /// `cascade.cuh:659-661` sets `bdy = num_heads` — a runtime value in a
    /// block extent — so the block is `bdx * num_heads` threads and 1,024 is
    /// the architectural cap. At head dim 64 that is 8 heads times 128,
    /// reached at 128 query heads; at 512 it is `bdx = 32` and the cap is 32
    /// heads.
    ///
    /// **A refusal upstream does not make.** `MergeStates` launches and
    /// returns `cudaErrorInvalidConfiguration`, which its callers pass to
    /// `FLASHINFER_CUDA_CALL`. Naming it here is a deliberate deviation and
    /// this is the record of it: the alternative was a driver error whose
    /// message says nothing about heads.
    BlockTooWide { bdx: u32, num_heads: u32 },
    /// A required device address was null.
    ///
    /// `v`, `v_merged` and — for [`variable_length`] — `indptr` are not
    /// nullable in the rows (`families::cascade`'s operand lists mark only
    /// `s_merged` and `seq_len`), so a zero here would be a fault inside the
    /// kernel rather than a bind error. `s` is included: all three kernels
    /// read it unconditionally.
    NullOperand { which: &'static str },
}

impl fmt::Display for Decline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::HeadDim { head_dim } => write!(
                f,
                "head dim {head_dim} has no cascade merge row -- \
                 `DISPATCH_HEAD_DIM` instantiates 64, 128, 256, 512"
            ),
            Self::NoWork { seq_len, num_heads } => {
                write!(f, "nothing to fold: seq_len {seq_len}, num_heads {num_heads}")
            }
            Self::BlockTooWide { bdx, num_heads } => write!(
                f,
                "`MergeStatesKernel`'s block is (bdx {bdx}, num_heads {num_heads}) = {} \
                 threads, over the 1024 cap (`cascade.cuh:661`)",
                bdx.saturating_mul(num_heads)
            ),
            Self::NullOperand { which } => write!(f, "`{which}` is null"),
        }
    }
}

/// `MergeStates`' operands, `cascade.cuh:638-640`.
///
/// One struct rather than eight positional arguments because four of them are
/// `u64` addresses and four are `u32` counts, and the two orders that
/// type-check are not the same order. `cascade.cuh` puts `v, s, v_merged,
/// s_merged` in that sequence and so does this; the names are its names.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Uniform {
    /// `DTypeIn*`, `[num_index_sets, seq_len, num_heads, head_dim]` partial
    /// outputs.
    pub v: u64,
    /// `float*`, `[num_index_sets, seq_len, num_heads]` partial log-sum-exps,
    /// base 2.
    pub s: u64,
    /// `DTypeO*`, `[seq_len, num_heads, head_dim]`, written.
    pub v_merged: u64,
    /// `float*`, `[seq_len, num_heads]`, written — or 0, which the kernels
    /// test (`cascade.cuh:298`, `:337`).
    pub s_merged: u64,
    /// How many states each row has. **The same for every row**, which is
    /// what makes this the single-request launcher; see the module header.
    pub num_index_sets: u32,
    /// Rows.
    pub seq_len: u32,
    /// Heads.
    pub num_heads: u32,
    /// 64, 128, 256 or 512.
    pub head_dim: u32,
}

/// `VariableLengthMergeStates`' operands, `cascade.cuh:687-690`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarLen {
    /// `DTypeIn*` partial outputs, ragged: row `pos` owns
    /// `[indptr[pos], indptr[pos + 1])`.
    pub v: u64,
    /// `float*` partial log-sum-exps, the same ragged shape.
    pub s: u64,
    /// `IdType*` — `int32_t`, `[max_seq_len + 1]` entries. Prefill passes
    /// `params.merge_indptr`, decode passes `params.o_indptr`.
    pub indptr: u64,
    /// `DTypeO*` `[seq_len, num_heads, head_dim]`, written.
    pub v_merged: u64,
    /// `float*` `[seq_len, num_heads]`, written — or 0.
    pub s_merged: u64,
    /// The row count the grid is sized against.
    pub max_seq_len: u32,
    /// `uint32_t*` — a DEVICE pointer to the real row count, or 0 to use
    /// `max_seq_len` (`cascade.cuh:375`). Prefill passes
    /// `params.total_num_rows`; decode passes null.
    pub seq_len: u64,
    /// Heads.
    pub num_heads: u32,
    /// 64, 128, 256 or 512.
    pub head_dim: u32,
}

/// `MergeStates`, `cascade.cuh:637-668` — the uniform-chunk-count fold.
///
/// The two host decisions the launcher makes, both of them plain Rust `if`s:
///
/// 1. **Empty work.** `dim3(seq_len, num_heads)` with either extent zero is a
///    launch the driver refuses; [`Decline::NoWork`] instead.
/// 2. **The arm**, `cascade.cuh:644` — `num_index_sets >= seq_len` picks
///    `MergeStatesLargeNumIndexSetsKernel`, otherwise `MergeStatesKernel`.
///    **Exactly one fires**, so there is no intermediate buffer and no
///    ordering to get wrong.
///
/// `table/attn.rs` recorded that arm as unstateable *"because it compares two
/// operands while every `Term` is unary"*. It is a comparison between two
/// numbers the host already holds, and this is what it looks like when host
/// composition is Rust: a `>=` and two branches.
///
/// # The two arms' geometry
///
/// `num_index_sets >= seq_len`, `cascade.cuh:645-657`:
///
/// | | |
/// |---|---|
/// | grid  | `(seq_len, num_heads, 1)` — `:647` |
/// | block | `(bdx, num_threads / bdx, 1)` — `:646`, `:648` |
/// | smem  | `4 * bdy * head_dim * 2 + 128 * 4` — `:653-654` |
///
/// otherwise, `:659-664`:
///
/// | | |
/// |---|---|
/// | grid  | `(seq_len, 1, 1)` — `:660` |
/// | block | `(bdx, num_heads, 1)` — `:659`, `:661` |
/// | smem  | 0 — `:664`'s launch passes it literally |
///
/// # Shared memory, and the `cudaFuncSetAttribute` that is not here
///
/// `:655-656` raises `cudaFuncAttributeMaxDynamicSharedMemorySize` before the
/// staged launch. The figure is 8,704 B at head dims 64, 128 and 256 and
/// 16,896 B at 512 — both under the 48 KB every architecture gives a block
/// without asking — so the call is a no-op and there is nothing for this
/// function to express. `families::cascade::smem_bytes` derives the numbers
/// and its tests pin them.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
/// A shape this launcher will not fire is a [`Decline`], not a panic.
///
/// # Safety
///
/// Every address in `job` must name device memory of the extent the kernel
/// reads or writes, and `stream` must outlive the launch — the same
/// assertion the caller made when it handed these pointers to a
/// `cudaLaunchKernel`.
pub unsafe fn merge_states(job: Uniform, stream: *mut c_void) -> Merged {
    let Some((_, bdx, bdy)) = cascade::geometry(job.head_dim) else {
        return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
    };
    if job.seq_len == 0 || job.num_heads == 0 {
        return Merged::Declined(Decline::NoWork {
            seq_len: job.seq_len,
            num_heads: job.num_heads,
        });
    }
    for (which, addr) in [("v", job.v), ("s", job.s), ("v_merged", job.v_merged)] {
        if addr == 0 {
            return Merged::Declined(Decline::NullOperand { which });
        }
    }

    if job.num_index_sets >= job.seq_len {
        // `cascade.cuh:645-657`. The staged arm.
        let Some(smem) = cascade::smem_bytes(job.head_dim) else {
            return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
        };
        let Some(symbol) = cascade::merge_states_large_symbol(job.head_dim) else {
            return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
        };
        // `:652`. Six operands: `head_dim` is `vec_size * bdx` as a
        // `constexpr` at `:288` and is not passed.
        let values = [
            ArgValue::Ptr(ptr(job.v)),
            ArgValue::Ptr(ptr(job.s)),
            ArgValue::Ptr(ptr(job.v_merged)),
            ArgValue::Ptr(ptr(job.s_merged)),
            ArgValue::U32(job.num_index_sets),
            ArgValue::U32(job.num_heads),
        ];
        let launch = Launch {
            grid: [job.seq_len, job.num_heads, 1],
            block: [bdx, bdy, 1],
            smem,
        };
        // SAFETY: the caller's contract, forwarded.
        super::hand::fire(symbol, launch, &values, stream);
        return Merged::Launched;
    }

    // `cascade.cuh:659-664`. `bdy` is `num_heads`, so the block width is a
    // runtime value and the cap is checkable only here.
    if bdx.saturating_mul(job.num_heads) > 1024 {
        return Merged::Declined(Decline::BlockTooWide { bdx, num_heads: job.num_heads });
    }
    let Some(symbol) = cascade::merge_states_symbol(job.head_dim) else {
        return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
    };
    // `:663`. Seven operands: this is the arm where `head_dim` is a
    // parameter rather than a template argument.
    let values = [
        ArgValue::Ptr(ptr(job.v)),
        ArgValue::Ptr(ptr(job.s)),
        ArgValue::Ptr(ptr(job.v_merged)),
        ArgValue::Ptr(ptr(job.s_merged)),
        ArgValue::U32(job.num_index_sets),
        ArgValue::U32(job.num_heads),
        ArgValue::U32(job.head_dim),
    ];
    let launch = Launch {
        grid: [job.seq_len, 1, 1],
        block: [bdx, job.num_heads, 1],
        smem: 0,
    };
    // SAFETY: as above.
    super::hand::fire(symbol, launch, &values, stream);
    Merged::Launched
}

/// `VariableLengthMergeStates`, `cascade.cuh:686-736` — the ragged fold, and
/// the one the FA2 split path calls.
///
/// One kernel, always: `PersistentVariableLengthMergeStatesKernel`. There is
/// no arm here — the raggedness is in `indptr` rather than in the launcher —
/// so the only host decision is the empty-work guard and the grid.
///
/// # The grid is a performance knob and not a correctness input
///
/// `:711` launches `num_sms * num_blocks_per_sm` blocks and `:398` runs a
/// grid-stride loop over `seq_len * num_heads`:
///
/// ```text
/// for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas)
/// ```
///
/// so **any positive grid computes the same answer** and a grid larger than
/// the work simply retires idle blocks. That is what makes it safe to
/// approximate `:707-708`'s
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` — and this does not
/// approximate it: it asks, through
/// `kernels_cuda_new::runtime::module::KernelModule::max_active_blocks_per_sm`
/// over the `CUfunction` the unit produced, which is the same query on the
/// same kernel. When the query cannot be made — no module, no such entry
/// point — it answers 1 block per SM, which is the conservative direction:
/// fewer, longer-lived blocks, all of them correct.
///
/// `:709` then bounds it by `ceil_div(max_seq_len * num_heads, num_sms)`, so
/// a small batch does not launch a full-device grid to retire most of it.
/// Both terms are here.
///
/// `num_sms` comes from `fire::flashinfer_fa2::plan_device`, which is
/// `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` asked once per
/// process and cached — `:693-694` asks it per call.
///
/// # Geometry
///
/// | | |
/// |---|---|
/// | grid  | `(num_sms * num_blocks_per_sm, 1, 1)` — `:711` |
/// | block | `(bdx, num_threads / bdx, 1)` — `:701`, `:712` |
/// | smem  | `4 * bdy * head_dim * 2 + 128 * 4` — `:703-704` |
///
/// The shared-memory figure is `merge_states`' staged arm's, identically:
/// 8,704 B at 64/128/256 and 16,896 B at 512, both under 48 KB, so `:715`'s
/// `cudaFuncSetAttribute` is the same no-op.
///
/// # PDL is not here
///
/// `:718-731` has a programmatic-dependent-launch path behind
/// `enable_pdl`. Both FA2 call sites pass `enable_pdl` through from their own
/// dispatch, and this driver's FA2 fires never set it — the lattice has no
/// PDL axis. So the `else` at `:732` is the only branch that has ever run
/// here and it is the only one ported. Adding PDL would be a
/// `cudaLaunchKernelEx` this crate's `fire` does not offer, which is a
/// separate change with its own measurement.
///
/// # Panics
///
/// If the kernel table and this driver disagree; see [`super::hand::fire`].
///
/// # Safety
///
/// As [`merge_states`].
pub unsafe fn variable_length(job: VarLen, stream: *mut c_void) -> Merged {
    let Some((_, bdx, bdy)) = cascade::geometry(job.head_dim) else {
        return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
    };
    let Some(smem) = cascade::smem_bytes(job.head_dim) else {
        return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
    };
    let Some(symbol) = cascade::merge_states_varlen_symbol(job.head_dim) else {
        return Merged::Declined(Decline::HeadDim { head_dim: job.head_dim });
    };
    if job.max_seq_len == 0 || job.num_heads == 0 {
        return Merged::Declined(Decline::NoWork {
            seq_len: job.max_seq_len,
            num_heads: job.num_heads,
        });
    }
    for (which, addr) in [
        ("v", job.v),
        ("s", job.s),
        ("indptr", job.indptr),
        ("v_merged", job.v_merged),
    ] {
        if addr == 0 {
            return Merged::Declined(Decline::NullOperand { which });
        }
    }

    let num_sms = super::flashinfer_fa2::plan_device().num_sm.max(1);
    let blocks = grid_blocks(symbol, smem, job.max_seq_len, job.num_heads, num_sms);

    // `:713`. Eight operands, `cascade.cuh:366-371`'s order.
    let values = [
        ArgValue::Ptr(ptr(job.v)),
        ArgValue::Ptr(ptr(job.s)),
        ArgValue::Ptr(ptr(job.indptr)),
        ArgValue::Ptr(ptr(job.v_merged)),
        ArgValue::Ptr(ptr(job.s_merged)),
        ArgValue::U32(job.max_seq_len),
        ArgValue::Ptr(ptr(job.seq_len)),
        ArgValue::U32(job.num_heads),
    ];
    let launch = Launch { grid: [blocks, 1, 1], block: [bdx, bdy, 1], smem };
    // SAFETY: the caller's contract, forwarded.
    super::hand::fire(symbol, launch, &values, stream);
    Merged::Launched
}

/// `cascade.cuh:707-711`'s grid, in one place because it is the only
/// arithmetic in this file a reader has to check twice.
///
/// Returns at least `num_sms`: the occupancy query failing is not a reason to
/// launch nothing, and one block per SM is a legal grid for a grid-stride
/// loop.
fn grid_blocks(
    symbol: &'static str,
    smem: u32,
    max_seq_len: u32,
    num_heads: u32,
    num_sms: u32,
) -> u32 {
    // `:709`'s bound. `max(1)` because `min(occupancy, 0)` would be a grid of
    // zero, and the empty-work guard above has already established that
    // neither factor is zero.
    let work_bound = max_seq_len
        .saturating_mul(num_heads)
        .div_ceil(num_sms)
        .max(1);
    let per_sm = occupancy(symbol, smem).min(work_bound);
    per_sm.saturating_mul(num_sms).max(num_sms)
}

/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cascade.cuh:707-708`.
///
/// 1 when it cannot be asked — see [`variable_length`]'s note on why that is
/// safe here and would not be in a launcher whose grid indexes the work.
fn occupancy(symbol: &'static str, smem: u32) -> u32 {
    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        return 1;
    };
    let Ok(module) = kernels_cuda_new::runtime::cache::module(index, unit) else {
        return 1;
    };
    match module.max_active_blocks_per_sm(symbol, NUM_THREADS, smem) {
        Ok(per_sm) => per_sm.max(1),
        Err(_) => 1,
    }
}

/// A device address as the pointer `ArgValue::Ptr` wants.
///
/// The FA2 seam carries addresses as `u64` — `plan_info`'s offsets are added
/// to a workspace base and never dereferenced on the host — so this is the
/// one place the width changes, rather than eight `as` casts in an argument
/// list where a transposition would not be visible.
fn ptr(addr: u64) -> *mut c_void {
    addr as usize as *mut c_void
}

#[cfg(test)]
mod tests {
    use super::{Decline, Merged, Uniform, VarLen, grid_blocks};

    /// The refusals are refusals, and a declined fold is not a launched one.
    ///
    /// `Merged` is `#[must_use]` and two-valued, so the thing worth checking
    /// is that `launched()` and the enum agree — a `Declined` that answered
    /// `true` would make every call site's `if` a no-op.
    #[test]
    fn a_decline_does_not_read_as_a_launch() {
        assert!(Merged::Launched.launched());
        assert!(!Merged::Declined(Decline::HeadDim { head_dim: 96 }).launched());
        assert!(!Merged::Declined(Decline::NoWork { seq_len: 0, num_heads: 8 }).launched());
    }

    /// Every `Decline` says a number, so a panic message is actionable.
    ///
    /// The failure this prevents is the one `DISPATCH_HEAD_DIM`'s
    /// `throw std::invalid_argument` had in practice: an abort with no
    /// message, from which the head dim had to be guessed.
    #[test]
    fn every_decline_names_what_it_refused() {
        let cases = [
            (Decline::HeadDim { head_dim: 96 }, "96"),
            (Decline::NoWork { seq_len: 0, num_heads: 8 }, "0"),
            (Decline::BlockTooWide { bdx: 8, num_heads: 256 }, "2048"),
            (Decline::NullOperand { which: "indptr" }, "indptr"),
        ];
        for (why, needle) in cases {
            let said = why.to_string();
            assert!(said.contains(needle), "`{said}` does not mention `{needle}`");
        }
    }

    /// `cascade.cuh:709`'s bound, on inputs where it BITES and inputs where
    /// it does not.
    ///
    /// The symbol handed in is deliberately not a row, so `occupancy` takes
    /// its `unit_of` miss and answers 1 — which makes this a test of `:709`'s
    /// `min` and `:711`'s product and of nothing else. Without that the two
    /// terms could be swapped and every input would still agree.
    #[test]
    fn the_grid_is_bounded_by_the_work_and_never_zero() {
        const NOT_A_ROW: &str = "cascade::not::a::row";
        // 132 SMs, 4 rows, 8 heads: 32 CTAs of work, `ceil_div(32, 132) = 1`,
        // so the bound bites and the grid is one block per SM.
        assert_eq!(grid_blocks(NOT_A_ROW, 8704, 4, 8, 132), 132);
        // A big batch: the bound is 63 and the occupancy stub is 1, so the
        // occupancy is what bites.
        assert_eq!(grid_blocks(NOT_A_ROW, 8704, 1024, 8, 132), 132);
        // One SM, one row, one head: still a launchable grid.
        assert_eq!(grid_blocks(NOT_A_ROW, 8704, 1, 1, 1), 1);
    }

    /// Both job structs default to all-zero, which every refusal path reads
    /// as "nothing to do" rather than as a shape.
    ///
    /// `Default` is derived so a caller can fill three fields and leave the
    /// rest; this is the assertion that the derived value is not accidentally
    /// a launchable one.
    #[test]
    fn a_default_job_names_no_shape() {
        assert_eq!(Uniform::default().head_dim, 0);
        assert_eq!(VarLen::default().head_dim, 0);
        assert_eq!(VarLen::default().seq_len, 0, "null, so `max_seq_len` stands");
    }
}
