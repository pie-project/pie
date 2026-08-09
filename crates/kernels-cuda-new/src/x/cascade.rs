//! `cascade.cuh`'s two merge launchers — the fold that turns a split-KV
//! attention's partial states back into one answer.
//!
//! **Not a family**, for `x::driver_internal`'s reason: no statement names a
//! merge. The fold is reached from inside an FA2 dispatch, which is a
//! `driver-cuda` decision made after the attention kernel has already been
//! planned — so there are `pub fn`s here and no `ROUTINES`, no `FAMILY` and no
//! line in `lib.rs`' `FAMILIES`. `fire/merge_states.rs` is the one caller, and
//! it calls by path.
//!
//! # THE SPECIFICATION NAMED THE WRONG FUNCTION
//!
//! The record that specified this fold asked for `MergeStates`
//! (`cascade.cuh:637-668`, upstream's numbering) and its
//! `num_index_sets >= seq_len` arm. That is a real launcher and it is
//! [`merge_states`] below — but **it is not the one the FA2 split path
//! calls.** Both batched dispatches call `VariableLengthMergeStates`:
//!
//! - `prefill.cuh:4350-4352` — `(tmp_v, tmp_s, params.merge_indptr, o, lse,
//!   params.max_total_num_rows, params.total_num_rows, num_qo_heads,
//!   HEAD_DIM_VO, ...)`
//! - `decode.cuh:822-824` — `(tmp_v, tmp_s, params.o_indptr, o, lse,
//!   params.paged_kv.batch_size, nullptr, num_qo_heads, HEAD_DIM, ...)`
//!
//! `MergeStates` is reached only from the SINGLE-request paths, where every
//! row was split into the same number of chunks and one `num_index_sets`
//! describes the batch.
//!
//! The difference is not a performance one. `MergeStatesKernel` folds
//! `num_index_sets` states for **every** row (`cascade.cuh:221`);
//! `PersistentVariableLengthMergeStatesKernel` reads each row's own count as
//! `indptr[pos + 1] - indptr[pos]` (`:395`). A batch whose requests have
//! different KV lengths has different chunk counts per row, so folding it
//! with a uniform count reads another row's partials and produces a wrong
//! answer that no assertion catches. Both are here;
//! [`merge_states_varlen`] is the one the FA2 seam uses.
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
//! kernel is not carried. If a variant with `use_softmax = false` is ever
//! added, this paragraph is what has to change first.
//!
//! # A merge's CALLERS live on every architecture
//!
//! The deleted `attention_merge_states.cu` was compiled UNCONDITIONALLY
//! rather than under `PIE_CUDA_FLASHINFER_HOPPER_SOURCE`, and the reasoning
//! for putting it under the gate — the KV split producing its inputs is the
//! **sm90 prefill's**, so nothing can reach a merge on an architecture where
//! that unit is stubbed — is **true on sm_80 and sm_90 and FALSE on sm_100.**
//! There the DECODE KV-split path calls a decode dispatch that is built on
//! every architecture and then merges. On Blackwell the dispatch succeeded
//! and the merge **threw**, poisoning the driver on the first fire and taking
//! gpt-oss and gemma-4 down with it.
//!
//! That is a statement about where a merge's callers live, not about where a
//! merge is fast, and it is why neither routine here may acquire an
//! architecture gate. A gate does not degrade to a slower path; it degrades
//! to a throw on the machine the split is most likely to be enabled for. The
//! refusals below are shape refusals for that reason and there is no
//! `compute_capability` read anywhere in this file.
//!
//! # A refusal is a refusal, never a fallback
//!
//! `DISPATCH_HEAD_DIM`'s `default:` is `throw std::invalid_argument`. An
//! exception crossing the C ABI is undefined behaviour and in this tree it
//! unwound to `SIGABRT` with no message. Here an uninstantiated head dim is a
//! refusal naming the four that are here, and the caller decides — every
//! caller in `driver-cuda` decides to stop. Nothing here substitutes a
//! different kernel for one it cannot fire.

#![allow(clippy::too_many_arguments)]

use core::ptr::NonNull;

use crate::jit::{Ctx, Launch, Root};
use crate::x::Abi;
use crate::x::abi::bf16;
use kernels::Refusal;

/// `cascade/merge_states.cuh` — the root these routines compile a symbol out
/// of.
///
/// `--device-as-default-execution-space` because the root is upstream's
/// header with three `using` declarations on top: without it NVRTC parses
/// `cascade.cuh`'s helpers as host functions. `.upstream()` because its one
/// `#include` reaches `attn/flashinfer/attention/cascade.cuh`, which is in
/// the carried upstream closure and not in the library set.
pub static ROOT: Root = Root::new(
    "cascade/merge_states",
    include_str!("../../csrc/src/cascade/merge_states.cuh"),
    "cascade/merge_states.cuh",
)
.options(&["--device-as-default-execution-space"])
.upstream();

/// The template-ids NVRTC is handed, spelled as it is handed them.
///
/// Three `__global__`s at four head dims. `MergeStatesKernel` takes `head_dim`
/// as a runtime parameter and is therefore parameterised by `vec_size` alone,
/// which is why there are ten and not twelve.
mod inst {
    /// `cascade.cuh:213` — `MergeStatesKernel` at `vec_size = 8`: head dims
    /// 64, 128 and 256.
    pub const MERGE_V8: &str = "::flashinfer::MergeStatesKernel<8, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO>";
    /// The same at `vec_size = 16`: head dim 512.
    pub const MERGE_V16: &str = "::flashinfer::MergeStatesKernel<16, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO>";

    /// `cascade.cuh:275-281` — `MergeStatesLargeNumIndexSetsKernel` at head
    /// dim 64: `<vec_size, bdx, bdy, num_smem_stages>`, [`super::geometry`]'s
    /// triple with the stage count after it.
    pub const LARGE_HD64: &str = "::flashinfer::MergeStatesLargeNumIndexSetsKernel<8, 8, 16, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO>";
    /// The same at head dim 128.
    pub const LARGE_HD128: &str = "::flashinfer::MergeStatesLargeNumIndexSetsKernel<8, 16, 8, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO>";
    /// The same at head dim 256.
    pub const LARGE_HD256: &str = "::flashinfer::MergeStatesLargeNumIndexSetsKernel<8, 32, 4, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO>";
    /// The same at head dim 512, where `vec_size` is 16.
    pub const LARGE_HD512: &str = "::flashinfer::MergeStatesLargeNumIndexSetsKernel<16, 32, 4, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO>";

    /// `cascade.cuh:366-371` — `PersistentVariableLengthMergeStatesKernel` at
    /// head dim 64. The third type argument is `indptr`'s.
    pub const VARLEN_HD64: &str = "::flashinfer::PersistentVariableLengthMergeStatesKernel<8, 8, 16, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO, \
         ::pie_cuda_driver::kernels::cascade::IdType>";
    /// The same at head dim 128.
    pub const VARLEN_HD128: &str = "::flashinfer::PersistentVariableLengthMergeStatesKernel<8, 16, 8, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO, \
         ::pie_cuda_driver::kernels::cascade::IdType>";
    /// The same at head dim 256.
    pub const VARLEN_HD256: &str = "::flashinfer::PersistentVariableLengthMergeStatesKernel<8, 32, 4, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO, \
         ::pie_cuda_driver::kernels::cascade::IdType>";
    /// The same at head dim 512.
    pub const VARLEN_HD512: &str = "::flashinfer::PersistentVariableLengthMergeStatesKernel<16, 32, 4, 4, \
         ::pie_cuda_driver::kernels::cascade::DTypeIn, \
         ::pie_cuda_driver::kernels::cascade::DTypeO, \
         ::pie_cuda_driver::kernels::cascade::IdType>";
}

/// What a head dim outside the lattice gets.
///
/// `DISPATCH_HEAD_DIM`'s `default:` as a value. The four it instantiates are
/// [`HEAD_DIMS`], and a fifth is a compile away rather than a fallback.
const NO_ROW: Refusal =
    Refusal::Unstated { what: "a cascade merge at this head dim -- 64, 128, 256 and 512 are here" };

/// The head dims `DISPATCH_HEAD_DIM` instantiates, in its order.
///
/// FA2's four, and they have to be: the buffers these kernels fold are the
/// ones an FA2 split fire wrote.
pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// `num_smem_stages`, as the two host launchers set it.
pub const NUM_SMEM_STAGES: u32 = 4;

/// `cascade.cuh:645` and `:700` — `constexpr uint32_t num_threads = 128`.
///
/// The staged block is `(bdx, bdy)` with `bdy = num_threads / bdx`, so the
/// product is this number exactly and the occupancy query's `blockSize` and
/// the launch's block extent cannot disagree. `fire/flashinfer_fa2.rs`'
/// `decode_max_grid_size` has to make that distinction because at GQA group 3
/// the FA2 decode block is 120 threads and upstream still queries 128; here
/// there is nothing to get wrong.
pub const NUM_THREADS: u32 = 128;

/// The widest block CUDA will launch, `cascade.cuh:661`'s implicit bound.
const MAX_BLOCK_THREADS: u32 = 1024;

/// `(vec_size, bdx, bdy)` for a head dim, as `MergeStates` derived it.
#[must_use]
pub const fn geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

/// The staged arms' dynamic shared memory, `cascade.cuh:653-654` and
/// `:703-704` — `num_smem_stages * bdy * head_dim * sizeof(DTypeIn) +
/// num_threads * sizeof(float)`.
///
/// 8,704 B at head dims 64, 128 and 256 and 16,896 B at 512 — both under the
/// 48 KB every architecture gives a block without asking, which is why
/// `:655-656`'s and `:715`'s `cudaFuncSetAttribute` are no-ops and why
/// nothing here raises a cap. The test below pins both figures.
#[must_use]
pub const fn smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = geometry(head_dim) else {
        return None;
    };
    Some(NUM_SMEM_STAGES * bdy * head_dim * 2 + NUM_THREADS * 4)
}

/// The instantiation that merges a uniform-chunk-count batch at `head_dim`.
const fn merge_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 | 128 | 256 => Some(inst::MERGE_V8),
        512 => Some(inst::MERGE_V16),
        _ => None,
    }
}

/// The instantiation for the staged arm at `head_dim`.
const fn merge_large_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(inst::LARGE_HD64),
        128 => Some(inst::LARGE_HD128),
        256 => Some(inst::LARGE_HD256),
        512 => Some(inst::LARGE_HD512),
        _ => None,
    }
}

/// The instantiation for the variable-length arm at `head_dim`.
const fn merge_varlen_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(inst::VARLEN_HD64),
        128 => Some(inst::VARLEN_HD128),
        256 => Some(inst::VARLEN_HD256),
        512 => Some(inst::VARLEN_HD512),
        _ => None,
    }
}

/// `MergeStates`, `cascade.cuh:637-668` — the uniform-chunk-count fold.
///
/// The two host decisions the launcher makes, both of them plain Rust `if`s:
///
/// 1. **Empty work.** `dim3(seq_len, num_heads)` with either extent zero is a
///    launch the driver refuses; [`Refusal::Empty`] instead.
/// 2. **The arm**, `cascade.cuh:644` — `num_index_sets >= seq_len` picks
///    `MergeStatesLargeNumIndexSetsKernel`, otherwise `MergeStatesKernel`.
///    **Exactly one fires**, so there is no intermediate buffer and no
///    ordering to get wrong.
///
/// # The two arms' geometry
///
/// `num_index_sets >= seq_len`, `cascade.cuh:645-657`:
///
/// | | |
/// |---|---|
/// | grid  | `(seq_len, num_heads, 1)` — `:647` |
/// | block | `(bdx, num_threads / bdx, 1)` — `:646`, `:648` |
/// | smem  | [`smem_bytes`] — `:653-654` |
///
/// otherwise, `:659-664`:
///
/// | | |
/// |---|---|
/// | grid  | `(seq_len, 1, 1)` — `:660` |
/// | block | `(bdx, num_heads, 1)` — `:659`, `:661` |
/// | smem  | 0 — `:664`'s launch passes it literally |
///
/// `num_index_sets == 0` is not a refusal: `cascade.cuh:221-229` writes zeros
/// to `v_merged` and `-inf` to `s_merged`, which is the right answer for a row
/// with no partials.
///
/// # Safety
///
/// `v` and `s` must address `num_index_sets * seq_len * num_heads * head_dim`
/// and `num_index_sets * seq_len * num_heads` live elements, `v_merged` and
/// `s_merged` the same extents without the leading factor, and `ctx`'s stream
/// must outlive the launch. `s_merged` may be null — `cascade.cuh:253`,
/// `:337` test it.
pub fn merge_states(
    ctx: &Ctx,
    v: *mut bf16,
    s: *mut f32,
    v_merged: *mut bf16,
    s_merged: *mut f32,
    num_index_sets: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), Refusal> {
    let (_, bdx, bdy) = geometry(head_dim).ok_or(NO_ROW)?;
    if seq_len == 0 {
        return Err(Refusal::Empty { what: "seq_len" });
    }
    if num_heads == 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    null_check(v.is_null(), "v")?;
    null_check(s.is_null(), "s")?;
    null_check(v_merged.is_null(), "v_merged")?;

    if num_index_sets >= seq_len {
        // `cascade.cuh:645-657`. The staged arm. Six operands: `head_dim` is
        // `vec_size * bdx` as a `constexpr` at `:288` and is not passed.
        let smem = smem_bytes(head_dim).ok_or(NO_ROW)?;
        let instantiation = merge_large_inst(head_dim).ok_or(NO_ROW)?;
        // SAFETY: the caller's contract -- every pointer bound here addresses
        // live device memory of the extent the kernel reads it as.
        return unsafe {
            ctx.launch(
                &ROOT,
                instantiation,
                Launch::grid([seq_len, num_heads, 1], [bdx, bdy, 1]).smem(smem),
                &[
                    v.arg(),
                    s.arg(),
                    v_merged.arg(),
                    NonNull::new(s_merged).arg(),
                    num_index_sets.arg(),
                    num_heads.arg(),
                ],
            )
        };
    }

    // `cascade.cuh:659-664`. `bdy` is `num_heads` here, so the block width is
    // a runtime value and the 1,024-thread cap is checkable only on the host.
    // At head dim 64 that is `bdx = 8` and the cap is reached at 128 query
    // heads; at 512 it is `bdx = 32` and the cap is 32 heads.
    //
    // **A refusal upstream does not make.** `MergeStates` launches and returns
    // `cudaErrorInvalidConfiguration`, which its callers hand to
    // `FLASHINFER_CUDA_CALL`; the alternative to refusing here was a driver
    // error whose message says nothing about heads.
    let threads = bdx.saturating_mul(num_heads);
    if threads > MAX_BLOCK_THREADS {
        return Err(Refusal::Wide {
            what: "threads per block, which `MergeStatesKernel` sizes by num_heads",
            at: i64::from(threads),
            max: i64::from(MAX_BLOCK_THREADS),
        });
    }
    let instantiation = merge_inst(head_dim).ok_or(NO_ROW)?;
    // `:663`. Seven operands: this is the arm where `head_dim` is a parameter
    // rather than a template argument.
    //
    // SAFETY: as above.
    unsafe {
        ctx.launch(
            &ROOT,
            instantiation,
            Launch::grid([seq_len, 1, 1], [bdx, num_heads, 1]),
            &[
                v.arg(),
                s.arg(),
                v_merged.arg(),
                NonNull::new(s_merged).arg(),
                num_index_sets.arg(),
                num_heads.arg(),
                head_dim.arg(),
            ],
        )
    }
}

/// `VariableLengthMergeStates`, `cascade.cuh:686-736` — the ragged fold, and
/// the one the FA2 split path calls.
///
/// One kernel, always: `PersistentVariableLengthMergeStatesKernel`. There is
/// no arm here — the raggedness is in `indptr` rather than in the launcher —
/// so the only host decisions are the empty-work guard and the grid.
///
/// # The grid is a performance knob and not a correctness input
///
/// `:711` launches `num_sms * num_blocks_per_sm` blocks and `:388` runs a
/// grid-stride loop over `seq_len * num_heads`:
///
/// ```text
/// for (uint32_t i = cta_id; i < seq_len * num_heads; i += num_ctas)
/// ```
///
/// so **any positive grid computes the same answer** and a grid larger than
/// the work simply retires idle blocks. That is what makes
/// `:707-708`'s `cudaOccupancyMaxActiveBlocksPerMultiprocessor` safe to
/// approximate — and [`blocks_per_sm`] does not approximate it, it asks the
/// same question of the same `CUfunction`. When it cannot be asked the answer
/// is 1 block per SM, which is the conservative direction: fewer,
/// longer-lived blocks, all of them correct.
///
/// `:709` then bounds it by `ceil_div(max_seq_len * num_heads, num_sms)`, so
/// a small batch does not launch a full-device grid to retire most of it.
/// Both terms are in [`grid_blocks`].
///
/// # Geometry
///
/// | | |
/// |---|---|
/// | grid  | `(num_sms * num_blocks_per_sm, 1, 1)` — `:711` |
/// | block | `(bdx, num_threads / bdx, 1)` — `:701`, `:712` |
/// | smem  | [`smem_bytes`] — `:703-704` |
///
/// # PDL is not here
///
/// `:718-731` has a programmatic-dependent-launch path behind `enable_pdl`.
/// Both FA2 call sites pass it through from their own dispatch and this
/// driver's FA2 fires never set it — the lattice has no PDL axis — so the
/// `else` at `:732` is the only branch that has ever run here and it is the
/// only one carried.
///
/// # Errors
///
/// [`Refusal::Device`] if the SM count cannot be read: the number is the grid,
/// and there is no defensible default for it.
///
/// # Safety
///
/// `v` and `s` are ragged — row `pos` owns `[indptr[pos], indptr[pos + 1])` —
/// and must be live for whatever `indptr`'s `max_seq_len + 1` entries
/// describe; `v_merged` and `s_merged` must be writable for
/// `[max_seq_len, num_heads, head_dim]` and `[max_seq_len, num_heads]`.
/// `seq_len` is a DEVICE `uint32_t*` overriding `max_seq_len`, or null
/// (`cascade.cuh:375`). `ctx`'s stream must outlive the launch.
pub fn merge_states_varlen(
    ctx: &Ctx,
    v: *mut bf16,
    s: *mut f32,
    indptr: *mut i32,
    v_merged: *mut bf16,
    s_merged: *mut f32,
    max_seq_len: u32,
    seq_len: *mut u32,
    num_heads: u32,
    head_dim: u32,
) -> Result<(), Refusal> {
    let (_, bdx, bdy) = geometry(head_dim).ok_or(NO_ROW)?;
    let smem = smem_bytes(head_dim).ok_or(NO_ROW)?;
    let instantiation = merge_varlen_inst(head_dim).ok_or(NO_ROW)?;
    if max_seq_len == 0 {
        return Err(Refusal::Empty { what: "max_seq_len" });
    }
    if num_heads == 0 {
        return Err(Refusal::Empty { what: "num_heads" });
    }
    null_check(v.is_null(), "v")?;
    null_check(s.is_null(), "s")?;
    null_check(indptr.is_null(), "indptr")?;
    null_check(v_merged.is_null(), "v_merged")?;

    let num_sms = ctx.multiprocessors()?.max(1);
    let blocks = grid_blocks(blocks_per_sm(instantiation, smem), max_seq_len, num_heads, num_sms);

    // `:713`. Eight operands, `cascade.cuh:366-371`'s order.
    //
    // SAFETY: the caller's contract -- every pointer bound here addresses live
    // device memory of the extent the kernel reads it as.
    unsafe {
        ctx.launch(
            &ROOT,
            instantiation,
            Launch::grid([blocks, 1, 1], [bdx, bdy, 1]).smem(smem),
            &[
                v.arg(),
                s.arg(),
                indptr.arg(),
                v_merged.arg(),
                NonNull::new(s_merged).arg(),
                max_seq_len.arg(),
                NonNull::new(seq_len).arg(),
                num_heads.arg(),
            ],
        )
    }
}

/// A required operand, refused by name rather than faulted inside the kernel.
///
/// `v`, `s`, `indptr` and `v_merged` are not nullable in either launcher —
/// only `s_merged` and the variable-length `seq_len` are — so a zero in one of
/// them is a bind error the host can name.
fn null_check(is_null: bool, which: &'static str) -> Result<(), Refusal> {
    if is_null { Err(Refusal::Null { what: which }) } else { Ok(()) }
}

/// `cascade.cuh:707-711`'s grid, in one place because it is the only
/// arithmetic here a reader has to check twice.
///
/// Returns at least `num_sms`: the occupancy query failing is not a reason to
/// launch nothing, and one block per SM is a legal grid for a grid-stride
/// loop.
fn grid_blocks(per_sm: u32, max_seq_len: u32, num_heads: u32, num_sms: u32) -> u32 {
    // `:709`'s bound. `max(1)` because `min(occupancy, 0)` would be a grid of
    // zero, and the empty-work guard has already established that neither
    // factor is zero.
    let work_bound = max_seq_len.saturating_mul(num_heads).div_ceil(num_sms).max(1);
    per_sm.min(work_bound).saturating_mul(num_sms).max(num_sms)
}

/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cascade.cuh:707-708`, on
/// the entry point this launch will use.
///
/// Compiling the instantiation is what the launch that follows does a moment
/// later, so asking here costs the ordering and nothing else. No shared-memory
/// cap is raised first: [`smem_bytes`] is under 48 KB at every head dim, which
/// is what a block gets without asking.
///
/// 1 when it cannot be asked — see [`merge_states_varlen`]'s note on why that
/// is safe here and would not be in a launcher whose grid indexes the work.
#[cfg(feature = "_cuda")]
fn blocks_per_sm(instantiation: &str, smem: u32) -> u32 {
    use cudarc::driver::sys as dr;

    let Ok(resolved) = crate::jit::cache::resolve(&ROOT, instantiation) else {
        return 1;
    };
    let mut blocks: core::ffi::c_int = 0;
    // SAFETY: `blocks` is a live out-parameter and the entry point came from a
    // module this process keeps loaded.
    let code = unsafe {
        dr::cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &raw mut blocks,
            resolved.function,
            i32::try_from(NUM_THREADS).unwrap_or(i32::MAX),
            usize::try_from(smem).unwrap_or(usize::MAX),
        )
    };
    if code != dr::CUresult::CUDA_SUCCESS {
        return 1;
    }
    u32::try_from(blocks).unwrap_or(1).max(1)
}

/// The same question where there is no device to ask.
#[cfg(not(feature = "_cuda"))]
fn blocks_per_sm(_instantiation: &str, _smem: u32) -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::{
        HEAD_DIMS, NUM_SMEM_STAGES, NUM_THREADS, geometry, grid_blocks, inst, merge_inst,
        merge_large_inst, merge_varlen_inst, smem_bytes,
    };

    /// Every instantiation, spelled as the record spells it.
    ///
    /// The constants above are written with line continuations and these are
    /// broken at different points, so the two spellings agree only if the
    /// STRING does. This is the last reader of the row list the unit carried:
    /// nothing else in the process now knows what NVRTC will be asked for.
    #[test]
    fn the_instantiations_are_the_ones_the_record_names() {
        let want: [(&str, &str); 10] = [
            (
                inst::MERGE_V8,
                "::flashinfer::MergeStatesKernel<8, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO>",
            ),
            (
                inst::MERGE_V16,
                "::flashinfer::MergeStatesKernel<16, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO>",
            ),
            (
                inst::LARGE_HD64,
                "::flashinfer::MergeStatesLargeNumIndexSetsKernel<8, 8, 16, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO>",
            ),
            (
                inst::LARGE_HD128,
                "::flashinfer::MergeStatesLargeNumIndexSetsKernel<8, 16, 8, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO>",
            ),
            (
                inst::LARGE_HD256,
                "::flashinfer::MergeStatesLargeNumIndexSetsKernel<8, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO>",
            ),
            (
                inst::LARGE_HD512,
                "::flashinfer::MergeStatesLargeNumIndexSetsKernel<16, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO>",
            ),
            (
                inst::VARLEN_HD64,
                "::flashinfer::PersistentVariableLengthMergeStatesKernel<8, 8, 16, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO, ::pie_cuda_driver::kernels::cascade::IdType>",
            ),
            (
                inst::VARLEN_HD128,
                "::flashinfer::PersistentVariableLengthMergeStatesKernel<8, 16, 8, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO, ::pie_cuda_driver::kernels::cascade::IdType>",
            ),
            (
                inst::VARLEN_HD256,
                "::flashinfer::PersistentVariableLengthMergeStatesKernel<8, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO, ::pie_cuda_driver::kernels::cascade::IdType>",
            ),
            (
                inst::VARLEN_HD512,
                "::flashinfer::PersistentVariableLengthMergeStatesKernel<16, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ::pie_cuda_driver::kernels::cascade::DTypeO, ::pie_cuda_driver::kernels::cascade::IdType>",
            ),
        ];
        for (have, spelled) in want {
            assert_eq!(have, spelled);
        }
    }

    /// Every head dim resolves all three arms, and nothing else resolves.
    #[test]
    fn every_head_dim_has_all_three_arms() {
        for &hd in HEAD_DIMS {
            assert!(merge_inst(hd).is_some(), "{hd}");
            assert!(merge_large_inst(hd).is_some(), "{hd}");
            assert!(merge_varlen_inst(hd).is_some(), "{hd}");
            assert!(geometry(hd).is_some(), "{hd}");
            assert!(smem_bytes(hd).is_some(), "{hd}");
        }
        for hd in [0u32, 32, 96, 120, 1024] {
            assert!(merge_inst(hd).is_none(), "{hd}");
            assert!(merge_large_inst(hd).is_none(), "{hd}");
            assert!(merge_varlen_inst(hd).is_none(), "{hd}");
            assert!(geometry(hd).is_none(), "{hd}");
            assert!(smem_bytes(hd).is_none(), "{hd}");
        }
    }

    /// [`geometry`] and the instantiations agree.
    ///
    /// The staged kernels take `<vec_size, bdx, bdy, num_smem_stages, ...>`,
    /// which is exactly what this function derives — so the derivation and the
    /// template-id are two spellings of one fact and this is where they meet.
    #[test]
    fn the_instantiations_match_the_derivation() {
        for &hd in HEAD_DIMS {
            let (vec_size, bdx, bdy) = geometry(hd).unwrap();
            assert_eq!(bdx * vec_size, hd, "bdx * vec_size is the head dim at {hd}");
            assert_eq!(bdx * bdy, NUM_THREADS, "the staged block is 128 threads at {hd}");

            let want = format!("<{vec_size}, {bdx}, {bdy}, {NUM_SMEM_STAGES}, ");
            for instantiation in [merge_large_inst(hd).unwrap(), merge_varlen_inst(hd).unwrap()] {
                assert!(instantiation.contains(&want), "{instantiation}: {want:?}");
            }
            assert!(merge_inst(hd).unwrap().contains(&format!("<{vec_size}, ")), "{hd}");
        }
    }

    /// The two shared-memory figures, and that neither needs a cap raised.
    #[test]
    fn the_shared_memory_is_the_figure_the_record_carries() {
        assert_eq!(smem_bytes(64), Some(8_704));
        assert_eq!(smem_bytes(128), Some(8_704));
        assert_eq!(smem_bytes(256), Some(8_704));
        assert_eq!(smem_bytes(512), Some(16_896));
        for &hd in HEAD_DIMS {
            assert!(smem_bytes(hd).unwrap() < 48 * 1024, "{hd}");
        }
    }

    /// `cascade.cuh:709`'s bound, on inputs where it BITES and inputs where it
    /// does not.
    ///
    /// Without both terms the grid would still be positive on every input, so
    /// this is the assertion that the `min` and the product are the ones
    /// upstream wrote and not each other.
    #[test]
    fn the_grid_is_bounded_by_the_work_and_never_zero() {
        // 132 SMs, 4 rows, 8 heads: 32 CTAs of work, `ceil_div(32, 132) = 1`,
        // so the bound bites and the grid is one block per SM.
        assert_eq!(grid_blocks(6, 4, 8, 132), 132);
        // A big batch: the bound is 63 and the occupancy is 6, so the
        // occupancy is what bites.
        assert_eq!(grid_blocks(6, 1024, 8, 132), 6 * 132);
        // One SM, one row, one head: still a launchable grid.
        assert_eq!(grid_blocks(1, 1, 1, 1), 1);
        // The query answered zero, which is not a grid.
        assert_eq!(grid_blocks(0, 1024, 8, 132), 132);
    }
}
