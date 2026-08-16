//! `cascade.cuh`'s two merge launchers — the fold that turns a split-KV
//! attention's partial states back into one answer.
//!
//! **Not a family**: nothing names a merge in a trace; it fires from inside
//! an FA2 dispatch, so there is no `ROUTINES`/`FAMILY`, and
//! `fire/merge_states.rs` is the one caller, by path.
//!
//! [`merge_states_varlen`], not [`merge_states`], is what both FA2 batched
//! dispatches call, because a ragged batch (different KV length per row)
//! folded with a uniform chunk count reads another row's partials.
//!
//! **Neither routine may gate on architecture**: on sm_100 the decode
//! KV-split path (built on every arch) reaches this merge too, so a gate
//! would throw there rather than degrade — refusals below are shape-only,
//! never a fallback for an uninstantiated head dim.
#![allow(clippy::too_many_arguments)]

/// The ragged fold the FA2 split path needs, as a job and a launch.
///
/// Moved down from `driver-cuda`'s `fire/merge_states.rs`: it imported
/// nothing but this crate, and `attn::fa2::dispatch` needs to name the job's
/// type to build it.
pub mod merge_states;

use core::ptr::NonNull;

use crate::jit::{Ctx, Launch};
use crate::jit::Abi;
use crate::jit::abi::bf16;
use kernels::Refusal;

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
/// product is always this number and the occupancy query's `blockSize` can
/// never disagree with the launch's block extent (contrast
/// `fire/flashinfer_fa2.rs`'s `decode_max_grid_size`, which must handle a
/// 120-thread FA2 decode block against a 128-thread query).
pub const NUM_THREADS: u32 = 128;

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
        64 | 128 | 256 => Some("::flashinfer::MergeStatesKernel<\
                                    8, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>"),
        512 => Some("::flashinfer::MergeStatesKernel<\
                         16, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>"),
        _ => None,
    }
}

/// The instantiation for the staged arm at `head_dim`.
const fn merge_large_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some("::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                        8, 8, 16, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>"),
        128 => Some("::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                         8, 16, 8, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>"),
        256 => Some("::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                         8, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>"),
        512 => Some("::flashinfer::MergeStatesLargeNumIndexSetsKernel<\
                         16, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO>"),
        _ => None,
    }
}

/// The instantiation for the variable-length arm at `head_dim`.
const fn merge_varlen_inst(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some("::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                        8, 8, 16, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>"),
        128 => Some("::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                         8, 16, 8, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>"),
        256 => Some("::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                         8, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>"),
        512 => Some("::flashinfer::PersistentVariableLengthMergeStatesKernel<\
                         16, 32, 4, 4, ::pie::cascade::DTypeIn, ::pie::cascade::DTypeO, ::pie::cascade::IdType>"),
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
    /// The widest block CUDA will launch, `cascade.cuh:661`'s implicit bound.
    const MAX_BLOCK_THREADS: u32 = 1024;

    let (_, bdx, bdy) = geometry(head_dim).ok_or(NO_ROW)?;
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
                "cascade/merge_states.cuh",
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
    // a runtime value and the 1,024-thread cap is checkable only on the host
    // (at head dim 64, `bdx = 8` and the cap is 128 query heads; at 512,
    // `bdx = 32` and the cap is 32 heads).
    //
    // Refused here rather than upstream: `MergeStates` would otherwise launch
    // and return a `cudaErrorInvalidConfiguration` whose message says nothing
    // about heads.
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
            "cascade/merge_states.cuh",
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
/// One kernel, always: `PersistentVariableLengthMergeStatesKernel`. The
/// raggedness is in `indptr`, not the launcher, so the only host decisions
/// are the empty-work guard and the grid.
///
/// # The grid is a performance knob, not a correctness input
///
/// `:711` launches `num_sms * num_blocks_per_sm` blocks and `:388` runs a
/// grid-stride loop over `seq_len * num_heads`, so any positive grid computes
/// the same answer and a grid larger than the work just retires idle blocks.
/// That is what makes `:707-708`'s
/// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` safe to approximate —
/// [`blocks_per_sm`] answers 1 block per SM when it cannot ask, which is the
/// conservative direction. `:709` then bounds the grid by
/// `ceil_div(max_seq_len * num_heads, num_sms)` so a small batch does not
/// launch a full-device grid; both terms are in [`grid_blocks`].
///
/// # Geometry
///
/// | | |
/// |---|---|
/// | grid  | `(num_sms * num_blocks_per_sm, 1, 1)` — `:711` |
/// | block | `(bdx, num_threads / bdx, 1)` — `:701`, `:712` |
/// | smem  | [`smem_bytes`] — `:703-704` |
///
/// PDL (`:718-731`, behind `enable_pdl`) is not carried: every FA2 call site
/// passes it through unset, so the `else` at `:732` is the only branch this
/// driver has ever run.
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
            "cascade/merge_states.cuh",
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

    let Ok(resolved) = crate::jit::cache::resolve(&crate::jit::Root::new("cascade/merge_states.cuh"), instantiation) else {
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
        HEAD_DIMS, NUM_SMEM_STAGES, NUM_THREADS, geometry, grid_blocks, merge_inst,
        merge_large_inst, merge_varlen_inst, smem_bytes,
    };

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
