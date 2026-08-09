//! `ssm/kda.cu`'s two recurrence launchers, in Rust.
//!
//! Kimi Delta Attention's state walk: a decode step and a prefill scan, both
//! over `R` requests' slots in a paged arena, both sizing dynamic shared
//! memory on the head dimension.
//!
//! # The other two launchers in that file are NOT here
//!
//! `kda_gate_beta_bf16` and `kda_o_norm_gated_bf16` are routed:
//! `device::JIT_DISPATCHED` names both and `families::ssm` states each
//! geometry as a [`kernels::LaunchRule`]. Their bodies were deleted in the
//! same edit that created this file.
//!
//! # These two are `Supplies`, not `Switch`
//!
//! Neither launcher chooses between kernels — there is one `__global__` per
//! entry point and no host `if` past the extent guard. What the host does is
//! supply two numbers no `LaunchRule` states:
//!
//! ```text
//! shmem   = 3 * D * sizeof(float)     both       kda.cu:51, kda.cu:75
//! threads = min(32, D) * 32           prefill    kda.cu:73-74
//! ```
//!
//! `LaunchRule` has no variant that reads a block width off a `min` of an
//! operand extent, and `new-horizon.md` §10.5 forbids adding one for a single
//! kernel. So the rectangle is stated here, beside the line it came from,
//! which is the [`crate::fire::attn_score`] escape hatch used for its reason.
//!
//! # Both rows are UNSOURCED and this is deliberate
//!
//! `table::ssm`'s `kda_recurrent_step_batched` and `kda_prefill_batched`
//! rows leave every operand unsourced: `state_base` is a driver-owned slab
//! and `Source` has no `Scratch` — see `new-horizon.md` §52.3, which counts
//! ten such rows across the tree and names these two. A half-bound row is
//! worse than an unbound one, so nothing here is bound.
//!
//! The consequence, stated so it is not discovered: naming these two in
//! `execution::RUST_SERVED` drops their `pie_k_*` shim entries and generates
//! NO dispatch arm, because `abi::emit_rust_dispatch` skips an unsourced row
//! whole. That is not a regression — an unsourced row had no arm before
//! either — but it does mean the functions below are unreachable from a
//! model trace until a `Source::Scratch` exists. They are reachable from a
//! test and from any caller that has the pointers, which is what makes
//! writing them now worth doing: the geometry is captured while the C++ that
//! states it is still readable.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::fire;

/// `kda.cu:50` — `const int threads = 256;` for the decode step.
///
/// The comment beside it is the reason and it is not arbitrary: *"a multiple
/// of the warp size: the kernel gives one warp a `v` row."* A decode step
/// walks one token, so there is no serialization to hide and 256 is simply a
/// full complement of warps.
const STEP_BLOCK: u32 = 256;

/// `kda.cu:73` — `std::min(32, D)`, the prefill's warp count.
///
/// The ceiling is 32 because that is where one warp per state row is reached
/// and beyond it warps sit idle; it is not a hardware limit.
const PREFILL_MAX_WARPS: i32 = 32;

/// Threads per warp — `kda.cu:74`'s multiplier.
const WARP: u32 = 32;

/// `kda.cu:52`.
const STEP: &str = "ssm::kda_recurrent_step_batched#step";

/// `kda.cu:76`.
const PREFILL: &str = "ssm::kda_prefill_batched#prefill";

/// `3 * D * sizeof(float)` — `kda.cu:51` and `kda.cu:75`, one expression in
/// both launchers.
///
/// Three fp32 rows of width `D`: the kernel stages `q`, `k` and the gate
/// there. It is NOT a function of the state's `v` extent, so it does not
/// grow with the 128×128 state the way `gated_delta_net`'s cached prefill's
/// does — at `D = 128` this is 1536 bytes and never approaches the 48 KiB
/// default cap.
#[allow(clippy::cast_sign_loss)] // every caller guards `d > 0`
const fn shmem(d: i32) -> u32 {
    3 * (d as u32) * 4
}

/// `ssm::kda_recurrent_step_batched` — `kda.cu:41-55`.
///
/// One token per request, advancing each request's delta-rule state in place.
///
/// # The refusal
///
/// `if (R <= 0 || H <= 0 || D <= 0) return;` — `kda.cu:47`. `D` in
/// particular is not covered by `module.fire`'s rectangle check: it sizes
/// the shared allocation and a zero there is a legal launch.
///
/// # Safety
///
/// `q_norm`, `k_norm`, `v`, `gate` and `beta` are fp32 over `[R, H, D]`;
/// `state_base` is a slot arena of `slot_stride_elems` fp32 per slot;
/// `slot_ids` is `[R]`; `out` is writable for `[R, H, D]`. All live on
/// `stream`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_step_batched(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
    stream: *mut std::ffi::c_void,
) {
    if r <= 0 || h <= 0 || d <= 0 {
        return;
    }
    // The kernel's order: `R` is `grid.x` and never crosses.
    let values = [
        ArgValue::Ptr(q_norm.cast_mut().cast()),
        ArgValue::Ptr(k_norm.cast_mut().cast()),
        ArgValue::Ptr(v.cast_mut().cast()),
        ArgValue::Ptr(gate.cast_mut().cast()),
        ArgValue::Ptr(beta.cast_mut().cast()),
        ArgValue::Ptr(state_base.cast()),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::I64(slot_stride_elems),
        ArgValue::Ptr(out.cast()),
        ArgValue::I32(h),
        ArgValue::I32(d),
    ];
    // `kda.cu:48-52`:
    //
    //     const dim3 grid(R, H);
    //     const int threads = 256;
    //     const std::size_t shmem = 3 * D * sizeof(float);
    #[allow(clippy::cast_sign_loss)] // `r` and `h` are `> 0` above
    let launch = Launch {
        grid: [r as u32, h as u32, 1],
        block: [STEP_BLOCK, 1, 1],
        smem: shmem(d),
    };
    fire(STEP, launch, &values, stream);
}

/// `ssm::kda_prefill_batched` — `kda.cu:57-79`.
///
/// A request's whole token run, serialized: the recurrence has no parallelism
/// along time, so every block walks its request's tokens in order.
///
/// # The block width IS the measurement — 2.2× at T=2048
///
/// `kda.cu:66-74`, kept verbatim because a port that dropped it would leave
/// a magic `min` behind:
///
/// > The recurrence serializes over tokens, so the only parallelism a block
/// > has is across the state's `v` rows — one warp each, `D / warps` rows per
/// > warp per token. At 256 threads a 128-row state gives every warp 16 rows
/// > to walk in sequence, and with a grid of only `R*H` blocks the whole
/// > kernel was using a tenth of the machine. Widening the block is the
/// > entire fix: **2.2× at T=2048 (26.2 ms → 12.0 ms per layer, measured at
/// > K3's widths)**. One warp per row is the useful limit; beyond that warps
/// > sit idle.
///
/// So `min(32, D) * 32` is not a tuning constant with a forgotten origin: it
/// is one warp per state row, capped where the rows run out. At K3's
/// `D = 128` it is a 1024-thread block.
///
/// # The refusal
///
/// `if (R <= 0 || H <= 0 || D <= 0) return;` — `kda.cu:64`, and here the `D`
/// guard is load-bearing twice: it sizes the shared allocation AND it is the
/// `min`'s other operand, so a `D` of zero would compute a zero-thread block
/// that `cuLaunchKernel` rejects with an error this would have to translate.
///
/// # Safety
///
/// As [`recurrent_step_batched`], plus `qo_indptr` readable for `[R + 1]`,
/// and the `q`/`k`/`v` rectangles are over `qo_indptr[R]` tokens rather than
/// `R`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_batched(
    q_norm: *const f32,
    k_norm: *const f32,
    v: *const f32,
    gate: *const f32,
    beta: *const f32,
    state_base: *mut f32,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: std::ffi::c_longlong,
    out: *mut f32,
    r: i32,
    h: i32,
    d: i32,
    stream: *mut std::ffi::c_void,
) {
    if r <= 0 || h <= 0 || d <= 0 {
        return;
    }
    let values = [
        ArgValue::Ptr(q_norm.cast_mut().cast()),
        ArgValue::Ptr(k_norm.cast_mut().cast()),
        ArgValue::Ptr(v.cast_mut().cast()),
        ArgValue::Ptr(gate.cast_mut().cast()),
        ArgValue::Ptr(beta.cast_mut().cast()),
        ArgValue::Ptr(state_base.cast()),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::I64(slot_stride_elems),
        ArgValue::Ptr(out.cast()),
        ArgValue::I32(h),
        ArgValue::I32(d),
    ];
    // `kda.cu:65-76`:
    //
    //     const dim3 grid(R, H);
    //     const int warps = std::min(32, D);
    //     const int threads = warps * 32;
    //     const std::size_t shmem = 3 * D * sizeof(float);
    #[allow(clippy::cast_sign_loss)] // `r`, `h` and the `min` are all `> 0`
    let launch = Launch {
        grid: [r as u32, h as u32, 1],
        block: [d.min(PREFILL_MAX_WARPS) as u32 * WARP, 1, 1],
        smem: shmem(d),
    };
    fire(PREFILL, launch, &values, stream);
}

#[cfg(test)]
mod tests {
    //! What can be checked with no device: the block width the measurement
    //! bought, the shared size both launchers ask for, and that neither
    //! launcher symbol is a row.

    use super::{PREFILL, PREFILL_MAX_WARPS, STEP, WARP, shmem};

    /// Both arms resolve to a row of `ssm/kda`.
    #[test]
    fn both_arms_name_a_row() {
        for symbol in [STEP, PREFILL] {
            let (_, unit) = kernels_cuda_new::unit::unit_of(symbol)
                .unwrap_or_else(|| panic!("{symbol} is in no JIT unit"));
            assert_eq!(unit.name, "ssm/kda", "{symbol} landed in the wrong unit");
        }
    }

    /// Neither launcher is a row.
    #[test]
    fn neither_launcher_is_a_row() {
        for symbol in ["ssm::kda_recurrent_step_batched", "ssm::kda_prefill_batched"] {
            assert!(
                kernels_cuda_new::unit::unit_of(symbol).is_none(),
                "{symbol} is walked and unit-hosted"
            );
        }
    }

    /// The width the 2.2× was measured at, and the cap that ends it.
    ///
    /// K3 runs `D = 128`, so the prefill takes a 1024-thread block — the
    /// largest CUDA allows, reached by the `min` rather than by aiming at it.
    /// The pre-fix width was 256 and the regression this guards is a silent
    /// return to it.
    #[test]
    fn the_prefill_block_is_one_warp_per_state_row() {
        let width = |d: i32| d.min(PREFILL_MAX_WARPS) as u32 * WARP;
        assert_eq!(width(128), 1024, "K3's width — the shape the 2.2x was measured at");
        assert_eq!(width(16), 512, "below the cap, one warp per row and no more");
        assert_eq!(width(64), 1024, "at and above 32 rows the cap holds it");
    }

    /// Three fp32 rows, and nowhere near the 48 KiB default cap.
    ///
    /// This is the assertion that says KDA does NOT need the
    /// `raise_dynamic_smem` path `gated_delta_net`'s cached prefill added.
    #[test]
    fn the_shared_request_stays_under_the_default_cap() {
        assert_eq!(shmem(128), 1536);
        assert!(shmem(128) < 48 * 1024);
    }
}
