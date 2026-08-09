//! The MoE launchers the driver owns, in Rust.
//!
//! Every kernel fired here is compiled by NVRTC from `.cuh` device text in
//! `kernels-cuda-new/csrc/src/moe/`. What used to sit in
//! `kernels-cuda/csrc/src/moe/*.cu` beside a `<<<>>>` — a support predicate,
//! a grid no `LaunchRule` states, a tile constant — is host code, and host
//! code is this crate's.
//!
//! # The rule these launchers are written to
//!
//! The owner's constraint: **the model compiler must not be able to tell
//! whether a symbol is cuBLAS or a JIT'd kernel.** `KernelSig` is unchanged.
//! `moe::moe_grouped_gemm_bf16` is a row of `table::moe` exactly as it was;
//! `execution::RUST_SERVED` names it, so `abi::emit_c_shim` drops its shim
//! entry and `abi::emit_dispatch` writes an arm calling
//! `crate::bind::service::moe_moe_grouped_gemm_bf16`. Nothing above this
//! crate moved.
//!
//! # Every number here is a citation
//!
//! The constants below each name the line of the deleted `.cu` they came
//! from, and `families::moe`'s `MOE_GROUPED_GEMM_SIGS` carries the same
//! `<<<>>>` quoted whole. A geometry that is derived rather than cited is
//! how a kernel gets fired on a grid nobody measured — see
//! `fire/attn_score.rs`, which says the same thing about its `64`.
//!
//! # A refusal is never a fallback
//!
//! `moe_grouped_gemm_bf16_supported` is a SHAPE test, and a shape it refuses
//! is one this kernel cannot compute — not one it computes slowly. The Rust
//! returns [`Grouped::Declined`] and the caller keeps whatever path it had.
//! A broken JIT unit is the opposite case and panics: that is drift between
//! this driver and its kernel table, and answering it with a decline would
//! report a missing kernel as an unsupported shape.

use kernels_cuda_new::runtime::{ArgValue, Args, Launch, Stream, cache};

/// The M fragment the mma path is built on — `moe_grouped_gemm.cuh:74`'s
/// `constexpr int kFrag = 16`.
///
/// Load-bearing twice, which is why it is one constant: the support test
/// requires `M == kFrag` exactly (the kernel computes one fragment of rows
/// per block and has no tail path) and requires `K % kFrag == 0` (the
/// mainloop steps K by a fragment and never checks a remainder).
const FRAG: i32 = 16;

/// Warps per block — `moe_grouped_gemm.cuh:76`'s `constexpr int kGemmWarps
/// = 4`. The launch was `<<<grid, device::kGemmWarps * 32, ...>>>`, so the
/// block is 128 threads, and the header's `__launch_bounds__(kGemmWarps *
/// 32)` states the same number to the compiler.
const GEMM_WARPS: u32 = 4;

/// The N tile — `moe_grouped_gemm.cuh:78`'s `constexpr int kNTile = kFrag *
/// kGemmWarps`, so 64.
///
/// Spelled as the product rather than as `64` because that is how the header
/// spells it: each of the four warps owns one 16-wide fragment of the N
/// axis. `grid.x` is `N / kNTile` with no rounding, which is why the support
/// test demands `N % kNTile == 0` — a rounded-up grid would have a fourth
/// warp writing past the row and a rounded-down one would leave the tail
/// unwritten, and the kernel bounds-checks neither.
const N_TILE: i32 = FRAG * GEMM_WARPS as i32;

/// Above this K, cuBLAS's tuned mainloop beats the early exit —
/// `moe_grouped_gemm.cu:16`'s `constexpr int kShortK = 512`.
///
/// A HOST constant because the decision is the launcher's: the kernel is
/// correct at any K and this is the bound at which firing it stops paying.
/// Measured on Qwen3.6-35B-A3B tp2 decode against cuBLAS —
/// `down K=256 7.94 -> 5.91 ms` taken, `gate_up K=2048 11.08 -> 11.98` left
/// on cuBLAS (`moe_grouped_gemm.cu:19-21`).
const SHORT_K: i32 = 512;

/// The JIT row's symbol, which is deliberately NOT the stated one.
///
/// `table::moe`'s row is `moe::moe_grouped_gemm_bf16` and this is
/// `moe::moe_grouped_gemm_wmma_bf16`, because `execution`'s
/// `a_walk_is_only_a_walk` asserts a walked symbol has no unit: a `Walk` is
/// a host program and `fire` takes a `Dims` that has no meaning for one. A
/// walk may drive JIT'd kernels; it may not be one. Same split as
/// `fire/lm_head_argmax.rs`'s `GEMV_SYMBOL`, whose walk is
/// `sample::lm_head_gemv_argmax_int8` and whose row is `..._int8_bf16`.
const SYMBOL: &str = "moe::moe_grouped_gemm_wmma_bf16";

/// What [`moe_grouped_gemm_bf16`] did.
///
/// A two-state answer rather than a `bool`, for `fire/gemv.rs`'s reason: the
/// C++ returned `void` and swallowed its own refusal, so a caller could not
/// distinguish "ran" from "declined and wrote nothing into `c`". The
/// declines are the whole interface of this kernel — it is an *alternative*
/// to a cuBLAS grouped GEMM, not a replacement for one.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Grouped {
    /// The kernel was fired on the stream.
    Launched,
    /// Nothing was launched, and `c` is untouched. The caller must run the
    /// general path.
    Declined(Decline),
}

/// Why [`moe_grouped_gemm_bf16`] declined.
///
/// Each variant is one conjunct of `moe_grouped_gemm_bf16_supported`
/// (`moe/moe_grouped_gemm.cu:18-24`) or one of the launcher's two emptiness
/// guards, kept apart so a caller that logs one can say which shape it was.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decline {
    /// `max_blocks <= 0` — the padded batch is empty, so there is nothing to
    /// launch over. `moe_grouped_gemm.cu:37`.
    NoBlocks,
    /// `M != kFrag`. The kernel computes exactly one 16-row fragment per
    /// block and has no tail path.
    RowsNotOneFragment,
    /// `N <= 0 || K <= 0` — an empty rectangle.
    Empty,
    /// `K > kShortK`. Not a correctness bound: see [`SHORT_K`].
    ReductionTooLong,
    /// `N % kNTile != 0`. See [`N_TILE`].
    WidthNotTiled,
    /// `K % kFrag != 0`. The mainloop steps K by a fragment and never checks
    /// a remainder.
    ReductionNotFragmented,
}

/// Whether this kernel can compute the rectangle at all, and whether it is
/// worth firing.
///
/// `moe/moe_grouped_gemm.cu:18-24`, one conjunct at a time so the answer
/// says which. The order is the C++'s: `M == kFrag && N > 0 && K > 0 &&
/// K <= kShortK && (N % kNTile) == 0 && (K % kFrag) == 0`.
#[must_use]
pub const fn supported(m: i32, n: i32, k: i32) -> Result<(), Decline> {
    if m != FRAG {
        return Err(Decline::RowsNotOneFragment);
    }
    if n <= 0 || k <= 0 {
        return Err(Decline::Empty);
    }
    if k > SHORT_K {
        return Err(Decline::ReductionTooLong);
    }
    if n % N_TILE != 0 {
        return Err(Decline::WidthNotTiled);
    }
    if k % FRAG != 0 {
        return Err(Decline::ReductionNotFragmented);
    }
    Ok(())
}

/// The short-K grouped GEMM: one launch over a padded, expert-sorted batch.
///
/// `a` is `[aligned_rows, K]`, `weight_base` the per-expert weight bank,
/// `c` is `[aligned_rows, N]`, and `expert_ids[b]` names the expert of
/// padded block `b` — negative for a padding block, which the kernel exits
/// on immediately (`moe_grouped_gemm.cuh:129`, *"padding block: the whole
/// point of this kernel"*).
///
/// # The grid
///
/// `moe/moe_grouped_gemm.cu:40-41`, verbatim:
///
/// ```text
/// const dim3 grid(N / device::kNTile, max_blocks);
/// device::moe_grouped_gemm<device::bf16><<<grid, device::kGemmWarps * 32, 0, stream>>>(
/// ```
///
/// `max_blocks` is a host-side bound on the padded batch, not an extent of
/// any operand, which is why no launch rule states this grid and why the row
/// is `LaunchRule::Unstated`.
///
/// # Panics
///
/// When the symbol is in no JIT unit, when the unit will not compile or
/// load, or when the argument list disagrees with the row's signature. Every
/// one of those is drift between this driver and its kernel table, and none
/// of them may be answered with [`Grouped::Declined`] — see the module
/// header.
///
/// # Safety
///
/// The four pointers must be device allocations of the shapes above, live on
/// `stream` until the launch completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn moe_grouped_gemm_bf16(
    a: *const std::ffi::c_void,
    weight_base: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
    expert_ids: *const i32,
    max_blocks: i32,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut std::ffi::c_void,
) -> Grouped {
    if max_blocks <= 0 {
        return Grouped::Declined(Decline::NoBlocks);
    }
    if let Err(why) = supported(m, n, k) {
        return Grouped::Declined(why);
    }
    let values = [
        ArgValue::Ptr(a.cast_mut()),
        ArgValue::Ptr(weight_base.cast_mut()),
        ArgValue::Ptr(c),
        ArgValue::Ptr(expert_ids.cast_mut().cast()),
        ArgValue::I32(n),
        ArgValue::I32(k),
    ];
    // `dim3 grid(N / kNTile, max_blocks)` — `moe_grouped_gemm.cu:40`. The
    // division is exact: `supported` rejected every `N` for which it is not.
    #[allow(clippy::cast_sign_loss)] // both are `> 0` above
    let launch = Launch {
        grid: [(n / N_TILE) as u32, max_blocks as u32, 1],
        // `device::kGemmWarps * 32` — `moe_grouped_gemm.cu:41`, and the
        // header's `__launch_bounds__` states the same product.
        block: [GEMM_WARPS * 32, 1, 1],
        // No dynamic shared memory: the mma fragments are registers and the
        // staging tile is a static `__shared__` array in the kernel.
        smem: 0,
    };
    fire(SYMBOL, launch, &values, stream);
    Grouped::Launched
}

/// Resolve a row through the JIT table, bind the operands, launch.
///
/// The same helper `fire/gemv.rs` carries, for the same reason: `Args::bind`
/// checks `values` against the row's signature, so a drift between the list
/// built above and `families::moe`'s `MOE_GROUPED_GEMM_SIGS` is a refusal
/// here rather than a shifted argument at the kernel.
///
/// # Panics
///
/// Every failure on this path is drift between this driver and its kernel
/// table, or a unit that will not compile.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // the stream is borrowed, never read
fn fire(symbol: &'static str, launch: Launch, values: &[ArgValue], stream: *mut std::ffi::c_void) {
    let Some((index, unit)) = kernels_cuda_new::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    let mut args = match Args::bind(sig, values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller holds the fire's stream live across the launch —
    // the same assertion it made when it handed the stream to a C++ launcher
    // that put it in a `<<<>>>`.
    let stream = unsafe { Stream::from_runtime(stream) };
    if let Err(why) = module.fire(sig, launch, &mut args, stream) {
        panic!("{symbol}: {why}");
    }
}

#[cfg(test)]
mod tests {
    //! What can be checked without a device: that the support predicate says
    //! what `moe/moe_grouped_gemm.cu:18-24` said.

    use super::{Decline, supported};

    /// The shipping shapes from the launcher's own measurement table.
    #[test]
    fn the_measured_shapes_answer_as_they_were_measured() {
        // `down K=256`, taken: M is one fragment, N a multiple of 64.
        assert_eq!(supported(16, 2048, 256), Ok(()));
        // `gate_up K=2048`, left on cuBLAS by the K bound and nothing else.
        assert_eq!(supported(16, 2048, 2048), Err(Decline::ReductionTooLong));
    }

    /// Each conjunct refuses on its own, and says which.
    #[test]
    fn every_conjunct_is_its_own_decline() {
        assert_eq!(supported(32, 2048, 256), Err(Decline::RowsNotOneFragment));
        assert_eq!(supported(16, 0, 256), Err(Decline::Empty));
        assert_eq!(supported(16, 2048, 0), Err(Decline::Empty));
        assert_eq!(supported(16, 100, 256), Err(Decline::WidthNotTiled));
        assert_eq!(supported(16, 2048, 24), Err(Decline::ReductionNotFragmented));
    }
}
