//! `ssm/causal_conv1d.cu`'s prefill launcher, in Rust.
//!
//! One symbol, two `__global__`s, and a host `if` on the request count.
//! `kernels-cuda/csrc/src/ssm/causal_conv1d.cu:52-90` was the whole program;
//! this is that program with the `<<<>>>` replaced by
//! [`crate::fire::hand::fire`] and every constant carrying the line it came
//! from.
//!
//! # The other launcher in that file is NOT here
//!
//! `causal_conv1d_update_batched_bf16` is routed —
//! `device::JIT_DISPATCHED` names it, the shim emits no entry, and
//! `families::ssm`'s `gdn_conv_update` row states its geometry as
//! [`kernels::LaunchRule::SplitPacked`]. A decode step advances one conv
//! window per request and that IS a rule; a prefill folds a variable-length
//! token run into the window and picks a grid from how many requests there
//! are, and that is not.
//!
//! # The `if` is a switch between two kernels, not two speeds
//!
//! §30's precedent for GDN's SMEM step — two arms measured byte-identical,
//! so the selector was a deletion rather than a port — was checked here and
//! does not apply. The arms index differently:
//!
//! ```text
//! causal_conv1d_prefill_batched              c = blockIdx.x                      (.cuh:225)
//! causal_conv1d_prefill_batched_channel_tile c = blockIdx.x * blockDim.x + tid    (.cuh:310)
//! ```
//!
//! Each is correct only under its own grid. There is no shape at which they
//! agree while both being launched as written, so there is nothing to
//! measure and both kernels stay.

use kernels_cuda_new::runtime::{ArgValue, Launch};

use crate::fire::hand::fire;

/// `causal_conv1d.cu:64` — `constexpr int TILE = 128;`, the channel tile's
/// block AND the divisor of its `grid.x`. One number, so it is one constant.
const TILE: u32 = 128;

/// `causal_conv1d.cu:78` — `constexpr int BLOCK = 64;`, the per-channel
/// arm's block. A block per channel, so the width is a thread budget for the
/// K-tap loop rather than a cut of any axis.
const PER_CHANNEL_BLOCK: u32 = 64;

/// `causal_conv1d.cu:65` — the threshold, and the only thing the host
/// decides.
///
/// At eight requests or more the channel-tiled kernel amortises its wider
/// block: `ceil(C / 128) * R` blocks against `C * R`, which at a 4096-channel
/// conv is 32R blocks rather than 4096R, each doing 128 channels' work. Below
/// that the grid is too short to fill the machine and a block per channel
/// keeps occupancy — `C * R` blocks of 64 threads, where the tiled form would
/// launch 32R blocks and leave most SMs idle.
const CHANNEL_TILE_FROM: i32 = 8;

/// The channel-tiled arm's row — `causal_conv1d.cu:67`.
const CHANNEL_TILE: &str = "ssm::causal_conv1d_prefill_batched_bf16#channel_tile";

/// The per-channel arm's row — `causal_conv1d.cu:81`.
const PER_CHANNEL: &str = "ssm::causal_conv1d_prefill_batched_bf16#per_channel";

/// `ssm::causal_conv1d_prefill_batched_bf16` — the whole launcher.
///
/// Folds each request's token run through the depthwise conv and leaves the
/// trailing `K`-tap window in that request's conv-state slot, found through
/// `slot_ids[r]` in a paged arena.
///
/// # The refusal
///
/// `if (R <= 0 || C <= 0 || K <= 0) return;` — `causal_conv1d.cu:63`, kept as
/// an early return and NOT as a fallback. A zero extent launches nothing and
/// `cuLaunchKernel` reports success for it, so the C++ refused before the
/// launch and so does this. `module.fire` would refuse the rectangle too, but
/// only for `R` and `C`: a `K` of zero produces a legal grid and a kernel
/// that reads a zero-tap filter.
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the kernel
/// reads, on `stream`: `x` and `y` are `[qo_indptr[R], C]` bf16,
/// `weight` is `[C, K]` bf16, `bias` is `[C]` bf16 or null,
/// `state_out_base` is a slot arena of `slot_stride_elems` per slot,
/// `slot_ids` is `[R]`, `qo_indptr` is `[R + 1]`, and `commit_len` and
/// `write_state_mask` are `[R]` or null. The same assertion the caller made
/// when it handed these to a C++ launcher.
#[allow(clippy::too_many_arguments)]
pub unsafe fn prefill_batched_bf16(
    x: *const std::ffi::c_void,
    weight: *const std::ffi::c_void,
    bias: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    state_out_base: *mut std::ffi::c_void,
    slot_ids: *const i32,
    qo_indptr: *const u32,
    slot_stride_elems: std::ffi::c_longlong,
    r: i32,
    c: i32,
    k: i32,
    stream: *mut std::ffi::c_void,
    write_state: bool,
    commit_len: *const i32,
    write_state_mask: *const u8,
) {
    if r <= 0 || c <= 0 || k <= 0 {
        return;
    }
    // The kernel's parameter order, which is NOT the launcher's: `R` is
    // `grid.y` and never reaches the device, and the mask precedes the commit
    // lengths. `Args::bind` checks this list against the row, so a
    // transposition is a refusal here rather than a shifted cell at the
    // kernel.
    let values = [
        ArgValue::Ptr(x.cast_mut()),
        ArgValue::Ptr(weight.cast_mut()),
        ArgValue::Ptr(bias.cast_mut()),
        ArgValue::Ptr(y),
        ArgValue::Ptr(state_out_base),
        ArgValue::Ptr(slot_ids.cast_mut().cast()),
        ArgValue::Ptr(qo_indptr.cast_mut().cast()),
        ArgValue::I64(slot_stride_elems),
        ArgValue::I32(c),
        ArgValue::I32(k),
        ArgValue::Bool(write_state),
        ArgValue::Ptr(write_state_mask.cast_mut().cast()),
        ArgValue::Ptr(commit_len.cast_mut().cast()),
    ];
    #[allow(clippy::cast_sign_loss)] // `r`, `c` and `k` are `> 0` above
    let (symbol, launch) = if r >= CHANNEL_TILE_FROM {
        // `causal_conv1d.cu:65-67`:
        //
        //     dim3 grid((C + TILE - 1) / TILE, R);
        //     dim3 block(TILE);
        (
            CHANNEL_TILE,
            Launch {
                grid: [(c as u32).div_ceil(TILE), r as u32, 1],
                block: [TILE, 1, 1],
                smem: 0,
            },
        )
    } else {
        // `causal_conv1d.cu:79-81`:
        //
        //     dim3 grid(C, R);
        //     dim3 block(BLOCK);
        (
            PER_CHANNEL,
            Launch { grid: [c as u32, r as u32, 1], block: [PER_CHANNEL_BLOCK, 1, 1], smem: 0 },
        )
    };
    fire(symbol, launch, &values, stream);
}

#[cfg(test)]
mod tests {
    //! What can be checked with no device: that the threshold picks the arm
    //! `causal_conv1d.cu:65` picked, and that the two rows exist under the
    //! names this file fires.

    use super::{CHANNEL_TILE, CHANNEL_TILE_FROM, PER_CHANNEL};

    /// Both arms resolve to a row of `ssm/causal_conv1d`.
    ///
    /// The failure this catches is a `#suffix` typo, which `hand::fire`
    /// reports only at the first fire — on a serving process, the first
    /// prefill of some request an hour in.
    #[test]
    fn both_arms_name_a_row() {
        for symbol in [CHANNEL_TILE, PER_CHANNEL] {
            let (_, unit) = kernels_cuda_new::unit::unit_of(symbol)
                .unwrap_or_else(|| panic!("{symbol} is in no JIT unit"));
            assert_eq!(unit.name, "ssm/causal_conv1d", "{symbol} landed in the wrong unit");
        }
    }

    /// The threshold is the launcher's, inclusive on the tiled side.
    ///
    /// `if (R >= 8)` — so eight requests takes the tile and seven does not.
    /// An off-by-one here is not a wrong answer; it is the slower arm at one
    /// shape, which is exactly the kind of drift nothing reports.
    #[test]
    fn eight_requests_take_the_tile() {
        assert_eq!(CHANNEL_TILE_FROM, 8);
    }

    /// The launcher's symbol is NOT a row.
    ///
    /// `execution::WALKED` states `ssm::causal_conv1d_prefill_batched_bf16`
    /// and `a_walk_is_only_a_walk` asserts a walked symbol is not
    /// unit-hosted. This is the same assertion from the driver's side, where
    /// the `#` suffixes are actually spelled.
    #[test]
    fn the_launcher_is_not_a_row() {
        assert!(
            kernels_cuda_new::unit::unit_of("ssm::causal_conv1d_prefill_batched_bf16").is_none(),
            "the walked symbol is unit-hosted, so a trace could fire an arm directly"
        );
    }
}
