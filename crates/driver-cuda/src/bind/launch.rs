//! Tier A: how a rectangle becomes a CUDA launch.
//!
//! The Metal driver has had this file for as long as it has had a generic
//! executor -- `driver-metal/src/lowering/launch.rs`, which turns a
//! [`kernels::LaunchRule`] into a thread grid. CUDA has not, because on CUDA
//! the grid was computed in C++, inside the launcher, one `(H + BLOCK - 1) /
//! BLOCK` per kernel. The two backends were not disagreeing about
//! arithmetic; only one of them had written the arithmetic down anywhere a
//! table could point at.
//!
//! So the rule vocabulary is the shared crate's and the arithmetic is this
//! backend's, which is the split [`kernels::LaunchRule`]'s own doc describes:
//! *"This is data. The arithmetic each variant names stays in the driver,
//! beside the doc comment that explains it."*
//!
//! # One axis Metal does not have
//!
//! [`Launch`] carries `smem`, and Metal's does not. Dynamic shared memory is
//! a launch parameter on CUDA and a threadgroup-memory binding on Metal, so
//! the rule has to produce it here or the reduction kernels cannot run. It is
//! the only structural difference the port found, and it is why [`Launch`] is
//! this crate's type rather than one lifted from `kernels`.
//!
//! # What is ported
//!
//! The four rules `kernels_cuda::norm_device::ENTRIES` states. Every other
//! variant answers [`Ungeometric::Unported`] rather than a guess: a rule this
//! backend has not written the arithmetic for is not a rule with a default,
//! and the whole reason the table can be trusted is that a driver refuses
//! what it cannot state.

use kernels::LaunchRule as Rule;

/// The fire-time quantities a CUDA launch rule may read.
///
/// A subset of the ten fields `driver-metal`'s `Dims` carries -- the head and
/// expert counts are not read by any rule Tier A ports, and a field no rule
/// reads is a field whose meaning nothing checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims {
    /// Rows the rectangle covers.
    pub rows: u32,
    /// Elements per row of the launch's last widthed operand — its output.
    pub width: u32,
    /// Elements per row of its first widthed operand — its input. Read by
    /// the rules that size on what a launch READS, which is what a statement
    /// that unpacks one buffer into a wider one needs.
    pub in_width: u32,
}

/// A launch, in CUDA's spelling: blocks, threads per block, dynamic shared
/// bytes.
///
/// `grid` is BLOCKS and `block` is THREADS, which is the one place a reader
/// coming from the Metal side has to stop: `dispatchThreads` takes a total
/// thread count and `cuLaunchKernel` takes a block count, so the same rule
/// produces numbers that differ by a factor of the block width. Writing one
/// where the other is meant launches `grid.x` threads instead of `grid.x`
/// blocks — a real fire, a real result, and every row past the first
/// untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
}

/// Why a rule could not produce a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// The row states no rule, so nothing can be dispatched from it. Drift,
    /// not a runtime condition — the same meaning `Source::Unbound` has for
    /// an operand.
    Unstated,
    /// The rule is real and this backend has not ported its arithmetic.
    ///
    /// Distinct from [`Ungeometric::Unstated`] because they are different
    /// bugs: an unstated row is a table that has not been filled in, and an
    /// unported rule is a driver that has not caught up with one. Only the
    /// second is fixed here.
    Unported(Rule),
    /// A launch over an empty extent.
    ///
    /// Refused rather than clamped. A zero grid launches nothing and returns
    /// success, so a fire whose rectangle collapsed would look exactly like a
    /// fire that ran — which is the failure `program::run::launch` already
    /// refuses for the PTIR lane, for the same reason.
    Empty,
}

/// Threads per block for the pointwise passes.
///
/// 256 because that is what every launcher in `norm/altup_aux.cu` used, and
/// the port's first duty is to reproduce today's launches rather than to
/// improve them. It is a tuning constant with one reader, which is the shape
/// a tuning constant should have.
const BLOCK: u32 = 256;

/// The widest block CUDA will launch.
const MAX_BLOCK: u32 = 1024;

/// Threads per warp — the unit `block_sum`'s shared scratch is counted in.
const WARP: u32 = 32;

/// One block per row, [`BLOCK`] wide, with scratch for the warp combine.
///
/// The width is fixed rather than sized on the row because the reduction is
/// ORDER-SENSITIVE: `block_sum` folds warp by warp, so a different block
/// width sums the same values in a different order and answers with a
/// different last bit. Sizing it on `width` is the obvious improvement and it
/// is deliberately not taken here — a port that changes the arithmetic cannot
/// be checked against the arithmetic it replaced.
fn rms(rows: u32) -> Launch {
    Launch {
        grid: [rows, 1, 1],
        block: [BLOCK, 1, 1],
        // `block_sum` writes one float per warp and reads them back from
        // lane 0 of the first. Sizing this on anything but the block width
        // is a race the hardware does not report.
        smem: (BLOCK / WARP) * 4,
    }
}

/// Flat pointwise: `n` elements, [`BLOCK`] per block, rounded up.
fn elementwise(n: u32) -> Launch {
    Launch {
        grid: [n.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// Pointwise with the row on its own grid axis.
///
/// What a pass whose rows are not contiguous needs: `mean_streams` reads
/// `[K, T, H]` and writes `[T, H]`, so a flat index over the output would
/// have to be divided back into a row and a channel by the kernel. The row
/// axis is `grid.x` and the channel axis is `grid.y`, which is the same
/// shape `LaunchRule::ElementwiseRows` names on Metal.
fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [rows, width.div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// One block per row, as wide as the row, rounded up to a warp and capped.
///
/// The cap is safe only because the kernels stride: `unpack_predict_coefs`
/// walks `kk += blockDim.x`, so a block narrower than the row computes all
/// of it in several passes. Before the stride loop this cap would have
/// silently computed a prefix — see `altup_aux_device.cuh`.
fn route_rows(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [rows, 1, 1],
        block: [width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK), 1, 1],
        smem: 0,
    }
}

/// The launch `rule` produces for `dims`.
///
/// A free function rather than a method on [`Rule`] because the rule is
/// `kernels`' and the arithmetic is this backend's — the same split
/// `driver-metal` makes, so that the two can disagree about numbers without
/// disagreeing about vocabulary.
///
/// # Errors
///
/// [`Ungeometric`], and every variant of it is drift rather than a condition
/// a fire can be in.
pub fn eval(rule: Rule, dims: Dims) -> Result<Launch, Ungeometric> {
    // A rectangle covers at least one row. Zero is refused rather than
    // floored: the callers that legitimately launch a single row state one,
    // and a rectangle that collapsed to nothing is a lowering bug that a
    // floor would hide behind a kernel doing one row of work.
    if dims.rows == 0 {
        return Err(Ungeometric::Empty);
    }
    Ok(match rule {
        Rule::Unstated => return Err(Ungeometric::Unstated),
        Rule::Rms => rms(dims.rows),
        Rule::Elementwise => {
            let n = dims.rows.checked_mul(dims.width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise(n)
        }
        Rule::ElementwiseRows => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise_rows(dims.rows, dims.width)
        }
        Rule::RouteRows => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            route_rows(dims.rows, dims.width)
        }
        other => return Err(Ungeometric::Unported(other)),
    })
}

#[cfg(test)]
mod tests {
    use super::{Dims, Launch, Rule, Ungeometric, eval};

    /// gemma-3n's shape: four AltUp streams, 2048 hidden, sixteen tokens.
    const T: u32 = 16;
    const H: u32 = 2048;
    const K: u32 = 4;

    fn dims(rows: u32, width: u32) -> Dims {
        Dims { rows, width, in_width: width }
    }

    /// Every rule the Tier A table states evaluates. The table and the
    /// driver are two crates and nothing but this test makes them agree
    /// about which rules are live.
    #[test]
    fn every_tier_a_rule_is_ported() {
        for k in kernels_cuda::norm_device::ENTRIES {
            let d = dims(T, H);
            assert!(
                !matches!(eval(k.launch, d), Err(Ungeometric::Unported(_))),
                "{} states {:?}, which this driver has not ported",
                k.symbol,
                k.launch
            );
        }
    }

    /// The reduction pair reproduces `compute_rms_bf16`'s launcher exactly:
    /// `compute_rms_kernel<<<T, 256, (256 / 32) * sizeof(float), stream>>>`.
    ///
    /// Transcribed from `norm/altup_aux.cu` at the commit this pilot forked
    /// from, and it is the whole precondition of the migration — a rule that
    /// does not reproduce the launcher it replaces is a rewrite, and a
    /// rewrite cannot be A/B'd against the thing it rewrote.
    #[test]
    fn rms_reproduces_the_cpp_launcher() {
        assert_eq!(
            eval(Rule::Rms, dims(T, H)),
            Ok(Launch { grid: [T, 1, 1], block: [256, 1, 1], smem: 32 })
        );
    }

    /// `tanh_kernel<<<(numel + 255) / 256, 256, 0, stream>>>`, where the
    /// row's `numel` is `rows * width`.
    #[test]
    fn elementwise_reproduces_the_cpp_launcher() {
        let numel = T * H;
        assert_eq!(
            eval(Rule::Elementwise, dims(T, H)),
            Ok(Launch { grid: [numel.div_ceil(256), 1, 1], block: [256, 1, 1], smem: 0 })
        );
    }

    /// `mean_streams_bf16` launched `dim3(T, (H + 127) / 128)` with 128
    /// threads; the rule says 256. The kernel is a pure map guarded by
    /// `h >= H`, so the two cover the same channels and answer the same
    /// bits — what a test can hold is the COVERAGE, which is the property
    /// the guard depends on.
    #[test]
    fn elementwise_rows_covers_every_channel() {
        let l = eval(Rule::ElementwiseRows, dims(T, H)).expect("rule evaluates");
        assert_eq!(l.grid[0], T, "one block per row");
        assert!(
            l.grid[1] * l.block[0] >= H,
            "grid.y ({}) x block ({}) must cover H ({H})",
            l.grid[1],
            l.block[0]
        );
        // And not by more than one block, which is what "rounded up" means
        // and what a rule that merely over-covered would fail.
        assert!((l.grid[1] - 1) * l.block[0] < H);
    }

    /// `unpack_predict_coefs_kernel<<<T, K * K>>>` — one block per row, as
    /// wide as the row. The rule rounds up to a warp, which the C++ did not,
    /// and the stride loop is what makes the two agree: sixteen elements
    /// over thirty-two threads leaves sixteen idle rather than sixteen
    /// unwritten.
    #[test]
    fn route_rows_covers_the_row_it_is_given() {
        let predict = eval(Rule::RouteRows, dims(T, K * K)).expect("rule evaluates");
        assert_eq!(predict.grid, [T, 1, 1]);
        assert!(predict.block[0] >= K * K);
        assert_eq!(predict.block[0] % 32, 0, "a partial warp is a wasted scheduler slot");

        // A row wider than any block still gets a legal launch, because the
        // kernel strides. This is the case the pre-stride kernel could not
        // have been given at all: `<<<T, 4096>>>` is a launch failure.
        let wide = eval(Rule::RouteRows, dims(T, 4096)).expect("rule evaluates");
        assert_eq!(wide.block, [1024, 1, 1]);
    }

    /// A collapsed rectangle is refused, not floored.
    #[test]
    fn an_empty_extent_is_refused() {
        assert_eq!(eval(Rule::Rms, dims(0, H)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::ElementwiseRows, dims(T, 0)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::RouteRows, dims(T, 0)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::Elementwise, dims(T, 0)), Err(Ungeometric::Empty));
    }

    /// An unported rule says which one, and an unstated row says neither.
    #[test]
    fn the_two_refusals_are_different_sentences() {
        assert_eq!(eval(Rule::Unstated, dims(T, H)), Err(Ungeometric::Unstated));
        assert_eq!(eval(Rule::Qmv, dims(T, H)), Err(Ungeometric::Unported(Rule::Qmv)));
    }
}
