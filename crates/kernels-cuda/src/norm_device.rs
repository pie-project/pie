//! Tier A: the six AltUp auxiliary kernels, stated as launches rather than
//! as calls.
//!
//! Each row here has a twin in [`crate::norm`], and the diff between the two
//! is the whole experiment. The twin describes a C++ *function*: what to pass
//! it, in the order its header declares. A row here describes a *launch*: the
//! kernel's own arguments, and the rule that turns a rectangle into a grid.
//!
//! Three things fall out of the shorter list, and none of them was designed:
//!
//! * **`stream` is gone.** It was never an operand. A stream is where a
//!   launch goes, and the only reason it appeared in an argument list is that
//!   a C++ launcher is a function and a function's arguments are the only
//!   channel it has. `cuLaunchKernel` takes it in the position it belongs in.
//! * **Four of the five `T`s are gone**, because `LaunchRule::Rms` and
//!   `LaunchRule::ElementwiseRows` put one block on each row: the kernel's
//!   `blockIdx.x < T` was checking the launcher's arithmetic, not the fire.
//!   `mean_streams` kept its `T` because there it is a stride -- see
//!   `altup_aux_device.cuh`.
//! * **No row needed a new rule.** All six land on four rules the Metal
//!   tables already state, which is the claim [`kernels::LaunchRule`] makes
//!   about itself -- *"a new kernel that launches like an existing one costs
//!   nothing"* -- tested for the first time against a second backend.
//!
//! # Why this is a separate table
//!
//! It is not the end state. When the C++ launcher for a row is deleted, its
//! row in [`crate::norm`] loses `stream`, gains `launch`, and this file goes
//! away. Two tables exist only while both paths must run, because the
//! measurement is a numeric A/B between them and an A/B needs both arms.
//!
//! Deliberately absent from [`crate::KERNELS`] for the same reason: a symbol
//! `model-compiler` can state must have exactly one contract, and until the
//! shim path retires, the twin is it.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

/// AltUp's epsilon, which is the ALGORITHM's and not the model's -- the same
/// constant, and the same argument, as [`crate::norm`]'s.
const ALTUP_EPS: f32 = 1e-5;

/// The `pie_g_*` entry points `csrc/src/norm/altup_aux_entry.cu` defines.
#[rustfmt::skip]
pub static ENTRIES: &[KernelSig] = &[
    // One block per row, and the row width is read by a stride loop, so
    // `h` is the only extent the kernel sees. `Rms` on CUDA means what it
    // means on Metal -- a row-wise reduction, one group per row -- and the
    // arithmetic that picks the block width is `driver-cuda`'s, beside the
    // sentence that explains it.
    kernel!(compute_rms "norm::compute_rms_bf16",
        file = Some("norm/altup_aux_entry.cu"),
        launch = LaunchRule::Rms,
        operands = operands![
            reference: Buf <- Source::In(0),
            target_rms_out: F32sMut <- Source::Out(0),
            h: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Lit(Lit::F32(ALTUP_EPS)),
        ]),
    kernel!(magnitude_rescale "norm::magnitude_rescale_bf16",
        file = Some("norm/altup_aux_entry.cu"),
        launch = LaunchRule::Rms,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            target_rms: F32s <- Source::In(1),
            h: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Lit(Lit::F32(ALTUP_EPS)),
        ]),
    // `t` SURVIVES here and nowhere else: `streams` is `[K, T, H]`, so the
    // k-th plane begins at `k * T * H` and the kernel cannot address its
    // input without it. The grid still covers the rows -- what is passed is
    // a stride that happens to equal an extent, which is exactly the pair
    // the old signatures could not tell apart.
    kernel!(mean_streams "norm::mean_streams_bf16",
        file = Some("norm/altup_aux_entry.cu"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            streams: Buf <- Source::In(0),
            out: BufMut <- Source::Out(0),
            k: I32 <- Source::CtxNonZero("altup_streams"),
            t: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(0),
        ]),
    // `RouteRows` -- one block per row, as wide as the row. The row is
    // `K*K` wide and `K` is its integer square root, which is the same
    // `Source` the twin states; nothing about how an operand is SOURCED
    // changes in Tier A, only which operands there are.
    kernel!(altup_unpack_predict_coefs "norm::altup_unpack_predict_coefs",
        file = Some("norm/altup_aux_entry.cu"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp32: F32sMut <- Source::Out(0),
            k: I32 <- Source::Isqrt(&Source::Width(&Source::In(0))),
        ]),
    kernel!(altup_unpack_correct_coefs "norm::altup_unpack_correct_coefs",
        file = Some("norm/altup_aux_entry.cu"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp32: F32sMut <- Source::Out(0),
            k: I32 <- Source::InWidth(0),
        ]),
    // The flat one: rows stack, so the extent is elements and the guard is
    // the kernel's own. `numel` is not geometry the rule can recover --
    // `Elementwise` reads `rows * width` and this operand says the same
    // number -- so it stays an argument.
    kernel!(tanh "norm::tanh_bf16",
        file = Some("norm/altup_aux_entry.cu"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            numel: I32 <- Source::OutElements(0),
        ]),
];

#[cfg(test)]
mod tests {
    use super::ENTRIES;
    use kernels::LaunchRule;

    /// Every Tier A row states a rule. `Unstated` is what a row says when it
    /// has not been ported, and a row in THIS table that said it would be
    /// launched by a driver that has nothing to launch it with.
    #[test]
    fn every_entry_states_its_launch() {
        for k in ENTRIES {
            assert_ne!(k.launch, LaunchRule::Unstated, "{} states no rule", k.symbol);
        }
    }

    /// No row here needed a rule the Metal tables had not already stated.
    ///
    /// This is the pilot's headline and it is cheap to keep honest: if a
    /// later port adds a variant to serve one CUDA kernel, this fails and
    /// the claim gets re-measured rather than repeated.
    #[test]
    fn the_pilot_added_no_launch_rules() {
        const REUSED: &[LaunchRule] = &[
            LaunchRule::Rms,
            LaunchRule::ElementwiseRows,
            LaunchRule::RouteRows,
            LaunchRule::Elementwise,
        ];
        for k in ENTRIES {
            assert!(
                REUSED.contains(&k.launch),
                "{} states {:?}, which is not one of the rules Metal already had",
                k.symbol,
                k.launch
            );
        }
    }

    /// A stream is not an operand, and a Tier A row may not say it is.
    #[test]
    fn no_entry_takes_a_stream() {
        for k in ENTRIES {
            assert!(
                k.operands.iter().all(|o| o.ty != kernels::Ty::Stream),
                "{} takes a stream as an operand",
                k.symbol
            );
        }
    }

    /// Each row is shorter than its twin, which is the deletion the
    /// experiment claims. Stated as a total so the number is in the test
    /// output rather than in a commit message.
    ///
    /// Thirty-one operands become twenty-one. The ten are the six streams
    /// and the four extents the rules recover -- which is to say every one
    /// of them was a fact the table already held, spelled a second time
    /// because a C++ function had no other way to receive it.
    #[test]
    fn tier_a_rows_are_shorter_than_their_twins() {
        let mine: usize = ENTRIES.iter().map(|k| k.operands.len()).sum();
        let twins: usize = crate::norm::KERNELS
            .iter()
            .filter(|t| ENTRIES.iter().any(|k| k.symbol == t.symbol))
            .map(|k| k.operands.len())
            .sum();
        assert_eq!(ENTRIES.len(), 6);
        assert_eq!((twins, mine), (31, 21));
    }
}
