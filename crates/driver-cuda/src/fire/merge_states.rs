//! The split-KV fold, as the FA2 seam reaches it.
//!
//! The kernels, their arms, their geometry and their refusals are
//! [`kernels_cuda_new::x::cascade`] — that module's header is where the
//! `MergeStates`/`VariableLengthMergeStates` distinction, the architecture
//! argument and the occupancy note live. What is here is the seam: the plan's
//! `u64` addresses, and an answer the seven call sites cannot mistake for a
//! success.
//!
//! # Why this file still exists
//!
//! `flashinfer_fa2_dispatch`'s [`Partials`](super::flashinfer_fa2_dispatch::Partials)
//! carries device addresses as `u64` — `plan_info`'s offsets are added to a
//! workspace base and never dereferenced on the host — and a routine takes
//! pointers. That widening, and the `#[must_use]` answer below, are the
//! driver's half; everything else went down.

use std::ffi::c_void;

use kernels_cuda_new::jit::Ctx;
use kernels_cuda_new::x::Refusal;
use kernels_cuda_new::x::abi::bf16;
use kernels_cuda_new::x::cascade;

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
    Declined(Refusal),
}

impl Merged {
    /// Panic unless the fold ran, naming the caller.
    ///
    /// The six FA2 call sites all want this and none of them can carry on
    /// without the merge: the alternative is returning an attention output
    /// that is a workspace scratch buffer. `what` is the dispatch's own name
    /// so the message says which of the six, which a panic from inside the
    /// routine could not.
    ///
    /// # Panics
    ///
    /// If the fold declined. `kernels_cuda_new::x::cascade` names the shapes
    /// it will not fire.
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

/// `VariableLengthMergeStates`' operands, `cascade.cuh:687-690`.
///
/// One struct rather than nine positional arguments because five of them are
/// `u64` addresses and three are `u32` counts, and the orders that type-check
/// are not the same order. The names are `cascade.cuh`'s.
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

/// [`kernels_cuda_new::x::cascade::merge_states_varlen`], over a plan's
/// addresses.
///
/// The ragged fold, and the one the FA2 split path calls: `prefill.cuh:4350-4352`
/// and `decode.cuh:822-824` fire exactly this, in exactly this position.
///
/// # Safety
///
/// Every address in `job` must name device memory of the extent the kernel
/// reads or writes, and `stream` must outlive the launch — the same assertion
/// the caller made when it handed these pointers to a `cudaLaunchKernel`.
pub unsafe fn variable_length(job: VarLen, stream: *mut c_void) -> Merged {
    // SAFETY: the caller's contract — `stream` is live across the launch,
    // which is what a context is used for and all it is used for.
    let ctx = unsafe { Ctx::on(stream) };
    let fired = cascade::merge_states_varlen(
        &ctx,
        ptr::<bf16>(job.v),
        ptr::<f32>(job.s),
        ptr::<i32>(job.indptr),
        ptr::<bf16>(job.v_merged),
        ptr::<f32>(job.s_merged),
        job.max_seq_len,
        ptr::<u32>(job.seq_len),
        job.num_heads,
        job.head_dim,
    );
    match fired {
        Ok(()) => Merged::Launched,
        Err(why) => Merged::Declined(why),
    }
}

/// A device address as the routine wants it.
///
/// The FA2 seam carries addresses as `u64`, so this is the one place the width
/// changes, rather than five `as` casts in an argument list where a
/// transposition would not be visible.
fn ptr<T>(addr: u64) -> *mut T {
    addr as usize as *mut T
}

#[cfg(test)]
mod tests {
    use super::{Merged, VarLen};
    use kernels_cuda_new::x::Refusal;

    /// A declined fold is not a launched one.
    ///
    /// `Merged` is `#[must_use]` and two-valued, so the thing worth checking
    /// is that `launched()` and the enum agree — a `Declined` that answered
    /// `true` would make every call site's `if` a no-op.
    #[test]
    fn a_decline_does_not_read_as_a_launch() {
        assert!(Merged::Launched.launched());
        assert!(!Merged::Declined(Refusal::Empty { what: "num_heads" }).launched());
    }

    /// A decline says what it refused, so a panic message is actionable.
    ///
    /// The failure this prevents is the one `DISPATCH_HEAD_DIM`'s
    /// `throw std::invalid_argument` had in practice: an abort with no
    /// message, from which the shape had to be guessed.
    #[test]
    fn a_decline_names_what_it_refused() {
        let said = Merged::Declined(Refusal::Null { what: "indptr" });
        let Merged::Declined(why) = said else { panic!("it declined") };
        assert!(why.to_string().contains("indptr"), "{why}");
    }

    /// The job defaults to all-zero, which every refusal path reads as
    /// "nothing to do" rather than as a shape.
    ///
    /// `Default` is derived so a caller can fill three fields and leave the
    /// rest; this is the assertion that the derived value is not accidentally
    /// a launchable one.
    #[test]
    fn a_default_job_names_no_shape() {
        assert_eq!(VarLen::default().head_dim, 0);
        assert_eq!(VarLen::default().max_seq_len, 0);
        assert_eq!(VarLen::default().seq_len, 0, "null, so `max_seq_len` stands");
    }
}
