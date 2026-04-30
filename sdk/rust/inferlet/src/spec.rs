//! Speculator trait for speculative-decoding drafters.
//!
//! Plug a [`Speculator`] into a [`Generator`](crate::gen::Generator) via
//! [`Generator::speculator`](crate::gen::Generator::speculator) to drive
//! draft tokens off your own logic.
//!
//! For host-driven speculation (where the runtime returns next-iter draft
//! tokens via the forward-pass output's spec channel), call
//! [`Generator::system_speculation`](crate::gen::Generator::system_speculation)
//! instead — that mode is built into the Generator and does not need a
//! `Speculator` impl.

/// A speculative-decoding drafter. Each iteration the [`Generator`] asks
/// for `draft()` tokens, runs the verifier, then reports `accept()`. On
/// rejection the Generator calls `rollback()` so the speculator can
/// truncate any state it grew during drafting.
///
/// [`Generator`]: crate::gen::Generator
pub trait Speculator: Send {
    /// Produce draft tokens and their absolute positions for the next
    /// forward pass. Empty vec means "no speculation this step."
    fn draft(&mut self) -> (Vec<u32>, Vec<u32>);

    /// Called with the verifier's accepted token sequence. The first
    /// accepted token corresponds to the anchor's own next-token
    /// prediction; the rest (if any) are matched drafts.
    fn accept(&mut self, accepted: &[u32]);

    /// Roll back the last `n` drafted tokens — used when the verifier
    /// rejects the tail of the draft sequence and the speculator's own
    /// internal context needs to mirror that truncation.
    fn rollback(&mut self, n: u32) {
        let _ = n;
    }

    /// Reset the speculator to its initial state.
    fn reset(&mut self) {}
}
