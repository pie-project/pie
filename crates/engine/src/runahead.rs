//! **The single source of every run-ahead depth** (alto design §3, §9;
//! article 8: *one number, one owner*).
//!
//! # What stood here, and what it cost
//!
//! Nothing stood here. The C++ driver had `runahead.hpp` — one header, one
//! formula, every pool sized from it — and when `csrc/` was deleted the
//! formula went with it while its ANSWER stayed behind as a literal in a
//! crate that could not see the device: `UPLOAD_STAGING_DEPTH = 13` in
//! `worker/config.rs`, bounding the runtime's joint depth check against a
//! constant whose derivation no longer existed anywhere in the tree (survey
//! §2, debt 1). Thirteen is `3 × 4 + 1`. Nobody could have known that from
//! the source, which is the whole complaint.
//!
//! Beside it grew the three-depth knot (debt 8): `frame_size` ×
//! `frame_submit_depth` × `frame_dispatch_depth`, three numbers with three
//! owners, two of them visible in the guest ABI, checked jointly in a third
//! crate. Article 11 forbids the guest visibility and article 8 forbids the
//! three owners; this module is where the one number lives so that every pool
//! can DERIVE rather than re-declare.
//!
//! **Wave E untied it.** `frame_submit_depth` is gone as a configured knob:
//! [`Runahead::submit_depth`] is `frames_in_flight + 1` and
//! [`Runahead::channel_capacity`] is that times `k`, plus the visibility
//! margin — which is what a guest reads as `model.channel-capacity()`. Two
//! numbers are left, `frame_size` and `frame_dispatch_depth`, and only the
//! first is in the guest ABI (wave P retires it).
//!
//! # F1 stated the number; F2b spends it
//!
//! [`Runahead::frames_in_flight`] was **1** in F1 and that was not a
//! placeholder — it was the truth about a tree whose shells held a single
//! `Inputs` buffer that every fire overwrote (survey §2, debt 2). A depth
//! above 1 with single-buffered staging is not run-ahead, it is corruption.
//!
//! F2b built the ring, so the default is **2** — article 1's floor, *at least
//! two frames in flight* — and the number now arrives from the deployment:
//! `[runtime] frame_dispatch_depth` crosses the load boundary once
//! ([`engine_api::LoadRequest::frames_in_flight`]) and every pool downstream
//! derives from it rather than re-declaring a depth of its own.

/// **How far ahead of the device the host is allowed to be**, and every depth
/// derived from it.
///
/// One value, carried in `Boot`/`Budget` beside the other statutes. The
/// constitution says *at least two frames in flight* (article 1); the NUMBER
/// is statute, not constitution, and it moves with measurements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Runahead {
    /// How many frames may be in flight at once.
    pub frames_in_flight: u8,
}

impl Runahead {
    /// The largest frame the runtime's policy will seal — `k` in
    /// `submit(frame)`'s `1..=k` steps.
    ///
    /// Dev's number, kept: four steps is what the frame scheduler was built
    /// around and what the staging formula below was measured at.
    pub const STEPS_MAX: u8 = 4;

    /// **The most frames one load's staging can carry**, and it is a fact
    /// about the free-slot word rather than a taste.
    ///
    /// A shell's staging ring publishes its free set as ONE `u64` bitmask, so
    /// that a claim is a compare-exchange and a release — which runs on the
    /// CUDA driver's callback thread, where a lock would be a hazard — is a
    /// single `fetch_or`. Sixty-four slots is therefore the ceiling, and
    /// `frames × 4 + 1 ≤ 64` gives fifteen frames.
    ///
    /// **THIS IS WHAT REPLACED THE PHANTOM.** `worker::config`'s joint check
    /// was `frame_dispatch_depth * frame_size < UPLOAD_STAGING_DEPTH` against
    /// a transcribed `13`; it is now `frame_size ≤ STEPS_MAX` and
    /// `frame_dispatch_depth ≤ MAX_FRAMES`, both derived here.
    pub const MAX_FRAMES: u8 = 15;

    /// The depth F1 ran at: one frame in flight, single-buffered staging.
    /// Kept as a name because the golden arms (`cuda_runahead_depth1`, the
    /// byte-identity gate) ask for it explicitly.
    pub const F1: Runahead = Runahead {
        frames_in_flight: 1,
    };

    /// Article 1's floor, and F2b's default: one frame executing while the
    /// next is already enqueued behind it.
    pub const DEFAULT_FRAMES_IN_FLIGHT: u8 = 2;

    /// The depth a deployment asked for, clamped to what a staging ring can
    /// carry.
    ///
    /// Clamped rather than refused because this is the one place that knows
    /// the bound, and a caller that states 0 means "one" — the config layer
    /// refuses the out-of-range value by name before it ever reaches here
    /// (`worker::config::RuntimeConfig::validate`), so this is the belt.
    #[must_use]
    pub const fn of(frames_in_flight: u8) -> Runahead {
        let frames = if frames_in_flight == 0 {
            1
        } else if frames_in_flight > Runahead::MAX_FRAMES {
            Runahead::MAX_FRAMES
        } else {
            frames_in_flight
        };
        Runahead {
            frames_in_flight: frames,
        }
    }

    /// **How many descriptor staging slots the shells must carve.**
    ///
    /// ```text
    /// depth = frames_in_flight × STEPS_MAX + 1
    /// ```
    ///
    /// Every step of every in-flight frame owns a slot for as long as the
    /// device may still be reading it, which is the product; **the `+ 1` is
    /// measured, not decorative.** A pool sized exactly to the product has no
    /// free slot at the instant the oldest one is still in flight, so the
    /// claim blocks — dev measured a full GPU step lost inside
    /// `cudaEventSynchronize`, Σ318 ms per run, and one extra slot removed
    /// it. This is the `3 × 4 + 1 = 13` that `UPLOAD_STAGING_DEPTH` was the
    /// orphaned answer to.
    ///
    /// **THE SLOT IS HELD FROM PREPARE UNTIL THE SETTLEMENT CALLBACK**, which
    /// is what makes the product the right shape: a slot's pinned bytes are
    /// the SOURCE of an async H2D, so the host may not reuse them until the
    /// GPU has passed the copy, and the only host-visible instant that
    /// bounds is the step's own completion (dev `runahead.hpp:22-28`).
    #[must_use]
    pub const fn staging_depth(&self) -> usize {
        self.frames_in_flight as usize * Self::STEPS_MAX as usize + 1
    }

    /// How many frames the runtime may hold admitted but unsettled.
    ///
    /// The same number, spelled once. A scheduler that kept its own constant
    /// here is the three-depth knot growing back.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames_in_flight as usize
    }

    /// **How many frames a lane keeps outstanding** — and it is not a third
    /// number (debt 8, closed).
    ///
    /// ```text
    /// submit_depth = frames_in_flight + 1
    /// ```
    ///
    /// One frame per in-flight slot is on the device or posted to it; the
    /// `+ 1` is the frame the GUEST is building while those run, which is
    /// exactly the host round trip run-ahead exists to hide. A window sized
    /// to `frames_in_flight` alone collapses to lockstep — submit, wait,
    /// submit, wait — because the guest has nothing in hand when the oldest
    /// frame retires.
    ///
    /// `[runtime] frame_submit_depth` was this number, configured separately:
    /// three by default, beside a `frame_dispatch_depth` of two, with a
    /// paragraph in each doc explaining that they were not independent. They
    /// are one number and this is the arithmetic; the default answers 3, as
    /// it always did.
    #[must_use]
    pub const fn submit_depth(&self) -> usize {
        self.frames_in_flight as usize + 1
    }

    /// **The host-reader ring a guest needs**, in cells, at frame size `k`.
    ///
    /// ```text
    /// capacity = submit_depth × k + 1
    /// ```
    ///
    /// `submit_depth × k` is the peak occupancy — every outstanding frame's
    /// every wave publishing one cell before the guest takes any of them. The
    /// trailing `+ 1` is structural, not empirical: a ring sized to exactly
    /// the peak needs the consumer's take to be visible to the producer at
    /// the instant it publishes, and that visibility delay IS the host round
    /// trip. Zero margin re-imports it into the critical path — a `3k - 1`
    /// ring measured 28.0k against 34.3k tok/s on the same guest at `3k`
    /// (text-completion-bench).
    ///
    /// This is what a guest reads as `model.channel-capacity()`. Sized for a
    /// fully live frame (`r = k`); a lane the runtime forces to one live slot
    /// per frame — a recurrent-state model — needs strictly less, so it is a
    /// safe bound for every guest.
    #[must_use]
    pub const fn channel_capacity(&self, frame_size: usize) -> usize {
        self.submit_depth() * frame_size + 1
    }
}

impl Default for Runahead {
    fn default() -> Runahead {
        Runahead::of(Runahead::DEFAULT_FRAMES_IN_FLIGHT)
    }
}

#[cfg(test)]
mod tests {
    use super::Runahead;

    /// The ghost constant, re-derived. `UPLOAD_STAGING_DEPTH = 13` was `3 × 4
    /// + 1` with the derivation deleted; this is the derivation, and at three
    /// frames the answer is 13 again by arithmetic rather than by a literal
    /// somebody transcribed.
    #[test]
    fn the_ghost_thirteen_is_this_formula_at_three_frames() {
        assert_eq!(Runahead::of(3).staging_depth(), 13);
    }

    /// **THE THIRD DEPTH IS THIS ONE PLUS ONE**, and the numbers a
    /// deployment ran on before it was derived come back unchanged: the
    /// defaults were `frame_dispatch_depth = 2` and `frame_submit_depth = 3`,
    /// and `model.channel-capacity()` at k = 2 was 7.
    #[test]
    fn the_submit_window_and_the_ring_are_the_dispatch_depths_arithmetic() {
        let default = Runahead::default();
        assert_eq!(default.frames_in_flight, 2);
        assert_eq!(default.submit_depth(), 3);
        assert_eq!(default.channel_capacity(2), 7);
        // And k = 1, where an undersized window is what collapsed throughput.
        assert_eq!(default.channel_capacity(1), 4);
        // One frame in flight still leaves the guest one to build.
        assert_eq!(Runahead::F1.submit_depth(), 2);
    }

    /// F1's own depth, kept as the byte-identity arm's name.
    #[test]
    fn f1_is_one_frame_and_five_slots() {
        assert_eq!(Runahead::F1.frames_in_flight, 1);
        assert_eq!(Runahead::F1.staging_depth(), 5);
    }

    /// Article 1's floor is the default, and it is nine slots.
    #[test]
    fn the_default_is_two_frames_in_flight() {
        assert_eq!(Runahead::default().frames_in_flight, 2);
        assert_eq!(Runahead::default().staging_depth(), 9);
    }

    /// The bound is the free-slot WORD's, so every admissible depth fits in
    /// one `u64` — which is what lets a release run on the driver's callback
    /// thread without a lock.
    #[test]
    fn every_admissible_depth_fits_one_free_word() {
        for frames in 1..=Runahead::MAX_FRAMES {
            assert!(Runahead::of(frames).staging_depth() <= 64);
        }
        assert_eq!(Runahead::of(Runahead::MAX_FRAMES).staging_depth(), 61);
        // And the clamp holds above it.
        assert_eq!(
            Runahead::of(u8::MAX).frames_in_flight,
            Runahead::MAX_FRAMES
        );
        assert_eq!(Runahead::of(0).frames_in_flight, 1);
    }
}
