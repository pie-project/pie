//! # Pinned ring-word reader (Runtime–Driver Boundary, B8/B9/B13)
//!
//! [`PinnedRingWord`] wraps a live pinned ring word the CUDA driver publishes:
//! on a fire commit the driver stores the committed ring index into the
//! instance's pinned word region (release, B11) *before* signalling, so reading
//! it is a plain acquire load of a host word — never a call back through the
//! driver. The value path never travels through the driver; only the *signal*
//! ("instance N committed") does.
//!
//! Soundness rests on B6: the frame address never moves for the instance's
//! lifetime, so the pinned word stays valid and fixed. The X0 waker table
//! ([`crate::driver::waker`]) is the wake mechanism; the direct
//! driver-side per-instance wake (`pie_wake_past` from the driver's instance
//! table) is the boundary-migration end state.

use std::sync::atomic::{AtomicU64, Ordering};

/// The **pinned ring word the CUDA driver publishes** — the X2 device path.
/// [`ControlPlane::bind_instance`](crate::driver::ControlPlane::bind_instance)
/// returns the instance's [`FrameAddresses`](crate::driver::FrameAddresses); the
/// driver stores `word[word_index] = committed_index` into the pinned word
/// region at `word_base` *before* it signals (B9/B11), so reading it is a plain
/// acquire load of that pinned host word — never a call back through the driver.
///
/// Soundness rests on B6: the frame address never moves for the instance's
/// lifetime, so the pinned word stays valid and fixed. Channel-word layout
/// (host + driver agree, [`WordLayout`](crate::driver::WordLayout)): `word[0]`
/// is the instance's pacing counter; for host-visible channel `c` the committed
/// head is at `word_index = 1 + 2*c` and the tail at `2 + 2*c`.
pub struct PinnedRingWord {
    /// Points into the instance's pinned word region (`word_base + offset`).
    /// The driver only ever advances it; the consumer only ever reads it.
    word: *const AtomicU64,
}

// SAFETY: `word` addresses pinned host memory the driver keeps live and fixed
// for the instance's lifetime (B6). The consumer only acquire-loads it (never
// writes, never frees) and the driver publishes with a release store before
// signalling (B11), so sharing/sending the read handle across runtime threads
// is sound.
unsafe impl Send for PinnedRingWord {}
unsafe impl Sync for PinnedRingWord {}

impl PinnedRingWord {
    /// Size of one pinned ring word (the driver's `u64` word).
    pub const WORD_BYTES: usize = std::mem::size_of::<u64>();

    /// Wrap a raw pointer to a live pinned ring word.
    ///
    /// # Safety
    /// `word` must be a valid, aligned, live pinned-word address that stays
    /// valid for as long as this `PinnedRingWord` is read (the instance's
    /// lifetime, B6).
    pub unsafe fn from_raw(word: *const AtomicU64) -> Self {
        PinnedRingWord { word }
    }

    /// Derive the pinned-word address from the X2 `word_base` and a per-channel
    /// `word_index` (`word_base + word_index * WORD_BYTES`).
    ///
    /// # Safety
    /// `word_base` must be the instance's real pinned word-region base (from
    /// [`bind_instance`](crate::driver::ControlPlane::bind_instance)) and
    /// `word_index` within that region's word count.
    pub unsafe fn from_word_base(word_base: u64, word_index: usize) -> Self {
        let addr = word_base + (word_index * Self::WORD_BYTES) as u64;
        // SAFETY: forwarded to the caller's `from_word_base` contract above.
        unsafe { Self::from_raw(addr as *const AtomicU64) }
    }

    /// Acquire-load the current committed ring index. Pairs with the driver's
    /// release publish (B11).
    pub fn load(&self) -> u64 {
        // SAFETY: `word` is a live, aligned pinned cell per the constructor
        // contract; this acquire load pairs with the driver's release publish.
        unsafe { (*self.word).load(Ordering::Acquire) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn pinned_ring_word_acquire_loads_the_published_index() {
        let word = AtomicU64::new(0);
        // SAFETY: `word` outlives the reader in this test.
        let reader = unsafe { PinnedRingWord::from_raw(&word as *const AtomicU64) };
        assert_eq!(reader.load(), 0);
        word.store(7, Ordering::Release);
        assert_eq!(reader.load(), 7);
    }

    #[test]
    fn from_word_base_offsets_by_word_index() {
        let words = [AtomicU64::new(11), AtomicU64::new(22), AtomicU64::new(33)];
        let base = words.as_ptr() as u64;
        // SAFETY: indices are within the live `words` array for the test.
        let w1 = unsafe { PinnedRingWord::from_word_base(base, 1) };
        let w2 = unsafe { PinnedRingWord::from_word_base(base, 2) };
        assert_eq!(w1.load(), 22);
        assert_eq!(w2.load(), 33);
    }
}
