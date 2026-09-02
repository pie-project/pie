//! Single source of every run-ahead depth; `submit_depth`, `staging_depth` and `channel_capacity` all derive from `frames_in_flight`.

/// How far ahead of the device the host is allowed to be, and every depth
/// derived from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Runahead {
    /// How many frames may be in flight at once.
    pub frames_in_flight: u8,
}

impl Runahead {
    /// Largest frame the runtime's policy will seal: `k` in `submit(frame)`'s `1..=k` steps.
    pub const STEPS_MAX: u8 = 4;

    /// Most frames one load's staging can carry. The free set is one `u64` bitmask, giving a 64-slot ceiling (`frames * 4 + 1 <= 64` = 15).
    pub const MAX_FRAMES: u8 = 15;

    /// One frame in flight, single-buffered staging.
    pub const F1: Runahead = Runahead {
        frames_in_flight: 1,
    };

    /// The default: one frame executing while the next is already enqueued
    /// behind it.
    pub const DEFAULT_FRAMES_IN_FLIGHT: u8 = 2;

    /// Clamped to what a staging ring can carry; 0 means "one".
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

    /// `depth = frames_in_flight * STEPS_MAX + 1`; the `+1` avoids blocking when the oldest slot is still in flight.
    #[must_use]
    pub const fn staging_depth(&self) -> usize {
        self.frames_in_flight as usize * Self::STEPS_MAX as usize + 1
    }

    /// How many frames the runtime may hold admitted but unsettled.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames_in_flight as usize
    }

    /// `submit_depth = frames_in_flight + 1`; the `+1` is the frame the guest is building while the rest run.
    #[must_use]
    pub const fn submit_depth(&self) -> usize {
        self.frames_in_flight as usize + 1
    }

    /// Host-reader ring size in cells at frame size `k`: `capacity = submit_depth * k + 1`; the `+1` covers producer/consumer visibility delay.
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

    #[test]
    fn every_admissible_depth_fits_one_free_word() {
        for frames in 1..=Runahead::MAX_FRAMES {
            assert!(Runahead::of(frames).staging_depth() <= 64);
        }
        assert_eq!(Runahead::of(Runahead::MAX_FRAMES).staging_depth(), 61);
        assert_eq!(
            Runahead::of(u8::MAX).frames_in_flight,
            Runahead::MAX_FRAMES
        );
        assert_eq!(Runahead::of(0).frames_in_flight, 1);
    }
}
