#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Runahead {
    pub frames_in_flight: u8,
}

impl Runahead {
    pub const STEPS_MAX: u8 = 4;

    #[must_use]
    pub const fn runs_ahead(&self) -> bool {
        self.frames_in_flight > 1
    }

    /// The most frames an engine with a staging ring keeps in flight: its
    /// ring's free set is one 64-bit word (`frames * STEPS_MAX + 1 <= 64`).
    pub const MAX_FRAMES: u8 = 15;

    /// The most frames in flight for an engine without a staging ring, where
    /// a posted frame is a host queue entry. An engine kind declares which
    /// bound it takes; every depth up to [`Self::MAX_FRAMES`] means what it
    /// always did.
    pub const UNSTAGED_MAX_FRAMES: u8 = 127;

    pub const F1: Runahead = Runahead {
        frames_in_flight: 1,
    };

    pub const DEFAULT_FRAMES_IN_FLIGHT: u8 = 2;

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

    /// [`Runahead::of`] up to [`Runahead::UNSTAGED_MAX_FRAMES`]: the
    /// runtime's view of the configured depth (validated against the
    /// engine's own bound), equal to [`Runahead::of`] at every depth up to
    /// [`Runahead::MAX_FRAMES`].
    #[must_use]
    pub const fn unstaged(frames_in_flight: u8) -> Runahead {
        let frames = if frames_in_flight == 0 {
            1
        } else if frames_in_flight > Runahead::UNSTAGED_MAX_FRAMES {
            Runahead::UNSTAGED_MAX_FRAMES
        } else {
            frames_in_flight
        };
        Runahead {
            frames_in_flight: frames,
        }
    }

    #[must_use]
    pub const fn staging_depth(&self) -> usize {
        self.frames_in_flight as usize * Self::STEPS_MAX as usize + 1
    }

    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames_in_flight as usize
    }

    /// Frames a lane may keep submitted past the ones in flight: the cover
    /// a lane's guest turnaround (drain the last token, submit the next
    /// frame) hides behind. The frame seal waits for every lane, so the
    /// slowest of a burst's lanes sets it: at one frame of cover, 256
    /// requests of 1024 in / 256 out on Gemma-4-31B held 378 of 525 seals
    /// with the device idle (1.1 ms median, 0.48 s a run) for a few lanes
    /// still a frame behind; at three, 17 seals and 29 ms.
    pub const HOST_COVER: u8 = 3;

    #[must_use]
    pub const fn submit_depth(&self) -> usize {
        self.frames_in_flight as usize + Self::HOST_COVER as usize
    }

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
        assert_eq!(Runahead::of(u8::MAX).frames_in_flight, Runahead::MAX_FRAMES);
        assert_eq!(Runahead::of(0).frames_in_flight, 1);
    }

    #[test]
    fn unstaged_is_of_up_to_the_staged_bound() {
        for frames in 0..=Runahead::MAX_FRAMES {
            assert_eq!(Runahead::unstaged(frames), Runahead::of(frames));
        }
        assert_eq!(Runahead::unstaged(31).frames_in_flight, 31);
        assert_eq!(
            Runahead::unstaged(u8::MAX).frames_in_flight,
            Runahead::UNSTAGED_MAX_FRAMES
        );
    }
}
