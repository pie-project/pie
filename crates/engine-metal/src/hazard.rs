//! Which launches must wait for the ones before them.
//!
//! A concurrent compute pass runs its dispatches with no ordering between
//! them until a memory barrier says otherwise; an indirect command buffer's
//! commands are the same, with a barrier flag per command. A barrier drains
//! everything before it, so it is owed exactly where a launch touches bytes
//! that a launch since the last barrier also touched and one of the two
//! writes them: a read after a write, a write after a read, a write after a
//! write. Two launches that only read the same bytes, or touch disjoint
//! bytes, run side by side.
//!
//! The bytes are ranges of a reservation, not reservations: the arena is one
//! buffer, and every value in it is a view at its own offset. A launch whose
//! touches are not all known — a binding with no length, a dispatch encoded
//! outside the tracked path — is [`Hazards::opaque`]: it waits for
//! everything before it and everything after it waits for it.

/// One argument's bytes: `[start, end)` of reservation `slab`, read or
/// written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub slab: u64,
    pub start: u64,
    pub end: u64,
    pub write: bool,
}

impl Access {
    /// `len` bytes at `offset`; `None` is a length nobody stated, and stands
    /// for every byte from `offset` on.
    #[must_use]
    pub fn of(slab: u64, offset: u64, len: Option<u64>, write: bool) -> Access {
        let end = len.map_or(u64::MAX, |len| offset.saturating_add(len.max(1)));
        Access {
            slab,
            start: offset,
            end,
            write,
        }
    }

    /// Every byte of `slab`.
    #[must_use]
    pub fn whole(slab: u64, write: bool) -> Access {
        Access {
            slab,
            start: 0,
            end: u64::MAX,
            write,
        }
    }

    fn overlaps(&self, other: &Access) -> bool {
        self.slab == other.slab && self.start < other.end && other.start < self.end
    }
}

/// What the launches since the last barrier read and wrote.
#[derive(Debug, Default)]
pub struct Hazards {
    reads: Vec<Access>,
    writes: Vec<Access>,
    /// A launch since the last barrier touched bytes nobody listed.
    blind: bool,
    /// Launches since the last barrier.
    since: usize,
    launches: u64,
    barriers: u64,
}

impl Hazards {
    #[must_use]
    pub fn new() -> Hazards {
        Hazards::default()
    }

    /// Admits a launch touching `accesses`, and says whether a barrier is
    /// owed ahead of it. The sets restart at a barrier, holding this launch.
    pub fn admit(&mut self, accesses: &[Access]) -> bool {
        let owed = self.blind
            || accesses.iter().any(|access| {
                self.writes.iter().any(|held| held.overlaps(access))
                    || (access.write && self.reads.iter().any(|held| held.overlaps(access)))
            });
        if owed {
            self.restart();
        }
        for access in accesses {
            if access.write {
                self.writes.push(*access);
            } else {
                self.reads.push(*access);
            }
        }
        self.note(owed)
    }

    /// Admits a launch whose touches are not known: it waits for whatever
    /// ran since the last barrier, and the next launch waits for it.
    pub fn opaque(&mut self) -> bool {
        let owed = self.since > 0;
        self.restart();
        self.blind = true;
        self.note(owed)
    }

    /// Forgets the sets: a new pass, which nothing before it overlaps.
    pub fn reset(&mut self) {
        self.restart();
    }

    /// Launches admitted and barriers owed, since this tracker was made.
    #[must_use]
    pub fn census(&self) -> (u64, u64) {
        (self.launches, self.barriers)
    }

    fn restart(&mut self) {
        self.reads.clear();
        self.writes.clear();
        self.blind = false;
        self.since = 0;
    }

    fn note(&mut self, owed: bool) -> bool {
        self.since += 1;
        self.launches += 1;
        if owed {
            self.barriers += 1;
        }
        owed
    }
}

#[cfg(test)]
mod tests {
    use super::{Access, Hazards};

    fn read(slab: u64, start: u64, len: u64) -> Access {
        Access::of(slab, start, Some(len), false)
    }

    fn write(slab: u64, start: u64, len: u64) -> Access {
        Access::of(slab, start, Some(len), true)
    }

    #[test]
    fn a_read_of_what_was_just_written_waits() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[read(1, 0, 64), write(1, 64, 64)]));
        assert!(hazards.admit(&[read(1, 64, 64), write(1, 128, 64)]));
    }

    #[test]
    fn a_write_over_what_was_just_read_or_written_waits() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[read(1, 0, 64), write(1, 64, 64)]));
        assert!(hazards.admit(&[write(1, 0, 64)]));
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 64, 64)]));
        assert!(hazards.admit(&[write(1, 64, 64)]));
    }

    #[test]
    fn disjoint_ranges_of_one_reservation_run_together() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[read(1, 0, 64), write(1, 64, 64)]));
        assert!(!hazards.admit(&[read(1, 0, 64), write(1, 128, 64)]));
        assert!(!hazards.admit(&[read(2, 64, 64), write(1, 192, 64)]));
        assert_eq!(hazards.census(), (3, 0));
    }

    #[test]
    fn two_reads_of_the_same_bytes_run_together() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[read(1, 0, 64), write(1, 64, 64)]));
        assert!(!hazards.admit(&[read(1, 0, 64), write(1, 128, 64)]));
    }

    #[test]
    fn the_same_offset_of_another_reservation_is_other_bytes() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 0, 64)]));
        assert!(!hazards.admit(&[read(2, 0, 64), write(3, 0, 64)]));
    }

    #[test]
    fn an_overlapping_sub_range_waits() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 0, 256)]));
        assert!(hazards.admit(&[read(1, 128, 4)]));
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 128, 4)]));
        assert!(hazards.admit(&[read(1, 0, 256)]));
    }

    #[test]
    fn ranges_that_only_touch_end_to_start_do_not_overlap() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 0, 64)]));
        assert!(!hazards.admit(&[read(1, 64, 64)]));
    }

    #[test]
    fn an_unstated_length_runs_to_the_end_of_the_reservation() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 1 << 20, 64)]));
        assert!(hazards.admit(&[Access::of(1, 0, None, false)]));
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 0, 64)]));
        assert!(hazards.admit(&[Access::whole(1, false)]));
    }

    #[test]
    fn an_opaque_launch_waits_and_is_waited_for() {
        let mut hazards = Hazards::new();
        assert!(!hazards.opaque());
        assert!(hazards.admit(&[read(9, 0, 4)]));
        assert!(!hazards.admit(&[read(8, 0, 4)]));
        assert!(hazards.opaque());
        assert!(hazards.opaque());
        assert!(hazards.admit(&[]));
    }

    #[test]
    fn a_barrier_forgets_what_came_before_it() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 0, 64)]));
        assert!(hazards.admit(&[read(1, 0, 64), write(2, 0, 64)]));
        // The first write is behind the barrier, so a second read of its
        // bytes runs beside the first; the second launch's write is held.
        assert!(!hazards.admit(&[read(1, 0, 64)]));
        assert!(hazards.admit(&[read(2, 0, 64)]));
    }

    #[test]
    fn a_reset_pass_starts_clean() {
        let mut hazards = Hazards::new();
        assert!(!hazards.admit(&[write(1, 0, 64)]));
        hazards.reset();
        assert!(!hazards.admit(&[read(1, 0, 64)]));
        assert_eq!(hazards.census(), (2, 0));
    }
}
