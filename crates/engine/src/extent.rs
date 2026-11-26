//! What a lane's attention may read of its working set: the slots the set
//! holds written, by its own fires or in the prefix it shares, and never a
//! slot past them. A page a set takes from the pool holds what its last
//! owner left there until the set writes it, so a fire whose rows read past
//! the set's written end would read another owner's keys and values.
//!
//! The runtime states, on every lane it stages, the set's written end as it
//! knows it then ([`KvDelta::readable`]) and the set ([`KvDelta::set`]).
//! Where the device places a lane's rows, the runtime learns where they
//! landed only once the fire settles, so a fire staged behind one still in
//! flight finds `readable` short of what that one wrote. An engine that
//! places rows the device resolves keeps, in a [`Written`], the written end
//! each set reached in the fires it admitted, and admits a lane only where
//! its writes run on from that end without a gap and its reach stops at the
//! end they leave ([`FireExtents::check`]). A refused lane is refused by
//! name, never served with the slots past its end hidden.

use std::collections::HashMap;
use std::ops::Range;

#[cfg(doc)]
use crate::fire::KvDelta;

/// The written end of each set an engine has admitted fires on, kept at
/// each seat its lanes took in the fire (a set holds its seats while it
/// lives, its first lane the first of them every fire). A seat another set
/// takes again starts from what the runtime states.
#[derive(Debug, Default, Clone)]
pub struct Written {
    ends: HashMap<u32, (u64, u32)>,
}

impl Written {
    /// The set's written end as a lane seated at `seat` finds it: what the
    /// runtime states, or further where this engine admitted writes past it.
    #[must_use]
    pub fn end_of(&self, seat: u32, set: u64, readable: u32) -> u32 {
        match self.ends.get(&seat) {
            Some(&(kept, end)) if kept == set => end.max(readable),
            _ => readable,
        }
    }

    /// A fire `fire` checked ([`FireExtents::check`]) lands: each set's end
    /// moves to where its writes left it. An end no further than what the
    /// runtime stated needs no keeping.
    pub fn commit(&mut self, fire: FireExtents) {
        for set in fire.sets {
            for seat in set.seats {
                if set.after > set.stated {
                    self.ends.insert(seat, (set.set, set.after));
                } else {
                    self.ends.remove(&seat);
                }
            }
        }
    }

    /// The sets kept now.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }
}

/// One fire's lanes, checked whole: every row of a fire lands its keys
/// before any row reads, so the lanes of one set in one fire read the union
/// of their writes.
#[derive(Debug, Default)]
pub struct FireExtents {
    sets: Vec<SetExtent>,
}

#[derive(Debug)]
struct SetExtent {
    set: u64,
    seats: Vec<u32>,
    /// What the runtime stated.
    stated: u32,
    /// The written end the fire found.
    found: u32,
    /// Where the fire's writes leave it.
    after: u32,
    writes: Vec<Range<u32>>,
    /// `(lane, reach)`: lane `lane` reads every slot below `reach`.
    reads: Vec<(usize, u32)>,
}

impl FireExtents {
    /// Lane `lane`, seated at `seat`, of the set and written end its
    /// runtime states ([`KvDelta::set`], [`KvDelta::readable`]), writes the
    /// slots `writes` and reads every slot below `reach`. A lane that names
    /// no set (`set` 0) is not held.
    pub fn lane(
        &mut self,
        written: &Written,
        lane: usize,
        seat: u32,
        (set, readable): (u64, u32),
        writes: impl IntoIterator<Item = Range<u32>>,
        reach: u32,
    ) {
        // No set's runtime staged it: an engine's own fire (one it arms or
        // measures by), whose rows are its own.
        if set == 0 {
            return;
        }
        let at = match self.sets.iter().position(|kept| kept.set == set) {
            Some(at) => at,
            None => {
                self.sets.push(SetExtent {
                    set,
                    seats: Vec::new(),
                    stated: readable,
                    found: readable,
                    after: readable,
                    writes: Vec::new(),
                    reads: Vec::new(),
                });
                self.sets.len() - 1
            }
        };
        let kept = &mut self.sets[at];
        // Whichever of its lanes the fire lists first, the set's end is
        // the furthest any of its seats kept.
        kept.found = kept.found.max(written.end_of(seat, set, readable));
        if !kept.seats.contains(&seat) {
            kept.seats.push(seat);
        }
        let set = kept;
        set.writes
            .extend(writes.into_iter().filter(|span| span.start < span.end));
        set.reads.push((lane, reach));
    }

    /// Refuses, by name, a fire whose writes leave a gap past what a set
    /// holds written, or one of whose lanes reads past where they leave it.
    pub fn check(&mut self) -> Result<(), String> {
        for set in &mut self.sets {
            set.writes.sort_by_key(|span| span.start);
            let mut end = set.found;
            for span in &set.writes {
                if span.start > end {
                    return Err(format!(
                        "a lane writes slots {}..{} of its working set, which holds slots \
                         0..{end} written: the slots between were never written by it, and \
                         a page holds what its last owner left until its set writes it",
                        span.start, span.end
                    ));
                }
                end = end.max(span.end);
            }
            set.after = end;
            if let Some(&(lane, reach)) = set.reads.iter().find(|&&(_, reach)| reach > end) {
                return Err(format!(
                    "lane {lane} reads {reach} slot(s) of its working set, which holds slots \
                     0..{end} written once this fire lands: a lane's attention reads no slot \
                     its set has not written"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{FireExtents, Written};

    fn kv(set: u64, readable: u32) -> (u64, u32) {
        (set, readable)
    }

    fn fire(
        written: &mut Written,
        lanes: &[(u32, (u64, u32), Vec<std::ops::Range<u32>>, u32)],
    ) -> Result<(), String> {
        let mut extents = FireExtents::default();
        for (at, (seat, kv, writes, reach)) in lanes.iter().enumerate() {
            extents.lane(written, at, *seat, *kv, writes.iter().cloned(), *reach);
        }
        extents.check()?;
        written.commit(extents);
        Ok(())
    }

    #[test]
    fn extent_every_case() {
        a_lane_reads_what_its_set_wrote_and_no_further();
        a_write_past_the_written_end_is_refused();
        a_fire_behind_one_in_flight_reads_what_that_one_wrote();
        lanes_of_one_set_read_the_union_of_their_writes();
        a_refused_fire_moves_no_end();
    }

    fn a_lane_reads_what_its_set_wrote_and_no_further() {
        let mut written = Written::default();
        assert_eq!(
            fire(&mut written, &[(0, kv(7, 10), vec![10..12], 12)]),
            Ok(())
        );
        let past = fire(&mut written, &[(0, kv(7, 10), vec![12..13], 20)]);
        assert!(past.is_err_and(|why| why.contains("reads 20 slot(s)")));
    }

    fn a_write_past_the_written_end_is_refused() {
        let mut written = Written::default();
        let gap = fire(&mut written, &[(0, kv(7, 10), vec![15..16], 16)]);
        assert!(gap.is_err_and(|why| why.contains("slots 15..16")));
        // A rewind writes inside what it holds.
        assert_eq!(
            fire(&mut written, &[(0, kv(7, 10), vec![4..6], 10)]),
            Ok(())
        );
    }

    fn a_fire_behind_one_in_flight_reads_what_that_one_wrote() {
        let mut written = Written::default();
        assert_eq!(
            fire(&mut written, &[(3, kv(7, 10), vec![10..11], 11)]),
            Ok(())
        );
        // Staged before the first settled: the runtime still states 10.
        assert_eq!(
            fire(&mut written, &[(3, kv(7, 10), vec![11..12], 12)]),
            Ok(())
        );
        assert_eq!(written.len(), 1);
        // The runtime caught up: nothing past it to keep.
        assert_eq!(fire(&mut written, &[(3, kv(7, 12), vec![], 12)]), Ok(()));
        assert!(written.is_empty());
    }

    fn lanes_of_one_set_read_the_union_of_their_writes() {
        let mut written = Written::default();
        let siblings = [(0, kv(7, 4), vec![6..8], 8), (1, kv(7, 4), vec![4..6], 6)];
        assert_eq!(fire(&mut written, &siblings), Ok(()));
        // Another set's writes fill no gap of this one's.
        let strangers = [
            (0, kv(7, 8), vec![10..12], 12),
            (1, kv(9, 12), vec![8..10], 10),
        ];
        assert!(fire(&mut written, &strangers).is_err());
    }

    fn a_refused_fire_moves_no_end() {
        let mut written = Written::default();
        let refused = fire(&mut written, &[(0, kv(7, 0), vec![0..4], 9)]);
        assert!(refused.is_err());
        assert!(written.is_empty(), "nothing of a refused fire is kept");
    }
}
