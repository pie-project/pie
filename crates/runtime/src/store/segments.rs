//! The segments a chunk pool grows and shrinks by.
//!
//! A segment is a run of `segment_chunks` physical chunks the backend maps or
//! unmaps as one; a chunk never straddles two. Only a committed segment's
//! chunks are handed out. A segment goes back by evacuation: its held chunks
//! move, whole, into free chunks of lower segments (a batched device copy and
//! an update of the engine's logical -> physical chunk table, ordered on the
//! stream), the segment drains until the fire that carried the copy retires,
//! and only then is it released. Nothing here maps memory; the pool says what
//! to commit ahead of need, so nothing is mapped on the fire path.

use std::ops::Range;

/// Where a segment stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentState {
    /// No memory under it; its chunks are not handed out.
    Uncommitted,
    /// Mapped, and its free chunks are handed out.
    Committed,
    /// Evacuated: still mapped, handing nothing out, until the fire at
    /// `until` retires; then it may be released.
    Draining { until: u64 },
}

/// One chunk's move in an evacuation: its logical id, which holders and
/// lanes keep, stays; its physical position goes from `from` to `to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkMove {
    pub logical: u32,
    pub from: u32,
    pub to: u32,
}

/// Every held chunk of `segment`, moved into free chunks below it. The
/// engine copies `from` -> `to` and rewrites its chunk table, in stream order
/// before the next fire.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Evacuation {
    pub segment: u32,
    pub moves: Vec<ChunkMove>,
}

/// Why an evacuation or a segment change was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SegmentError {
    #[error("segment {0} does not exist")]
    Unknown(u32),
    #[error("segment {segment} is {state:?}, not {wanted}")]
    State {
        segment: u32,
        state: SegmentState,
        wanted: &'static str,
    },
    #[error("segment {segment} still holds {held} chunks")]
    Held { segment: u32, held: u32 },
    #[error("segment {segment} drains until epoch {until}; retired through {retired}")]
    Draining {
        segment: u32,
        until: u64,
        retired: u64,
    },
    #[error("an evacuation of segment {0} no longer matches the pool")]
    Stale(u32),
    #[error("releasing segment {segment} leaves {committed} chunks for {needed}")]
    Budget {
        segment: u32,
        committed: u32,
        needed: u32,
    },
}

/// The segment table: each segment's state and how many of its chunks are
/// held (reserved, or opened for a slot shape).
#[derive(Clone, Debug)]
pub struct Segments {
    segment_chunks: u32,
    total_chunks: u32,
    state: Vec<SegmentState>,
    used: Vec<u32>,
    /// Chunks of `Committed` segments: what the budget may spend.
    committed_chunks: u32,
}

impl Segments {
    /// `total_chunks` cut into segments of `segment_chunks` (the last may be
    /// shorter), the lowest `committed` of them committed.
    pub fn new(total_chunks: u32, segment_chunks: u32, committed: u32) -> Self {
        assert!(segment_chunks > 0, "a segment holds at least one chunk");
        let count = total_chunks.div_ceil(segment_chunks);
        assert!(
            committed <= count,
            "{committed} committed segments of {count}"
        );
        let mut segments = Self {
            segment_chunks,
            total_chunks,
            state: vec![SegmentState::Uncommitted; count as usize],
            used: vec![0; count as usize],
            committed_chunks: 0,
        };
        for segment in 0..committed {
            segments.set_state(segment, SegmentState::Committed);
        }
        segments
    }

    pub fn count(&self) -> u32 {
        self.state.len() as u32
    }

    pub fn segment_chunks(&self) -> u32 {
        self.segment_chunks
    }

    /// The physical chunks of `segment`.
    pub fn range(&self, segment: u32) -> Range<u32> {
        let start = segment * self.segment_chunks;
        start..(start + self.segment_chunks).min(self.total_chunks)
    }

    pub fn len(&self, segment: u32) -> u32 {
        let range = self.range(segment);
        range.end - range.start
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    /// The segment physical chunk `chunk` lies in.
    pub fn of(&self, chunk: u32) -> u32 {
        chunk / self.segment_chunks
    }

    pub fn state(&self, segment: u32) -> SegmentState {
        self.state[segment as usize]
    }

    /// Held chunks in `segment`.
    pub fn used(&self, segment: u32) -> u32 {
        self.used[segment as usize]
    }

    pub fn committed_chunks(&self) -> u32 {
        self.committed_chunks
    }

    pub fn committed_segments(&self) -> u32 {
        self.state
            .iter()
            .filter(|state| **state == SegmentState::Committed)
            .count() as u32
    }

    /// Segments with memory under them, draining ones included.
    pub fn mapped_segments(&self) -> u32 {
        self.state
            .iter()
            .filter(|state| **state != SegmentState::Uncommitted)
            .count() as u32
    }

    /// The committed segment holding the fewest chunks, the highest among
    /// equals: the cheapest to evacuate, and the top one when all tie.
    pub fn emptiest(&self) -> Option<u32> {
        (0..self.count())
            .rev()
            .filter(|&segment| self.state(segment) == SegmentState::Committed)
            .min_by_key(|&segment| self.used(segment))
    }

    /// The lowest uncommitted segments, in order.
    pub fn uncommitted(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.count()).filter(|&segment| self.state(segment) == SegmentState::Uncommitted)
    }

    pub(super) fn hold(&mut self, chunk: u32) {
        let segment = self.of(chunk) as usize;
        self.used[segment] += 1;
    }

    pub(super) fn let_go(&mut self, chunk: u32) {
        let segment = self.of(chunk) as usize;
        debug_assert!(self.used[segment] > 0);
        self.used[segment] -= 1;
    }

    pub(super) fn set_state(&mut self, segment: u32, state: SegmentState) {
        let len = self.len(segment);
        let was = std::mem::replace(&mut self.state[segment as usize], state);
        if was == SegmentState::Committed {
            self.committed_chunks -= len;
        }
        if state == SegmentState::Committed {
            self.committed_chunks += len;
        }
    }

    pub(super) fn check(&self, segment: u32) -> Result<(), SegmentError> {
        if segment < self.count() {
            Ok(())
        } else {
            Err(SegmentError::Unknown(segment))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_last_segment_counts_only_its_chunks() {
        let segments = Segments::new(10, 4, 3);
        assert_eq!(segments.count(), 3);
        assert_eq!(segments.range(2), 8..10);
        assert_eq!(segments.committed_chunks(), 10);
        assert_eq!(segments.of(9), 2);
    }

    #[test]
    fn the_emptiest_segment_is_the_highest_among_equals() {
        let mut segments = Segments::new(12, 4, 3);
        assert_eq!(segments.emptiest(), Some(2));
        segments.hold(9);
        segments.hold(1);
        assert_eq!(segments.emptiest(), Some(1));
        segments.set_state(1, SegmentState::Uncommitted);
        assert_eq!(segments.emptiest(), Some(2));
        assert_eq!(segments.committed_chunks(), 8);
        assert_eq!(segments.uncommitted().collect::<Vec<_>>(), vec![1]);
    }
}
