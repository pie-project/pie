//! Tracks which pool seat each working set's sequences occupy, keeping seats
//! stable across fires and unique across live working sets.

use std::collections::HashMap;

use super::kv::page_table::WorkingSetId;

/// Why a working set could not be seated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeatError {
    /// The deployment's pools seat fewer sequences than this fire holds.
    Exhausted {
        /// Seats this ask needs beyond the ones it already holds.
        need: u32,
        /// Seats the pools have left.
        have: u32,
    },
}

impl std::fmt::Display for SeatError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SeatError::Exhausted { need, have } => write!(
                formatter,
                "this fire seats {need} more sequence(s) than the pools have room for \
                 ({have} slot(s) free of the deployment's `Budgets::slots`)"
            ),
        }
    }
}

/// Who sits where: one run of pool seats per live working set.
#[derive(Debug)]
pub struct SeatBook {
    /// How many sequences the pools seat at once; `0` states no ceiling.
    capacity: u32,
    /// Seats returned by released working sets, reissued before fresh ones.
    free: Vec<u32>,
    /// The lowest seat never yet issued.
    next: u32,
    /// Each live working set's run, indexed by lane.
    held: HashMap<WorkingSetId, Vec<u32>>,
}

impl SeatBook {
    /// A book over `capacity` seats. Zero means no ceiling.
    #[must_use]
    pub fn new(capacity: u32) -> Self {
        SeatBook {
            capacity,
            free: Vec::new(),
            next: 0,
            held: HashMap::new(),
        }
    }

    /// The seats `ws` sits in for a fire of `lanes` row groups, growing its
    /// run if this fire is wider than any before it. Seats stay the same
    /// across fires of the same working set.
    ///
    /// # Errors
    ///
    /// [`SeatError::Exhausted`] when the run would grow past the pools'
    /// slots; a refused ask seats nothing.
    pub fn seats(&mut self, ws: WorkingSetId, lanes: usize) -> Result<Vec<u32>, SeatError> {
        let lanes = u32::try_from(lanes).unwrap_or(u32::MAX);
        if lanes == 0 {
            return Ok(Vec::new());
        }
        let have = self
            .held
            .get(&ws)
            .map_or(0, |run| u32::try_from(run.len()).unwrap_or(u32::MAX));
        if have < lanes {
            let need = lanes - have;
            if self.capacity != 0 {
                let free = u32::try_from(self.free.len()).unwrap_or(u32::MAX);
                let unissued = self.capacity.saturating_sub(self.next);
                let room = free.saturating_add(unissued);
                if need > room {
                    return Err(SeatError::Exhausted { need, have: room });
                }
            }
            let mut fresh = Vec::with_capacity(need as usize);
            for _ in 0..need {
                fresh.push(self.free.pop().unwrap_or_else(|| {
                    let seat = self.next;
                    self.next += 1;
                    seat
                }));
            }
            self.held.entry(ws).or_default().extend(fresh);
        }
        Ok(self.held[&ws][..lanes as usize].to_vec())
    }

    /// Give a released working set's seats back. Idempotent: releasing
    /// twice is a no-op.
    pub fn release(&mut self, ws: WorkingSetId) {
        if let Some(run) = self.held.remove(&ws) {
            self.free.extend(run);
        }
    }

    /// How many seats are neither held nor yet issued. `None` when the book
    /// states no ceiling.
    #[cfg(test)]
    #[must_use]
    pub fn available(&self) -> Option<u32> {
        (self.capacity != 0).then(|| {
            u32::try_from(self.free.len())
                .unwrap_or(u32::MAX)
                .saturating_add(self.capacity.saturating_sub(self.next))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::registry;

    /// Two working sets in one book never sit in one seat.
    #[test]
    fn two_working_sets_never_share_a_seat() {
        let model = registry::register_model(16, &[8], &[4]);
        let stores = registry::get(model, 0);
        let (a, b) = registry::with_kv_lock(&stores.kv, "test", |kv| {
            (kv.create_working_set(), kv.create_working_set())
        });
        let mut book = SeatBook::new(4);
        let first = book.seats(a, 1).expect("a seat");
        let second = book.seats(b, 1).expect("a seat");
        assert_ne!(first, second, "two live sequences, two seats");
    }

    /// The refusal states both numbers and seats nothing.
    #[test]
    fn a_fire_wider_than_the_pools_is_refused_by_name() {
        let model = registry::register_model(16, &[8], &[2]);
        let stores = registry::get(model, 0);
        let (a, b) = registry::with_kv_lock(&stores.kv, "test", |kv| {
            (kv.create_working_set(), kv.create_working_set())
        });
        let mut book = SeatBook::new(2);
        book.seats(a, 1).expect("the first seat");
        let refusal = book.seats(b, 3).expect_err("three seats, one left");
        assert_eq!(refusal, SeatError::Exhausted { need: 3, have: 1 });
        assert_eq!(
            book.available(),
            Some(1),
            "a refused ask seats nothing — the book is as it was"
        );
        assert!(refusal.to_string().contains("Budgets::slots"));
    }

    /// A released working set's seats are reissued to the next sequence.
    #[test]
    fn releasing_a_working_set_returns_its_seats() {
        let model = registry::register_model(16, &[8], &[2]);
        let stores = registry::get(model, 0);
        let (a, b) = registry::with_kv_lock(&stores.kv, "test", |kv| {
            (kv.create_working_set(), kv.create_working_set())
        });
        let mut book = SeatBook::new(2);
        let first = book.seats(a, 2).expect("both seats");
        assert_eq!(book.available(), Some(0));
        book.release(a);
        book.release(a);
        assert_eq!(book.available(), Some(2), "released once, returned once");
        let second = book.seats(b, 2).expect("the recycled seats");
        assert_eq!(
            second.iter().collect::<std::collections::HashSet<_>>(),
            first.iter().collect::<std::collections::HashSet<_>>(),
            "the seats a released sequence sat in are reissued"
        );
    }

}
