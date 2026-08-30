//! Which pool slot a sequence sits in — the runtime's half of
//! [`Lane::slot`](engine::fire::Lane).
//!
//! # What a slot is, and who was supposed to own it
//!
//! A slot is the per-sequence SEAT in the shell's pools: the block of kv
//! pages a shell-owned page table hands it, the row of the recurrent bank a
//! linear-attention scan reads and writes, and the row
//! `engine_cuda::store::Pools::clear` zeroes when a fresh sequence arrives
//! (`palo` build log 19). `engine::fire`'s header states the same thing from
//! the contract's side: "`Lane::slot` is the sequence's seat in BOTH pools".
//!
//! Two properties follow from that sentence, and this module exists to
//! provide them:
//!
//! * **A seat is the sequence's for as long as the sequence lives.** The
//!   shell clears a slot's recurrent banks on the fire where `held == 0` and
//!   never again, so a sequence that changed seats between two fires would
//!   continue somebody else's state. A seat assigned per FIRE — by position
//!   in the batch, say — cannot be right for a recurrent model.
//! * **No two live sequences share one.** `Step::validate` refuses
//!   a fire where "slot N appears twice", by name, because two lanes seated
//!   together would write one another's cache.
//!
//! The runtime's per-sequence identity is the KV working set: it IS the page
//! table of one sequence, minted at `create`/`fork`/`slice` and released when
//! the last handle to it goes. So the working set owns the seat, and this is
//! the book it owns it in.
//!
//! # A run per working set, not a seat
//!
//! A request is LANES, plural (`engine::fire`'s header): a beam fires B row
//! groups against one page table, and each of those rows is a sequence of its
//! own as far as the pools are concerned. So a working set holds a RUN of
//! seats — one per lane it has ever fired, grown on demand and kept — and
//! lane `i` of every fire of that working set sits in the same seat.
//!
//! # The ceiling
//!
//! `capacity` is what the deployment's engine advertises as
//! [`PoolFacts::state_slots`](engine::caps::PoolFacts) — the same number
//! the contract calls [`Budgets::slots`](engine::load::Budgets) ("how
//! many sequences the pools seat at once") and the same number that sizes
//! this store registry's `RsStore`. A fire that would seat more sequences
//! than that is refused HERE, by name and with both numbers, rather than
//! reaching the shell and coming back as a `Fault::Ceiling` naming a lane
//! index.
//!
//! **A capacity of zero states no ceiling rather than a ceiling of none.**
//! `offload::register_remote_store` registers a peer's stores with no slot
//! count because it has not asked one, and a deployment whose engine
//! advertises nothing would otherwise have every fire refused here. Seats are
//! still unique — that is this module's other property, and it does not need
//! a number — and the shell's own ceiling stays the backstop.

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
    /// A book over `capacity` seats. Zero states no ceiling (module header).
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
    /// run if this fire is wider than any before it.
    ///
    /// The answer is stable: fire `n + 1` of the same working set gets the
    /// same seats fire `n` did, which is what a recurrent bank row depends
    /// on.
    ///
    /// # Errors
    ///
    /// [`SeatError::Exhausted`] when the run would grow past the pools'
    /// slots. All-or-nothing: a refused ask seats nothing, so a fire that
    /// cannot be seated whole leaves the book as it found it.
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
            // Taken before the run is borrowed, so the refusal above is the
            // only way out and it has already returned.
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

    /// Give a released working set's seats back. Idempotent — a working set
    /// released twice returns nothing the second time, which is the shape
    /// `KvLifecycle`'s own idempotent release needs.
    pub fn release(&mut self, ws: WorkingSetId) {
        if let Some(run) = self.held.remove(&ws) {
            self.free.extend(run);
        }
    }

    /// How many seats are neither held nor yet issued. `None` when the book
    /// states no ceiling.
    ///
    /// Read by this module's own tests and nowhere else: a refusal already
    /// carries the number a caller would want ([`SeatError::Exhausted`]'s
    /// `have`), so there is no production reader to write this for. It is
    /// `cfg(test)` rather than `allow(dead_code)` so the absence is stated
    /// rather than masked.
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

    /// Two working sets in one book never sit in one seat — the property
    /// `Step::validate` refuses the absence of.
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

    /// A seat is the sequence's for as long as the sequence lives: the
    /// second fire of a working set sits where the first did, which is what
    /// the recurrent bank row depends on.
    #[test]
    fn a_working_sets_seat_is_the_same_at_every_fire() {
        let model = registry::register_model(16, &[8], &[4]);
        let stores = registry::get(model, 0);
        let ws = registry::with_kv_lock(&stores.kv, "test", |kv| kv.create_working_set());
        let mut book = SeatBook::new(4);
        let first = book.seats(ws, 1).expect("a seat");
        let second = book.seats(ws, 1).expect("a seat");
        assert_eq!(first, second);
        // A wider fire keeps the seats it had and grows the run.
        let wider = book.seats(ws, 3).expect("three seats");
        assert_eq!(wider[0], first[0], "lane 0 keeps its seat");
        assert_eq!(wider.len(), 3);
        assert_eq!(
            wider.iter().collect::<std::collections::HashSet<_>>().len(),
            3,
            "a beam's rows are three sequences and take three seats"
        );
    }

    /// The refusal is by name and states both numbers, and it seats nothing
    /// on the way out.
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

    /// A released working set's seats come back, and the next sequence sits
    /// in one of them.
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

    /// A deployment that states no slot count gets unique seats and no
    /// ceiling — `offload::register_remote_store`'s shape.
    #[test]
    fn a_book_with_no_stated_ceiling_still_seats_uniquely() {
        let model = registry::register_model(16, &[8], &[0]);
        let stores = registry::get(model, 0);
        let sets: Vec<_> = registry::with_kv_lock(&stores.kv, "test", |kv| {
            (0..64).map(|_| kv.create_working_set()).collect()
        });
        let mut book = SeatBook::new(0);
        let seats: std::collections::HashSet<u32> = sets
            .iter()
            .map(|&ws| book.seats(ws, 1).expect("no ceiling refuses nothing")[0])
            .collect();
        assert_eq!(seats.len(), 64);
        assert_eq!(book.available(), None, "no ceiling is not a ceiling of none");
    }
}
