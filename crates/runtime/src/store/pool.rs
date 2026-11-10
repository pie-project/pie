use std::marker::PhantomData;
use std::ops::Range;

pub trait PoolId: Copy {
    fn from_index(index: u32) -> Self;
    fn index(self) -> u32;
}

const WORD: u32 = u64::BITS;

/// The word of `free` that holds offset `offset` and its bit there.
fn locate(offset: u32) -> (usize, u64) {
    ((offset / WORD) as usize, 1 << (offset % WORD))
}

/// Ids handed out lowest first, so what is held packs toward the bottom of
/// the range and a tail stays empty to give back.
///
/// The kv store's host and disk slots come from here. The device pools take
/// their ids from [`super::chunks::ShapePool`]s, which are checked against
/// this pool's epochs and limits, call for call.
#[cfg_attr(not(test), allow(dead_code))]
pub struct Pool<I> {
    /// A set bit: the id at that offset past `base` is free.
    free: Vec<u64>,
    /// No word before this one has a free id.
    lowest: usize,
    /// Free ids under `limit`.
    available: usize,
    pending: Vec<(u64, Vec<I>)>,
    base: u32,
    capacity: u32,
    /// Offsets at and past this are not handed out.
    limit: u32,
    id: PhantomData<I>,
}

#[cfg_attr(not(test), allow(dead_code))]
impl<I: PoolId> Pool<I> {
    pub fn new(capacity: u32) -> Self {
        Self::new_range(0, capacity)
    }

    pub fn new_range(base: u32, capacity: u32) -> Self {
        base.checked_add(capacity)
            .expect("pool id range overflows u32");
        let mut free = vec![u64::MAX; capacity.div_ceil(WORD) as usize];
        if let Some(last) = free.last_mut()
            && !capacity.is_multiple_of(WORD)
        {
            *last = (1 << (capacity % WORD)) - 1;
        }
        Self {
            free,
            lowest: 0,
            available: capacity as usize,
            pending: Vec::new(),
            base,
            capacity,
            limit: capacity,
            id: PhantomData,
        }
    }

    /// The `n` lowest free ids, in ascending order, or none when fewer than
    /// `n` stand free under the limit: the lowest are then all under it.
    pub fn try_alloc_n(&mut self, n: usize) -> Option<Vec<I>> {
        if self.available < n {
            return None;
        }
        let mut taken = Vec::with_capacity(n);
        let mut at = self.lowest;
        while taken.len() < n {
            let word = &mut self.free[at];
            while *word != 0 && taken.len() < n {
                let bit = word.trailing_zeros();
                *word &= *word - 1;
                taken.push(I::from_index(self.base + at as u32 * WORD + bit));
            }
            if *word == 0 {
                at += 1;
            }
        }
        self.lowest = at;
        self.available -= n;
        Some(taken)
    }

    /// Takes `id` out of the free list, as a reservation; false when it is
    /// not free.
    pub fn take(&mut self, id: I) -> bool {
        let offset = id.index().wrapping_sub(self.base);
        if offset >= self.capacity {
            return false;
        }
        let (word, bit) = locate(offset);
        if self.free[word] & bit == 0 {
            return false;
        }
        self.free[word] &= !bit;
        if offset < self.limit {
            self.available -= 1;
        }
        true
    }

    pub fn recycle_after_epoch(&mut self, ids: Vec<I>, epoch: u64) {
        if !ids.is_empty() {
            self.pending.push((epoch, ids));
        }
    }

    pub fn release_reserved(&mut self, ids: Vec<I>) {
        self.give_back(ids);
    }

    pub fn retire_through(&mut self, epoch: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= epoch {
                let (_, ids) = self.pending.swap_remove(i);
                self.give_back(ids);
            } else {
                i += 1;
            }
        }
    }

    fn give_back(&mut self, ids: Vec<I>) {
        for id in ids {
            debug_assert!(self.contains(id));
            let offset = id.index() - self.base;
            let (word, bit) = locate(offset);
            debug_assert_eq!(self.free[word] & bit, 0, "id {} freed twice", id.index());
            self.free[word] |= bit;
            self.lowest = self.lowest.min(word);
            if offset < self.limit {
                self.available += 1;
            }
        }
    }

    /// Stops handing out ids at `limit` and up, or hands them out again.
    /// An id already out stays out until it is given back; [`Pool::tail_held`]
    /// counts those.
    pub fn set_limit(&mut self, limit: u32) {
        self.limit = limit.min(self.capacity);
        self.available = self.free_in(0..self.limit);
    }

    /// The ids at and past the limit still out, waiting on an epoch or on
    /// their holder.
    pub fn tail_held(&self) -> usize {
        (self.capacity - self.limit) as usize - self.free_in(self.limit..self.capacity)
    }

    /// One past the highest offset still out, at least the limit: what
    /// lies above it can go now, whatever the rest of the tail waits on.
    pub fn held_end(&self) -> u32 {
        self.high_end().max(self.limit)
    }

    /// One past the highest offset still out: how far a limit reaches
    /// without cutting anything off, however many holes lie below.
    pub fn high_end(&self) -> u32 {
        let (mut word, mut end) = (self.free.len(), 0);
        while word > 0 {
            word -= 1;
            let unused = (self.capacity - word as u32 * WORD).min(WORD);
            let mask = if unused == WORD {
                u64::MAX
            } else {
                (1 << unused) - 1
            };
            let out = !self.free[word] & mask;
            if out != 0 {
                end = word as u32 * WORD + (u64::BITS - out.leading_zeros());
                break;
            }
        }
        end
    }

    fn free_in(&self, offsets: Range<u32>) -> usize {
        offsets
            .filter(|&offset| {
                let (word, bit) = locate(offset);
                self.free[word] & bit != 0
            })
            .count()
    }

    pub fn available(&self) -> usize {
        self.available
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn limit(&self) -> u32 {
        self.limit
    }

    pub fn contains(&self, id: I) -> bool {
        (self.base..self.base.saturating_add(self.capacity)).contains(&id.index())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl PoolId for u32 {
        fn from_index(index: u32) -> Self {
            index
        }
        fn index(self) -> u32 {
            self
        }
    }

    #[test]
    fn a_freed_low_id_is_handed_out_before_any_higher_one() {
        let mut pool = Pool::<u32>::new_range(10, 130);
        assert_eq!(pool.try_alloc_n(3), Some(vec![10, 11, 12]));
        pool.release_reserved(vec![11]);
        assert_eq!(pool.try_alloc_n(2), Some(vec![11, 13]));
        assert_eq!(pool.available(), 126);
    }

    #[test]
    fn a_limit_holds_the_tail_back_and_lifting_it_returns_the_tail() {
        let mut pool = Pool::<u32>::new(130);
        let held = pool.try_alloc_n(100).unwrap();
        pool.set_limit(70);
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.tail_held(), 30);
        assert_eq!(pool.try_alloc_n(1), None);

        pool.release_reserved(held[60..].to_vec());
        assert_eq!(pool.tail_held(), 0);
        assert_eq!(pool.available(), 10);
        assert!(pool.try_alloc_n(10).unwrap().iter().all(|&id| id < 70));

        pool.set_limit(130);
        assert_eq!(pool.available(), 60);
        assert_eq!(pool.try_alloc_n(60).unwrap()[0], 70);
    }

    #[test]
    fn an_id_recycled_past_the_limit_waits_its_epoch_and_stays_out() {
        let mut pool = Pool::<u32>::new(8);
        let held = pool.try_alloc_n(8).unwrap();
        pool.set_limit(4);
        pool.recycle_after_epoch(held[4..].to_vec(), 7);
        assert_eq!(pool.tail_held(), 4, "a frame may still read them");
        pool.retire_through(7);
        assert_eq!(pool.tail_held(), 0);
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn the_held_end_is_past_the_highest_id_out_and_never_under_the_limit() {
        let mut pool = Pool::<u32>::new(130);
        let held = pool.try_alloc_n(100).unwrap();
        pool.set_limit(10);
        assert_eq!(pool.held_end(), 100);
        pool.release_reserved(held[50..].to_vec());
        assert_eq!(pool.held_end(), 50);
        pool.release_reserved(held[..50].to_vec());
        assert_eq!(pool.held_end(), 10);
    }

    #[test]
    fn taking_an_id_keeps_the_count_and_refuses_a_stranger() {
        let mut pool = Pool::<u32>::new_range(4, 4);
        assert!(pool.take(5));
        assert!(!pool.take(5));
        assert!(!pool.take(3));
        assert!(!pool.take(8));
        assert_eq!(pool.available(), 3);
        assert_eq!(pool.try_alloc_n(3), Some(vec![4, 6, 7]));
    }

    #[test]
    fn an_empty_pool_hands_out_nothing_and_stays_whole() {
        let mut pool = Pool::<u32>::new(0);
        assert_eq!(pool.try_alloc_n(0), Some(vec![]));
        assert_eq!(pool.try_alloc_n(1), None);
    }
}
