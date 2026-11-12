use std::marker::PhantomData;

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
/// the range.
///
/// The kv store's disk slots come from here. The device and host pools take
/// their ids from [`super::chunks::ShapePool`]s.
pub struct Pool<I> {
    /// A set bit: the id at that offset past `base` is free.
    free: Vec<u64>,
    /// No word before this one has a free id.
    lowest: usize,
    available: usize,
    base: u32,
    capacity: u32,
    id: PhantomData<I>,
}

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
            base,
            capacity,
            id: PhantomData,
        }
    }

    /// The `n` lowest free ids, in ascending order, or none when fewer than
    /// `n` stand free.
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
        self.available -= 1;
        true
    }

    pub fn release_reserved(&mut self, ids: Vec<I>) {
        self.give_back(ids);
    }

    fn give_back(&mut self, ids: Vec<I>) {
        for id in ids {
            debug_assert!(self.contains(id));
            let offset = id.index() - self.base;
            let (word, bit) = locate(offset);
            debug_assert_eq!(self.free[word] & bit, 0, "id {} freed twice", id.index());
            self.free[word] |= bit;
            self.lowest = self.lowest.min(word);
            self.available += 1;
        }
    }

    pub fn available(&self) -> usize {
        self.available
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
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
