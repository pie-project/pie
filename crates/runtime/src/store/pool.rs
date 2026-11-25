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
    /// Told each time ids come free ([`Self::set_room_signal`]): the disk
    /// tier's slots, whose return a waiter blocked on swap room waits on.
    signal: Option<super::slots::RoomSignal>,
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
            signal: None,
        }
    }

    /// Calls `signal` each time ids come free (lock-free: a give-back runs
    /// under its store's lock).
    pub fn set_room_signal(&mut self, signal: super::slots::RoomSignal) {
        self.signal = Some(signal);
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

    /// `n` runs of `span` adjacent free ids each, every run starting at a
    /// multiple of `span` (a whole chunk of a pool laid out in chunks of
    /// `span` ids), lowest first, as each run's first id; all or none.
    pub fn try_alloc_aligned_runs(&mut self, n: usize, span: u32) -> Option<Vec<I>> {
        let span = span.max(1);
        if self.available < n.saturating_mul(span as usize) {
            return None;
        }
        let free = |offset: u32| {
            let (word, bit) = locate(offset);
            self.free[word] & bit != 0
        };
        let firsts: Vec<u32> = (0..self.capacity / span)
            .map(|chunk| chunk * span)
            .filter(|&first| (first..first + span).all(free))
            .take(n)
            .collect();
        if firsts.len() < n {
            return None;
        }
        for &first in &firsts {
            for offset in first..first + span {
                let (word, bit) = locate(offset);
                self.free[word] &= !bit;
            }
        }
        self.available -= n * span as usize;
        while self.lowest < self.free.len() && self.free[self.lowest] == 0 {
            self.lowest += 1;
        }
        Some(
            firsts
                .into_iter()
                .map(|first| I::from_index(self.base + first))
                .collect(),
        )
    }

    /// Runs [`Self::try_alloc_aligned_runs`] took, each from its first id,
    /// free again.
    pub fn release_runs(&mut self, firsts: &[I], span: u32) {
        let ids = firsts
            .iter()
            .flat_map(|first| {
                let first = first.index();
                (first..first + span.max(1)).map(I::from_index)
            })
            .collect();
        self.give_back(ids);
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
        let freed = !ids.is_empty();
        for id in ids {
            debug_assert!(self.contains(id));
            let offset = id.index() - self.base;
            let (word, bit) = locate(offset);
            debug_assert_eq!(self.free[word] & bit, 0, "id {} freed twice", id.index());
            self.free[word] |= bit;
            self.lowest = self.lowest.min(word);
            self.available += 1;
        }
        if freed && let Some(signal) = &self.signal {
            (signal.0)(super::slots::RoomCount {
                tier: super::slots::RoomTier::Swap,
                chunks: 0,
                slots: self.available as u64,
            });
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

}
