#![allow(dead_code)]

pub trait PoolId: Copy {
    fn from_index(index: u32) -> Self;
    fn index(self) -> u32;
}

pub struct Pool<I> {
    free: Vec<I>,
    pending: Vec<(u64, Vec<I>)>,
    base: u32,
    capacity: u32,
}

impl<I: PoolId> Pool<I> {
    pub fn new(capacity: u32) -> Self {
        Self::new_range(0, capacity)
    }

    pub fn new_range(base: u32, capacity: u32) -> Self {
        let end = base
            .checked_add(capacity)
            .expect("pool id range overflows u32");
        Self {
            free: (base..end).rev().map(I::from_index).collect(),
            pending: Vec::new(),
            base,
            capacity,
        }
    }

    pub fn try_alloc_n(&mut self, n: usize) -> Option<Vec<I>> {
        if self.free.len() < n {
            return None;
        }
        let at = self.free.len() - n;
        Some(self.free.split_off(at))
    }

    pub fn recycle_after_epoch(&mut self, ids: Vec<I>, epoch: u64) {
        if !ids.is_empty() {
            self.pending.push((epoch, ids));
        }
    }

    pub fn release_reserved(&mut self, ids: Vec<I>) {
        debug_assert!(ids.iter().all(|id| {
            id.index() >= self.base && id.index() < self.base.saturating_add(self.capacity)
        }));
        debug_assert!(
            ids.iter()
                .all(|id| !self.free.iter().any(|free| free.index() == id.index()))
        );
        self.free.extend(ids);
    }

    pub fn retire_through(&mut self, epoch: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= epoch {
                let (_, ids) = self.pending.swap_remove(i);
                self.free.extend(ids);
            } else {
                i += 1;
            }
        }
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn pending_recycle(&self) -> usize {
        self.pending.iter().map(|(_, ids)| ids.len()).sum()
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}
