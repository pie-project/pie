use crate::api;
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Errors that can occur while using the ID pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResourcePoolError {
    /// The new capacity is smaller than the number of already allocated IDs.
    CapacityTooSmall,
    /// No more IDs can be allocated.
    PoolExhausted,
    /// The given ID was never allocated or is not currently active.
    IdNotAllocated,
    /// The given ID was already freed and is in the pool.
    IdAlreadyFreed,
}

impl fmt::Display for ResourcePoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResourcePoolError::CapacityTooSmall => write!(f, "New capacity is too small"),
            ResourcePoolError::PoolExhausted => write!(f, "ID pool exhausted"),
            ResourcePoolError::IdNotAllocated => write!(f, "ID was never allocated"),
            ResourcePoolError::IdAlreadyFreed => write!(f, "ID was already freed"),
        }
    }
}

impl std::error::Error for ResourcePoolError {}

pub type Result<T> = std::result::Result<T, ResourcePoolError>;

pub trait Allocator {
    fn allocate(&self, queue: &api::Queue, ids: &[u32]) -> Result<()>;
    fn deallocate(&self, queue: &api::Queue, ids: &[u32]) -> Result<()>;
    fn import(&self, queue: &api::Queue, ids: &[u32], name: &str) -> Result<()>;
}

/// A fast, bounded ID pool that always returns the smallest available ID.
#[derive(Debug)]
pub struct ResourcePool<T: Allocator> {
    allocator: T,
    next: u32,
    free: BTreeSet<u32>,
    max_capacity: u32,
    /// If true, the `allocator` is invoked for IDs reused from the `free` set.
    /// If false, it's only invoked for brand new IDs from the `next` sequence.
    tight: bool,
    /// The number of freed IDs required to trigger tail optimization.
    tail_optimization_threshold: usize,
}

impl<T> ResourcePool<T>
where
    T: Allocator,
{
    /// Creates a new ID pool.
    pub fn new(
        allocator: T,
        max_capacity: u32,
        tight: bool,
        tail_optimization_threshold: usize,
    ) -> Self {
        Self {
            allocator,
            next: 0,
            free: BTreeSet::new(),
            max_capacity,
            tight,
            tail_optimization_threshold,
        }
    }

    /// Sets a new capacity for the pool.
    pub fn set_capacity(&mut self, capacity: u32) -> Result<()> {
        let allocated_count = self.next as usize - self.free.len();
        if capacity < allocated_count as u32 {
            return Err(ResourcePoolError::CapacityTooSmall);
        }
        if let Some(&max_freed) = self.free.iter().next_back() {
            if capacity <= max_freed {
                return Err(ResourcePoolError::CapacityTooSmall);
            }
        }
        self.max_capacity = capacity;
        Ok(())
    }

    /// Acquires and returns the smallest available ID.
    pub fn acquire(&mut self, queue: &api::Queue) -> Result<u32> {
        if let Some(id) = self.free.pop_first() {
            if self.tight {
                self.allocator.allocate(queue, &[id])?;
            }
            Ok(id)
        } else if self.next < self.max_capacity {
            let id = self.next;
            self.allocator.allocate(queue, &[id])?;
            self.next += 1;
            Ok(id)
        } else {
            Err(ResourcePoolError::PoolExhausted)
        }
    }

    /// Acquires many IDs at once.
    pub fn acquire_many(&mut self, queue: &api::Queue, count: usize) -> Result<Vec<u32>> {
        if self.available() < count {
            return Err(ResourcePoolError::PoolExhausted);
        }

        // Step 1: Tentatively select IDs without modifying state.
        let ids_from_free: Vec<u32> = self.free.iter().take(count).copied().collect();
        let remaining_count = count.saturating_sub(ids_from_free.len());

        let next_id_start = self.next;
        let next_id_end = self
            .next
            .saturating_add(remaining_count as u32)
            .min(self.max_capacity);
        let ids_from_next: Vec<u32> = (next_id_start..next_id_end).collect();

        let mut result = ids_from_free.clone();
        result.extend_from_slice(&ids_from_next);

        if result.len() < count {
            return Err(ResourcePoolError::PoolExhausted);
        }

        // Step 2: Perform the fallible allocation.
        let mut to_alloc = Vec::new();
        if self.tight {
            to_alloc.extend_from_slice(&ids_from_free);
        }
        to_alloc.extend_from_slice(&ids_from_next);

        if !to_alloc.is_empty() {
            self.allocator.allocate(queue, &to_alloc)?;
        }

        // Step 3: Commit state changes only after success.
        for id in &ids_from_free {
            self.free.remove(id);
        }
        self.next = next_id_end;

        Ok(result)
    }

    pub fn import(&mut self, queue: &api::Queue, count: usize, name: &str) -> Result<Vec<u32>> {
        // Step 1: Tentatively select IDs without modifying state.
        let ids_from_free: Vec<u32> = self.free.iter().take(count).copied().collect();
        let remaining_count = count.saturating_sub(ids_from_free.len());

        let next_id_start = self.next;
        let next_id_end = self
            .next
            .saturating_add(remaining_count as u32)
            .min(self.max_capacity);
        let ids_from_next: Vec<u32> = (next_id_start..next_id_end).collect();

        let mut result = ids_from_free.clone();
        result.extend_from_slice(&ids_from_next);

        if result.len() < count {
            return Err(ResourcePoolError::PoolExhausted);
        }

        // Step 2: Perform the fallible allocation.
        let mut to_alloc = Vec::new();
        if self.tight {
            to_alloc.extend_from_slice(&ids_from_free);
        }
        to_alloc.extend_from_slice(&ids_from_next);

        if !to_alloc.is_empty() {
            self.allocator.import(queue, &to_alloc, name)?;
        }

        // Step 3: Commit state changes only after success.
        for id in &ids_from_free {
            self.free.remove(id);
        }
        self.next = next_id_end;

        Ok(result)
    }

    /// Releases an ID back into the pool.
    pub fn release(&mut self, queue: &api::Queue, id: u32) -> Result<()> {
        if id >= self.next {
            return Err(ResourcePoolError::IdNotAllocated);
        }
        if !self.free.insert(id) {
            return Err(ResourcePoolError::IdAlreadyFreed);
        }

        if self.tight {
            self.allocator.deallocate(queue, &[id])?;
        }
        if self.free.len() > self.tail_optimization_threshold {
            self.tail_optimization(queue)?;
        }
        Ok(())
    }

    /// Releases multiple IDs back into the pool.
    pub fn release_many(&mut self, queue: &api::Queue, ids: &[u32]) -> Result<()> {
        for &id in ids {
            if id >= self.next {
                return Err(ResourcePoolError::IdNotAllocated);
            }
            if self.free.contains(&id) {
                return Err(ResourcePoolError::IdAlreadyFreed);
            }
        }

        let mut to_deallocate = Vec::new();
        for &id in ids {
            self.free.insert(id);
            if self.tight {
                to_deallocate.push(id);
            }
        }

        if self.tight && !to_deallocate.is_empty() {
            self.allocator.deallocate(queue, &to_deallocate)?;
        }

        if self.free.len() > self.tail_optimization_threshold {
            self.tail_optimization(queue)?;
        }
        Ok(())
    }

    /// Returns the number of IDs available for allocation.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity.saturating_sub(self.next)) as usize
    }

    /// If the largest freed IDs are contiguous with `next`, collapses them back into the main pool.
    fn tail_optimization(&mut self, queue: &api::Queue) -> Result<()> {
        let cur_next = self.next;
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - 1 {
                self.free.pop_last();
                self.next -= 1;
            } else {
                break;
            }
        }
        let diff = cur_next - self.next;

        if diff > 0 && !self.tight {
            let ids_to_dealloc = (self.next..cur_next).collect::<Vec<_>>();
            self.allocator.deallocate(queue, &ids_to_dealloc)?;
        }
        Ok(())
    }
}

/// A reference-counted ID pool that re-uses IDs only when their reference count drops to zero.
#[derive(Debug)]
pub struct RcResourcePool<T: Allocator> {
    pool: ResourcePool<T>,
    /// Stores the reference count for each active ID.
    ref_counts: HashMap<u32, u8>,
}

impl<T> RcResourcePool<T>
where
    T: Allocator,
{
    pub fn new(
        allocator: T,
        max_capacity: u32,
        tight: bool,
        tail_optimization_threshold: usize,
    ) -> Self {
        Self {
            pool: ResourcePool::new(allocator, max_capacity, tight, tail_optimization_threshold),
            ref_counts: HashMap::new(),
        }
    }

    pub fn set_capacity(&mut self, capacity: u32) -> Result<()> {
        self.pool.set_capacity(capacity)
    }

    pub fn increment_rc(&mut self, id: u32) {
        *self.ref_counts.entry(id).or_insert(0) += 1;
    }

    pub fn increment_rc_many(&mut self, ids: &[u32]) {
        for &id in ids {
            self.increment_rc(id);
        }
    }

    pub fn acquire(&mut self, queue: &api::Queue) -> Result<u32> {
        let id = self.pool.acquire(queue)?;
        self.increment_rc(id);
        Ok(id)
    }

    pub fn acquire_many(&mut self, queue: &api::Queue, count: usize) -> Result<Vec<u32>> {
        let ids = self.pool.acquire_many(queue, count)?;
        self.increment_rc_many(&ids);
        Ok(ids)
    }

    pub fn import(&mut self, queue: &api::Queue, count: usize, name: &str) -> Result<Vec<u32>> {
        let ids = self.pool.import(queue, count, name)?;
        self.increment_rc_many(&ids);
        Ok(ids)
    }

    /// Decrements an ID's reference count, releasing it to the pool if the count reaches zero.
    pub fn release(&mut self, queue: &api::Queue, id: u32) -> Result<()> {
        use std::collections::hash_map::Entry;

        match self.ref_counts.entry(id) {
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count -= 1;
                if *count == 0 {
                    entry.remove();
                    self.pool.release(queue, id)?;
                }
                Ok(())
            }
            Entry::Vacant(_) => Err(ResourcePoolError::IdNotAllocated),
        }
    }

    /// Decrements reference counts for multiple IDs.
    pub fn release_many(&mut self, queue: &api::Queue, ids: &[u32]) -> Result<()> {
        let mut to_deallocate = Vec::new();

        for &id in ids {
            if self.ref_counts.get(&id).unwrap_or(&0) == &0 {
                return Err(ResourcePoolError::IdNotAllocated);
            }
        }

        for &id in ids {
            if let Some(count) = self.ref_counts.get_mut(&id) {
                *count -= 1;
                if *count == 0 {
                    to_deallocate.push(id);
                }
            }
        }

        if !to_deallocate.is_empty() {
            for &id in &to_deallocate {
                self.ref_counts.remove(&id);
            }
            self.pool.release_many(queue, &to_deallocate)?;
        }

        Ok(())
    }
}
