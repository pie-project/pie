//! Utility functions for the Management Service

use anyhow::{Error, Result};
use num_traits::PrimInt;
use std::collections::BTreeSet;
use std::hash::Hash;
use std::sync::atomic::{AtomicIsize, Ordering};

/// A fast, thread-safe counter.
#[derive(Debug)]
pub struct Counter {
    count: AtomicIsize,
}

impl Counter {
    /// Creates a new counter starting at the given initial value.
    pub fn new(initial: isize) -> Self {
        Self {
            count: AtomicIsize::new(initial),
        }
    }

    /// Increments the counter by 1.
    pub fn inc(&self) -> isize {
        // Using relaxed ordering because we only care about the counter's value.
        self.count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements the counter by 1.
    pub fn dec(&self) -> isize {
        self.count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Returns the current count.
    pub fn get(&self) -> isize {
        self.count.load(Ordering::Relaxed)
    }
}

/// A very fast bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
#[derive(Debug)]
pub struct IdPool<T> {
    /// The next (never‑allocated) ID.
    next: T,
    /// The set of freed IDs.
    free: BTreeSet<T>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: T,
}

impl<T> IdPool<T>
where
    T: PrimInt,
{
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: T) -> Self {
        Self {
            next: T::zero(),
            free: BTreeSet::new(),
            max_capacity,
        }
    }

    pub fn set_capacity(&mut self, capacity: T) -> Result<()> {
        if capacity < self.next {
            return Err(Error::msg("Cannot set capacity lower than the next ID"));
        }
        self.max_capacity = capacity;
        Ok(())
    }

    pub fn capacity(&self) -> T {
        self.max_capacity
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Some(id)` if an ID is available, or `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Result<T> {
        if let Some(&id) = self.free.iter().next() {
            // There is a freed ID available. Remove and return it.
            self.free.remove(&id);
            Ok(id)
        } else if self.next < self.max_capacity {
            // No freed IDs available; allocate a fresh one.
            let addr = self.next;
            self.next = self.next + T::one();
            Ok(addr)
        } else {
            // Pool is exhausted.
            Err(Error::msg("ID pool exhausted"))
        }
    }

    pub fn acquire_many(&mut self, count: usize) -> Result<Vec<T>> {
        // check if we have enough available ids
        if self.available() < count {
            return Err(Error::msg("ID pool exhausted"));
        }

        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.acquire()?);
        }
        Ok(result)
    }

    /// Release an ID back into the pool so it can be re-used.

    pub fn release(&mut self, addr: T) -> Result<()> {
        // Only allow releasing IDs that were allocated.
        if addr >= self.next {
            return Err(Error::msg("ID was never allocated"));
        }

        // Insert the id into the free set.
        self.free.insert(addr);

        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }

        Ok(())
    }

    pub fn release_many(&mut self, addrs: &[T]) -> Result<()> {
        for &addr in addrs {
            if addr >= self.next {
                return Err(Error::msg("ID was never allocated"));
            }
            self.free.insert(addr);
        }
        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }
        Ok(())
    }

    /// Returns the number of IDs that are available for allocation.
    ///
    /// This equals the number of IDs that have been freed plus the difference
    /// between the maximum capacity and the next never‑allocated ID.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity - self.next).to_usize().unwrap()
    }

    fn tail_optimization(&mut self) {
        // Tail optimization: if the largest freed id is exactly next-1,
        // collapse the free block by decrementing `next`.
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - T::one() {
                self.free.remove(&last);
                self.next = self.next - T::one();
            } else {
                break;
            }
        }
    }
}
