//! Utility functions for the Management Service

use anyhow::{Error, Result};
use num_traits::PrimInt;
use std::collections::BTreeSet;

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
