//! Utility functions for the Management Service

use anyhow::{Result, anyhow};
use num_traits::PrimInt;
use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
///
/// This pool uses a guard pattern - IDs are automatically released when the
/// guard is dropped, similar to `MutexGuard` in the standard library.
#[derive(Debug, Clone)]
pub struct IdPool<T> {
    inner: Arc<Mutex<IdPoolInner<T>>>,
}

#[derive(Debug)]
struct IdPoolInner<T> {
    /// The next (never‑allocated) ID.
    next: T,
    /// The set of freed IDs.
    free: BTreeSet<T>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: T,
}

/// A guard that holds an allocated ID and automatically releases it when dropped.
///
/// The ID can be accessed by dereferencing the guard.
#[derive(Debug)]
pub struct IdGuard<T: PrimInt + Send + 'static> {
    pool: Arc<Mutex<IdPoolInner<T>>>,
    id: T,
}

impl<T> IdPool<T>
where
    T: PrimInt,
{
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(IdPoolInner {
                next: T::zero(),
                free: BTreeSet::new(),
                max_capacity,
            })),
        }
    }

    /// Allocate and return the smallest available ID wrapped in a guard.
    ///
    /// The ID will be automatically released when the guard is dropped.
    /// Returns an error if the pool is exhausted.
    pub async fn acquire(&self) -> Result<IdGuard<T>>
    where
        T: Send + 'static,
    {
        let id = self.inner.lock().await.acquire_id()?;
        Ok(IdGuard {
            pool: Arc::clone(&self.inner),
            id,
        })
    }
}

impl<T> IdPoolInner<T>
where
    T: PrimInt,
{
    /// Internal method to allocate an ID.
    fn acquire_id(&mut self) -> Result<T> {
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
            Err(anyhow!("ID pool exhausted"))
        }
    }

    /// Internal method to release an ID back into the pool.
    fn release_id(&mut self, addr: T) {
        // Insert the id into the free set.
        self.free.insert(addr);

        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }
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

impl<T: PrimInt + Send + 'static> Deref for IdGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

impl<T: PrimInt + Send + 'static> Drop for IdGuard<T> {
    fn drop(&mut self) {
        // Spawn a task to release the ID asynchronously since we can't await in Drop.
        // In the future, when `std::future::AsyncDrop` is stable, we can use that instead,
        // and we can avoid wrapping `inner` in an Arc but use a reference instead.
        let pool = Arc::clone(&self.pool);
        let id = self.id;
        tokio::spawn(async move {
            pool.lock().await.release_id(id);
        });
    }
}
