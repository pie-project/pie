use crate::error::{Error, Result};
use num_traits::PrimInt;
use std::collections::BTreeSet;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone)]
pub struct IdPool<T> {
    inner: Arc<Mutex<IdPoolInner<T>>>,
}

#[derive(Debug)]
struct IdPoolInner<T> {
    next: T,
    free: BTreeSet<T>,
    max_capacity: T,
}

#[derive(Debug)]
pub struct IdGuard<T: PrimInt + Send + 'static> {
    pool: Arc<Mutex<IdPoolInner<T>>>,
    id: T,
}

impl<T> IdPool<T>
where
    T: PrimInt,
{
    pub fn new(max_capacity: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(IdPoolInner {
                next: T::zero(),
                free: BTreeSet::new(),
                max_capacity,
            })),
        }
    }

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
    fn acquire_id(&mut self) -> Result<T> {
        if let Some(&id) = self.free.iter().next() {
            self.free.remove(&id);
            Ok(id)
        } else if self.next < self.max_capacity {
            let addr = self.next;
            self.next = self.next + T::one();
            Ok(addr)
        } else {
            Err(Error::TooManyRequests)
        }
    }

    fn release_id(&mut self, addr: T) {
        self.free.insert(addr);

        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }
    }

    fn tail_optimization(&mut self) {
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
        let pool = Arc::clone(&self.pool);
        let id = self.id;
        tokio::spawn(async move {
            pool.lock().await.release_id(id);
        });
    }
}
