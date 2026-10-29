//! A counting semaphore that grants the waiter with the highest schedule key,
//! the earliest among equals, read at each grant. Only the first gate holds
//! waiters below the floor: one at a later gate holds an earlier permit.

use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::oneshot;

use super::ProcessId;
use crate::policy::schedule;

#[derive(Debug)]
pub(crate) struct OrderedSemaphore {
    state: Mutex<State>,
}

#[derive(Debug)]
struct State {
    permits: usize,
    holds: bool,
    next_seq: u64,
    waiters: Vec<Waiter>,
}

#[derive(Debug)]
struct Waiter {
    seq: u64,
    pid: ProcessId,
    granted: oneshot::Sender<()>,
}

pub(crate) struct OrderedPermit {
    semaphore: Arc<OrderedSemaphore>,
}

impl Drop for OrderedPermit {
    fn drop(&mut self) {
        self.semaphore.add_permits(1);
    }
}

impl OrderedSemaphore {
    pub(crate) fn new(permits: usize, holds: bool) -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(State {
                permits,
                holds,
                next_seq: 0,
                waiters: Vec::new(),
            }),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("state lock poisoned")
    }

    pub(crate) fn available_permits(&self) -> usize {
        self.lock().permits
    }

    pub(crate) fn add_permits(&self, n: usize) {
        let mut state = self.lock();
        state.permits += n;
        state.grant();
    }

    #[cfg_attr(
        target_arch = "wasm32",
        allow(dead_code, reason = "no policy runs in the browser")
    )]
    pub(crate) fn regrant(&self) {
        self.lock().grant();
    }

    /// Dropping the future gives up its place, or the permit if one was granted
    /// meanwhile.
    pub(crate) async fn acquire_owned(self: Arc<Self>, pid: ProcessId) -> OrderedPermit {
        let (granted, waiting) = oneshot::channel();
        let seq = {
            let mut state = self.lock();
            let seq = state.next_seq;
            state.next_seq += 1;
            state.waiters.push(Waiter { seq, pid, granted });
            state.grant();
            seq
        };
        let mut queued = Queued {
            semaphore: &self,
            seq,
            done: false,
        };
        // The sender lives in the waiter list until it grants: it is
        // dropped unsent only by `Queued`, which this future owns.
        let _ = waiting.await;
        queued.done = true;
        OrderedPermit {
            semaphore: Arc::clone(&self),
        }
    }
}

impl State {
    fn grant(&mut self) {
        while self.permits > 0 {
            let schedule = schedule::current();
            let best = self
                .waiters
                .iter()
                .enumerate()
                .filter(|(_, w)| !self.holds || !schedule.as_ref().is_some_and(|s| s.held(w.pid)))
                .max_by_key(|(_, w)| {
                    let key = schedule.as_ref().map_or(0, |s| s.key(w.pid));
                    (key, std::cmp::Reverse(w.seq))
                })
                .map(|(index, _)| index);
            let Some(index) = best else {
                return;
            };
            let waiter = self.waiters.swap_remove(index);
            if waiter.granted.send(()).is_ok() {
                self.permits -= 1;
            }
        }
    }
}

struct Queued<'a> {
    semaphore: &'a OrderedSemaphore,
    seq: u64,
    done: bool,
}

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let mut state = self.semaphore.lock();
        match state.waiters.iter().position(|w| w.seq == self.seq) {
            Some(index) => {
                state.waiters.swap_remove(index);
            }
            // Granted, and abandoned before it was taken.
            None => {
                state.permits += 1;
                state.grant();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use super::*;
    use crate::policy::schedule::Table;

    fn pid(n: u128) -> ProcessId {
        ProcessId::from_u128(n)
    }

    async fn grant_order(pids: &[ProcessId]) -> Vec<ProcessId> {
        let semaphore = OrderedSemaphore::new(0, true);
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let mut tasks = Vec::new();
        for &p in pids {
            let gate = Arc::clone(&semaphore);
            let tx = tx.clone();
            tasks.push(tokio::spawn(async move {
                let permit = gate.acquire_owned(p).await;
                tx.send(p).unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
                drop(permit);
            }));
            while semaphore.lock().waiters.len() < tasks.len() {
                tokio::task::yield_now().await;
            }
        }
        semaphore.add_permits(1);
        let mut order = Vec::new();
        for _ in pids {
            order.push(rx.recv().await.unwrap());
        }
        order
    }

    // One test owns the process-wide schedule, so the cases run in order.
    #[tokio::test]
    async fn grants_follow_the_schedule() {
        let _schedule = schedule::TEST_LOCK.write().await;
        let pids = [pid(1), pid(2), pid(3), pid(4)];
        let keys = HashMap::from([(pid(3), 10), (pid(2), 5)]);
        schedule::install(Some(Table::new(keys, None)));
        assert_eq!(
            grant_order(&pids).await,
            [pid(3), pid(2), pid(1), pid(4)],
            "highest key first, then arrival"
        );

        // A floor of 1 holds every unnamed process until it is named.
        let keys = HashMap::from([(pid(4), 1)]);
        schedule::install(Some(Table::new(keys, Some(1))));
        let semaphore = OrderedSemaphore::new(1, true);
        let held = tokio::spawn({
            let semaphore = Arc::clone(&semaphore);
            async move { semaphore.acquire_owned(pid(1)).await }
        });
        let named = Arc::clone(&semaphore).acquire_owned(pid(4)).await;
        drop(named);
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!held.is_finished(), "below the floor is held");
        schedule::install(Some(Table::new(HashMap::from([(pid(1), 1)]), Some(1))));
        semaphore.regrant();
        tokio::time::timeout(Duration::from_secs(1), held)
            .await
            .expect("naming it releases it")
            .unwrap();

        // A waiter that gives up returns its place, or its permit.
        schedule::install(None);
        let semaphore = OrderedSemaphore::new(0, true);
        let abandoned = tokio::spawn({
            let semaphore = Arc::clone(&semaphore);
            async move { semaphore.acquire_owned(pid(1)).await }
        });
        while semaphore.lock().waiters.is_empty() {
            tokio::task::yield_now().await;
        }
        abandoned.abort();
        let _ = abandoned.await;
        semaphore.add_permits(1);
        assert_eq!(semaphore.available_permits(), 1);
    }
}
