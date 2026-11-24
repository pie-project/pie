//! A counting semaphore that grants the waiter with the highest schedule key,
//! the earliest among equals, read at each grant. A gate that holds grants
//! no held waiter (prewarm, and execution: a held process is not started).
//!
//! Gates are passed in order, and no process passes one ahead of a process
//! of higher key waiting at an earlier gate (and not held there): a later
//! arrival of higher priority is not overtaken by processes that went
//! through the earlier gates before its row landed.

use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::oneshot;

use super::ProcessId;
use crate::policy::schedule;

#[derive(Debug)]
pub(crate) struct OrderedSemaphore {
    /// Which gate, for what a long wait says.
    name: &'static str,
    state: Mutex<State>,
    /// The gates a process passes before this one.
    upstream: Vec<Arc<OrderedSemaphore>>,
    /// The gates after this one: they may grant once a waiter here leaves.
    downstream: Mutex<Vec<std::sync::Weak<OrderedSemaphore>>>,
}

/// How long a waiter waits before it says where, and again after each
/// such span: what a stall at the gates is read from.
const WAIT_SAYS_AFTER: std::time::Duration = std::time::Duration::from_secs(5);

#[derive(Debug)]
struct State {
    permits: usize,
    /// Who holds this gate's permits.
    holders: Vec<ProcessId>,
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
    pid: ProcessId,
}

impl Drop for OrderedPermit {
    fn drop(&mut self) {
        self.semaphore.release(self.pid);
    }
}

impl OrderedSemaphore {
    pub(crate) fn new(name: &'static str, permits: usize, holds: bool) -> Arc<Self> {
        Self::after(name, permits, holds, Vec::new())
    }

    /// A gate passed after `upstream`: the gate just before it. (Looking
    /// further back would bar on processes that wait, at the gate between,
    /// for permits this gate's waiters hold.)
    pub(crate) fn after(
        name: &'static str,
        permits: usize,
        holds: bool,
        upstream: Vec<Arc<OrderedSemaphore>>,
    ) -> Arc<Self> {
        let gate = Arc::new(Self {
            name,
            state: Mutex::new(State {
                permits,
                holders: Vec::new(),
                holds,
                next_seq: 0,
                waiters: Vec::new(),
            }),
            upstream,
            downstream: Mutex::default(),
        });
        for earlier in &gate.upstream {
            earlier
                .downstream
                .lock()
                .expect("downstream lock poisoned")
                .push(Arc::downgrade(&gate));
        }
        gate
    }

    /// The highest key a process waits with at an earlier gate, not held
    /// there, that will get a permit there without waiting on this gate:
    /// the earlier gate has one free, or one of its holders is not waiting
    /// here (it gives its permit back without this gate's help). A gate
    /// whose every permit is held by waiters here bars nothing, so none
    /// waits on the other. Earlier gates are locked under this one's lock,
    /// never the other way round.
    fn bar(&self, here: &State) -> Option<i64> {
        let schedule = schedule::current();
        self.upstream
            .iter()
            .flat_map(|gate| {
                let state = gate.lock();
                let frees = state.permits > 0
                    || state
                        .holders
                        .iter()
                        .any(|holder| !here.waiters.iter().any(|w| w.pid == *holder));
                let holds = state.holds;
                let key = |pid| schedule.as_ref().map_or(0, |s| s.key(pid));
                // Past it and not yet here (setting up between a program's
                // bind and its first fire, which takes this gate): it is
                // on its way. An idle process gives up both seats at once.
                let on_the_way = state
                    .holders
                    .iter()
                    .filter(|pid| {
                        !here.holders.contains(pid) && !here.waiters.iter().any(|w| w.pid == **pid)
                    })
                    .map(|pid| key(*pid))
                    .max();
                let waiting = frees
                    .then(|| {
                        state
                            .waiters
                            .iter()
                            .filter(|w| !holds || !schedule.as_ref().is_some_and(|s| s.held(w.pid)))
                            .map(|w| key(w.pid))
                            .max()
                    })
                    .flatten();
                on_the_way.max(waiting)
            })
            .max()
    }

    /// A waiter left this gate: a later one may grant what it held back.
    fn wake_downstream(&self) {
        let later: Vec<_> = self
            .downstream
            .lock()
            .expect("downstream lock poisoned")
            .iter()
            .filter_map(std::sync::Weak::upgrade)
            .collect();
        for gate in later {
            gate.regrant();
        }
    }

    fn grant(&self, state: &mut State) {
        let bar = self.bar(state);
        state.grant(self.name, bar);
    }

    fn release(&self, pid: ProcessId) {
        {
            let mut state = self.lock();
            if let Some(at) = state.holders.iter().position(|h| *h == pid) {
                state.holders.swap_remove(at);
            }
            state.permits += 1;
            self.grant(&mut state);
        }
        self.wake_downstream();
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().expect("state lock poisoned")
    }

    pub(crate) fn available_permits(&self) -> usize {
        self.lock().permits
    }

    pub(crate) fn add_permits(&self, n: usize) {
        {
            let mut state = self.lock();
            state.permits += n;
            self.grant(&mut state);
        }
        self.wake_downstream();
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code, reason = "no policy runs in the browser"))]
    pub(crate) fn regrant(&self) {
        {
            let mut state = self.lock();
            self.grant(&mut state);
        }
        self.wake_downstream();
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
            self.grant(&mut state);
            seq
        };
        self.wake_downstream();
        let mut queued = Queued {
            semaphore: &self,
            pid,
            seq,
            done: false,
        };
        // The sender lives in the waiter list until it grants: it is
        // dropped unsent only by `Queued`, which this future owns.
        tokio::pin!(waiting);
        let mut waited = std::time::Duration::ZERO;
        loop {
            match crate::rt::time::timeout(WAIT_SAYS_AFTER, &mut waiting).await {
                Ok(_) => break,
                Err(_) => {
                    waited += WAIT_SAYS_AFTER;
                    let (permits, waiters, held) = {
                        let state = self.lock();
                        let schedule = schedule::current();
                        let held = state
                            .waiters
                            .iter()
                            .filter(|w| schedule.as_ref().is_some_and(|s| s.held(w.pid)))
                            .count();
                        (state.permits, state.waiters.len(), held)
                    };
                    tracing::warn!(
                        gate = self.name,
                        %pid,
                        waited_s = waited.as_secs(),
                        permits,
                        waiters,
                        held_below_floor = held,
                        "a process waits at an admission gate"
                    );
                }
            }
        }
        queued.done = true;
        OrderedPermit {
            semaphore: Arc::clone(&self),
            pid,
        }
    }
}

impl State {
    /// Grants while permits last, never to a waiter whose key is below
    /// `bar`, the highest key waiting at an earlier gate.
    fn grant(&mut self, gate: &'static str, bar: Option<i64>) {
        while self.permits > 0 {
            let schedule = schedule::current();
            let key = |pid| schedule.as_ref().map_or(0, |s| s.key(pid));
            let best = self
                .waiters
                .iter()
                .enumerate()
                .filter(|(_, w)| !self.holds || !schedule.as_ref().is_some_and(|s| s.held(w.pid)))
                .filter(|(_, w)| bar.is_none_or(|bar| key(w.pid) >= bar))
                .max_by_key(|(_, w)| {
                    let key = schedule.as_ref().map_or(0, |s| s.key(w.pid));
                    (key, std::cmp::Reverse(w.seq))
                })
                .map(|(index, _)| index);
            let Some(index) = best else {
                return;
            };
            let waiter = self.waiters.swap_remove(index);
            tracing::trace!(
                target: "pie::admission",
                gate,
                ?bar,
                pid = %waiter.pid,
                key = schedule.as_ref().map_or(0, |s| s.key(waiter.pid)),
                waiting = self.waiters.len(),
                "granted"
            );
            if waiter.granted.send(()).is_ok() {
                self.permits -= 1;
                self.holders.push(waiter.pid);
            }
        }
    }
}

struct Queued<'a> {
    semaphore: &'a OrderedSemaphore,
    pid: ProcessId,
    seq: u64,
    done: bool,
}

impl Drop for Queued<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        {
            let mut state = self.semaphore.lock();
            match state.waiters.iter().position(|w| w.seq == self.seq) {
                Some(index) => {
                    state.waiters.swap_remove(index);
                }
                // Granted, and abandoned before it was taken.
                None => {
                    if let Some(at) = state.holders.iter().position(|h| *h == self.pid) {
                        state.holders.swap_remove(at);
                    }
                    state.permits += 1;
                    self.semaphore.grant(&mut state);
                }
            }
        }
        self.semaphore.wake_downstream();
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
        let semaphore = OrderedSemaphore::new("test", 0, true);
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

    #[tokio::test]
    async fn an_earlier_gate_with_no_permit_bars_nothing() {
        let _schedule = schedule::TEST_LOCK.write().await;
        let (low, high) = (pid(11), pid(12));
        schedule::install(Some(Table::new(
            HashMap::from([(low, 1), (high, 9)]),
            None,
            HashMap::new(),
            None,
        )));
        let bind = OrderedSemaphore::new("bind", 0, false);
        let execution = OrderedSemaphore::after("execution", 1, false, vec![Arc::clone(&bind)]);
        let behind = tokio::spawn({
            let bind = Arc::clone(&bind);
            async move { bind.acquire_owned(high).await }
        });
        while bind.lock().waiters.is_empty() {
            tokio::task::yield_now().await;
        }
        // Bind has no permit: the high one cannot pass it, so it bars nothing.
        let first = Arc::clone(&execution).acquire_owned(low).await;
        drop(first);
        // Given a permit, bind lets the high one through.
        bind.add_permits(1);
        let high_bound = tokio::time::timeout(Duration::from_secs(1), behind)
            .await
            .unwrap()
            .unwrap();
        drop(high_bound);
        schedule::install(None);
    }

    #[tokio::test]
    async fn no_process_passes_a_gate_ahead_of_a_higher_key_that_could_pass_the_earlier_one() {
        let _schedule = schedule::TEST_LOCK.write().await;
        let (low, high) = (pid(21), pid(22));
        schedule::install(Some(Table::new(
            HashMap::from([(low, 1), (high, 9)]),
            None,
            HashMap::new(),
            None,
        )));
        // The high one waits at bind, which has a permit it is about to take;
        // execution has one too, and the low one asks for it first.
        let bind = OrderedSemaphore::new("bind", 0, false);
        let execution = OrderedSemaphore::after("execution", 1, false, vec![Arc::clone(&bind)]);
        {
            let mut state = bind.lock();
            let (granted, _) = oneshot::channel();
            state.waiters.push(Waiter {
                seq: 0,
                pid: high,
                granted,
            });
            state.permits = 1;
        }
        let low_waits = tokio::spawn({
            let execution = Arc::clone(&execution);
            async move { execution.acquire_owned(low).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!low_waits.is_finished(), "barred by the higher key at bind");
        bind.lock().waiters.clear();
        bind.regrant();
        tokio::time::timeout(Duration::from_secs(1), low_waits)
            .await
            .expect("passes once it has gone")
            .unwrap();
        schedule::install(None);
    }

    #[tokio::test]
    async fn a_higher_key_between_two_gates_is_not_overtaken() {
        let _schedule = schedule::TEST_LOCK.write().await;
        let (low, high) = (pid(31), pid(32));
        schedule::install(Some(Table::new(
            HashMap::from([(low, 1), (high, 9)]),
            None,
            HashMap::new(),
            None,
        )));
        let bind = OrderedSemaphore::new("bind", 1, false);
        let execution = OrderedSemaphore::after("execution", 1, false, vec![Arc::clone(&bind)]);
        // The high one is bound, setting up, not yet at execution.
        let bound = Arc::clone(&bind).acquire_owned(high).await;
        let low_waits = tokio::spawn({
            let execution = Arc::clone(&execution);
            async move { execution.acquire_owned(low).await }
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(
            !low_waits.is_finished(),
            "the low one waits for the high one on its way"
        );
        let running = Arc::clone(&execution).acquire_owned(high).await;
        // Done, it gives up its seat: bind first, then execution.
        drop(bound);
        drop(running);
        tokio::time::timeout(Duration::from_secs(1), low_waits)
            .await
            .expect("then it passes")
            .unwrap();
        schedule::install(None);
    }

    #[tokio::test]
    async fn waiters_holding_the_gate_before_are_not_barred_by_who_waits_for_it() {
        // The stall VTC found: execution's waiters hold every bind permit;
        // higher keys wait at bind, holding prewarm. Execution has permits
        // and must use them, or no one moves.
        let _schedule = schedule::TEST_LOCK.write().await;
        let (low, high) = (pid(41), pid(42));
        schedule::install(Some(Table::new(
            HashMap::from([(low, 1), (high, 9)]),
            None,
            HashMap::new(),
            None,
        )));
        let prewarm = OrderedSemaphore::new("prewarm", 1, true);
        let bind = OrderedSemaphore::after("bind", 1, false, vec![Arc::clone(&prewarm)]);
        let execution = OrderedSemaphore::after("execution", 1, true, vec![Arc::clone(&bind)]);
        let low_bound = Arc::clone(&bind).acquire_owned(low).await;
        let high_warm = Arc::clone(&prewarm).acquire_owned(high).await;
        let high_waits = tokio::spawn({
            let bind = Arc::clone(&bind);
            async move { bind.acquire_owned(high).await }
        });
        while bind.lock().waiters.is_empty() {
            tokio::task::yield_now().await;
        }
        let low_runs = tokio::time::timeout(
            Duration::from_secs(1),
            Arc::clone(&execution).acquire_owned(low),
        )
        .await
        .expect("execution is not barred by a waiter for the bind permit it holds");
        drop(low_bound);
        drop(low_runs);
        let high_bound = tokio::time::timeout(Duration::from_secs(1), high_waits)
            .await
            .unwrap()
            .unwrap();
        drop((high_bound, high_warm));
        schedule::install(None);
    }

    // One test owns the process-wide schedule, so the cases run in order.
    #[tokio::test]
    async fn grants_follow_the_schedule() {
        let _schedule = schedule::TEST_LOCK.write().await;
        let pids = [pid(1), pid(2), pid(3), pid(4)];
        let keys = HashMap::from([(pid(3), 10), (pid(2), 5)]);
        schedule::install(Some(Table::new(keys, None, HashMap::new(), None)));
        assert_eq!(
            grant_order(&pids).await,
            [pid(3), pid(2), pid(1), pid(4)],
            "highest key first, then arrival"
        );

        // A hold default holds every process its own row does not release.
        let holds = HashMap::from([(pid(4), false)]);
        schedule::install(Some(Table::new(HashMap::new(), None, holds, Some(true))));
        let semaphore = OrderedSemaphore::new("test", 1, true);
        let held = tokio::spawn({
            let semaphore = Arc::clone(&semaphore);
            async move { semaphore.acquire_owned(pid(1)).await }
        });
        let named = Arc::clone(&semaphore).acquire_owned(pid(4)).await;
        drop(named);
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!held.is_finished(), "a held process is not granted");
        let holds = HashMap::from([(pid(1), false)]);
        schedule::install(Some(Table::new(HashMap::new(), None, holds, Some(true))));
        semaphore.regrant();
        tokio::time::timeout(Duration::from_secs(1), held)
            .await
            .expect("its row releases it")
            .unwrap();

        // A waiter that gives up returns its place, or its permit.
        schedule::install(None);
        let semaphore = OrderedSemaphore::new("test", 0, true);
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
