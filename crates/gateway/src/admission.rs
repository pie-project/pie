use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use controller_api::{Health, RoutingTable};
use tokio::sync::{Mutex, watch};
use tokio::time::Instant;

const RECHECK: Duration = Duration::from_millis(250);

const SETTLE: Duration = Duration::from_millis(150);

#[derive(Debug, Clone, Copy)]
pub struct AdmissionConfig {
    pub kv_saturate_bucket: u8,
    pub max_inflight_per_worker: u32,
    pub wait: Duration,
}

impl Default for AdmissionConfig {
    fn default() -> Self {
        Self {
            kv_saturate_bucket: 240,
            max_inflight_per_worker: 256,
            wait: Duration::from_secs(90),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionDecision {
    Admit,
    Reject(RejectReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    ClusterSaturated,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::ClusterSaturated => {
                f.write_str("cluster saturated: no healthy worker has KV/seq headroom")
            }
        }
    }
}

pub fn admit(table: &RoutingTable, cfg: &AdmissionConfig) -> AdmissionDecision {
    let has_headroom = table.workers.iter().any(|w| {
        w.health == Health::Healthy
            && w.coarse_load.kv_pressure_bucket < cfg.kv_saturate_bucket
            && w.coarse_load.inflight < cfg.max_inflight_per_worker
    });
    if has_headroom {
        AdmissionDecision::Admit
    } else {
        AdmissionDecision::Reject(RejectReason::ClusterSaturated)
    }
}

struct Waiting<'a>(&'a AtomicUsize);

impl<'a> Waiting<'a> {
    fn enter(counter: &'a AtomicUsize) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        Self(counter)
    }
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

#[derive(Clone, Default)]
pub struct AdmissionQueue {
    turnstile: Arc<Mutex<()>>,
    waiting: Arc<AtomicUsize>,
}

impl AdmissionQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }

    pub async fn admit(
        &self,
        routing: &mut watch::Receiver<RoutingTable>,
        cfg: &AdmissionConfig,
    ) -> AdmissionDecision {
        let deadline = Instant::now() + cfg.wait;
        let refused = AdmissionDecision::Reject(RejectReason::ClusterSaturated);
        let _waiting = Waiting::enter(&self.waiting);
        let Ok(turn) = tokio::time::timeout_at(deadline, self.turnstile.clone().lock_owned()).await
        else {
            return refused;
        };
        loop {
            let decision = admit(&routing.borrow_and_update(), cfg);
            if decision == AdmissionDecision::Admit {
                if self.waiting.load(Ordering::Acquire) > 1 {
                    tokio::spawn(async move {
                        tokio::time::sleep(SETTLE).await;
                        drop(turn);
                    });
                }
                return decision;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return decision;
            }
            tokio::select! {
                changed = routing.changed() => {
                    if changed.is_err() {
                        return refused;
                    }
                }
                _ = tokio::time::sleep(RECHECK.min(remaining)) => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use controller_api::{Role, RoutableWorker, WorkerStatus};
    use ids::WorkerId;
    use std::sync::Mutex as StdMutex;

    fn table(kv: u8, inflight: u32) -> RoutingTable {
        RoutingTable {
            epoch: 1,
            workers: vec![RoutableWorker {
                id: WorkerId(1),
                addr: "10.0.0.1:7000".into(),
                role: Role::Decode,
                model: "m".into(),
                health: Health::Healthy,
                coarse_load: WorkerStatus {
                    kv_pressure_bucket: kv,
                    inflight,
                },
            }],
        }
    }

    fn cfg(wait_ms: u64) -> AdmissionConfig {
        AdmissionConfig {
            wait: Duration::from_millis(wait_ms),
            ..AdmissionConfig::default()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn admits_at_once_with_headroom() {
        let (_tx, mut rx) = watch::channel(table(10, 0));
        let queue = AdmissionQueue::new();
        assert_eq!(
            queue.admit(&mut rx, &cfg(1_000)).await,
            AdmissionDecision::Admit
        );
        assert_eq!(queue.waiting(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn refuses_only_after_the_wait_expires() {
        let (_tx, mut rx) = watch::channel(table(255, 0));
        let queue = AdmissionQueue::new();
        let started = Instant::now();
        let decision = queue.admit(&mut rx, &cfg(2_000)).await;
        assert_eq!(
            decision,
            AdmissionDecision::Reject(RejectReason::ClusterSaturated)
        );
        assert!(started.elapsed() >= Duration::from_millis(2_000));
        assert_eq!(queue.waiting(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn zero_wait_refuses_immediately() {
        let (_tx, mut rx) = watch::channel(table(255, 0));
        let queue = AdmissionQueue::new();
        let started = Instant::now();
        let decision = queue.admit(&mut rx, &cfg(0)).await;
        assert!(matches!(decision, AdmissionDecision::Reject(_)));
        assert!(started.elapsed() < Duration::from_millis(1));
    }

    #[tokio::test(start_paused = true)]
    async fn wakes_when_headroom_appears() {
        let (tx, mut rx) = watch::channel(table(255, 0));
        let queue = AdmissionQueue::new();
        let waiter = tokio::spawn({
            let queue = queue.clone();
            async move { queue.admit(&mut rx, &cfg(60_000)).await }
        });
        tokio::time::sleep(Duration::from_millis(40)).await;
        assert_eq!(queue.waiting(), 1);
        let started = Instant::now();
        tx.send(table(20, 0)).unwrap();
        assert_eq!(waiter.await.unwrap(), AdmissionDecision::Admit);
        assert!(started.elapsed() < Duration::from_millis(10));
    }

    #[tokio::test(start_paused = true)]
    async fn a_change_between_check_and_wait_is_not_lost() {
        let (tx, mut rx) = watch::channel(table(255, 0));
        let queue = AdmissionQueue::new();
        let waiter = tokio::spawn({
            let queue = queue.clone();
            async move { queue.admit(&mut rx, &cfg(60_000)).await }
        });
        for _ in 0..8 {
            tokio::task::yield_now().await;
            tx.send(table(255, 0)).unwrap();
        }
        tx.send(table(0, 0)).unwrap();
        assert_eq!(waiter.await.unwrap(), AdmissionDecision::Admit);
    }

    #[tokio::test(start_paused = true)]
    async fn waiters_are_admitted_in_arrival_order() {
        let (tx, rx) = watch::channel(table(255, 0));
        let queue = AdmissionQueue::new();
        let order = Arc::new(StdMutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for i in 0..5u32 {
            let queue = queue.clone();
            let mut rx = rx.clone();
            let order = order.clone();
            tasks.push(tokio::spawn(async move {
                let decision = queue.admit(&mut rx, &cfg(60_000)).await;
                order.lock().unwrap().push(i);
                decision
            }));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(queue.waiting(), 5);
        tx.send(table(0, 0)).unwrap();
        for task in tasks {
            assert_eq!(task.await.unwrap(), AdmissionDecision::Admit);
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        assert_eq!(queue.waiting(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn a_waiter_that_gives_up_does_not_block_the_line() {
        let (tx, rx) = watch::channel(table(255, 0));
        let queue = AdmissionQueue::new();
        let head = tokio::spawn({
            let queue = queue.clone();
            let mut rx = rx.clone();
            async move { queue.admit(&mut rx, &cfg(60_000)).await }
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        let next = tokio::spawn({
            let queue = queue.clone();
            let mut rx = rx.clone();
            async move { queue.admit(&mut rx, &cfg(60_000)).await }
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        head.abort();
        let _ = head.await;
        tx.send(table(0, 0)).unwrap();
        assert_eq!(next.await.unwrap(), AdmissionDecision::Admit);
        assert_eq!(queue.waiting(), 0);
    }
}
