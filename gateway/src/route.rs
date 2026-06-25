//! Worker selection + dispatch (the gateway data-plane router, design §7).
//!
//! Per turn the gateway runs `admission → route → dispatch`. This module owns the
//! **route** + **dispatch-with-retry** steps; [`admission`](crate::admission)
//! owns the first. All three are reached through one [`RoutingHandle`] so the
//! cluster-table read is single-sourced.
//!
//! # Routing (design §7)
//!
//! Selection runs over `Healthy ∩ connected` workers — the controller's pushed
//! [`RoutingTable`] (who exists + coarse load, soft/fail-static state) intersected
//! with the live dial-in set (who is actually reachable; foxtrot's worker
//! registry, surfaced as a `connected` watch). Two strategies:
//!
//! - **Soft affinity** (a multi-turn session): a stable rendezvous
//!   (highest-random-weight) ranking keyed by the session, so a session prefers
//!   its warm-KV worker across turns. It is *never* a hard bind — a worker going
//!   unavailable just promotes the deterministic next in the ranking (minimal
//!   reshuffle), so the session re-routes freely and rebuilds KV cold rather than
//!   stalling.
//! - **No affinity** (a fresh one-shot): **power-of-two-choices** — sample two
//!   candidates, take the less loaded — to spread load without herding onto a
//!   single (possibly stale) "least-loaded" worker.
//!
//! # Dispatch (design §7/§8)
//!
//! The gateway's pick is a *hint*; the worker has final admission. So
//! [`dispatch_with_retry`](RoutingHandle::dispatch_with_retry) walks the ordered
//! candidates, dispatching the turn until one accepts. A worker may
//! [`Reject`](pie_worker_rpc::Accepted::Reject) /
//! [`Redirect`](pie_worker_rpc::Accepted::Redirect) (advance to the next
//! candidate), or the dispatch may fail at the registry/transport layer
//! (not-connected / no-ack — also advance, the idempotent re-route of §8). The
//! turn's `ReqId` is minted once by the session *before* this loop, so re-routing
//! across candidates is idempotent and never double-binds.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use pie_controller_rpc::{Health, RoutableWorker, RoutingTable};
use pie_ids::WorkerId;
use pie_worker_rpc::{Accepted, Request};
use tokio::sync::watch;

use crate::admission::{AdmissionConfig, AdmissionDecision, admit};

/// A soft-affinity key — an opaque hash of the logical session, minted by the
/// session layer (stable across a session's turns). `route` only feeds it to the
/// rendezvous hash; it never inspects or transmits it.
pub type AffinityKey = u64;

/// Upper bound on dispatch attempts per turn, so a fully-rejecting or flapping
/// fleet cannot spin the retry loop. Comfortably above realistic per-turn
/// candidate counts for the spine.
const MAX_DISPATCH_ATTEMPTS: usize = 8;

/// What `route` needs from the worker connection registry (foxtrot's
/// `worker.rs`): dispatch one turn to a specific, currently-connected worker.
///
/// Kept as a one-method seam (the registry implements it) so `route` depends only
/// on the interface data floor and compiles / tests in isolation, independent of
/// the connection-registry mechanism. The associated [`Err`](WorkerDispatch::Err)
/// lets the registry keep its own error type; `route` only needs it to be
/// loggable, since *any* `Err` means the same thing — advance to the next
/// candidate (registry-level not-connected / transport / no-ack, design §8),
/// distinct from a worker's own [`Accepted`] answer.
pub trait WorkerDispatch {
    /// The registry's dispatch error (e.g. `not-connected` / `transport`).
    type Err: std::fmt::Display;

    /// Dispatch `req` to worker `id`. `Ok(Accepted)` is the worker's *answer*
    /// (accept / reject / redirect); `Err` is a registry/transport failure.
    fn dispatch(
        &self,
        id: WorkerId,
        req: Request,
    ) -> impl std::future::Future<Output = Result<Accepted, Self::Err>> + Send;
}

/// The outcome of a successful dispatch: which worker bound the turn (so the
/// session can target [`cancel`]/[`set_priority`] at it) plus the worker's
/// `Accepted` answer.
///
/// [`cancel`]: pie_worker_rpc::WorkerControl::cancel
/// [`set_priority`]: pie_worker_rpc::WorkerControl::set_priority
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dispatched {
    /// The worker the turn is now bound to.
    pub worker_id: WorkerId,
    /// The worker's accept answer (its `Accepted::Ok { worker }`).
    pub accepted: Accepted,
}

/// Why a turn could not be placed on any worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// No `Healthy ∩ connected` worker existed to try (empty/stale table, or no
    /// worker has dialed in yet).
    NoCandidate,
    /// Every candidate rejected or failed within the attempt budget.
    Exhausted { attempts: usize },
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteError::NoCandidate => {
                f.write_str("no healthy, connected worker available to route")
            }
            RouteError::Exhausted { attempts } => {
                write!(
                    f,
                    "no worker accepted the turn after {attempts} dispatch attempt(s)"
                )
            }
        }
    }
}

impl std::error::Error for RouteError {}

// ─────────────────────────────── routing handle ───────────────────────────────

/// The gateway's routing/admission brain. Holds the controller's pushed
/// [`RoutingTable`] (soft state, fail-static) and the live `connected`-worker
/// snapshot (foxtrot's registry), and drives the §7 per-turn sequence.
///
/// Constructed from injected watch receivers — never dials the controller
/// itself — so it is fully test-injectable (a stub routing watch + a stub
/// connected watch drive selection with no live controller). It is the
/// `routing` field of `GatewayState`; cloning is cheap (two watch receivers).
#[derive(Clone)]
pub struct RoutingHandle {
    /// Latest routing table (worker roster + coarse load), pushed by the
    /// controller backend via `watch_gateway` (or a stub in tests).
    routing: watch::Receiver<RoutingTable>,
    /// Live set of dialed-in workers, bumped by the worker registry on
    /// dial-in / drop. The hot-path membership filter for selection.
    connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
    /// Thresholds for the coarse admission gate.
    admission: AdmissionConfig,
}

impl RoutingHandle {
    /// Build from the routing-table watch (controller backend) and the
    /// connected-worker watch (worker registry). Uses the default admission
    /// thresholds; see [`with_admission`](Self::with_admission).
    pub fn new(
        routing: watch::Receiver<RoutingTable>,
        connected: watch::Receiver<Arc<HashSet<WorkerId>>>,
    ) -> Self {
        Self {
            routing,
            connected,
            admission: AdmissionConfig::default(),
        }
    }

    /// Override the coarse admission thresholds.
    pub fn with_admission(mut self, admission: AdmissionConfig) -> Self {
        self.admission = admission;
        self
    }

    /// Coarse cluster admission (design §7, first step) — see
    /// [`admission`](crate::admission). Takes the turn's [`Request`] for
    /// forward-compatibility (per-tenant token-budget accounting reads it in a
    /// later graduation); v1 gates only on the cluster's coarse load.
    pub fn admit(&self, _req: &Request) -> AdmissionDecision {
        admit(&self.routing.borrow(), &self.admission)
    }

    /// Compute this turn's ordered worker candidates (most-preferred first) for
    /// the given soft-affinity key — `None` for a fresh one-shot (power-of-two).
    /// A pure read of the routing + connected watches; exposed for inspection
    /// and tests (the retry loop uses it internally).
    pub fn select_worker(&self, affinity: Option<AffinityKey>) -> Vec<WorkerId> {
        let table = self.routing.borrow();
        let connected = self.connected.borrow();
        let mut rng = next_rand;
        select_candidates(&table, &connected, None, affinity, &mut rng)
    }

    /// Route + dispatch the turn with worker-final-admission retry (design §7/§8):
    /// walk the ordered candidates, dispatching `req` to each until one accepts.
    ///
    /// - `Ok(Accepted::Ok { worker })` ⇒ bound — return [`Dispatched`].
    /// - `Ok(Accepted::Reject)` and `Ok(Accepted::Redirect { .. })` ⇒ advance to
    ///   the next candidate. (v1 treats `Redirect` like `Reject`; honoring the
    ///   suggested partner target is the prefill/decode-pairing graduation.)
    /// - `Err(..)` (registry not-connected / transport / no-ack) ⇒ advance — the
    ///   idempotent re-route of §8.
    ///
    /// `req` is cloned per attempt (it carries the gateway-minted `ReqId`, bound
    /// once before this loop), so retries never double-bind: only the accepting
    /// worker ever streams this `ReqId`.
    pub async fn dispatch_with_retry<W: WorkerDispatch>(
        &self,
        workers: &W,
        req: &Request,
        affinity: Option<AffinityKey>,
    ) -> Result<Dispatched, RouteError> {
        // Snapshot candidates first; the watch borrows must not be held across
        // the dispatch awaits.
        let candidates = {
            let table = self.routing.borrow();
            let connected = self.connected.borrow();
            let mut rng = next_rand;
            select_candidates(&table, &connected, None, affinity, &mut rng)
        };
        if candidates.is_empty() {
            return Err(RouteError::NoCandidate);
        }

        let mut attempts = 0usize;
        for id in candidates.into_iter().take(MAX_DISPATCH_ATTEMPTS) {
            attempts += 1;
            match workers.dispatch(id, req.clone()).await {
                Ok(accepted @ Accepted::Ok { .. }) => {
                    return Ok(Dispatched {
                        worker_id: id,
                        accepted,
                    });
                }
                // Worker declined (just filled / draining), or redirected to a
                // partner (v1: not honored — treat as a decline). Try the next.
                Ok(Accepted::Reject) | Ok(Accepted::Redirect { .. }) => {
                    tracing::debug!(%id, req_id = %req.req_id, "worker declined turn; trying next candidate");
                }
                // Registry/transport failure: not-connected or no-ack. Re-route
                // (idempotent — the turn's ReqId is stable, §8).
                Err(e) => {
                    tracing::debug!(%id, req_id = %req.req_id, error = %e, "dispatch failed; trying next candidate");
                }
            }
        }
        Err(RouteError::Exhausted { attempts })
    }
}

// ───────────────────────────── selection core (pure) ─────────────────────────────

/// Coarse load ordering key — lower is less loaded. Tie-broken on `WorkerId` for
/// a deterministic total order (stable fallback tails across calls).
fn load_key(w: &RoutableWorker) -> (u8, u32, u64) {
    (
        w.coarse_load.kv_pressure_bucket,
        w.coarse_load.inflight,
        w.id.0,
    )
}

/// splitmix64 finalizer — a strong 64-bit mixer for rendezvous (HRW) scoring and
/// the power-of-two RNG.
fn mix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Rendezvous (highest-random-weight) score of a worker for an affinity key.
/// Deterministic in `(key, worker)`, well-distributed, and stable under churn:
/// when the top worker leaves the candidate set the next-highest score is the
/// natural fallback — exactly soft-affinity's "prefer warm, re-route freely,
/// never hard-bind".
fn hrw_score(key: u64, worker: WorkerId) -> u64 {
    mix64(key ^ mix64(worker.0))
}

/// Process-global RNG state for power-of-two-choices, seeded per-process so that
/// gateway replicas don't herd on an identical pick sequence.
fn next_rand() -> u64 {
    static STATE: AtomicU64 = AtomicU64::new(0);
    // Lazily seed from the process's randomized `RandomState` on first use.
    let mut s = STATE.load(Ordering::Relaxed);
    if s == 0 {
        use std::hash::{BuildHasher, Hasher};
        let seed = std::collections::hash_map::RandomState::new()
            .build_hasher()
            .finish()
            | 1;
        // Best-effort: a race just means two threads seed; both seeds are fine.
        STATE.store(seed, Ordering::Relaxed);
        s = seed;
    }
    let prev = STATE.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed);
    mix64(prev.wrapping_add(s))
}

/// Ordered worker candidates for one turn (most-preferred first), for the retry
/// loop to walk.
///
/// Filter `Healthy ∩ connected ∩ model?` (model filtering is an optional hook —
/// `None` for the spine, matching health-only selection; the multi-model /
/// prefill-decode graduation supplies it), then order:
/// - `affinity = Some(key)` → stable HRW (rendezvous) ranking, warmest-first.
/// - `affinity = None`      → power-of-two-choices primary + load-ordered tail.
fn select_candidates(
    table: &RoutingTable,
    connected: &HashSet<WorkerId>,
    want_model: Option<&str>,
    affinity: Option<AffinityKey>,
    rng: &mut dyn FnMut() -> u64,
) -> Vec<WorkerId> {
    let mut eligible: Vec<&RoutableWorker> = table
        .workers
        .iter()
        .filter(|w| w.health == Health::Healthy)
        .filter(|w| connected.contains(&w.id))
        .filter(|w| want_model.is_none_or(|m| w.model == m))
        .collect();

    if eligible.is_empty() {
        return Vec::new();
    }

    match affinity {
        // Soft affinity: a stable per-session rendezvous ranking. The retry loop
        // walks down it (warmest first); across a session's turns the same key
        // yields the same ranking, so the worker the session lands on stays its
        // warm-KV preference — a worker going unavailable just promotes the
        // deterministic HRW-next (minimal reshuffle), never a load-driven churn
        // that would scatter the session's KV across the fleet.
        Some(key) => {
            eligible.sort_by_key(|w| (std::cmp::Reverse(hrw_score(key, w.id)), w.id.0));
            eligible.iter().map(|w| w.id).collect()
        }
        // No session to keep warm (fresh one-shot): power-of-two-choices primary
        // to avoid herding on a stale "least-loaded", then load-ordered fallbacks.
        None => {
            let primary_pos = p2c_pick(&eligible, rng);
            let primary = eligible.remove(primary_pos);
            eligible.sort_by_key(|w| load_key(w));
            std::iter::once(primary.id)
                .chain(eligible.iter().map(|w| w.id))
                .collect()
        }
    }
}

/// Power-of-two-choices: sample two distinct candidates uniformly, return the
/// index of the less-loaded one. With a single candidate, that one.
fn p2c_pick(eligible: &[&RoutableWorker], rng: &mut dyn FnMut() -> u64) -> usize {
    let n = eligible.len();
    if n == 1 {
        return 0;
    }
    let i = (rng() % n as u64) as usize;
    // Draw the second from [0, n-1) then skip over `i` to keep it uniform + distinct.
    let mut j = (rng() % (n as u64 - 1)) as usize;
    if j >= i {
        j += 1;
    }
    if load_key(eligible[i]) <= load_key(eligible[j]) {
        i
    } else {
        j
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_client_api::ClientMessage;
    use pie_controller_rpc::{Role, WorkerStatus};
    use pie_ids::{ReqId, SessionId, TenantId};
    use pie_worker_rpc::Priority;
    use std::sync::Mutex;

    // ── fixtures ──

    fn worker(id: u64, model: &str, health: Health, kv: u8, inflight: u32) -> RoutableWorker {
        RoutableWorker {
            id: WorkerId(id),
            addr: format!("10.0.0.{id}:7000"),
            role: Role::Decode,
            model: model.to_string(),
            health,
            coarse_load: WorkerStatus {
                kv_pressure_bucket: kv,
                inflight,
            },
        }
    }

    fn table(workers: Vec<RoutableWorker>) -> RoutingTable {
        RoutingTable { epoch: 1, workers }
    }

    fn connset(ids: &[u64]) -> HashSet<WorkerId> {
        ids.iter().map(|&i| WorkerId(i)).collect()
    }

    /// Deterministic scripted RNG for power-of-two tests.
    fn scripted(seq: Vec<u64>) -> impl FnMut() -> u64 {
        let mut it = seq.into_iter();
        move || it.next().unwrap_or(0)
    }

    // ── selection: affinity (HRW) ──

    #[test]
    fn affinity_is_stable_for_a_key() {
        let t = table(
            (1..=4)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let conn = connset(&[1, 2, 3, 4]);
        let mut rng = scripted(vec![]);
        let key = 0xDEAD_BEEF;
        let a = select_candidates(&t, &conn, Some("m"), Some(key), &mut rng);
        let b = select_candidates(&t, &conn, Some("m"), Some(key), &mut rng);
        assert_eq!(a, b, "same affinity key must yield the same ranking");
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn different_keys_spread_primaries() {
        let t = table(
            (1..=4)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let conn = connset(&[1, 2, 3, 4]);
        let mut rng = scripted(vec![]);
        let mut seen = HashSet::new();
        for key in 0..200u64 {
            seen.insert(select_candidates(&t, &conn, Some("m"), Some(key), &mut rng)[0]);
        }
        assert!(
            seen.len() >= 3,
            "HRW should spread primaries, saw {:?}",
            seen
        );
    }

    #[test]
    fn affinity_reroutes_to_hrw_next_when_primary_disconnects() {
        let t = table(
            (1..=4)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let mut rng = scripted(vec![]);
        let key = 12345;
        let full = connset(&[1, 2, 3, 4]);
        let full_order = select_candidates(&t, &full, Some("m"), Some(key), &mut rng);
        let primary = full_order[0];

        let mut reduced = full.clone();
        reduced.remove(&primary);
        let rerouted = select_candidates(&t, &reduced, Some("m"), Some(key), &mut rng);
        assert_ne!(rerouted[0], primary);
        assert_eq!(
            rerouted[0], full_order[1],
            "HRW fallback must be the next-highest-scoring worker"
        );
    }

    // ── selection: no affinity (power-of-two) ──

    #[test]
    fn p2c_picks_lighter_of_the_two_sampled() {
        let t = table(vec![
            worker(1, "m", Health::Healthy, 200, 50),
            worker(2, "m", Health::Healthy, 10, 1),
            worker(3, "m", Health::Healthy, 250, 99),
        ]);
        let conn = connset(&[1, 2, 3]);
        // rng → i=0 (id1), j=0→skip→1 (id2); id2 lighter ⇒ primary id2.
        let mut rng = scripted(vec![0, 0]);
        let c = select_candidates(&t, &conn, Some("m"), None, &mut rng);
        assert_eq!(c[0], WorkerId(2));
        // tail load-ordered: id3(30..) heavier than id1(200)? id1 kv200<id3 kv250 ⇒ [id1, id3]
        assert_eq!(c, vec![WorkerId(2), WorkerId(1), WorkerId(3)]);
    }

    // ── selection: filtering ──

    #[test]
    fn filters_unhealthy_disconnected_and_model() {
        let t = table(vec![
            worker(1, "m", Health::Unreachable, 0, 0), // unhealthy
            worker(2, "m", Health::Healthy, 0, 0),     // not connected
            worker(3, "other", Health::Healthy, 0, 0), // wrong model
            worker(4, "m", Health::Healthy, 0, 0),     // ✓ only eligible
        ]);
        let conn = connset(&[1, 3, 4]); // 2 omitted
        let mut rng = scripted(vec![]);
        assert_eq!(
            select_candidates(&t, &conn, Some("m"), Some(7), &mut rng),
            vec![WorkerId(4)]
        );
    }

    #[test]
    fn empty_when_nothing_eligible() {
        let t = table(vec![worker(1, "m", Health::Unreachable, 0, 0)]);
        let conn = connset(&[1]);
        let mut rng = scripted(vec![]);
        assert!(select_candidates(&t, &conn, Some("m"), Some(1), &mut rng).is_empty());
        assert!(select_candidates(&t, &conn, None, None, &mut rng).is_empty());
    }

    // ── dispatch_with_retry (against a stub WorkerDispatch) ──

    fn req() -> Request {
        Request {
            req_id: ReqId(7),
            session: SessionId(1),
            tenant: TenantId("t".to_string()),
            priority: Priority::Normal,
            blobs: Vec::new(),
            message: ClientMessage::Query {
                corr_id: 1,
                subject: "s".to_string(),
                record: "r".to_string(),
            },
        }
    }

    /// Stub registry: per-worker scripted dispatch answers, recording the order.
    struct StubRegistry {
        answers: std::collections::HashMap<WorkerId, Result<Accepted, String>>,
        calls: Mutex<Vec<WorkerId>>,
    }

    impl WorkerDispatch for StubRegistry {
        type Err = String;
        async fn dispatch(&self, id: WorkerId, _req: Request) -> Result<Accepted, String> {
            self.calls.lock().unwrap().push(id);
            self.answers
                .get(&id)
                .cloned()
                .unwrap_or(Err("not-connected".to_string()))
        }
    }

    fn handle_with(table_v: RoutingTable, connected: &[u64]) -> RoutingHandle {
        // Senders are dropped at end of scope; `watch::Receiver::borrow()` still
        // reads the last-sent value, which is all the routing reads need.
        let (_rt, rr) = watch::channel(table_v);
        let (_ct, cr) = watch::channel(Arc::new(connset(connected)));
        RoutingHandle::new(rr, cr)
    }

    #[tokio::test]
    async fn dispatch_returns_dispatched_on_first_accept() {
        let t = table(vec![worker(1, "m", Health::Healthy, 0, 0)]);
        let h = handle_with(t, &[1]);
        let reg = StubRegistry {
            answers: [(
                WorkerId(1),
                Ok(Accepted::Ok {
                    worker: WorkerId(1),
                }),
            )]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let d = h.dispatch_with_retry(&reg, &req(), Some(42)).await.unwrap();
        assert_eq!(d.worker_id, WorkerId(1));
        assert_eq!(
            d.accepted,
            Accepted::Ok {
                worker: WorkerId(1)
            }
        );
    }

    #[tokio::test]
    async fn dispatch_advances_past_reject_and_transport_error() {
        let t = table(
            (1..=3)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let h = handle_with(t, &[1, 2, 3]);
        // Use the deterministic HRW order so the accepting worker is last — this
        // forces the loop to advance past both a transport error and a reject.
        let order = h.select_worker(Some(99));
        assert_eq!(order.len(), 3);
        let (first, second, accepting) = (order[0], order[1], order[2]);
        let reg = StubRegistry {
            answers: [
                (first, Err("not-connected".to_string())),
                (second, Ok(Accepted::Reject)),
                (accepting, Ok(Accepted::Ok { worker: accepting })),
            ]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let d = h.dispatch_with_retry(&reg, &req(), Some(99)).await.unwrap();
        assert_eq!(d.worker_id, accepting);
        assert_eq!(*reg.calls.lock().unwrap(), vec![first, second, accepting]);
    }

    #[tokio::test]
    async fn dispatch_treats_redirect_like_reject_in_v1() {
        let t = table(
            (1..=2)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let h = handle_with(t, &[1, 2]);
        let reg = StubRegistry {
            answers: [
                (
                    WorkerId(1),
                    Ok(Accepted::Redirect {
                        worker: WorkerId(2),
                    }),
                ),
                (
                    WorkerId(2),
                    Ok(Accepted::Ok {
                        worker: WorkerId(2),
                    }),
                ),
            ]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let d = h.dispatch_with_retry(&reg, &req(), Some(1)).await.unwrap();
        // Redirect is not honored as a jump in v1; we just advance and the next
        // candidate happens to be the accepting one.
        assert_eq!(d.worker_id, WorkerId(2));
    }

    #[tokio::test]
    async fn dispatch_no_candidate_when_none_connected() {
        let t = table(vec![worker(1, "m", Health::Healthy, 0, 0)]);
        let h = handle_with(t, &[]); // nothing dialed in
        let reg = StubRegistry {
            answers: Default::default(),
            calls: Mutex::new(Vec::new()),
        };
        let err = h.dispatch_with_retry(&reg, &req(), None).await.unwrap_err();
        assert_eq!(err, RouteError::NoCandidate);
    }

    #[tokio::test]
    async fn dispatch_exhausted_when_all_reject() {
        let t = table(
            (1..=2)
                .map(|i| worker(i, "m", Health::Healthy, 10, 0))
                .collect(),
        );
        let h = handle_with(t, &[1, 2]);
        let reg = StubRegistry {
            answers: [
                (WorkerId(1), Ok(Accepted::Reject)),
                (WorkerId(2), Ok(Accepted::Reject)),
            ]
            .into(),
            calls: Mutex::new(Vec::new()),
        };
        let err = h
            .dispatch_with_retry(&reg, &req(), Some(5))
            .await
            .unwrap_err();
        assert_eq!(err, RouteError::Exhausted { attempts: 2 });
    }
}
