//! `session.rs` — THE internal serving abstraction (design §6/§7/§10).
//!
//! Everything the gateway does for a live request funnels through one [`Session`]
//! model: one-shot (REST/SSE) is a 1-turn session, interactive (WebSocket) is a
//! multi-turn session. The user-facing adapters in [`ingress`](crate::ingress)
//! are thin shells over this; the worker-facing token push
//! ([`GatewayInbound`](pie_dispatch::GatewayInbound)) demuxes back into here. The
//! route → admission → dispatch → token-stream pipe is **one path**.
//!
//! # Surfaces
//!
//! - **ingress (delta):** [`Sessions::create`] → `(`[`SessionHandle`]`, `[`TokenRx`]`)`,
//!   [`SessionHandle::turn`] (WS next turn), [`SessionHandle::cancel`],
//!   [`SessionHandle::close`], [`TokenRx::recv`].
//! - **worker.rs (foxtrot):** [`Sessions::feed`] (the `push_tokens` demux, the
//!   backpressure point) and [`Sessions::redirect`].
//! - **route/admission (alpha) + worker registry (foxtrot):** reached through the
//!   [`TurnRouter`] seam — admission gate, select+retry dispatch, immediate
//!   cancel, and the live connected-worker watch. At the crate root an adapter
//!   impls `TurnRouter` over the real `RoutingHandle` + `WorkerRegistry`; unit
//!   tests here use a mock.
//!
//! # Token pipe, backpressure, eos vs abort
//!
//! Each in-flight turn owns a **bounded** [`mpsc`] pipe keyed by its
//! [`ReqId`](pie_schema::gateway::ReqId). [`feed`](Sessions::feed) forwards a
//! [`Tokens`] chunk into it and **awaits on a full pipe** — that await is the
//! backpressure point: a slow consumer stalls `push_tokens`, which stalls the
//! worker's push pump, which backpressures generation (design §6). The consumer
//! distinguishes a **clean** end from an **abort**:
//!
//! - clean: the worker's terminal [`Tokens::Eos`] is forwarded in-band, so
//!   [`recv`](TokenRx::recv) yields `Some(Tokens::Eos)` and then the pipe closes.
//! - abort: every abort path (cancel / drain / already-emitted worker-drop) drops
//!   the pipe sender **without** an `Eos`, so `recv` yields a bare `None`.
//!
//! Because `Eos` is always observed before a clean close and never before an
//! abort close, "did I see an `Eos`?" is the unambiguous discriminator.
//!
//! # Mid-turn worker loss (design §8/§10, manager-ratified)
//!
//! [`Sessions`] watches [`TurnRouter::connected`]; when a live turn's bound
//! worker leaves the connected set the response is split by whether the turn has
//! **emitted** yet: not-emitted → re-dispatch the stored `Request` (idempotent,
//! same `ReqId`, no double-emit); already-emitted → fail the turn clean (drop the
//! pipe → consumer `None`). True mid-stream resume needs a runtime
//! resume-from-position hook and is deferred.

use std::collections::HashMap;
use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use pie_schema::control::WorkerId;
use pie_schema::gateway::{BlobRef, Priority, ReqId, Request, SessionId, TenantId, Tokens};
use pie_schema::message::ClientMessage;
use tokio::sync::{mpsc, watch};

/// Default bounded token-pipe capacity (chunks). The pipe is the per-turn
/// backpressure buffer; a slow consumer fills it and throttles the worker.
const DEFAULT_PIPE_CAP: usize = 256;

/// Poll interval while [`Sessions::drain`] waits for live sessions to finish.
const DRAIN_POLL: Duration = Duration::from_millis(50);

// ───────────────────────────── ingress seam types ─────────────────────────────

/// The edge-supplied principal for a turn, extracted by `ingress/identity.rs`
/// from the trusted edge header (design §5 — a light identity gate, **not**
/// authentication). `session.rs` consumes only [`tenant`](Identity::tenant) for
/// the dispatched [`Request`]; the rest is carried for tracing / quota.
#[derive(Debug, Clone)]
pub struct Identity {
    /// Tenant the turn is attributed to (routing / quota / isolation).
    pub tenant: TenantId,
    /// Edge-supplied user/principal id.
    pub user: String,
    /// Origin client IP (left-most `X-Forwarded-For`), if present and parseable.
    pub client_ip: Option<IpAddr>,
    /// Edge tracing id (`X-Request-Id`), if present.
    pub request_id: Option<String>,
}

/// The user-turn content ingress hands to [`Sessions::create`] /
/// [`SessionHandle::turn`]. Gateway-internal (never crosses the gateway↔worker
/// wire — only the assembled [`Request`] does), so it lives here, not on the
/// `pie_schema::gateway` floor. `session.rs` stamps `{req_id, session, tenant}`
/// onto the `Request`; ingress never mints those.
#[derive(Debug, Clone)]
pub struct TurnInput {
    /// The turn payload in the existing client-message vocabulary.
    pub message: ClientMessage,
    /// Out-of-band binary inputs (image/audio) as references, never bytes.
    pub blobs: Vec<BlobRef>,
    /// Scheduling priority (default [`Priority::Normal`]).
    pub priority: Priority,
}

/// The §7 routing mode for a session, chosen by the ingress adapter (which knows
/// one-shot from multi-turn) and threaded to the dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Affinity {
    /// Fresh one-shot (REST/SSE): no warm KV to stick to ⇒ load-aware
    /// power-of-two-choices. Front-loads load-awareness and avoids herding.
    Ephemeral,
    /// Multi-turn session (WS): stick to the warm-KV worker across turns ⇒ stable
    /// HRW on the [`SessionId`], re-routed only if that worker is gone.
    Sticky,
}

/// Why a turn could not be started.
#[derive(Debug, Clone)]
pub enum SessionError {
    /// The cluster-level admission gate rejected the turn (resource saturation).
    /// Maps to an over-capacity user response (e.g. HTTP 429/503).
    Admission(String),
    /// No healthy, connected worker accepted the turn after retries (HTTP 503).
    NoWorker,
    /// The gateway is draining and is not accepting new sessions (HTTP 503).
    Draining,
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::Admission(r) => write!(f, "admission rejected: {r}"),
            SessionError::NoWorker => f.write_str("no worker available"),
            SessionError::Draining => f.write_str("gateway draining"),
        }
    }
}

impl std::error::Error for SessionError {}

/// The consumer end of one turn's bounded token pipe (owned by ingress).
///
/// `recv` yields `Some(Tokens::Chunk(..))` per output frame, `Some(Tokens::Eos)`
/// at a **clean** turn end, then `None`; a bare `None` with no preceding `Eos`
/// is an **abort**. Owned + `&mut`, so it composes in a `select!` alongside
/// `&self` control calls on [`SessionHandle`] without a borrow clash.
pub struct TokenRx {
    rx: mpsc::Receiver<Tokens>,
}

impl TokenRx {
    /// Await the next token-stream item. See [`TokenRx`] for the
    /// clean-end-vs-abort contract.
    pub async fn recv(&mut self) -> Option<Tokens> {
        self.rx.recv().await
    }
}

// ───────────────────────────── route/dispatch seam ─────────────────────────────

/// Admission rejected the turn — carries a human-readable reason.
#[derive(Debug, Clone)]
pub struct AdmitReject(pub String);

/// Dispatch exhausted all candidate workers (none accepted / reachable).
#[derive(Debug, Clone)]
pub struct DispatchFail;

/// The orchestration backend `session.rs` drives — the in-proc seam onto alpha's
/// `route.rs`/`admission.rs` and foxtrot's `WorkerRegistry`. Kept as a trait so
/// the (bug-prone) registry/pipe/lifecycle logic is unit-testable in isolation
/// with a mock, and so `session.rs` carries no upward edge to those modules. The
/// crate root provides the real adapter:
///
/// ```ignore
/// struct RouteBackend { routing: RoutingHandle, workers: WorkerRegistry }
/// #[async_trait::async_trait]
/// impl TurnRouter for RouteBackend {
///     async fn admit(&self, req: &Request) -> Result<(), AdmitReject> { self.routing.admit(req)... }
///     async fn dispatch(&self, req: &Request, affinity: Option<u64>) -> Result<WorkerId, DispatchFail> { self.routing.dispatch_with_retry(&self.workers, req, affinity)... }
///     async fn cancel(&self, w: WorkerId, r: ReqId) { if let Some(c) = self.workers.client(w) { c.cancel(.., r).await; } }
///     fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> { self.workers.connected_watch() }
/// }
/// ```
#[async_trait::async_trait]
pub trait TurnRouter: Send + Sync + 'static {
    /// Coarse cluster admission gate (runs before routing; design §7).
    async fn admit(&self, req: &Request) -> Result<(), AdmitReject>;

    /// Select a worker and dispatch with the worker-final-admission retry loop,
    /// returning the bound worker. `affinity` is the §7 routing mode: `Some(key)`
    /// → stable HRW (warm-KV sticky, for a multi-turn session), `None` →
    /// power-of-two-choices (load-aware, for a fresh one-shot).
    async fn dispatch(
        &self,
        req: &Request,
        affinity: Option<u64>,
    ) -> Result<WorkerId, DispatchFail>;

    /// Immediately abort an in-flight turn on a specific worker (reverse-channel
    /// `WorkerControl::cancel`, for when the piggybacked `Control::Abort` can't
    /// reach a non-pushing worker promptly).
    async fn cancel(&self, worker: WorkerId, req: ReqId);

    /// The live connected-worker set (foxtrot's `connected_watch`). Drives
    /// mid-turn worker-drop detection.
    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>>;
}

// ───────────────────────────── internal state ─────────────────────────────

/// Per in-flight turn bookkeeping in the registry.
struct TurnState {
    /// The bound worker (set once dispatch succeeds; cleared while re-dispatching).
    worker: Option<WorkerId>,
    /// The dispatched request, retained for idempotent re-dispatch on worker loss.
    request: Request,
    /// The §7 routing mode for this turn (`None` = p2c, `Some(key)` = HRW),
    /// retained so a re-dispatch (redirect / worker-drop) keeps the same policy.
    affinity: Option<u64>,
    /// Set once the first chunk is delivered to the consumer (in
    /// [`Sessions::feed`], after the send succeeds — never for a bare `Eos`). The
    /// §10 re-dispatch-vs-fail discriminator: unset ⇒ a worker drop can
    /// re-dispatch this turn (no output seen yet); set ⇒ it must fail clean, since
    /// a fresh dispatch would re-emit tokens the user already received.
    emitted: bool,
    /// Producer end of the turn's bounded pipe.
    sink: mpsc::Sender<Tokens>,
}

/// Shared gateway-global session state behind [`Sessions`] / [`SessionHandle`].
struct Inner {
    turns: Mutex<HashMap<ReqId, TurnState>>,
    router: Arc<dyn TurnRouter>,
    next_req: AtomicU64,
    next_session: AtomicU64,
    draining: AtomicBool,
    live: AtomicUsize,
    pipe_cap: usize,
}

impl Inner {
    /// Mint a `ReqId`, build the `Request`, gate admission, bind the bounded pipe
    /// **before** dispatch (so an early `push_tokens` finds it), then dispatch.
    async fn run_turn(
        &self,
        session: SessionId,
        tenant: &TenantId,
        input: TurnInput,
        affinity: Option<u64>,
    ) -> Result<(ReqId, TokenRx), SessionError> {
        let req_id = ReqId(self.next_req.fetch_add(1, Ordering::Relaxed));
        let request = Request {
            req_id,
            session,
            tenant: tenant.clone(),
            priority: input.priority,
            blobs: input.blobs,
            message: input.message,
        };

        self.router
            .admit(&request)
            .await
            .map_err(|r| SessionError::Admission(r.0))?;

        let (tx, rx) = mpsc::channel(self.pipe_cap);
        {
            let mut turns = self.turns.lock().unwrap();
            turns.insert(
                req_id,
                TurnState {
                    worker: None,
                    request: request.clone(),
                    affinity,
                    emitted: false,
                    sink: tx,
                },
            );
        }

        match self.router.dispatch(&request, affinity).await {
            Ok(worker) => {
                let mut turns = self.turns.lock().unwrap();
                if let Some(turn) = turns.get_mut(&req_id) {
                    turn.worker = Some(worker);
                }
                Ok((req_id, TokenRx { rx }))
            }
            Err(DispatchFail) => {
                self.turns.lock().unwrap().remove(&req_id);
                Err(SessionError::NoWorker)
            }
        }
    }

    /// Tear a turn out of the registry (dropping its sink → consumer observes the
    /// channel-close abort) and return its bound worker, if any.
    fn abort_turn(&self, req_id: ReqId) -> Option<WorkerId> {
        self.turns
            .lock()
            .unwrap()
            .remove(&req_id)
            .and_then(|t| t.worker)
    }
}

// ───────────────────────────── public handles ─────────────────────────────

/// Gateway-global session manager — the `GatewayState.sessions` handle.
///
/// `Clone` (cheap `Arc`), so it injects into the axum router state and the
/// worker-RPC server alike. Serves both faces: ingress
/// ([`create`](Sessions::create)/[`drain`](Sessions::drain)) and the worker
/// token demux ([`feed`](Sessions::feed)/[`redirect`](Sessions::redirect)).
#[derive(Clone)]
pub struct Sessions {
    inner: Arc<Inner>,
}

impl Sessions {
    /// Build the manager over a [`TurnRouter`] backend and spawn the worker-drop
    /// watcher. Must be called on a Tokio runtime.
    pub fn new(router: Arc<dyn TurnRouter>) -> Self {
        Self::with_pipe_cap(router, DEFAULT_PIPE_CAP)
    }

    /// As [`new`](Sessions::new) with an explicit bounded-pipe capacity.
    pub fn with_pipe_cap(router: Arc<dyn TurnRouter>, pipe_cap: usize) -> Self {
        let inner = Arc::new(Inner {
            turns: Mutex::new(HashMap::new()),
            router,
            next_req: AtomicU64::new(0),
            next_session: AtomicU64::new(0),
            draining: AtomicBool::new(false),
            live: AtomicUsize::new(0),
            pipe_cap,
        });
        spawn_drop_watcher(inner.clone());
        Self { inner }
    }

    /// Begin a session and dispatch its first turn (one-shot = the whole session;
    /// WS = the first of many). `affinity` is the §7 routing mode the ingress
    /// adapter picks: [`Affinity::Ephemeral`] for a one-shot (→ p2c),
    /// [`Affinity::Sticky`] for a multi-turn session (→ HRW on the session id).
    /// Runs admission → route → dispatch internally (§6/§7 one-path). Errors if
    /// the gateway is draining, admission rejects, or no worker accepts.
    pub async fn create(
        &self,
        ident: Identity,
        first: TurnInput,
        affinity: Affinity,
    ) -> Result<(SessionHandle, TokenRx), SessionError> {
        if self.inner.draining.load(Ordering::Acquire) {
            return Err(SessionError::Draining);
        }
        let session = SessionId(self.inner.next_session.fetch_add(1, Ordering::Relaxed));
        // Sticky multi-turn sessions key affinity on the stable session id; a
        // fresh one-shot has no warm KV to prefer, so it routes load-aware (p2c).
        let affinity_key = match affinity {
            Affinity::Ephemeral => None,
            Affinity::Sticky => Some(session.0),
        };
        self.inner.live.fetch_add(1, Ordering::Relaxed);
        let (req_id, rx) = match self
            .inner
            .run_turn(session, &ident.tenant, first, affinity_key)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                self.inner.live.fetch_sub(1, Ordering::Relaxed);
                return Err(e);
            }
        };
        let handle = SessionHandle {
            inner: self.inner.clone(),
            session,
            tenant: ident.tenant,
            affinity_key,
            current: Mutex::new(Some(req_id)),
        };
        Ok((handle, rx))
    }

    /// Worker → gateway token push (`GatewayInbound::push_tokens`): route a chunk
    /// to its turn's bounded pipe and answer with the [`Control`] for the worker.
    ///
    /// [`Control`](pie_schema::gateway::Control)`::Continue` while the pipe
    /// accepts; `Abort` when the turn is gone or the consumer dropped. The send
    /// **awaits on a full pipe** — the backpressure point. A forwarded
    /// [`Tokens::Eos`] cleanly ends the turn (the pipe closes after the consumer
    /// observes it).
    pub async fn feed(&self, req_id: ReqId, chunk: Tokens) -> pie_schema::gateway::Control {
        use pie_schema::gateway::Control;

        let sink = {
            let turns = self.inner.turns.lock().unwrap();
            match turns.get(&req_id) {
                Some(turn) => turn.sink.clone(),
                None => return Control::Abort,
            }
        };

        let is_eos = matches!(chunk, Tokens::Eos);
        match sink.send(chunk).await {
            Ok(()) => {
                {
                    let mut turns = self.inner.turns.lock().unwrap();
                    if is_eos {
                        // Clean end: drop the registry's sender so the consumer
                        // sees the buffered Eos, then the channel-close `None`.
                        turns.remove(&req_id);
                    } else if let Some(turn) = turns.get_mut(&req_id) {
                        turn.emitted = true;
                    }
                }
                Control::Continue
            }
            Err(_) => {
                // Consumer dropped (user gone / turn aborted) → stop the worker.
                self.inner.turns.lock().unwrap().remove(&req_id);
                Control::Abort
            }
        }
    }

    /// Worker → gateway redirect (`GatewayInbound::redirect`): the worker can no
    /// longer serve an accepted turn. Re-dispatch the stored `Request`
    /// (idempotent, same `ReqId`) to another worker; if none accepts, abort the
    /// turn. Fire-and-forget (the re-dispatch runs on a spawned task).
    pub fn redirect(&self, req_id: ReqId) {
        let inner = self.inner.clone();
        let entry = {
            let mut turns = inner.turns.lock().unwrap();
            match turns.get_mut(&req_id) {
                Some(turn) => {
                    turn.worker = None;
                    Some((turn.request.clone(), turn.affinity))
                }
                None => None,
            }
        };
        let Some((request, affinity)) = entry else {
            return;
        };
        tokio::spawn(async move {
            match inner.router.dispatch(&request, affinity).await {
                Ok(worker) => {
                    let mut turns = inner.turns.lock().unwrap();
                    if let Some(turn) = turns.get_mut(&req_id) {
                        turn.worker = Some(worker);
                    }
                }
                Err(DispatchFail) => {
                    inner.turns.lock().unwrap().remove(&req_id);
                }
            }
        });
    }

    /// Graceful drain (design §10): stop accepting new sessions, wait up to `max`
    /// for live sessions to finish, then force-close any stragglers (their token
    /// streams abort).
    pub async fn drain(&self, max: Duration) {
        self.inner.draining.store(true, Ordering::Release);
        let deadline = Instant::now() + max;
        while self.inner.live.load(Ordering::Acquire) > 0 {
            if Instant::now() >= deadline {
                break;
            }
            tokio::time::sleep(DRAIN_POLL).await;
        }
        self.inner.turns.lock().unwrap().clear();
    }

    /// Number of live (not-yet-dropped) sessions. Exposed for drain/observability.
    pub fn live(&self) -> usize {
        self.inner.live.load(Ordering::Acquire)
    }
}

/// Controls one live session (the ingress adapter holds it). Control methods take
/// `&self` so they compose in a `select!` while the owned [`TokenRx`] is borrowed
/// mutably. Dropping the handle ends the session (cancels any in-flight turn).
pub struct SessionHandle {
    inner: Arc<Inner>,
    session: SessionId,
    tenant: TenantId,
    /// The session's §7 affinity key, reused for every turn (a multi-turn session
    /// sticks to its warm-KV worker; `None` for a one-shot).
    affinity_key: Option<u64>,
    current: Mutex<Option<ReqId>>,
}

impl SessionHandle {
    /// Submit the next turn on this session (WS multi-turn). Re-runs admission →
    /// route → dispatch; the session's stable affinity key keeps it on its
    /// warm-KV worker. Returns the new turn's token stream.
    pub async fn turn(&self, next: TurnInput) -> Result<TokenRx, SessionError> {
        let (req_id, rx) = self
            .inner
            .run_turn(self.session, &self.tenant, next, self.affinity_key)
            .await?;
        *self.current.lock().unwrap() = Some(req_id);
        Ok(rx)
    }

    /// Cancel the current in-flight turn (immediate abort on the worker; the
    /// turn's `TokenRx` aborts). The session stays open — a WS client may submit
    /// another [`turn`](SessionHandle::turn).
    pub async fn cancel(&self) {
        let req_id = self.current.lock().unwrap().take();
        if let Some(req_id) = req_id
            && let Some(worker) = self.inner.abort_turn(req_id)
        {
            self.inner.router.cancel(worker, req_id).await;
        }
    }

    /// End the session: cancel any in-flight turn. Idempotent; [`Drop`] also runs
    /// this, so an early `close` and the final drop are both safe.
    pub async fn close(&self) {
        self.cancel().await;
    }

    /// The logical session id (stable across this session's turns).
    pub fn id(&self) -> SessionId {
        self.session
    }
}

impl Drop for SessionHandle {
    fn drop(&mut self) {
        self.inner.live.fetch_sub(1, Ordering::Relaxed);
        let req_id = self.current.lock().unwrap().take();
        if let Some(req_id) = req_id
            && let Some(worker) = self.inner.abort_turn(req_id)
        {
            let router = self.inner.router.clone();
            tokio::spawn(async move { router.cancel(worker, req_id).await });
        }
    }
}

// ───────────────────────────── worker-drop watcher ─────────────────────────────

/// Background task: when a live turn's bound worker leaves the connected set,
/// re-dispatch the turn if it hasn't emitted, else fail it clean (design §10).
fn spawn_drop_watcher(inner: Arc<Inner>) {
    let mut connected = inner.router.connected();
    tokio::spawn(async move {
        loop {
            if connected.changed().await.is_err() {
                break; // router gone → gateway shutting down.
            }
            let live: Arc<HashSet<WorkerId>> = connected.borrow_and_update().clone();

            // Collect affected turns under the lock; do the async re-dispatch
            // outside it.
            let affected: Vec<(ReqId, bool, Request, Option<u64>)> = {
                let turns = inner.turns.lock().unwrap();
                turns
                    .iter()
                    .filter(|(_, t)| t.worker.is_some_and(|w| !live.contains(&w)))
                    .map(|(id, t)| (*id, t.emitted, t.request.clone(), t.affinity))
                    .collect()
            };

            for (req_id, emitted, request, affinity) in affected {
                if emitted {
                    // Tokens already reached the user → cannot resume cleanly.
                    inner.turns.lock().unwrap().remove(&req_id);
                    continue;
                }
                // Not emitted → re-dispatch the same ReqId (idempotent).
                {
                    let mut turns = inner.turns.lock().unwrap();
                    match turns.get_mut(&req_id) {
                        Some(turn) => turn.worker = None,
                        None => continue, // raced with completion/abort.
                    }
                }
                match inner.router.dispatch(&request, affinity).await {
                    Ok(worker) => {
                        let mut turns = inner.turns.lock().unwrap();
                        if let Some(turn) = turns.get_mut(&req_id) {
                            turn.worker = Some(worker);
                        }
                    }
                    Err(DispatchFail) => {
                        inner.turns.lock().unwrap().remove(&req_id);
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use pie_schema::gateway::Control;
    use pie_schema::message::ServerMessage;

    /// Mock [`TurnRouter`]: admission/dispatch outcomes are scriptable, dispatched
    /// `ReqId`s and cancels are recorded, and the connected set is controllable.
    struct MockRouter {
        admit_ok: AtomicBool,
        dispatch_worker: Mutex<Option<WorkerId>>, // Some → Ok(worker); None → DispatchFail
        dispatched: Mutex<Vec<ReqId>>,
        affinities: Mutex<Vec<Option<u64>>>,
        cancels: Mutex<Vec<(WorkerId, ReqId)>>,
        connected_tx: watch::Sender<Arc<HashSet<WorkerId>>>,
        connected_rx: watch::Receiver<Arc<HashSet<WorkerId>>>,
    }

    impl MockRouter {
        fn new(worker: Option<WorkerId>) -> Arc<Self> {
            let init: Arc<HashSet<WorkerId>> = Arc::new(worker.into_iter().collect::<HashSet<_>>());
            let (tx, rx) = watch::channel(init);
            Arc::new(Self {
                admit_ok: AtomicBool::new(true),
                dispatch_worker: Mutex::new(worker),
                dispatched: Mutex::new(Vec::new()),
                affinities: Mutex::new(Vec::new()),
                cancels: Mutex::new(Vec::new()),
                connected_tx: tx,
                connected_rx: rx,
            })
        }

        fn set_connected(&self, workers: &[WorkerId]) {
            let set: Arc<HashSet<WorkerId>> = Arc::new(workers.iter().copied().collect());
            self.connected_tx.send(set).unwrap();
        }
    }

    #[async_trait::async_trait]
    impl TurnRouter for MockRouter {
        async fn admit(&self, _req: &Request) -> Result<(), AdmitReject> {
            if self.admit_ok.load(Ordering::Acquire) {
                Ok(())
            } else {
                Err(AdmitReject("full".into()))
            }
        }
        async fn dispatch(
            &self,
            req: &Request,
            affinity: Option<u64>,
        ) -> Result<WorkerId, DispatchFail> {
            self.dispatched.lock().unwrap().push(req.req_id);
            self.affinities.lock().unwrap().push(affinity);
            match *self.dispatch_worker.lock().unwrap() {
                Some(w) => Ok(w),
                None => Err(DispatchFail),
            }
        }
        async fn cancel(&self, worker: WorkerId, req: ReqId) {
            self.cancels.lock().unwrap().push((worker, req));
        }
        fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
            self.connected_rx.clone()
        }
    }

    fn ident() -> Identity {
        Identity {
            tenant: TenantId("t".into()),
            user: "u".into(),
            client_ip: None,
            request_id: None,
        }
    }

    fn input() -> TurnInput {
        TurnInput {
            message: ClientMessage::Ping { corr_id: 1 },
            blobs: Vec::new(),
            priority: Priority::Normal,
        }
    }

    fn chunk() -> Tokens {
        Tokens::Chunk(ServerMessage::Response {
            corr_id: 1,
            ok: true,
            result: String::new(),
        })
    }

    #[tokio::test]
    async fn create_then_stream_chunk_and_eos() {
        let router = MockRouter::new(Some(WorkerId(7)));
        let sessions = Sessions::new(router.clone());
        let (_h, mut rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();

        let req_id = router.dispatched.lock().unwrap()[0];
        assert_eq!(sessions.feed(req_id, chunk()).await, Control::Continue);
        assert!(matches!(rx.recv().await, Some(Tokens::Chunk(_))));

        // Eos is delivered in-band, then the pipe closes (clean end).
        assert_eq!(sessions.feed(req_id, Tokens::Eos).await, Control::Continue);
        assert!(matches!(rx.recv().await, Some(Tokens::Eos)));
        assert!(rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn feed_unknown_req_aborts() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router);
        assert_eq!(sessions.feed(ReqId(999), chunk()).await, Control::Abort);
    }

    #[tokio::test]
    async fn dropped_consumer_aborts_feed() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router.clone());
        let (_h, rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let req_id = router.dispatched.lock().unwrap()[0];
        drop(rx); // user disconnected
        assert_eq!(sessions.feed(req_id, chunk()).await, Control::Abort);
    }

    #[tokio::test]
    async fn admission_reject_surfaces() {
        let router = MockRouter::new(Some(WorkerId(1)));
        router.admit_ok.store(false, Ordering::Release);
        let sessions = Sessions::new(router);
        let res = sessions.create(ident(), input(), Affinity::Sticky).await;
        assert!(matches!(res, Err(SessionError::Admission(_))));
        assert_eq!(sessions.live(), 0); // failed create does not leak a live slot
    }

    #[tokio::test]
    async fn no_worker_surfaces() {
        let router = MockRouter::new(None); // dispatch always fails
        let sessions = Sessions::new(router);
        let res = sessions.create(ident(), input(), Affinity::Sticky).await;
        assert!(matches!(res, Err(SessionError::NoWorker)));
    }

    #[tokio::test]
    async fn cancel_aborts_turn_and_signals_worker() {
        let router = MockRouter::new(Some(WorkerId(3)));
        let sessions = Sessions::new(router.clone());
        let (h, mut rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let req_id = router.dispatched.lock().unwrap()[0];

        h.cancel().await;
        assert!(rx.recv().await.is_none()); // aborted: bare None, no Eos
        assert_eq!(
            router.cancels.lock().unwrap().as_slice(),
            &[(WorkerId(3), req_id)]
        );
        assert_eq!(sessions.feed(req_id, chunk()).await, Control::Abort);
    }

    #[tokio::test]
    async fn ws_multi_turn_reuses_session_id() {
        let router = MockRouter::new(Some(WorkerId(5)));
        let sessions = Sessions::new(router.clone());
        let (h, mut rx1) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let r1 = router.dispatched.lock().unwrap()[0];
        sessions.feed(r1, Tokens::Eos).await;
        assert!(matches!(rx1.recv().await, Some(Tokens::Eos)));

        let mut rx2 = h.turn(input()).await.unwrap();
        let r2 = router.dispatched.lock().unwrap()[1];
        assert_ne!(r1, r2, "fresh ReqId per turn");
        sessions.feed(r2, chunk()).await;
        assert!(matches!(rx2.recv().await, Some(Tokens::Chunk(_))));
    }

    #[tokio::test]
    async fn affinity_mode_maps_to_dispatch_key() {
        // Ephemeral one-shot → None (alpha's p2c, load-aware).
        let r1 = MockRouter::new(Some(WorkerId(1)));
        let s1 = Sessions::new(r1.clone());
        let _ = s1
            .create(ident(), input(), Affinity::Ephemeral)
            .await
            .unwrap();
        assert_eq!(r1.affinities.lock().unwrap()[0], None);

        // Sticky WS → Some(session id), reused on every turn (HRW warm-KV).
        let r2 = MockRouter::new(Some(WorkerId(1)));
        let s2 = Sessions::new(r2.clone());
        let (h, _rx) = s2.create(ident(), input(), Affinity::Sticky).await.unwrap();
        let key = r2.affinities.lock().unwrap()[0];
        assert!(key.is_some(), "sticky session keys affinity");
        let _ = h.turn(input()).await.unwrap();
        assert_eq!(
            r2.affinities.lock().unwrap()[1],
            key,
            "turn reuses the session's affinity key"
        );
    }

    #[tokio::test]
    async fn draining_rejects_new_sessions() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router);
        sessions.drain(Duration::from_millis(0)).await;
        let res = sessions.create(ident(), input(), Affinity::Sticky).await;
        assert!(matches!(res, Err(SessionError::Draining)));
    }

    #[tokio::test]
    async fn worker_drop_before_emit_redispatches() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router.clone());
        let (_h, _rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let req_id = router.dispatched.lock().unwrap()[0];

        // Worker 1 drops, worker 2 is now the only one connected.
        *router.dispatch_worker.lock().unwrap() = Some(WorkerId(2));
        router.set_connected(&[WorkerId(2)]);

        // The watcher re-dispatches the same ReqId (not emitted yet).
        tokio::time::sleep(Duration::from_millis(50)).await;
        let dispatched = router.dispatched.lock().unwrap().clone();
        assert_eq!(dispatched, vec![req_id, req_id], "same ReqId re-dispatched");
    }

    #[tokio::test]
    async fn worker_drop_after_emit_aborts() {
        let router = MockRouter::new(Some(WorkerId(1)));
        let sessions = Sessions::new(router.clone());
        let (_h, mut rx) = sessions
            .create(ident(), input(), Affinity::Sticky)
            .await
            .unwrap();
        let req_id = router.dispatched.lock().unwrap()[0];

        // A token reached the consumer → the turn has emitted.
        sessions.feed(req_id, chunk()).await;
        assert!(matches!(rx.recv().await, Some(Tokens::Chunk(_))));

        router.set_connected(&[WorkerId(2)]); // worker 1 drops
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Emitted turn is failed clean (abort), not re-dispatched.
        assert!(rx.recv().await.is_none());
        assert_eq!(
            router.dispatched.lock().unwrap().len(),
            1,
            "no re-dispatch after emit"
        );
    }
}
