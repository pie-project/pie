//! Worker-facing **data plane** — the gateway is the SERVER; workers dial IN.
//!
//! Post-inversion (Pie 0.5.0, design §4/§8) the topology flips: instead of the
//! gateway dialing each worker per session, every worker dials INTO the gateway
//! (1:N fan-in). The gateway is the stable listening endpoint; the heavy token
//! traffic flows worker→gateway on the plain client→server direction
//! ([`GatewayInbound::push_tokens`]), and the latency-sensitive commands flow
//! reverse ([`WorkerControl`]) over the SAME connection, split at accept time by
//! [`accept_gateway_link`] (the `spawn_twoway` mux, isolated in `pie-dispatch`).
//!
//! This module owns:
//! - [`WorkerRegistry`] — the live `WorkerId → WorkerControlClient` map plus a
//!   coalesced `connected_watch` of the connected set (the `(b)` half of routing:
//!   a worker is dispatchable only once it has dialed in and `register`ed).
//! - [`serve`] — the accept loop: per connection split the link, serve
//!   `GatewayInbound`, and on the first `register` move the reverse
//!   `WorkerControlClient` into the registry; on connection drop, evict it (the
//!   worker-drop signal alpha's selector + charlie's `Sessions` re-dispatch react
//!   to via `connected_watch`).

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use futures::StreamExt;
use pie_dispatch::{GatewayInbound, WorkerControlClient, accept_gateway_link};
use pie_schema::control::{WorkerId, WorkerStatus};
use pie_schema::gateway::{Accepted, Control, ReqId, Request, Tokens};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Json;
use tokio::net::ToSocketAddrs;
use tokio::sync::watch;

use crate::session::Sessions;

/// Max frame on the worker link. Far smaller than the old poll-based edge-rpc
/// (which batched 64 × 256 KiB chunks): token chunks are small and large blobs
/// ride out-of-band HTTP (`BlobRef`, design §9), so 8 MiB is ample headroom.
const WORKER_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// Why a [`WorkerRegistry::dispatch`] could not reach the worker — distinct from
/// the worker's own [`Accepted`] answer. Both variants mean "advance to the next
/// candidate" for alpha's retry loop (worker-drop re-route, design §8, idempotent
/// on the stable per-turn `ReqId`).
#[derive(Debug)]
pub enum DispatchErr {
    /// No live client for this `WorkerId` (never dialed in, or dropped between
    /// the selector's read and the dispatch — the select→dispatch TOCTOU).
    NotConnected,
    /// The reverse-channel RPC failed mid-dispatch — treat as a worker drop and
    /// re-route (the re-sent `Request` carries the same `ReqId`, so it's safe).
    Transport(String),
}

impl std::fmt::Display for DispatchErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchErr::NotConnected => f.write_str("worker not connected"),
            DispatchErr::Transport(e) => write!(f, "worker dispatch transport error: {e}"),
        }
    }
}

impl std::error::Error for DispatchErr {}

/// The live, dialed-in worker connections (the data-plane `(b)` view). Cloneable
/// (`Arc`-backed) so the accept loop (writer), alpha's `route.rs` (reader of
/// [`connected_watch`]), and charlie's `Sessions` (caller of [`dispatch`] /
/// [`client`]) all share one instance.
///
/// [`connected_watch`]: WorkerRegistry::connected_watch
/// [`dispatch`]: WorkerRegistry::dispatch
/// [`client`]: WorkerRegistry::client
#[derive(Clone)]
pub struct WorkerRegistry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    clients: RwLock<HashMap<WorkerId, WorkerControlClient>>,
    /// The connected set, republished (coalesced) on every dial-in / drop. Held
    /// as an `Arc<HashSet>` so a reader's per-turn `borrow()` is a pointer clone,
    /// not a set clone.
    connected_tx: watch::Sender<Arc<HashSet<WorkerId>>>,
}

impl WorkerRegistry {
    /// A fresh, empty registry.
    pub fn new() -> Self {
        let (connected_tx, _rx) = watch::channel(Arc::new(HashSet::new()));
        Self {
            inner: Arc::new(RegistryInner {
                clients: RwLock::new(HashMap::new()),
                connected_tx,
            }),
        }
    }

    /// Subscribe to the connected-worker set. alpha's selector borrows the latest
    /// `Arc<HashSet>` once per turn (lock-free) and filters
    /// `RoutingTable.healthy ∩ connected`.
    pub fn connected_watch(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
        self.inner.connected_tx.subscribe()
    }

    /// Whether this worker is currently dialed in (cheap membership check).
    pub fn is_connected(&self, id: WorkerId) -> bool {
        self.inner.clients.read().unwrap().contains_key(&id)
    }

    /// The reverse-channel client for `id`, if connected — for the single-shot
    /// commands (`cancel` / `set_priority` / `drain`). `None` ⇒ the worker is
    /// gone; the caller treats it as a re-route signal.
    pub fn client(&self, id: WorkerId) -> Option<WorkerControlClient> {
        self.inner.clients.read().unwrap().get(&id).cloned()
    }

    /// Dispatch a turn to `id`. The registry owns the client lookup and classifies
    /// a missing/failed client as [`DispatchErr`] — distinct from the worker's
    /// [`Accepted`] answer, so alpha's loop branches cleanly: `Err(..)` ⇒ next
    /// candidate (idempotent re-dispatch); `Ok(Accepted::{Reject|Redirect})` ⇒ a
    /// real worker answer (also retry, but a different class).
    ///
    /// Exposed via the [`WorkerDispatch`](crate::route::WorkerDispatch) seam (which
    /// `route::dispatch_with_retry` is generic over) rather than an inherent method,
    /// so `route` compiles floor-only without depending on the registry mechanism.
    fn dispatch_impl(
        &self,
        id: WorkerId,
        req: Request,
    ) -> impl std::future::Future<Output = Result<Accepted, DispatchErr>> + Send {
        let client = self.client(id);
        async move {
            let client = client.ok_or(DispatchErr::NotConnected)?;
            client
                .dispatch(tarpc::context::current(), req)
                .await
                .map_err(|e| DispatchErr::Transport(e.to_string()))
        }
    }

    /// Bind a freshly registered worker's reverse client. Called from the
    /// `register` handler (the first `GatewayInbound` call on a dialed-in
    /// connection). Republishes the connected set.
    fn insert(&self, id: WorkerId, client: WorkerControlClient) {
        let mut clients = self.inner.clients.write().unwrap();
        clients.insert(id, client);
        Self::publish(&self.inner.connected_tx, &clients);
    }

    /// Evict a worker on connection drop (the worker-drop signal). Republishes
    /// the connected set so the selector stops picking it.
    fn remove(&self, id: WorkerId) {
        let mut clients = self.inner.clients.write().unwrap();
        clients.remove(&id);
        Self::publish(&self.inner.connected_tx, &clients);
    }

    fn publish(
        tx: &watch::Sender<Arc<HashSet<WorkerId>>>,
        clients: &HashMap<WorkerId, WorkerControlClient>,
    ) {
        // `send_replace` updates the stored value regardless of live receivers
        // (unlike `send`, which errors when none are subscribed), so a late
        // `connected_watch()` subscriber always sees the current set.
        tx.send_replace(Arc::new(clients.keys().copied().collect()));
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The registry is the [`WorkerDispatch`](crate::route::WorkerDispatch) backend
/// alpha's `dispatch_with_retry` is generic over — so `route` depends only on the
/// `pie-schema` floor, not the connection-registry mechanism.
impl crate::route::WorkerDispatch for WorkerRegistry {
    type Err = DispatchErr;

    fn dispatch(
        &self,
        id: WorkerId,
        req: Request,
    ) -> impl std::future::Future<Output = Result<Accepted, Self::Err>> + Send {
        self.dispatch_impl(id, req)
    }
}

/// A running worker-facing server: its bound address (resolved, so an ephemeral
/// `:0` bind surfaces the real port — needed by the single-node launcher to point
/// its in-proc worker at the gateway, and by the smoke harness to dial it) and
/// the accept-loop task.
pub struct WorkerServer {
    pub bound: SocketAddr,
    pub task: tokio::task::JoinHandle<()>,
}

/// Bind the worker-facing listener on `bind` and serve dialed-in worker
/// connections until the task is dropped. Each connection is split with
/// [`accept_gateway_link`]: this end serves [`GatewayInbound`] over the
/// server-half and holds the reverse [`WorkerControlClient`], which the first
/// `register` call binds into `registry`.
pub async fn serve(
    bind: impl ToSocketAddrs,
    sessions: Sessions,
    registry: WorkerRegistry,
) -> Result<WorkerServer> {
    // Self-describing codec (JSON), NOT bincode: the data plane carries
    // `Request{ message: ClientMessage }` / `Tokens::Chunk(ServerMessage)`, whose
    // vocab enums are `#[serde(tag = "type")]` (internally tagged, for the
    // self-describing client wire) — bincode cannot decode them
    // (`deserialize_any` is unsupported). The worker's dial-in side must match.
    // (MessagePack is the tracked hot-path perf graduation, task #21.)
    let mut incoming = tcp::listen(bind, Json::default)
        .await
        .context("bind worker-facing listener")?;
    incoming
        .config_mut()
        .max_frame_length(WORKER_MAX_FRAME_BYTES);
    let bound = incoming.local_addr();

    let task = tokio::spawn(async move {
        while let Some(conn) = incoming.next().await {
            let transport = match conn {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(error = %e, "worker accept error");
                    continue;
                }
            };
            // Split the one connection: serve GatewayInbound here, hold the
            // reverse WorkerControl client for the registry.
            let (server_half, wc_client) = accept_gateway_link(transport);
            let sessions = sessions.clone();
            let registry = registry.clone();
            tokio::spawn(async move {
                let conn_state = Arc::new(ConnState {
                    client: wc_client,
                    worker_id: Mutex::new(None),
                });
                let server = InboundServer {
                    sessions,
                    registry: registry.clone(),
                    conn: conn_state.clone(),
                };
                BaseChannel::with_defaults(server_half)
                    .execute(server.serve())
                    .for_each_concurrent(None, |req| async move {
                        tokio::spawn(req);
                    })
                    .await;
                // Connection closed → worker drop. Evict so the selector stops
                // picking it and charlie's `Sessions` re-dispatches its in-flight,
                // not-yet-emitted turns (design §8/§10).
                if let Some(id) = *conn_state.worker_id.lock().unwrap() {
                    registry.remove(id);
                    tracing::info!(worker = %id, "worker link closed; evicted from registry");
                }
            });
        }
    });

    Ok(WorkerServer { bound, task })
}

/// Per-connection state shared between the [`GatewayInbound`] handlers and the
/// post-serve eviction: the reverse client to register, and the `WorkerId` once
/// `register` lands (so a drop knows which entry to evict).
struct ConnState {
    client: WorkerControlClient,
    worker_id: Mutex<Option<WorkerId>>,
}

/// The gateway's [`GatewayInbound`] server, one per dialed-in worker connection.
#[derive(Clone)]
struct InboundServer {
    sessions: Sessions,
    registry: WorkerRegistry,
    conn: Arc<ConnState>,
}

impl GatewayInbound for InboundServer {
    async fn register(self, _: tarpc::context::Context, worker_id: WorkerId) {
        // The register-first invariant: bind the reverse client into the registry
        // so this worker is now selectable + dispatchable, and remember the id for
        // eviction on drop.
        *self.conn.worker_id.lock().unwrap() = Some(worker_id);
        self.registry.insert(worker_id, self.conn.client.clone());
        tracing::info!(worker = %worker_id, "worker dialed in + registered");
    }

    async fn push_tokens(
        self,
        _: tarpc::context::Context,
        req_id: ReqId,
        chunk: Tokens,
    ) -> Control {
        // Route the chunk to its turn's bounded pipe. Awaiting a full pipe here is
        // THE backpressure point (§6): it stalls this reply → stalls the worker's
        // push pump → fills its outbox → backpressures generation. The returned
        // `Control` piggybacks ordinary cancel back to the worker.
        self.sessions.feed(req_id, chunk).await
    }

    async fn report(self, _: tarpc::context::Context, worker_id: WorkerId, status: WorkerStatus) {
        // Freshness-only (design §5, manager ruling): admission gates off the
        // controller's `RoutingTable` coarse load, so the dial-in report is not a
        // hard dependency. Logged for observability; a gateway-local freshness
        // cache can layer in later without a contract change.
        tracing::trace!(
            worker = %worker_id,
            kv = status.kv_pressure_bucket,
            inflight = status.inflight,
            "worker load report (freshness)"
        );
    }

    async fn redirect(self, _: tarpc::context::Context, req_id: ReqId) {
        // Post-hoc final-admission reject: the worker accepted then could no longer
        // serve the turn. Hand it back to the session to re-route (§7/§8).
        self.sessions.redirect(req_id);
    }
}
