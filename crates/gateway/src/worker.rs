//! Worker-facing data plane — the gateway is the server; workers dial in
//! (1:N fan-in) over one connection split into a forward `GatewayInbound`
//! side and a reverse `WorkerControl` side via [`accept_gateway_link`].
//! [`WorkerRegistry`] tracks the live, dialed-in `WorkerId -> WorkerControlClient`
//! map; [`serve`] is the accept loop that populates and evicts it.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Context, Result};
use controller_api::WorkerStatus;
use futures::StreamExt;
use ids::{ReqId, WorkerId};
use tarpc::serde_transport::tcp;
use tarpc::server::{BaseChannel, Channel};
use tokio::net::ToSocketAddrs;
use tokio::sync::watch;
use worker_api::{
    Accepted, Control, GatewayInbound, Request, Tokens, WorkerControlClient, accept_gateway_link,
    dispatch_codec,
};

use crate::session::Sessions;

/// Max frame on the worker link. Token chunks are small; large blobs ride
/// out-of-band HTTP, so 8 MiB is ample headroom.
///
/// **This is a hard edge, not a soft one.** A `dispatch` that exceeds it fails
/// at the codec and takes the tarpc connection with it, so nothing a client can
/// send may be allowed to reach this size — see [`MAX_CLIENT_FRAME_BYTES`],
/// which the ingress enforces against exactly this number.
pub(crate) const WORKER_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

/// The most one client frame may carry, enforced at the ingress.
///
/// One MiB under the worker frame, which is orders of magnitude more than the
/// envelope the gateway wraps a turn in (a `ReqId`, a session, a tenant, a
/// priority) — so a frame this size always dispatches, while the limit stays as
/// close as it safely can to what the transport already carried. Cutting it
/// harder would refuse traffic that works today: before this existed everything
/// under `WORKER_MAX_FRAME_BYTES` was served, and only what crossed it broke the
/// worker link permanently and answered `no worker available`, a cluster-outage
/// message for a client-side error.
pub const MAX_CLIENT_FRAME_BYTES: usize = WORKER_MAX_FRAME_BYTES - (1024 * 1024);

/// What the WebSocket transport will still *receive* before closing the socket.
///
/// Deliberately above [`MAX_CLIENT_FRAME_BYTES`]: a frame in the gap is read in
/// full so the client can be told its request is too large and keep its session,
/// which a transport-level close cannot do. Beyond this the socket closes, which
/// is rude but bounded — and 256 KiB still stands between it and the worker's
/// frame cap, the one edge that must never be reached.
pub const MAX_CLIENT_FRAME_RECV_BYTES: usize = WORKER_MAX_FRAME_BYTES - (256 * 1024);

/// Why a [`WorkerRegistry::dispatch`] could not reach the worker — distinct
/// from the worker's own [`Accepted`] answer. Both mean "advance to the next
/// candidate" for the retry loop (idempotent on the stable per-turn `ReqId`).
#[derive(Debug)]
pub enum DispatchErr {
    /// No live client for this `WorkerId` (never dialed in, or dropped
    /// between the selector's read and the dispatch).
    NotConnected,
    /// The reverse-channel RPC failed mid-dispatch — treat as a worker drop
    /// and re-route.
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

/// The live, dialed-in worker connections. Cloneable (`Arc`-backed) so the
/// accept loop, `route.rs`, and `Sessions` all share one instance.
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

    /// Subscribe to the connected-worker set. The selector borrows the latest
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

    /// Dispatch a turn to `id`. Classifies a missing/failed client as
    /// [`DispatchErr`], distinct from the worker's own [`Accepted`] answer:
    /// `Err(..)` means next candidate (idempotent re-dispatch);
    /// `Ok(Accepted::{Reject|Redirect})` is a real worker answer (also retried).
    ///
    /// Exposed via [`WorkerDispatch`](crate::route::WorkerDispatch) rather than
    /// an inherent method, so `route` doesn't depend on the registry mechanism.
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
/// `dispatch_with_retry` is generic over, so `route` depends only on the
/// interface, not the connection-registry mechanism.
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

/// A running worker-facing server: its resolved bound address (so an
/// ephemeral `:0` bind surfaces the real port) and the accept-loop task.
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
    // MessagePack (`dispatch_codec`), not bincode: the wire vocab enums are
    // internally tagged (`#[serde(tag = "type")]`), which bincode can't decode.
    let mut incoming = tcp::listen(bind, dispatch_codec)
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
                // Connection closed -> worker drop. Evict so the selector stops
                // picking it and `Sessions` re-dispatches its in-flight turns.
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
        // Route the chunk to its turn's bounded pipe. Awaiting a full pipe here
        // is the backpressure point: it stalls this reply, which stalls the
        // worker's push pump. `Control` piggybacks ordinary cancel back to it.
        self.sessions.feed(req_id, chunk).await
    }

    async fn report(self, _: tarpc::context::Context, worker_id: WorkerId, status: WorkerStatus) {
        // Freshness-only: admission gates off the controller's RoutingTable
        // coarse load, so this report is not a hard dependency. Logged only.
        tracing::trace!(
            worker = %worker_id,
            kv = status.kv_pressure_bucket,
            inflight = status.inflight,
            "worker load report (freshness)"
        );
    }

    async fn redirect(self, _: tarpc::context::Context, req_id: ReqId) {
        // Post-hoc final-admission reject: hand it back to the session to re-route.
        self.sessions.redirect(req_id);
    }
}
