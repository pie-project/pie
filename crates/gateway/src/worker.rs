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

pub(crate) const WORKER_MAX_FRAME_BYTES: usize = 8 * 1024 * 1024;

pub const MAX_CLIENT_FRAME_BYTES: usize = WORKER_MAX_FRAME_BYTES - (1024 * 1024);

pub const MAX_CLIENT_FRAME_RECV_BYTES: usize = WORKER_MAX_FRAME_BYTES - (256 * 1024);

#[derive(Debug)]
pub enum DispatchErr {
    NotConnected,
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

#[derive(Clone)]
pub struct WorkerRegistry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    clients: RwLock<HashMap<WorkerId, WorkerControlClient>>,
    connected_tx: watch::Sender<Arc<HashSet<WorkerId>>>,
}

impl WorkerRegistry {
    pub fn new() -> Self {
        let (connected_tx, _rx) = watch::channel(Arc::new(HashSet::new()));
        Self {
            inner: Arc::new(RegistryInner {
                clients: RwLock::new(HashMap::new()),
                connected_tx,
            }),
        }
    }

    pub fn connected_watch(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
        self.inner.connected_tx.subscribe()
    }

    pub fn is_connected(&self, id: WorkerId) -> bool {
        self.inner.clients.read().unwrap().contains_key(&id)
    }

    pub fn client(&self, id: WorkerId) -> Option<WorkerControlClient> {
        self.inner.clients.read().unwrap().get(&id).cloned()
    }

    pub fn clients(&self) -> Vec<WorkerControlClient> {
        self.inner
            .clients
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

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

    fn insert(&self, id: WorkerId, client: WorkerControlClient) {
        let mut clients = self.inner.clients.write().unwrap();
        clients.insert(id, client);
        Self::publish(&self.inner.connected_tx, &clients);
    }

    fn remove(&self, id: WorkerId) {
        let mut clients = self.inner.clients.write().unwrap();
        clients.remove(&id);
        Self::publish(&self.inner.connected_tx, &clients);
    }

    fn publish(
        tx: &watch::Sender<Arc<HashSet<WorkerId>>>,
        clients: &HashMap<WorkerId, WorkerControlClient>,
    ) {
        tx.send_replace(Arc::new(clients.keys().copied().collect()));
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

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

pub struct WorkerServer {
    pub bound: SocketAddr,
    pub task: tokio::task::JoinHandle<()>,
}

pub async fn serve(
    bind: impl ToSocketAddrs,
    sessions: Sessions,
    registry: WorkerRegistry,
) -> Result<WorkerServer> {
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
            let _ = transport.get_ref().set_nodelay(true);
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
                if let Some(id) = *conn_state.worker_id.lock().unwrap() {
                    registry.remove(id);
                    tracing::info!(worker = %id, "worker link closed; evicted from registry");
                }
            });
        }
    });

    Ok(WorkerServer { bound, task })
}

struct ConnState {
    client: WorkerControlClient,
    worker_id: Mutex<Option<WorkerId>>,
}

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
        self.sessions.feed(req_id, chunk).await
    }

    async fn report(self, _: tarpc::context::Context, worker_id: WorkerId, status: WorkerStatus) {
        tracing::trace!(
            worker = %worker_id,
            kv = status.kv_pressure_bucket,
            inflight = status.inflight,
            "worker load report (freshness)"
        );
    }

    async fn redirect(self, _: tarpc::context::Context, req_id: ReqId) {
        self.sessions.redirect(req_id);
    }
}
