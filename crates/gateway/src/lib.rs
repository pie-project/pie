pub mod admission;
pub mod blob;
pub mod controller;
pub mod directory;
pub mod ingress;
pub mod policy;
pub mod route;
pub mod session;
pub mod worker;

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use axum::Router;
use controller_api::GatewayInfo;
use ids::{ReqId, SessionId, WorkerId};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use tokio::sync::{Notify, watch};
use worker_api::Request;

use crate::admission::AdmissionDecision;
use crate::blob::{BlobStore, GatewayOriginStore};
use crate::policy::wit::{By, Outcome};
use crate::route::RoutingHandle;
use crate::session::{AdmitReject, DispatchFail, Sessions, TurnRouter};
use crate::worker::WorkerRegistry;

pub use crate::controller::GatewayControl;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    #[serde(default = "default_worker_listen")]
    pub worker_listen: SocketAddr,
    #[serde(default = "default_controller")]
    pub controller: String,
    /// The cluster policies: each a built-in policy's name or a component
    /// path, no two granted the same write; none keeps the gateway's own
    /// order.
    #[serde(default)]
    pub policies: Vec<String>,
}

fn default_listen() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 8080))
}
fn default_worker_listen() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 8081))
}
fn default_controller() -> String {
    "127.0.0.1:7000".to_string()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            worker_listen: default_worker_listen(),
            controller: default_controller(),
            policies: Vec::new(),
        }
    }
}

impl Config {
    pub fn parse(s: &str) -> Result<Config> {
        toml::from_str(s).context("parse gateway config (TOML)")
    }
}

pub type GatewayConfig = Config;

#[derive(Clone)]
pub struct GatewayState {
    pub sessions: Sessions,
    pub routing: RoutingHandle,
    pub workers: WorkerRegistry,
    pub blobs: Arc<dyn BlobStore>,
    pub router: Option<Arc<policy::Cluster>>,
}

struct RouteBackend {
    routing: RoutingHandle,
    workers: WorkerRegistry,
    located: Mutex<Located>,
    router: Option<Arc<policy::Cluster>>,
}

impl RouteBackend {
    fn router_for(&self, req: &Request) -> Option<&policy::Cluster> {
        let launch = matches!(req.message, client_api::ClientMessage::LaunchProcess { .. });
        self.router.as_deref().filter(|_| launch)
    }
}

/// Where launched processes run, so a later turn about one goes there, not
/// where its session would.
#[derive(Default)]
struct Located {
    workers: HashMap<String, WorkerId>,
    launches: HashMap<ReqId, String>,
}

fn addressed(req: &Request) -> Option<&str> {
    use client_api::ClientMessage::*;
    match &req.message {
        AttachProcess { process_id, .. }
        | TerminateProcess { process_id, .. }
        | SignalProcess { process_id, .. }
        | TransferFile { process_id, .. } => Some(process_id),
        _ => None,
    }
}

#[async_trait::async_trait]
impl TurnRouter for RouteBackend {
    async fn admit(&self, req: &Request) -> std::result::Result<(), AdmitReject> {
        if let Some(router) = self.router_for(req) {
            return router.admit(req, &self.routing).await;
        }
        match self.routing.admit_queued(req).await {
            AdmissionDecision::Admit => Ok(()),
            AdmissionDecision::Reject(reason) => Err(AdmitReject::Full(reason.to_string())),
        }
    }

    async fn dispatch(
        &self,
        req: &Request,
        affinity: Option<u64>,
    ) -> std::result::Result<WorkerId, DispatchFail> {
        let router = self.router_for(req);
        let placed: Request;
        let (req, preferred) = match router {
            Some(router) => {
                let (preferred, cluster_props, executors) = router.placement(req);
                placed = Request {
                    cluster_props,
                    executors,
                    ..req.clone()
                };
                (&placed, preferred)
            }
            None => {
                let located = &self.located.lock().unwrap().workers;
                let preferred = addressed(req).and_then(|process| located.get(process).copied());
                (req, preferred)
            }
        };
        let worker = self
            .routing
            .dispatch_with_retry(&self.workers, req, affinity, preferred)
            .await
            .map_err(|_| DispatchFail)?
            .worker_id;
        if let Some(router) = router {
            router.dispatched(req, worker);
        }
        Ok(worker)
    }

    async fn cancel(&self, worker: WorkerId, req: ReqId) {
        if let Some(client) = self.workers.client(worker) {
            let _ = client.cancel(tarpc::context::current(), req).await;
        }
    }

    async fn end_session(&self, session: SessionId) {
        // A sticky session may have reached more than one worker over its
        // life; the one that never held it ignores the call.
        for client in self.workers.clients() {
            let _ = client.end_session(tarpc::context::current(), session).await;
        }
    }

    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
        self.workers.connected_watch()
    }

    fn launched(&self, req: ReqId, process: &str, worker: WorkerId) {
        let mut located = self.located.lock().unwrap();
        located.workers.insert(process.to_string(), worker);
        located.launches.insert(req, process.to_string());
    }

    fn counted(&self, req: &Request, prompt: u64, generated: u64) {
        if let Some(router) = self.router_for(req) {
            router.counted(req, prompt, generated);
        }
    }

    fn moved(&self, req: &Request, worker: Option<WorkerId>, why: &str) {
        if let Some(router) = self.router_for(req) {
            router.moved(req, worker, why);
        }
    }

    fn ended(&self, req: &Request, outcome: Outcome, _by: By) {
        {
            let mut located = self.located.lock().unwrap();
            if let Some(process) = located.launches.remove(&req.req_id) {
                located.workers.remove(&process);
            }
        }
        if let Some(router) = self.router_for(req) {
            router.exited(req, outcome);
        }
    }
}

pub struct Gateway {
    pub listen_addr: SocketAddr,
    pub worker_addr: SocketAddr,
    pub state: GatewayState,
    listener: TcpListener,
    app: Router,
    _worker_task: tokio::task::JoinHandle<()>,
}

fn no_nagle(listener: TcpListener) -> impl axum::serve::Listener<Addr = std::net::SocketAddr> {
    use axum::serve::ListenerExt;
    listener.tap_io(|tcp| {
        if let Err(error) = tcp.set_nodelay(true) {
            tracing::warn!(%error, "TCP_NODELAY refused on a client connection");
        }
    })
}

impl Gateway {
    pub async fn serve(self) -> Result<()> {
        axum::serve(no_nagle(self.listener), self.app)
            .await
            .context("gateway client-facing serve")?;
        Ok(())
    }

    pub fn into_handle(self) -> GatewayHandle {
        let shutdown = Arc::new(Notify::new());
        let listen_addr = self.listen_addr;
        let worker_addr = self.worker_addr;
        let worker_task = self._worker_task;
        let listener = self.listener;
        let app = self.app;
        let serve_shutdown = shutdown.clone();
        let serve_task = tokio::spawn(async move {
            let graceful = async move { serve_shutdown.notified().await };
            if let Err(e) = axum::serve(no_nagle(listener), app)
                .with_graceful_shutdown(graceful)
                .await
            {
                tracing::error!(error = %e, "gateway client-facing serve ended");
            }
        });
        GatewayHandle {
            listen_addr,
            worker_addr,
            shutdown,
            serve_task,
            worker_task,
        }
    }
}

pub struct GatewayHandle {
    pub listen_addr: SocketAddr,
    pub worker_addr: SocketAddr,
    shutdown: Arc<Notify>,
    serve_task: tokio::task::JoinHandle<()>,
    worker_task: tokio::task::JoinHandle<()>,
}

/// How long a shutting-down gateway lets open client streams run on.
pub const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(30);

impl GatewayHandle {
    /// Stops taking connections and lets open streams finish, for up to
    /// [`SHUTDOWN_GRACE`]: a stream that does not end in that time is cut.
    pub async fn shutdown(mut self) {
        self.shutdown.notify_one();
        if tokio::time::timeout(SHUTDOWN_GRACE, &mut self.serve_task)
            .await
            .is_err()
        {
            tracing::warn!("client streams still open after the shutdown grace; cutting them");
            self.serve_task.abort();
        }
        self.worker_task.abort();
        let _ = self.worker_task.await;
    }
}

pub async fn bind<C: GatewayControl>(config: Config, control: C) -> Result<Gateway> {
    let routing_rx = control.routing_watch();

    let workers = WorkerRegistry::new();
    let routing = RoutingHandle::new(routing_rx, workers.connected_watch());
    let router = if config.policies.is_empty() {
        None
    } else {
        let programs = config
            .policies
            .iter()
            .map(|spec| Ok((spec.clone(), builtins::policy_component(spec)?.into_owned())))
            .collect::<Result<Vec<_>>>()?;
        let (router, cancelled, moves) = policy::Cluster::start(
            programs,
            routing.table_watch(),
            workers.connected_watch(),
            workers.clone(),
        )
        .context("load the cluster policies")?;
        Some((router, cancelled, moves))
    };
    let policy_router = router.as_ref().map(|(router, _, _)| router.clone());
    let scale = policy_router.as_ref().map(|router| router.scale_watch());
    let sessions = Sessions::new(Arc::new(RouteBackend {
        routing: routing.clone(),
        workers: workers.clone(),
        located: Mutex::default(),
        router: policy_router.clone(),
    }));
    if let Some((_, mut cancelled, mut moves)) = router {
        let cancelling = sessions.clone();
        tokio::spawn(async move {
            while let Some(req) = cancelled.recv().await {
                cancelling.terminate(req).await;
            }
        });
        let moving = sessions.clone();
        tokio::spawn(async move {
            while let Some((req, target)) = moves.recv().await {
                let moving = moving.clone();
                tokio::spawn(async move {
                    if !moving.migrate(req, target).await {
                        tracing::debug!(%req, %target, "a turn did not move: it ended, has sent output, or is already there");
                    }
                });
            }
        });
    }
    let blobs: Arc<dyn BlobStore> =
        Arc::new(GatewayOriginStore::new(format!("http://{}", config.listen)));
    let state = GatewayState {
        sessions: sessions.clone(),
        routing,
        workers: workers.clone(),
        blobs: blobs.clone(),
        router: policy_router,
    };

    let worker_server = worker::serve(config.worker_listen, sessions, workers)
        .await
        .context("start worker-facing data-plane server")?;
    let worker_addr = worker_server.bound;
    tracing::info!(%worker_addr, "gateway worker-facing listener up (workers dial in)");

    let info = GatewayInfo {
        addr: worker_addr.to_string(),
        prior: None,
    };
    let gateway_id = control
        .register_gateway(info.clone())
        .await
        .context("register gateway with controller")?;
    tracing::info!(%gateway_id, %worker_addr, "gateway registered with controller");
    tokio::spawn(controller::heartbeat_loop(control, gateway_id, info, scale));

    let app = Router::new()
        .merge(ingress::router(state.clone()))
        .merge(blob::router(blobs));
    let listener = TcpListener::bind(config.listen)
        .await
        .with_context(|| format!("bind client-facing listener on {}", config.listen))?;
    let listen_addr = listener
        .local_addr()
        .context("client listener local_addr")?;
    tracing::info!(%listen_addr, "pie-gateway client-facing edge up");

    Ok(Gateway {
        listen_addr,
        worker_addr,
        state,
        listener,
        app,
        _worker_task: worker_server.task,
    })
}

pub async fn run(config: Config) -> Result<GatewayHandle> {
    let control = controller::Controller::connect(&config.controller).await?;
    run_with(config, control).await
}

pub async fn run_with<C: GatewayControl>(config: Config, control: C) -> Result<GatewayHandle> {
    Ok(bind(config, control).await?.into_handle())
}
