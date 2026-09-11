pub mod admission;
pub mod blob;
pub mod controller;
pub mod ingress;
pub mod route;
pub mod session;
pub mod worker;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

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
}

struct RouteBackend {
    routing: RoutingHandle,
    workers: WorkerRegistry,
}

#[async_trait::async_trait]
impl TurnRouter for RouteBackend {
    async fn admit(&self, req: &Request) -> std::result::Result<(), AdmitReject> {
        match self.routing.admit(req) {
            AdmissionDecision::Admit => Ok(()),
            AdmissionDecision::Reject(reason) => Err(AdmitReject(reason.to_string())),
        }
    }

    async fn dispatch(
        &self,
        req: &Request,
        affinity: Option<u64>,
    ) -> std::result::Result<WorkerId, DispatchFail> {
        self.routing
            .dispatch_with_retry(&self.workers, req, affinity)
            .await
            .map(|d| d.worker_id)
            .map_err(|_| DispatchFail)
    }

    async fn cancel(&self, worker: WorkerId, req: ReqId) {
        if let Some(client) = self.workers.client(worker) {
            let _ = client.cancel(tarpc::context::current(), req).await;
        }
    }

    async fn close_session(&self, worker: WorkerId, session: SessionId) {
        if let Some(client) = self.workers.client(worker) {
            let _ = client
                .close_session(tarpc::context::current(), session)
                .await;
        }
    }

    fn connected(&self) -> watch::Receiver<Arc<HashSet<WorkerId>>> {
        self.workers.connected_watch()
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

impl Gateway {
    pub async fn serve(self) -> Result<()> {
        axum::serve(self.listener, self.app)
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
            if let Err(e) = axum::serve(listener, app)
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

impl GatewayHandle {
    pub async fn shutdown(self) {
        self.shutdown.notify_one();
        let _ = self.serve_task.await;
        self.worker_task.abort();
        let _ = self.worker_task.await;
    }
}

pub async fn bind<C: GatewayControl>(config: Config, control: C) -> Result<Gateway> {
    let routing_rx = control.routing_watch();

    let workers = WorkerRegistry::new();
    let routing = RoutingHandle::new(routing_rx, workers.connected_watch());
    let sessions = Sessions::new(Arc::new(RouteBackend {
        routing: routing.clone(),
        workers: workers.clone(),
    }));
    let blobs: Arc<dyn BlobStore> =
        Arc::new(GatewayOriginStore::new(format!("http://{}", config.listen)));
    let state = GatewayState {
        sessions: sessions.clone(),
        routing,
        workers: workers.clone(),
        blobs: blobs.clone(),
    };

    let worker_server = worker::serve(config.worker_listen, sessions, workers)
        .await
        .context("start worker-facing data-plane server")?;
    let worker_addr = worker_server.bound;
    tracing::info!(%worker_addr, "gateway worker-facing listener up (workers dial in)");

    let info = GatewayInfo {
        addr: worker_addr.to_string(),
    };
    let gateway_id = control
        .register_gateway(info.clone())
        .await
        .context("register gateway with controller")?;
    tracing::info!(%gateway_id, %worker_addr, "gateway registered with controller");
    tokio::spawn(controller::heartbeat_loop(control, gateway_id, info));

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
    let control = controller::connect_controller(&config.controller).await?;
    run_with(config, control).await
}

pub async fn run_with<C: GatewayControl>(config: Config, control: C) -> Result<GatewayHandle> {
    Ok(bind(config, control).await?.into_handle())
}
