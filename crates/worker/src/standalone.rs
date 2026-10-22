//! `pie serve`'s single-node shape: the controller, the gateway and this
//! worker in one process.

use std::net::{Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use controller_api::{Ack, GatewayInfo, Neighbors, RoutingTable, WorkerInfo, WorkerStatus};
use ids::{GatewayId, NodeId, WorkerId};
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::ControlLink;
use crate::boot::Summary;

#[derive(Clone)]
struct EmbeddedControl(controller::Handle);

impl ControlLink for EmbeddedControl {
    async fn register_worker(&self, info: WorkerInfo) -> Result<WorkerId> {
        Ok(self.0.register_worker(info).await)
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        Ok(self.0.heartbeat(id).await)
    }

    async fn report_worker(&self, id: WorkerId, status: WorkerStatus) -> Result<()> {
        self.0.report_worker(id, status).await;
        Ok(())
    }

    fn neighbors_watch(&self, id: WorkerId) -> watch::Receiver<Neighbors> {
        self.0.worker_watch(id)
    }
}

impl gateway::GatewayControl for EmbeddedControl {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        Ok(self.0.register_gateway(info).await)
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        Ok(self.0.heartbeat(id).await)
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.0.gateway_watch()
    }

    async fn desire_scale(&self, entries: Vec<controller_api::ScaleEntry>) -> Result<()> {
        self.0.desire_scale(entries).await;
        Ok(())
    }
}

/// Swaps both planes' policies of a `pie serve` process.
#[derive(Clone)]
pub struct PolicyReload {
    policies: Option<std::sync::Arc<gateway::policy::Cluster>>,
}

impl PolicyReload {
    /// For those `config` names: the in-proc gateway's `[server] policies`,
    /// where it started with any, and the worker's `[runtime] policy`.
    pub async fn reload(&self, config: &crate::Config) -> Result<()> {
        match &self.policies {
            Some(policies) => gateway::reload_policies(policies, &config.server.policies).await?,
            None if !config.server.policies.is_empty() => {
                tracing::warn!("no cluster policies to reload: this server started with none");
            }
            None => {}
        }
        crate::reload_policy(config).await
    }
}

pub struct StandaloneHandle {
    pub listen_addr: SocketAddr,
    pub worker_addr: SocketAddr,
    policies: Option<std::sync::Arc<gateway::policy::Cluster>>,
    _controller: controller::Handle,
    worker: crate::WorkerHandle,
    gateway: JoinHandle<()>,
}

impl StandaloneHandle {
    /// What swaps this server's policies (a SIGHUP's reload).
    #[must_use]
    pub fn policy_reload(&self) -> PolicyReload {
        PolicyReload {
            policies: self.policies.clone(),
        }
    }

    pub fn summary(&self) -> &Summary {
        self.worker.summary()
    }

    pub async fn shutdown(self) {
        self.gateway.abort();
        self.worker.shutdown().await;
    }
}

/// `pie serve` in one process: the controller, the gateway with its HTTP
/// routes, and a worker linked to it.
pub async fn run_standalone(
    controller: controller::Config,
    mut gateway: gateway::Config,
    worker: crate::Config,
    home: &std::path::Path,
) -> Result<StandaloneHandle> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let handle = controller::embed(controller);
    let control = EmbeddedControl(handle.clone());

    gateway.worker_listen = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));

    let host: std::net::IpAddr = worker.server.host.parse().with_context(|| {
        format!(
            "[server] host {:?} is not an IP address",
            worker.server.host
        )
    })?;
    gateway.listen = SocketAddr::new(host, worker.server.port);
    let gw = gateway::bind(gateway, control.clone())
        .await
        .context("bind in-proc gateway")?;
    let listen_addr = gw.listen_addr;
    let worker_addr = gw.worker_addr;
    let policies = gw.state.router.clone();

    let worker = crate::run_with(
        worker,
        home,
        control,
        vec![format!("tcp://{worker_addr}")],
        Some(format!("ws://{listen_addr}")),
    )
    .await
    .context("boot embedded worker")?;

    let gateway = tokio::spawn(async move {
        if let Err(e) = gw.serve().await {
            tracing::error!(error = %e, "in-proc gateway exited");
        }
    });

    Ok(StandaloneHandle {
        listen_addr,
        worker_addr,
        policies,
        _controller: handle,
        worker,
        gateway,
    })
}

/// The three role configs one combined file yields: the worker reads the
/// whole file; the controller and gateway take their defaults.
pub fn derive_standalone(
    combined: &str,
) -> Result<(controller::Config, gateway::Config, crate::Config)> {
    let worker = crate::Config::parse(combined).context("parsing config")?;
    let controller = controller::Config::parse("").context("controller defaults")?;
    let mut gateway = gateway::Config::parse("").context("gateway defaults")?;
    gateway.policies = worker.server.policies.clone();
    Ok((controller, gateway, worker))
}
