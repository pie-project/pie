//! `pie serve`'s single-node shape: the controller, the gateway and this
//! worker in one process.

use std::net::{Ipv4Addr, SocketAddr};

use anyhow::{Context, Result};
use controller_api::{
    Ack, DirectoryView, GatewayInfo, Neighbors, RoutingTable, StateReport, WorkerInfo, WorkerStatus,
};
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

    async fn report_state(&self, id: WorkerId, report: StateReport) -> Result<()> {
        self.0.report_state(id, report);
        Ok(())
    }

    async fn locate(
        &self,
        asker: WorkerId,
        keys: Vec<[u8; 32]>,
    ) -> Result<Vec<(WorkerId, String, u32)>> {
        Ok(self.0.locate(asker, &keys))
    }

    async fn link(&self, worker: WorkerId) -> Result<Option<String>> {
        Ok(self.0.link(worker))
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

    async fn cluster_events(
        &self,
        gateway: GatewayId,
        events: Vec<controller_api::ProcessEvent>,
    ) -> Result<()> {
        self.0.cluster_events(gateway, events);
        Ok(())
    }

    fn decisions_watch(
        &self,
        gateway: GatewayId,
    ) -> watch::Receiver<controller_api::GatewayDecisions> {
        self.0.decisions_watch(gateway)
    }

    fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
        self.0.directory_watch()
    }

    async fn report_state(&self, worker: WorkerId, report: StateReport) -> Result<()> {
        self.0.report_state(worker, report);
        Ok(())
    }

    async fn forget_state(&self, worker: WorkerId) -> Result<()> {
        self.0.forget_state(worker);
        Ok(())
    }
}

/// Swaps both planes' policies of a `pie serve` process.
#[derive(Clone)]
pub struct PolicyReload {
    controller: controller::Handle,
}

impl PolicyReload {
    /// For those `config` names: the in-proc controller's cluster policies
    /// (`[server] policies`) and the worker's `[runtime] policy`.
    pub async fn reload(&self, config: &crate::Config) -> Result<()> {
        if !config.server.policies.is_empty() {
            self.controller
                .reload_policies(&config.server.policies)
                .await?;
        }
        crate::reload_policy(config).await
    }
}

pub struct StandaloneHandle {
    pub listen_addr: SocketAddr,
    pub worker_addr: SocketAddr,
    controller: controller::Handle,
    worker: crate::WorkerHandle,
    gateway: JoinHandle<()>,
}

impl StandaloneHandle {
    /// What swaps this server's policies (a SIGHUP's reload).
    #[must_use]
    pub fn policy_reload(&self) -> PolicyReload {
        PolicyReload {
            controller: self.controller.clone(),
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
        controller: handle,
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
    let mut controller = controller::Config::parse("").context("controller defaults")?;
    controller.policies = worker.server.policies.clone();
    let gateway = gateway::Config::parse("").context("gateway defaults")?;
    Ok((controller, gateway, worker))
}
