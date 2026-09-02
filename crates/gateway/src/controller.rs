//! Gateway control-plane client (`controller_api::ControlClient`). Registers
//! as a gateway, heartbeats for liveness, and long-polls `watch_gateway` for
//! the global [`RoutingTable`] — never asks the controller per request. The
//! [`GatewayControl`] seam abstracts the backend so the launcher can inject
//! either the dialed client (distributed) or an in-proc handle (single-node),
//! keeping `gateway` `controller`-free.

use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, GatewayInfo, RoutingTable};
use ids::{GatewayId, NodeId};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

/// Shorter than the controller's liveness-timeout window.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// Client deadline for the `watch_gateway` long-poll; on expiry we re-poll
/// with the same `since`.
const WATCH_DEADLINE: Duration = Duration::from_secs(300);

/// Backoff before retrying `watch_gateway` after a transport error.
const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// The control-plane backend the gateway drives: [`ControlClient`] (tarpc
/// RPCs, distributed) or an in-proc `EmbeddedControl(Handle)` (single-node).
pub trait GatewayControl: Clone + Send + Sync + 'static {
    /// Register this gateway; returns its controller-minted [`GatewayId`].
    fn register_gateway(&self, info: GatewayInfo)
    -> impl Future<Output = Result<GatewayId>> + Send;

    /// Liveness ping. [`Ack::ReRegister`] ⇒ the gateway must re-register.
    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    /// A receiver of the latest [`RoutingTable`].
    fn routing_watch(&self) -> watch::Receiver<RoutingTable>;
}

impl GatewayControl for ControlClient {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        // Inherent tarpc method shadows this trait method, so this dispatches to the RPC.
        self.register_gateway(tarpc::context::current(), info)
            .await
            .context("register_gateway rpc")
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        self.heartbeat(tarpc::context::current(), id)
            .await
            .context("heartbeat rpc")
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        let (tx, rx) = watch::channel(RoutingTable {
            epoch: 0,
            workers: Vec::new(),
        });
        tokio::spawn(watch_routing_loop(self.clone(), tx));
        rx
    }
}

/// Heartbeat the controller on a fixed interval. On [`Ack::ReRegister`] we
/// re-register and adopt the new [`GatewayId`].
pub(crate) async fn heartbeat_loop<C: GatewayControl>(
    control: C,
    mut id: GatewayId,
    info: GatewayInfo,
) {
    let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
    loop {
        ticker.tick().await;
        match control.heartbeat(NodeId::Gateway(id)).await {
            Ok(Ack::Ok) => {}
            Ok(Ack::ReRegister) => {
                tracing::warn!(%id, "controller lost our registration; re-registering");
                match control.register_gateway(info.clone()).await {
                    Ok(new_id) => {
                        id = new_id;
                        tracing::info!(%id, "gateway re-registered");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "re-register failed; retrying next tick");
                    }
                }
            }
            Err(e) => tracing::warn!(error = %e, "heartbeat transport error"),
        }
    }
}

/// Long-poll `watch_gateway`, publishing each new [`RoutingTable`] to the
/// shared channel. Re-polls with the returned epoch; backs off on error.
async fn watch_routing_loop(control: ControlClient, routing_tx: watch::Sender<RoutingTable>) {
    let mut since: u64 = 0;
    loop {
        let mut ctx = tarpc::context::current();
        ctx.deadline = Instant::now() + WATCH_DEADLINE;
        match control.watch_gateway(ctx, since).await {
            Ok(table) => {
                since = table.epoch;
                tracing::debug!(
                    epoch = since,
                    workers = table.workers.len(),
                    "routing table updated"
                );
                // All receivers dropped → the gateway is shutting down.
                if routing_tx.send(table).is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "watch_gateway error; retrying");
                tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
            }
        }
    }
}

/// Dial the controller's tarpc endpoint and spawn the request dispatcher.
/// Control messages are tiny → tarpc's default frame cap is fine. `addr` is
/// `tcp://host:port`, a bare `host:port`, or `unix:/path`.
pub(crate) async fn connect_controller(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        let conn = unix::connect(path, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlClient::new(cfg, conn).spawn())
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlClient::new(cfg, conn).spawn())
    }
}
