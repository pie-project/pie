use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, GatewayInfo, RoutingTable};
use ids::{GatewayId, NodeId};
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::tokio_serde::formats::Bincode;
use tokio::sync::watch;

const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

const WATCH_DEADLINE: Duration = Duration::from_secs(300);

const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

pub trait GatewayControl: Clone + Send + Sync + 'static {
    fn register_gateway(&self, info: GatewayInfo)
    -> impl Future<Output = Result<GatewayId>> + Send;

    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    fn routing_watch(&self) -> watch::Receiver<RoutingTable>;
}

impl GatewayControl for ControlClient {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
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

/// A link to the controller that is dialed again when it goes, so a
/// controller restarted on its address is found again: the routing watch
/// starts over, and the next heartbeat hears it does not know this gateway,
/// which registers again under its id.
#[derive(Clone)]
pub struct Controller {
    addr: std::sync::Arc<str>,
    client: std::sync::Arc<std::sync::Mutex<(ControlClient, u64)>>,
    dialing: std::sync::Arc<tokio::sync::Mutex<()>>,
}

impl Controller {
    pub async fn connect(addr: &str) -> Result<Self> {
        let client = connect_controller(addr).await?;
        Ok(Self {
            addr: addr.into(),
            client: std::sync::Arc::new(std::sync::Mutex::new((client, 0))),
            dialing: Default::default(),
        })
    }

    /// The client, and which dial it came from.
    fn client(&self) -> (ControlClient, u64) {
        self.client.lock().unwrap().clone()
    }

    /// After `error` on the link from dial `seen`: if the link is gone,
    /// dials again, unless another caller already has.
    async fn heal(&self, error: &tarpc::client::RpcError, seen: u64) {
        if !matches!(error, tarpc::client::RpcError::Shutdown) {
            return;
        }
        let _dialing = self.dialing.lock().await;
        if self.client().1 != seen {
            return;
        }
        match connect_controller(&self.addr).await {
            Ok(client) => {
                *self.client.lock().unwrap() = (client, seen + 1);
                tracing::info!(controller = %self.addr, "dialed the controller again");
            }
            Err(e) => tracing::debug!(error = %format!("{e:#}"), "the controller is not back yet"),
        }
    }
}

impl GatewayControl for Controller {
    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        let (client, seen) = self.client();
        match client
            .register_gateway(tarpc::context::current(), info)
            .await
        {
            Ok(id) => Ok(id),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("register_gateway rpc")
            }
        }
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        let (client, seen) = self.client();
        match client.heartbeat(tarpc::context::current(), id).await {
            Ok(ack) => Ok(ack),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("heartbeat rpc")
            }
        }
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        let (tx, rx) = watch::channel(RoutingTable {
            epoch: 0,
            workers: Vec::new(),
        });
        let control = self.clone();
        tokio::spawn(async move {
            let mut since: u64 = 0;
            loop {
                let (client, seen) = control.client();
                let mut ctx = tarpc::context::current();
                ctx.deadline = Instant::now() + WATCH_DEADLINE;
                match client.watch_gateway(ctx, since).await {
                    Ok(table) => {
                        since = table.epoch;
                        tracing::debug!(
                            epoch = since,
                            workers = table.workers.len(),
                            "routing table updated"
                        );
                        if tx.send(table).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "watch_gateway error; retrying");
                        control.heal(&e, seen).await;
                        if control.client().1 != seen {
                            // A new controller counts its epochs afresh.
                            since = 0;
                        }
                        tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
                    }
                }
            }
        });
        rx
    }
}

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
                let again = GatewayInfo {
                    prior: Some(id),
                    ..info.clone()
                };
                match control.register_gateway(again).await {
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

pub(crate) async fn connect_controller(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        #[cfg(unix)]
        {
            let conn = unix::connect(path, Bincode::default)
                .await
                .with_context(|| format!("dialing controller at {addr}"))?;
            Ok(ControlClient::new(cfg, conn).spawn())
        }
        #[cfg(not(unix))]
        {
            let _ = (path, cfg);
            anyhow::bail!(
                "{addr}: a `unix://` control address is distributed serving, which needs a unix-domain socket; this build is single-node and speaks `tcp://`"
            )
        }
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        let _ = conn.get_ref().set_nodelay(true);
        Ok(ControlClient::new(cfg, conn).spawn())
    }
}
