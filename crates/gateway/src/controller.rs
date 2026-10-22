use std::future::Future;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use controller_api::{Ack, ControlClient, GatewayInfo, RoutingTable, ScaleEntry};
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

    /// The cluster policies' scale table, whole (`cluster.scale`), from
    /// `gateway`, the author.
    fn desire_scale(
        &self,
        gateway: GatewayId,
        entries: Vec<ScaleEntry>,
    ) -> impl Future<Output = Result<()>> + Send;
}

/// Whether the gateway `me` authors the cluster-wide tables, as `routing`
/// names the author.
pub(crate) fn authors(routing: &RoutingTable, me: Option<GatewayId>) -> bool {
    me.is_some() && routing.author == me
}

fn no_routing() -> RoutingTable {
    RoutingTable {
        epoch: 0,
        workers: Vec::new(),
        scales: false,
        author: None,
    }
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

    async fn desire_scale(&self, gateway: GatewayId, entries: Vec<ScaleEntry>) -> Result<()> {
        self.desire_scale(tarpc::context::current(), gateway, entries)
            .await
            .context("desire_scale rpc")
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        let (tx, rx) = watch::channel(no_routing());
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

    async fn desire_scale(&self, gateway: GatewayId, entries: Vec<ScaleEntry>) -> Result<()> {
        let (client, seen) = self.client();
        match client
            .desire_scale(tarpc::context::current(), gateway, entries)
            .await
        {
            Ok(()) => Ok(()),
            Err(e) => {
                self.heal(&e, seen).await;
                Err(e).context("desire_scale rpc")
            }
        }
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        let (tx, rx) = watch::channel(no_routing());
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

/// Beats for the gateway, registering it again when the controller lost it,
/// and tells `me` its id; and, while the routing table names this gateway
/// the author, keeps the controller's scale table the policies' one: sent
/// once a program granted `scale` has written one, when it changes, again
/// when a send failed, again after a re-registration (a restarted controller
/// desires nothing), and as soon as this gateway becomes the author. A
/// gateway that is not the author, or whose programs wrote none, sends none.
pub(crate) async fn heartbeat_loop<C: GatewayControl>(
    control: C,
    mut id: GatewayId,
    info: GatewayInfo,
    mut scale: Option<watch::Receiver<Option<Vec<ScaleEntry>>>>,
    me: watch::Sender<Option<GatewayId>>,
    routing: watch::Receiver<RoutingTable>,
) {
    me.send_replace(Some(id));
    let mut routing = Some(routing);
    let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
    let mut owed = scale.is_some();
    loop {
        tokio::select! {
            _ = ticker.tick() => {}
            changed = async {
                match scale.as_mut() {
                    Some(rx) => rx.changed().await,
                    None => std::future::pending().await,
                }
            } => {
                if changed.is_err() {
                    scale = None;
                } else {
                    owed = true;
                }
                send_scale(&control, id, routing.as_ref(), scale.as_mut(), &mut owed).await;
                continue;
            }
            changed = async {
                match routing.as_mut() {
                    Some(rx) => rx.changed().await,
                    None => std::future::pending().await,
                }
            } => {
                if changed.is_err() {
                    routing = None;
                }
                send_scale(&control, id, routing.as_ref(), scale.as_mut(), &mut owed).await;
                continue;
            }
        }
        send_scale(&control, id, routing.as_ref(), scale.as_mut(), &mut owed).await;
        match control.heartbeat(NodeId::Gateway(id)).await {
            Ok(Ack::Ok) => {}
            Ok(Ack::ReRegister) => {
                owed = scale.is_some();
                tracing::warn!(%id, "controller lost our registration; re-registering");
                let again = GatewayInfo {
                    prior: Some(id),
                    ..info.clone()
                };
                match control.register_gateway(again).await {
                    Ok(new_id) => {
                        id = new_id;
                        me.send_replace(Some(id));
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

/// Sends the scale table if one is owed and this gateway, `id`, is the
/// author. Not the author, the controller holds another's table, so this
/// one is owed for when it is.
async fn send_scale<C: GatewayControl>(
    control: &C,
    id: GatewayId,
    routing: Option<&watch::Receiver<RoutingTable>>,
    scale: Option<&mut watch::Receiver<Option<Vec<ScaleEntry>>>>,
    owed: &mut bool,
) {
    let Some(scale) = scale else {
        return;
    };
    if !routing.is_some_and(|routing| authors(&routing.borrow(), Some(id))) {
        *owed = true;
        return;
    }
    if !*owed {
        return;
    }
    let Some(entries) = scale.borrow_and_update().clone() else {
        // No table written yet: nothing to speak for.
        *owed = false;
        return;
    };
    match control.desire_scale(id, entries).await {
        Ok(()) => *owed = false,
        Err(e) => {
            tracing::warn!(error = %format!("{e:#}"), "scale table not delivered; again next beat")
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    /// Each scale table sent, and by whom.
    type Sent = Vec<(GatewayId, Vec<ScaleEntry>)>;

    #[derive(Clone, Default)]
    struct Fake {
        sent: Arc<Mutex<Sent>>,
    }

    impl GatewayControl for Fake {
        async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
            Ok(info.prior.unwrap_or(GatewayId(0)))
        }

        async fn heartbeat(&self, _id: NodeId) -> Result<Ack> {
            Ok(Ack::Ok)
        }

        fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
            watch::channel(no_routing()).1
        }

        async fn desire_scale(&self, gateway: GatewayId, entries: Vec<ScaleEntry>) -> Result<()> {
            self.sent.lock().unwrap().push((gateway, entries));
            Ok(())
        }
    }

    fn table(workers: u32) -> Option<Vec<ScaleEntry>> {
        Some(vec![ScaleEntry {
            model: Some("m".into()),
            engine: "cuda_native".into(),
            workers,
        }])
    }

    fn routing(author: u64) -> RoutingTable {
        RoutingTable {
            author: Some(GatewayId(author)),
            ..no_routing()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn only_the_author_sends_its_scale_table_and_one_that_becomes_it_sends_at_once() {
        let control = Fake::default();
        let (scale_tx, scale) = watch::channel(table(2));
        let (routing_tx, routing_rx) = watch::channel(routing(1));
        let (me, me_rx) = watch::channel(None);
        let me_id = GatewayId(2);
        let info = GatewayInfo {
            addr: "g".into(),
            prior: None,
        };
        tokio::spawn(heartbeat_loop(
            control.clone(),
            me_id,
            info,
            Some(scale),
            me,
            routing_rx,
        ));
        let sent = || control.sent.lock().unwrap().clone();
        tokio::time::sleep(3 * HEARTBEAT_INTERVAL).await;
        assert_eq!(*me_rx.borrow(), Some(me_id), "told its id");
        assert!(sent().is_empty(), "gateway 1 is the author");
        scale_tx.send_replace(table(3));
        tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        assert!(sent().is_empty(), "a changed table is still not sent");

        // Gateway 1 is evicted: this one is named, and sends what it has.
        routing_tx.send_replace(routing(2));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert_eq!(sent(), [(me_id, table(3).unwrap())]);
        tokio::time::sleep(3 * HEARTBEAT_INTERVAL).await;
        assert_eq!(sent().len(), 1, "once, until it changes");

        // A lower one comes back and takes it; this one owes its table for
        // when it is named again.
        routing_tx.send_replace(routing(1));
        tokio::time::sleep(HEARTBEAT_INTERVAL).await;
        routing_tx.send_replace(routing(2));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        assert_eq!(sent().len(), 2, "sent again on becoming the author");
    }
}
