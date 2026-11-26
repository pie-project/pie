use std::future::Future;

use anyhow::Result;
use controller_api::{
    Ack, ControllerLink, DirectoryView, GatewayDecisions, GatewayInfo, HEARTBEAT_INTERVAL,
    ProcessEvent, RoutingTable, StateReport,
};
use futures::future::BoxFuture;
use ids::{GatewayId, NodeId, WorkerId};
use tokio::sync::watch;

pub trait GatewayControl: Clone + Send + Sync + 'static {
    fn register_gateway(&self, info: GatewayInfo)
    -> impl Future<Output = Result<GatewayId>> + Send;

    fn heartbeat(&self, id: NodeId) -> impl Future<Output = Result<Ack>> + Send;

    fn routing_watch(&self) -> watch::Receiver<RoutingTable>;

    /// What happened to this gateway's processes, in order, for the
    /// cluster policies.
    fn cluster_events(
        &self,
        gateway: GatewayId,
        events: Vec<ProcessEvent>,
    ) -> impl Future<Output = Result<()>> + Send;

    /// The cluster policies' decisions about this gateway's processes, as
    /// they change.
    fn decisions_watch(&self, gateway: GatewayId) -> watch::Receiver<GatewayDecisions>;

    /// The controller's directory, as it changes.
    fn directory_watch(&self) -> watch::Receiver<DirectoryView>;

    /// A report from a worker that links only to this gateway (a pool),
    /// passed on to the controller's directory.
    fn report_state(
        &self,
        worker: WorkerId,
        report: StateReport,
    ) -> impl Future<Output = Result<()>> + Send;

    /// That worker's link to this gateway went: it holds nothing any more.
    fn forget_state(&self, worker: WorkerId) -> impl Future<Output = Result<()>> + Send;

    /// Which controller this link reaches: a new one after it was dialed
    /// again, which has heard nothing of this gateway's processes.
    fn generation(&self) -> u64 {
        0
    }
}

/// The directory calls a gateway makes for the pools linked to it, behind
/// whichever [`GatewayControl`] it was bound with.
pub(crate) trait DirectorySink: Send + Sync {
    fn report_state(&self, worker: WorkerId, report: StateReport)
    -> BoxFuture<'static, Result<()>>;
    fn forget_state(&self, worker: WorkerId) -> BoxFuture<'static, Result<()>>;
}

impl<C: GatewayControl> DirectorySink for C {
    fn report_state(
        &self,
        worker: WorkerId,
        report: StateReport,
    ) -> BoxFuture<'static, Result<()>> {
        let control = self.clone();
        Box::pin(async move { GatewayControl::report_state(&control, worker, report).await })
    }

    fn forget_state(&self, worker: WorkerId) -> BoxFuture<'static, Result<()>> {
        let control = self.clone();
        Box::pin(async move { GatewayControl::forget_state(&control, worker).await })
    }
}

fn no_routing() -> RoutingTable {
    RoutingTable {
        epoch: 0,
        workers: Vec::new(),
        author: None,
    }
}

impl GatewayControl for ControllerLink {
    fn generation(&self) -> u64 {
        ControllerLink::generation(self)
    }

    async fn register_gateway(&self, info: GatewayInfo) -> Result<GatewayId> {
        self.call("register_gateway", |c, ctx| async move {
            c.register_gateway(ctx, info).await
        })
        .await
    }

    async fn heartbeat(&self, id: NodeId) -> Result<Ack> {
        self.call(
            "heartbeat",
            |c, ctx| async move { c.heartbeat(ctx, id).await },
        )
        .await
    }

    async fn cluster_events(&self, gateway: GatewayId, events: Vec<ProcessEvent>) -> Result<()> {
        self.call("cluster_events", |c, ctx| async move {
            c.cluster_events(ctx, gateway, events).await
        })
        .await
    }

    fn decisions_watch(&self, gateway: GatewayId) -> watch::Receiver<GatewayDecisions> {
        self.watch(
            "watch_decisions",
            GatewayDecisions::default(),
            |view| view.epoch,
            move |c, ctx, since| async move { c.watch_decisions(ctx, gateway, since).await },
        )
    }

    fn routing_watch(&self) -> watch::Receiver<RoutingTable> {
        self.watch(
            "watch_gateway",
            no_routing(),
            |table| table.epoch,
            |c, ctx, since| async move { c.watch_gateway(ctx, since).await },
        )
    }

    fn directory_watch(&self) -> watch::Receiver<DirectoryView> {
        self.watch(
            "watch_directory",
            DirectoryView::default(),
            |view| view.epoch,
            |c, ctx, since| async move { c.watch_directory(ctx, since).await },
        )
    }

    async fn report_state(&self, worker: WorkerId, report: StateReport) -> Result<()> {
        self.call("report_state", |c, ctx| async move {
            c.report_state(ctx, worker, report).await
        })
        .await
    }

    async fn forget_state(&self, worker: WorkerId) -> Result<()> {
        self.call("forget_state", |c, ctx| async move {
            c.forget_state(ctx, worker).await
        })
        .await
    }
}

/// Beats for the gateway, registering it again when the controller lost it,
/// and tells `me` its id.
pub(crate) async fn heartbeat_loop<C: GatewayControl>(
    control: C,
    mut id: GatewayId,
    info: GatewayInfo,
    me: watch::Sender<Option<GatewayId>>,
) {
    me.send_replace(Some(id));
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
