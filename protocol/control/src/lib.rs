//! Control-plane RPC contract — the `Control` tarpc service.
//!
//! Workers and gateways dial the controller through the macro-generated
//! `ControlClient`; the controller implements `Control`. The trait lives in this
//! thin crate (deps = `pie-schema` + `tarpc` only) so the data-type floor
//! `pie-schema` stays tarpc-free, while controller, worker, and gateway all
//! share one contract without depending on `pie-controller`.
//!
//! Wire data types are defined in [`pie_schema::control`].

use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};

/// The controller's RPC surface. Registry of workers + gateways; pushes neighbor
/// views / routing tables via long-poll watches; tracks liveness via heartbeats.
#[tarpc::service]
pub trait Control {
    /// Register a worker; returns its controller-minted [`WorkerId`]. The worker
    /// then calls `watch_worker(id, since = 0)` to get its initial neighbor view
    /// (returns immediately, since `0 < current_epoch`).
    async fn register_worker(info: WorkerInfo) -> WorkerId;

    /// Register a gateway; returns its controller-minted [`GatewayId`].
    async fn register_gateway(info: GatewayInfo) -> GatewayId;

    /// Liveness ping from either node kind (unified). [`Ack::ReRegister`] means
    /// the controller has no record of this id (it restarted / the node timed
    /// out) — the node must re-register. This is the sole eviction signal.
    async fn heartbeat(id: NodeId) -> Ack;

    /// Push a worker's coarse load (write-only, returns nothing). Separate from
    /// `heartbeat` so frequent load updates can be coalesced without disturbing
    /// membership.
    async fn report_worker(id: WorkerId, status: WorkerStatus);

    /// Long-poll a worker's neighbor view. Blocks until the worker epoch advances
    /// past `since`, then returns the scoped [`Neighbors`] (which carries the new
    /// epoch to re-poll with).
    async fn watch_worker(id: WorkerId, since: u64) -> Neighbors;

    /// Long-poll the global routing table. Blocks until the gateway epoch
    /// advances past `since`, then returns the [`RoutingTable`] (which carries
    /// the new epoch). Unscoped — every gateway gets the same global view.
    async fn watch_gateway(since: u64) -> RoutingTable;
}
