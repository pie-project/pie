//! Deferred HA seam — [`StateStore`].
//!
//! Today the controller is **soft-state**: the cluster lives only in the actor's
//! memory and is rebuilt by clients re-registering after a restart (the
//! `Ack::ReRegister` path). This trait is the seam where a future
//! highly-available controller would persist/replicate membership behind the
//! *same* single-writer actor, so a failover survivor keeps the registry without
//! waiting for every node to re-register.
//!
//! **Not implemented.** Stub + docs only — the actor does not call it yet. The
//! default [`SoftState`] is the explicit "persist nothing" choice.

use pie_schema::control::{GatewayId, WorkerId};

use crate::state::{Gateway, Worker};

/// A recovered membership snapshot: the workers and gateways a failover survivor
/// restores. Empty under the soft-state default.
pub type Recovered = (Vec<(WorkerId, Worker)>, Vec<(GatewayId, Gateway)>);

/// Persistence / replication seam for the cluster registry. A real impl (Raft
/// log, replicated KV, …) would be invoked by the actor on each membership
/// mutation (`put_*`); reads (`recover`) happen only on failover.
///
/// Load (frequent coarse-load reports) is intentionally **not** persisted — it
/// is soft by nature and reconstructs from the next report.
pub trait StateStore: Send + 'static {
    /// Persist a worker membership change. `None` = removal.
    fn put_worker(&mut self, id: WorkerId, worker: Option<&Worker>);

    /// Persist a gateway membership change. `None` = removal.
    fn put_gateway(&mut self, id: GatewayId, gateway: Option<&Gateway>);

    /// Recover persisted membership on failover. The soft-state default returns
    /// nothing (clients re-register).
    fn recover(&mut self) -> Recovered;
}

/// The default store: pure **soft-state**. Persists nothing, recovers empty.
/// Membership is rebuilt by clients re-registering after a controller restart.
#[derive(Debug, Default)]
pub struct SoftState;

impl StateStore for SoftState {
    fn put_worker(&mut self, _id: WorkerId, _worker: Option<&Worker>) {}
    fn put_gateway(&mut self, _id: GatewayId, _gateway: Option<&Gateway>) {}
    fn recover(&mut self) -> Recovered {
        (Vec::new(), Vec::new())
    }
}
