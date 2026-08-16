//! Deferred HA seam — [`StateStore`].
//!
//! The controller is **soft-state**: the cluster lives only in the actor's
//! memory and is rebuilt by clients re-registering after a restart (the
//! `Ack::ReRegister` path). This trait is the seam where a highly-available
//! controller would persist membership behind the *same* single-writer actor.
//! Nothing calls it yet; [`SoftState`] is the "persist nothing" default.

use ids::{GatewayId, WorkerId};

use crate::state::{Gateway, Worker};

/// A recovered membership snapshot; empty under the soft-state default.
pub type Recovered = (Vec<(WorkerId, Worker)>, Vec<(GatewayId, Gateway)>);

/// Persistence / replication seam for the cluster registry: `put_*` on each
/// membership mutation, `recover` only on failover. Load reports are
/// deliberately not persisted — they reconstruct from the next report.
pub trait StateStore: Send + 'static {
    /// Persist a worker membership change. `None` = removal.
    fn put_worker(&mut self, id: WorkerId, worker: Option<&Worker>);

    /// Persist a gateway membership change. `None` = removal.
    fn put_gateway(&mut self, id: GatewayId, gateway: Option<&Gateway>);

    /// Recover persisted membership on failover. The soft-state default
    /// returns nothing.
    fn recover(&mut self) -> Recovered;
}

/// The default store: pure **soft-state**. Persists nothing, recovers empty.
#[derive(Debug, Default)]
pub struct SoftState;

impl StateStore for SoftState {
    fn put_worker(&mut self, _id: WorkerId, _worker: Option<&Worker>) {}
    fn put_gateway(&mut self, _id: GatewayId, _gateway: Option<&Gateway>) {}
    fn recover(&mut self) -> Recovered {
        (Vec::new(), Vec::new())
    }
}
