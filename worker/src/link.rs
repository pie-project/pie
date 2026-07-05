//! The worker↔cluster **link** plane: how the worker talks to the control
//! plane (controller) and the data plane (gateway) post-inversion (M3).
//!
//! - [`control`] — the `ControlLink` seam + dialed `ControlClient` (register /
//!   heartbeat / report / neighbor-watch against the controller).
//! - [`gateway`] — the worker dials INTO the gateway, serves `WorkerControl`,
//!   streams tokens back via `GatewayInbound::push_tokens`.
//! - [`topology`] — the resolved control-plane `TopologyMode` + `Coordinator`.
//! - [`blob`] — out-of-band data-plane blob fetch (`GET /blob/{hash}`, §9).

pub mod blob;
pub mod control;
pub mod gateway;
pub mod topology;
