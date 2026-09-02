//! The worker↔cluster **link** plane: [`control`], the `ControlLink` seam + dialed
//! `ControlClient` (register, heartbeat, report, neighbor-watch); [`gateway`], where the worker
//! dials INTO the gateway, serves `WorkerControl` and streams tokens back via
//! `GatewayInbound::push_tokens`; [`topology`], the resolved `TopologyMode` + `Coordinator`; 
//! [`blob`], out-of-band data-plane blob fetch (`GET /blob/{hash}`); and
//! [`client`], the standalone client server that terminates client WebSockets
//! directly in local-inference mode (no gateway).

pub mod blob;
pub mod client;
pub mod control;
pub mod gateway;
pub mod partner;
pub mod topology;
