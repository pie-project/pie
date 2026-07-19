//! pie-client-api — the public client↔server message vocabulary and the
//! edge-session frames that carry it.
//!
//! This is the only public, versioned interface crate. It holds the
//! [`ClientMessage`]/[`ServerMessage`] envelope a client exchanges with the
//! server over the client-facing edge (WebSocket today; gateway/worker tarpc in
//! disaggregated serving), plus the [`edge`] frames that wrap them on the
//! gateway↔worker and worker↔local-client hops.
//!
//! Plain serde vocabulary, independent of the local runtime-driver ABI. Id atoms
//! (e.g. `SessionId`) live in `pie-ids`; this crate references none of them.

pub mod edge;
pub mod message;

pub use edge::{GatewayFrame, WorkerFrame};
pub use message::{ClientMessage, ServerMessage};
