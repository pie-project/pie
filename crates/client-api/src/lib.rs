//! The public, versioned client↔server message vocabulary: the [`ClientMessage`]/[`ServerMessage`]
//! envelope exchanged over the client-facing edge, plus the [`edge`] frames that wrap them on the
//! gateway↔worker and worker↔local-client hops. Plain serde, independent of the local
//! runtime-driver ABI; id atoms such as `SessionId` live in `ids` and are not referenced here.

pub mod edge;
pub mod message;

pub use edge::{GatewayFrame, WorkerFrame};
pub use message::{ClientMessage, ServerMessage};
