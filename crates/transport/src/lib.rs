pub mod backends;
pub mod core;
pub mod error;
#[cfg(feature = "tcp")]
pub mod tcp;

pub use crate::core::{
    Backend, BackendKind, Completion, PageSet, PeerConn, RegisteredHandle, TransferId, WorkerId,
};
#[cfg(feature = "nixl")]
pub use backends::nixl::{NixlBackend, objects::Objects};
pub use error::{Result, TransportError};
