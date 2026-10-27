use thiserror::Error;

pub type Result<T> = std::result::Result<T, TransportError>;

#[derive(Debug, Error)]
pub enum TransportError {
    #[error("transport path not available: {0}")]
    Unsupported(&'static str),

    #[error("KV transfer failed: {0}")]
    Transfer(String),

    #[error("source/destination KV layout mismatch")]
    LayoutMismatch,

    #[error("page {page} is out of bounds for the KV region")]
    PageOutOfBounds { page: u32 },

    #[error("no registered KV handle for worker {worker}")]
    UnknownPeer { worker: u64 },

    #[error("unknown transfer id {id}")]
    UnknownTransfer { id: u64 },
}
