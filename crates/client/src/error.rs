use std::path::PathBuf;

use tokio_tungstenite::tungstenite;

/// Everything a [`Client`](crate::client::Client), a
/// [`Process`](crate::client::Process) or a
/// [`ParsedPrivateKey`](crate::crypto::ParsedPrivateKey) can fail with.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The WebSocket handshake with the engine failed (bad URL, unreachable
    /// host, HTTP-level rejection, TLS, ...).
    #[error("failed to connect to the engine")]
    Connect(#[source] tungstenite::Error),

    /// The connection to the engine closed before the request completed.
    #[error("connection to the engine closed")]
    Disconnected,

    /// The process's event stream has ended: its terminal event was already
    /// delivered, or the connection to the engine closed.
    #[error("process event channel closed")]
    EventsClosed,

    /// The engine refused the client's identity or proof of it.
    #[error("{0}")]
    Auth(String),

    /// The engine answered a request with a failure.
    #[error("{request} failed: {message}")]
    Request {
        /// Which request failed, e.g. `"Launch process"`.
        request: &'static str,
        /// The engine's explanation.
        message: String,
    },

    /// The inferlet finished with an error instead of a return value.
    #[error("inferlet returned an error: {0}")]
    Inferlet(String),

    /// The engine sent something the client cannot interpret.
    #[error("{0}")]
    Protocol(String),

    /// An argument the caller passed is malformed (identity, program
    /// identifier, process id, path).
    #[error("{0}")]
    InvalidInput(String),

    /// A private key could not be parsed, is unsupported, or failed to sign.
    #[error("{0}")]
    PrivateKey(String),

    /// Reading a local file failed.
    #[error("failed to read {path:?}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// More requests are in flight than there are correlation ids.
    #[error("ID pool exhausted")]
    TooManyRequests,

    /// A client message could not be serialized.
    #[error("failed to encode client message")]
    Encode(#[source] rmp_serde::encode::Error),

    /// A bug in this client, not in the engine or the caller's input.
    #[error("internal client error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub(crate) fn request(request: &'static str, message: String) -> Self {
        Self::Request { request, message }
    }
}
