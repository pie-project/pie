//! Gateway↔worker data-plane wire vocabulary: the turn types
//! [`GatewayInbound`](crate::GatewayInbound) / [`WorkerControl`](crate::WorkerControl)
//! carry, flat-re-exported at the crate root. CODEC CONSTRAINT: a *self-describing*
//! codec ([`dispatch_codec`](crate::dispatch_codec)'s MessagePack, NOT bincode):
//! [`Request`] and [`Tokens`] need `deserialize_any`.

use serde::{Deserialize, Serialize};

use ids::{ReqId, SessionId, TenantId, WorkerId};
use client_api::{ClientMessage, ServerMessage};


/// A reference to a large binary input. Blob bytes never travel the command
/// path: the worker pulls them out-of-band (`GET {origin}/blob/{hash}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    /// Content address: blake3-256, hex-encoded. Also the `{hash}` URL segment.
    pub hash: String,
    /// Byte length -- lets the worker pre-size the fetch and sanity-check it.
    pub size: u64,
    /// MIME type, e.g. `"image/jpeg"`. A string, not an enum, for forward-compat.
    pub kind: String,
    /// Base URL of the origin gateway's blob endpoint, so any worker can fetch.
    pub origin: String,
}


/// Scheduling priority for a turn, set at dispatch and adjustable via
/// [`WorkerControl::set_priority`](crate::WorkerControl::set_priority). Ordered
/// `Low < Normal < High` by declaration order; `Normal` is the default.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum Priority {
    /// Below `Normal`; yields to any other waiting turn.
    Low,
    /// The default.
    #[default]
    Normal,
    /// Above `Normal`; scheduled ahead of the rest.
    High,
}


/// One dispatched turn (gateway → worker via
/// [`WorkerControl::dispatch`](crate::WorkerControl::dispatch)). Self-describing,
/// idempotent and ack-based: a re-sent `Request` with the same `req_id` is the
/// same turn, and large binaries ride [`blobs`](Request::blobs) as references.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// This turn's id (idempotency key for re-dispatch / cancel / redirect).
    pub req_id: ReqId,
    /// The logical session this turn belongs to (warm-KV affinity across turns).
    pub session: SessionId,
    /// Who the turn is attributed to (tenant/user; isolation & quota).
    pub tenant: TenantId,
    /// Scheduling priority; `Normal` unless the dispatch states otherwise.
    pub priority: Priority,
    /// Out-of-band binary inputs for this turn (images/audio) — references only.
    pub blobs: Vec<BlobRef>,
    /// The turn's payload, in the existing client-message vocabulary.
    pub message: ClientMessage,
}

/// The worker's final-admission answer to [`dispatch`](crate::WorkerControl::dispatch):
/// the gateway's pick is a hint, and transport failure is the call's `Err`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Accepted {
    /// Accepted; the worker streams this turn's tokens via `push_tokens`.
    Ok { worker: WorkerId },
    /// Declined (just filled / draining); the gateway routes elsewhere (p2c).
    Reject,
    /// Declined with a suggested target; the gateway should try `worker` next.
    Redirect { worker: WorkerId },
}


/// One item on a turn's output stream (worker → gateway via
/// [`push_tokens`](crate::GatewayInbound::push_tokens)): [`Chunk`](Tokens::Chunk)s
/// terminated by exactly one [`Eos`](Tokens::Eos). Abort is deliberately not a
/// variant: it rides the session's channel-close, so a bare `None` is an abort.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Tokens {
    /// One output frame; `WireServerMessage` is exactly `ServerMessage`, so
    /// the worker's push pump forwards with no translation.
    Chunk(ServerMessage),
    /// Clean end of turn — the worker finished generating.
    Eos,
}

/// The gateway's reply to each
/// [`push_tokens`](crate::GatewayInbound::push_tokens), piggybacking ordinary
/// cancel onto the existing response. Immediate abort goes the reverse channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Control {
    /// Keep generating.
    Continue,
    /// Stop generating for this turn. A free, piggybacked cancel.
    Abort,
}
