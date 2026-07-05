//! Gateway↔worker data-plane wire vocabulary — the turn types the
//! [`GatewayInbound`](crate::GatewayInbound) / [`WorkerControl`](crate::WorkerControl)
//! calls carry.
//!
//! The gateway terminates user protocols (REST/SSE, WebSocket), gates admission
//! on cluster *resources*, routes each turn to a worker, and pipes the resulting
//! token stream back. Post-inversion (Pie 0.5.0) the worker dials INTO the
//! gateway: the gateway is the listening server for the data plane, the worker
//! the client. These are the shared data types the two services carry; they are
//! flat-re-exported at the crate root (`pie_worker_rpc::Request`, …).
//!
//! The id atoms (`ReqId`/`SessionId`/`TenantId`/`WorkerId`) come from
//! [`pie_ids`]. The turn payload vocabulary (`ClientMessage`/`ServerMessage`)
//! lives in `pie-client-api`.
//!
//! CODEC CONSTRAINT: the gateway↔worker data plane MUST use a *self-describing*
//! codec (MessagePack via [`dispatch_codec`](crate::dispatch_codec), NOT
//! bincode), because [`Request`] / [`Tokens`] embed the internally-tagged
//! `ClientMessage` / `ServerMessage` (`#[serde(tag = "type")]`), which need
//! `deserialize_any` — bincode structurally can't decode them. The codec is
//! single-sourced in this crate (see [`link`](crate::link)) so the two ends
//! can't diverge.
//!
//! This is plain serde vocabulary, independent of the local driver ABI.

use serde::{Deserialize, Serialize};

use pie_ids::{ReqId, SessionId, TenantId, WorkerId};
// Wave-A BRIDGE → pie_client_api::{ClientMessage, ServerMessage} at Wave B.
use pie_client_api::{ClientMessage, ServerMessage};

// ──────────────────────────────── blob ────────────────────────────────

/// A reference to a large binary input (e.g. a user image). Blob bytes never
/// travel the command path — `dispatch` carries only this reference and the
/// worker pulls the bytes out-of-band over plain HTTP (`GET {origin}/blob/{hash}`,
/// design §9). Content-addressed, so integrity is free: the fetcher verifies
/// `hash(bytes) == hash`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobRef {
    /// Content address: blake3-256, hex-encoded. Also the `{hash}` URL segment.
    pub hash: String,
    /// Byte length — lets the worker pre-size / bound the fetch and sanity-check.
    pub size: u64,
    /// MIME type, e.g. `"image/jpeg"`. A string (not an enum) for forward-compat:
    /// new media types add no match-churn across consumers.
    pub kind: String,
    /// Base URL of the origin gateway's blob endpoint, e.g.
    /// `"http://10.0.0.5:8080"`. Tells any worker (even one dispatched by a
    /// different gateway) where to fetch — no shared directory needed.
    pub origin: String,
}

// ────────────────────────────── priority ──────────────────────────────

/// Scheduling priority for a turn, set at dispatch and adjustable via
/// [`WorkerControl::set_priority`](crate::WorkerControl::set_priority). Ordered
/// `Low < Normal < High` (derived from declaration order); `Normal` is the
/// default.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize,
)]
pub enum Priority {
    Low,
    #[default]
    Normal,
    High,
}

// ───────────────────────────── dispatch ───────────────────────────────

/// One dispatched turn (gateway → worker via
/// [`WorkerControl::dispatch`](crate::WorkerControl::dispatch)).
///
/// Self-describing: it carries everything the worker needs for this turn with no
/// reliance on a prior `open` (the old poll-based session lifecycle is now
/// gateway-internal). The turn's payload is the existing `ClientMessage` vocab —
/// the worker feeds it straight into the runtime broker, so no new runtime
/// vocabulary is introduced. Large binaries ride [`blobs`](Request::blobs) as
/// references, never inline bytes.
///
/// `dispatch` is idempotent + ack-based: a re-sent `Request` (same `req_id`) is
/// the same turn, so the gateway can re-route on a transport/no-ack failure
/// without duplicating work (design §8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// This turn's id (idempotency key for re-dispatch / cancel / redirect).
    pub req_id: ReqId,
    /// The logical session this turn belongs to (warm-KV affinity across turns).
    pub session: SessionId,
    /// Who the turn is attributed to (tenant/user; isolation & quota).
    pub tenant: TenantId,
    /// Scheduling priority for this turn.
    pub priority: Priority,
    /// Out-of-band binary inputs for this turn (images/audio) — references only.
    pub blobs: Vec<BlobRef>,
    /// The turn's payload, in the existing client-message vocabulary.
    pub message: ClientMessage,
}

/// The worker's final-admission answer to
/// [`dispatch`](crate::WorkerControl::dispatch) (design §7: the gateway's pick is
/// a hint; the worker decides authoritatively).
///
/// This is the *worker's answer* — accept / reject / redirect. A registry-level
/// transport or not-connected failure is a *different* class (the gateway's
/// idempotent re-dispatch path) and is NOT modeled here; it surfaces as the
/// dispatch call's `Err`, distinct from any `Accepted` variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Accepted {
    /// Accepted; the worker will stream this turn's tokens via `push_tokens`.
    /// Carries the accepting worker as a dedupe-safe ack (confirms the binding,
    /// even if the dispatch was a retry that fanned across candidates).
    Ok { worker: WorkerId },
    /// Declined (just filled / draining); the gateway routes elsewhere (p2c).
    Reject,
    /// Declined with a suggested target (e.g. a decode worker pointing at its
    /// prefill partner); the gateway should try `worker` next.
    Redirect { worker: WorkerId },
}

// ───────────────────────────── token stream ───────────────────────────

/// One item on a turn's output stream (worker → gateway via
/// [`push_tokens`](crate::GatewayInbound::push_tokens)). A turn is a sequence of
/// [`Chunk`](Tokens::Chunk)s terminated by exactly one [`Eos`](Tokens::Eos):
/// `GatewayInbound` has no separate `finish` call, so the clean terminal rides
/// here. These are the only two things ever sent on the wire.
///
/// Abnormal termination is deliberately NOT a variant: abort is a
/// gateway-internal session decision (cancel / already-emitted worker-drop /
/// drain), never a worker-originated wire event, so it rides the session's
/// channel-close (the consumer's `TokenRx::recv()` observes a bare `None` with
/// no preceding `Eos`). Keeping `Tokens` to `{ Chunk, Eos }` leaves it purely
/// wire-meaningful; clean-vs-abort stays unambiguous because `Eos` is always
/// observed before a clean close and never before an abort close.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Tokens {
    /// One output frame. The runtime's `WireServerMessage` is exactly
    /// `ServerMessage`, so the worker's push pump forwards with no translation.
    /// (Singular per the spec's `Chunk(WireServerMessage)` shape; the worker
    /// pipelines pushes rather than blocking on each `Control` reply, so the
    /// per-frame round-trip doesn't serialize the stream. Batching to
    /// `Chunk(Vec<..>)` is a deferred perf option, not needed for the spine.)
    Chunk(ServerMessage),
    /// Clean end of turn — the worker finished generating.
    Eos,
}

/// The gateway's reply to each
/// [`push_tokens`](crate::GatewayInbound::push_tokens), piggybacking ordinary
/// cancel onto the existing response (design §8): no extra round-trip for the
/// common case. Immediate abort (when the worker isn't pushing, e.g. mid-prefill)
/// goes the reverse channel via
/// [`WorkerControl::cancel`](crate::WorkerControl::cancel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Control {
    /// Keep streaming this turn's tokens.
    Continue,
    /// Stop generating for this turn (user disconnected / budget hit). A free,
    /// piggybacked cancel.
    Abort,
}
