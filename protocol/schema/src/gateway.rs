//! Gateway data-plane wire contract — the user-facing serving hot path.
//!
//! The gateway terminates user protocols (REST/SSE, WebSocket), gates admission
//! on cluster *resources*, routes each turn to a worker, and pipes the resulting
//! token stream back. Post-inversion (Pie 0.5.0) the worker dials INTO the
//! gateway: the gateway is the listening server for the data plane, the worker
//! the client. Two `#[tarpc::service]` traits ride one worker-initiated
//! connection (split with `spawn_twoway`):
//!
//! - [`GatewayInbound`] — served by the gateway, called by the worker (bulk,
//!   forward: token push + register + load report + redirect).
//! - [`WorkerControl`] — served by the worker, called by the gateway
//!   (latency-sensitive, varied: dispatch + cancel + set_priority + drain).
//!
//! Both traits live in the thin `pie-dispatch` crate so this floor stays
//! tarpc-free; these are the shared data types those calls carry.
//!
//! Like [`control`](crate::control)/[`cluster`](crate::cluster)/
//! [`message`](crate::message), this is plain serde vocabulary — deliberately
//! NOT `#[schema]`/rkyv (it never rides the zero-copy tensor ring), so it is NOT
//! part of `SCHEMA_HASH`.
//!
//! These types are reached as `pie_schema::gateway::*` (NOT flat-re-exported at
//! the crate root) — same discipline as [`control`](crate::control): keep the
//! module path explicit so a `gateway::SessionId` can coexist with the legacy
//! [`edge::SessionId`](crate::edge::SessionId) it supersedes until the old
//! poll-based session path (`edge_rpc` / `WorkerSessionApi`) is removed.

use serde::{Deserialize, Serialize};

use crate::control::WorkerId;
use crate::message::{ClientMessage, ServerMessage};

// ───────────────────────────── opaque ids ─────────────────────────────

/// Per-turn id, gateway-minted at dispatch. Keys a single in-flight turn across
/// `dispatch` / `push_tokens` / `cancel` / `set_priority` / `redirect`. A
/// multi-turn WS session produces one fresh `ReqId` per user prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ReqId(pub u64);

impl std::fmt::Display for ReqId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "req#{}", self.0)
    }
}

/// Logical session id, gateway-minted and stable across the turns of one
/// session (one-shot = a 1-turn session; WS = many). Carried on every
/// [`Request`] so a worker can bind warm KV across a multi-turn session without
/// any prior `open` handshake (the turn is self-describing). Distinct from
/// [`ReqId`]: `SessionId` spans turns, `ReqId` is one turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SessionId(pub u64);

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sess#{}", self.0)
    }
}

/// The edge-supplied principal a turn is attributed to (tenant / user id),
/// extracted by the light identity gate (`ingress/identity.rs`) from the trusted
/// edge header. Used for routing, quota, and isolation — NOT authentication (the
/// edge already authed; see design §5). An opaque string so the gateway does not
/// pin a tenant scheme.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

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
/// [`WorkerControl::set_priority`]. Ordered `Low < Normal < High` (derived from
/// declaration order); `Normal` is the default.
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

/// One dispatched turn (gateway → worker via [`WorkerControl::dispatch`]).
///
/// Self-describing: it carries everything the worker needs for this turn with no
/// reliance on a prior `open` (the old poll-based session lifecycle is now
/// gateway-internal). The turn's payload is the existing [`ClientMessage`] vocab
/// — the worker feeds it straight into the runtime broker, so no new runtime
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

/// The worker's final-admission answer to [`dispatch`](WorkerControl::dispatch)
/// (design §7: the gateway's pick is a hint; the worker decides authoritatively).
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
/// [`push_tokens`](GatewayInbound::push_tokens)). A turn is a sequence of
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
    /// [`ServerMessage`], so the worker's push pump forwards with no translation.
    /// (Singular per the spec's `Chunk(WireServerMessage)` shape; the worker
    /// pipelines pushes rather than blocking on each `Control` reply, so the
    /// per-frame round-trip doesn't serialize the stream. Batching to
    /// `Chunk(Vec<..>)` is a deferred perf option, not needed for the spine.)
    Chunk(ServerMessage),
    /// Clean end of turn — the worker finished generating.
    Eos,
}

/// The gateway's reply to each [`push_tokens`](GatewayInbound::push_tokens),
/// piggybacking ordinary cancel onto the existing response (design §8): no extra
/// round-trip for the common case. Immediate abort (when the worker isn't
/// pushing, e.g. mid-prefill) goes the reverse channel via
/// [`WorkerControl::cancel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Control {
    /// Keep streaming this turn's tokens.
    Continue,
    /// Stop generating for this turn (user disconnected / budget hit). A free,
    /// piggybacked cancel.
    Abort,
}
