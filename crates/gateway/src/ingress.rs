//! `ingress/` — user-facing protocol adapters. Each converges onto the one
//! `Session`: one-shot (`http.rs`) is the degenerate 1-turn form of the
//! general multi-turn (`ws.rs`). The edge has already authed; we only trust +
//! extract identity (`identity.rs`).
//!
//! Per the "one server, one router" rule this module **contributes routes**
//! ([`router`]) but does not own the listener — `lib.rs` merges this with
//! `blob::router` and owns the single `axum::serve` over [`crate::GatewayState`].

pub mod http;
pub mod identity;
pub mod ws;

use axum::{
    Router,
    routing::{get, post},
};

use crate::GatewayState;

/// The ingress route set. Mounted by `lib.rs` via `.merge(ingress::router(state))`.
/// Reads only `GatewayState.sessions` (+ `blobs` for image ingest) — routing /
/// admission / dispatch live behind `Session`, never in ingress.
pub fn router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/generate", post(http::generate)) // REST + SSE, one-shot
        .route("/v1/ws", get(ws::ws)) // WebSocket, multi-turn
        .with_state(state)
}
