//! SCAFFOLD STUB — owned by **delta** (`ingress/`, landed real @`a76fd6d1`);
//! replaced at consolidation.
//!
//! Delta's real `ingress` is the user-facing protocol adapters — `http.rs`
//! (REST + SSE one-shot), `ws.rs` (WebSocket multi-turn), `identity.rs` (the
//! light edge-header identity gate) — all converging onto charlie's one
//! `Session`. Only the `router` provider foxtrot merges is stubbed here, so the
//! crate `cargo check`s while delta ports their `create` calls onto charlie's
//! affinity-mode signature.

use crate::GatewayState;

/// Route provider: mounts the REST/SSE + WS upgrade routes. foxtrot merges this
/// onto the one client-facing `axum::serve` with the shared [`GatewayState`].
pub fn router(_state: GatewayState) -> axum::Router {
    axum::Router::new()
}
