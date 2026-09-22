pub mod compat;
pub mod http;
pub mod identity;
pub mod status;
pub mod ws;

use axum::{
    Router,
    routing::{get, post},
};

use crate::GatewayState;

pub fn router(state: GatewayState) -> Router {
    let router = Router::new()
        .route("/v1/generate", post(http::generate)) // REST + SSE, one-shot
        .route("/v1/ws", get(ws::ws)) // WebSocket, multi-turn
        .route("/v1/models", get(compat::models))
        .route("/status", get(status::status));
    // The API-compatible routes: each a relay to one built-in inferlet.
    compat::ROUTES
        .iter()
        .fold(router, |router, (path, inferlet)| {
            router.route(path, compat::route(inferlet))
        })
        .with_state(state)
}
