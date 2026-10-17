//! Session props and `close` between a session's processes (PLEX 1.0 §9), to
//! the router and every worker's engine policy.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use client_api::Props;
use serde_json::json;

use crate::GatewayState;
use crate::ingress::compat::tenant;

pub async fn session_props(
    State(state): State<GatewayState>,
    Path(session): Path<String>,
    headers: HeaderMap,
    Json(changes): Json<Props>,
) -> Response {
    if let Err(why) = check(&session, &changes) {
        return refused(why);
    }
    let id = format!("{}/{session}", tenant(&headers));
    let text = serde_json::to_string(&changes).unwrap_or_default();
    if let Some(router) = &state.router {
        router.session_props(id.clone(), changes);
    }
    fan_out(&state, id, Some(text)).await;
    Json(json!({"ok": true})).into_response()
}

pub async fn session_close(
    State(state): State<GatewayState>,
    Path(session): Path<String>,
    headers: HeaderMap,
) -> Response {
    if let Err(why) = client_api::props::check_session(Some(&session)) {
        return refused(why);
    }
    let id = format!("{}/{session}", tenant(&headers));
    if let Some(router) = &state.router {
        router.session_closed(id.clone());
    }
    fan_out(&state, id, None).await;
    Json(json!({"ok": true})).into_response()
}

fn check(session: &str, changes: &Props) -> Result<(), String> {
    client_api::props::check_session(Some(session))?;
    client_api::props::from_client(&mut changes.clone())?;
    if let Some(name) = changes
        .keys()
        .find(|n| n.starts_with(client_api::props::ROUTER_PREFIX))
    {
        return Err(format!("`{name}` is the router's to set"));
    }
    Ok(())
}

async fn fan_out(state: &GatewayState, session: String, changes: Option<String>) {
    let calls = state.workers.clients().into_iter().map(|client| {
        let (session, changes) = (session.clone(), changes.clone());
        async move {
            let _ = client
                .plex_session(tarpc::context::current(), session, changes)
                .await;
        }
    });
    futures::future::join_all(calls).await;
}

fn refused(why: String) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": {"message": why, "type": "invalid_request_error"}})),
    )
        .into_response()
}
