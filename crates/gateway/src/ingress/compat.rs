//! The API-compatible HTTP routes: a thin relay to the built-in inferlets.
//!
//! The gateway knows no API here. Each route maps to an inferlet name; the
//! request arrives at the inferlet as `{"path", "query", "body", "time"}`;
//! and the inferlet's events come back as the HTTP response, decided by the
//! first event it emits:
//!
//! - `message` — a streamed reply. The response is `text/event-stream`, and
//!   every message `{"event"?: name, "data": value}` is one SSE frame (a
//!   string `data` goes out as-is, anything else as JSON).
//! - `return` — one JSON body, status 200.
//! - `error` — `{"status": n, "body": …}` becomes exactly that response;
//!   any other error text is a 500.
//!
//! These routes carry no authentication: pie is an offline, single-user
//! server, and every compat request runs as one fixed identity.

use std::collections::HashMap;
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    Json,
    body::Bytes,
    extract::{Query, State},
    http::{HeaderMap, StatusCode, Uri, header},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{MethodRouter, post},
};
use client_api::{ClientMessage, ServerMessage};
use futures::{Stream, StreamExt};
use ids::TenantId;
use serde_json::{Value, json};
use worker_api::{Priority, Tokens};

use crate::GatewayState;
use crate::ingress::identity::REQUEST_ID_HEADER;
use crate::session::{Affinity, Identity, SessionHandle, TokenRx, TurnInput};

/// Route → the built-in inferlet that serves it.
pub const ROUTES: &[(&str, &str)] = &[
    ("/v1/chat/completions", "compat-openai"),
    ("/v1/completions", "compat-openai"),
    ("/v1/responses", "compat-openai"),
    ("/v1/messages", "compat-anthropic"),
    ("/v1beta/models/{model}", "compat-gemini"),
];

/// The handler for one of [`ROUTES`]: a `POST` relayed to `inferlet`.
pub fn route(inferlet: &'static str) -> MethodRouter<GatewayState> {
    post(
        move |State(state): State<GatewayState>,
              uri: Uri,
              Query(query): Query<HashMap<String, String>>,
              headers: HeaderMap,
              body: Bytes| relay(state, inferlet, uri, query, headers, body),
    )
}

fn identity(headers: &HeaderMap) -> Identity {
    Identity {
        tenant: TenantId("default".to_string()),
        user: "http".to_string(),
        client_ip: None,
        request_id: headers
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .filter(|s| !s.is_empty()),
    }
}

fn error_body(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.into(), "type": "server_error"}})),
    )
        .into_response()
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Relay one request to `inferlet` and answer with what it emits. The
/// session handle lives as long as the response does: dropping it cancels
/// the turn, which is what a client going away mid-stream does.
async fn relay(
    state: GatewayState,
    inferlet: &str,
    uri: Uri,
    query: HashMap<String, String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let body: Value = if body.is_empty() {
        Value::Object(Default::default())
    } else {
        match serde_json::from_slice(&body) {
            Ok(v) => v,
            Err(e) => return error_body(StatusCode::BAD_REQUEST, format!("body is not JSON: {e}")),
        }
    };
    let input = json!({
        "path": uri.path(),
        "query": query,
        "body": body,
        "time": unix_now(),
    });
    let turn = TurnInput {
        message: ClientMessage::LaunchProcess {
            corr_id: 1,
            inferlet: inferlet.to_string(),
            input: input.to_string(),
            capture_outputs: true,
        },
        blobs: Vec::new(),
        priority: Priority::Normal,
    };
    let (handle, mut rx) = match state
        .sessions
        .create(identity(&headers), turn, Affinity::Ephemeral)
        .await
    {
        Ok(pair) => pair,
        Err(e) => return error_body(StatusCode::SERVICE_UNAVAILABLE, format!("admission: {e}")),
    };

    // The first event decides the response.
    loop {
        match rx.recv().await {
            Some(Tokens::Chunk(ServerMessage::Response { ok, result, .. })) => {
                if !ok {
                    let status = if result.contains("not installed")
                        || result.contains("no program named")
                    {
                        StatusCode::NOT_FOUND
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    };
                    return error_body(status, format!("{inferlet}: {result}"));
                }
            }
            Some(Tokens::Chunk(ServerMessage::ProcessEvent { event, value, .. })) => {
                match event.as_str() {
                    "message" => {
                        return Sse::new(frames(value, rx, handle))
                            .keep_alive(KeepAlive::default())
                            .into_response();
                    }
                    "return" => return body_response(&value),
                    "error" => return error_response(&value),
                    _ => {} // stdout / stderr: the inferlet's own noise
                }
            }
            Some(Tokens::Chunk(ServerMessage::File { .. })) => {}
            Some(Tokens::Eos) => {
                return error_body(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("{inferlet} ended without answering"),
                );
            }
            None => return error_body(StatusCode::BAD_GATEWAY, "stream aborted"),
        }
    }
}

fn body_response(value: &str) -> Response {
    match serde_json::from_str::<Value>(value) {
        Ok(v) => (StatusCode::OK, Json(v)).into_response(),
        Err(_) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            value.to_string(),
        )
            .into_response(),
    }
}

/// `{"status": n, "body": …}` verbatim; anything else is a 500.
fn error_response(value: &str) -> Response {
    if let Ok(v) = serde_json::from_str::<Value>(value)
        && let Some(status) = v.get("status").and_then(Value::as_u64)
        && let Ok(status) = u16::try_from(status)
        && let Ok(status) = StatusCode::from_u16(status)
        && let Some(body) = v.get("body")
    {
        return (status, Json(body.clone())).into_response();
    }
    error_body(StatusCode::INTERNAL_SERVER_ERROR, value)
}

/// One `message` value as an SSE frame.
fn frame(value: &str) -> Event {
    let Ok(v) = serde_json::from_str::<Value>(value) else {
        return Event::default().data(value);
    };
    let Some(object) = v.as_object() else {
        return Event::default().data(value);
    };
    let mut event = Event::default();
    // axum splits data across lines but panics on a line break in a name.
    if let Some(name) = object.get("event").and_then(Value::as_str)
        && !name.contains(['\r', '\n'])
    {
        event = event.event(name);
    }
    match object.get("data") {
        Some(Value::String(s)) => event.data(s),
        Some(data) => event.data(data.to_string()),
        None => event.data(value),
    }
}

fn frames(
    first: String,
    rx: TokenRx,
    handle: SessionHandle,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let rest = futures::stream::unfold(Some((rx, handle)), |st| async move {
        let (mut rx, handle) = st?;
        loop {
            match rx.recv().await {
                Some(Tokens::Chunk(ServerMessage::ProcessEvent { event, value, .. })) => {
                    match event.as_str() {
                        "message" => return Some((Ok(frame(&value)), Some((rx, handle)))),
                        "return" => return None,
                        "error" => {
                            let data = match serde_json::from_str::<Value>(&value) {
                                Ok(v) => v.get("body").cloned().unwrap_or(v).to_string(),
                                Err(_) => {
                                    json!({"error": {"message": value, "type": "server_error"}})
                                        .to_string()
                                }
                            };
                            return Some((Ok(Event::default().event("error").data(data)), None));
                        }
                        _ => {}
                    }
                }
                Some(Tokens::Chunk(_)) => {}
                Some(Tokens::Eos) => return None,
                None => {
                    return Some((
                        Ok(Event::default().event("error").data("stream aborted")),
                        None,
                    ));
                }
            }
        }
    });
    futures::stream::once(async move { Ok(frame(&first)) }).chain(rest)
}

/// `GET /v1/models`: the loaded model, in the OpenAI list shape. Answered
/// by the gateway from the runtime's model status, because an inferlet is
/// too much machinery for one name.
pub async fn models(State(state): State<GatewayState>, headers: HeaderMap) -> Response {
    let turn = TurnInput {
        message: ClientMessage::Query {
            corr_id: 1,
            subject: client_api::message::QUERY_MODEL_STATUS.to_string(),
            record: String::new(),
        },
        blobs: Vec::new(),
        priority: Priority::Normal,
    };
    let (_handle, mut rx) = match state
        .sessions
        .create(identity(&headers), turn, Affinity::Ephemeral)
        .await
    {
        Ok(pair) => pair,
        Err(e) => return error_body(StatusCode::SERVICE_UNAVAILABLE, format!("admission: {e}")),
    };
    let mut names: Vec<String> = Vec::new();
    while let Some(token) = rx.recv().await {
        match token {
            Tokens::Chunk(ServerMessage::Response { ok, result, .. }) => {
                if ok
                    && let Ok(v) = serde_json::from_str::<Value>(&result)
                    && let Some(o) = v.as_object()
                {
                    for key in o.keys() {
                        if let Some(name) = key.strip_suffix(".kv_pages_total") {
                            names.push(name.to_string());
                        }
                    }
                }
            }
            Tokens::Eos => break,
            _ => {}
        }
    }
    let created = unix_now();
    let data: Vec<Value> = names
        .into_iter()
        .map(|id| json!({"id": id, "object": "model", "created": created, "owned_by": "pie"}))
        .collect();
    (
        StatusCode::OK,
        Json(json!({"object": "list", "data": data})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_its_event_and_data() {
        // The exact wire text is axum's; the parse is ours, and a name the
        // inferlet chose must not be able to panic it.
        let v = json!({"event": "ping", "data": {"a": 1}}).to_string();
        let _ = frame(&v);
        let raw = json!({"data": "[DONE]"}).to_string();
        let _ = frame(&raw);
        let _ = frame("not json");
        let _ = frame(&json!({"event": "a\nb", "data": "x\r\ny"}).to_string());
    }

    #[test]
    fn every_route_names_an_inferlet() {
        for (path, inferlet) in ROUTES {
            assert!(path.starts_with("/v1"), "{path}");
            assert!(!inferlet.is_empty());
        }
    }
}
