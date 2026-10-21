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
//! A streamed reply whose worker is lost after the client had output goes
//! on elsewhere: each frame names, under `continuation`, the tokens it
//! covers (taken off before the client sees it), and the request is
//! launched again with them, the new run dropping the frames the client
//! has. These APIs alone are continued, because their meaning is known
//! here; any other program's turn ends and its client retries.
//!
//! These routes carry no authentication: pie is an offline, single-user
//! server, and every compat request runs as tenant `default` unless an
//! `x-plex-tenant` header names another.
//!
//! Props come from `x-plex-prop-<name>` headers (JSON if it parses) over a
//! `plex` body object; `x-plex-session` names the session; `prompt` defaults to
//! the request's text.

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
use client_api::{ClientMessage, Props, ServerMessage};
use futures::{Stream, StreamExt};
use ids::TenantId;
use serde_json::{Value, json};
use worker_api::{Priority, Tokens};

use crate::GatewayState;
use crate::ingress::identity::REQUEST_ID_HEADER;
use crate::session::{Affinity, Identity, SessionError, SessionHandle, TokenRx, TurnInput};

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

pub fn tenant(headers: &HeaderMap) -> String {
    headers
        .get(TENANT_HEADER)
        .and_then(|v| v.to_str().ok())
        .filter(|t| !t.is_empty() && !t.contains('/'))
        .unwrap_or("default")
        .to_string()
}

fn identity(headers: &HeaderMap) -> Identity {
    Identity {
        tenant: TenantId(tenant(headers)),
        user: "http".to_string(),
        client_ip: None,
        request_id: headers
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .filter(|s| !s.is_empty()),
    }
}

const PROP_HEADER_PREFIX: &str = "x-plex-prop-";
const TENANT_HEADER: &str = "x-plex-tenant";
const SESSION_HEADER: &str = "x-plex-session";

fn props(headers: &HeaderMap, body: &Value) -> Props {
    let mut props: Props = match body.get("plex") {
        Some(Value::Object(plex)) => plex.clone().into_iter().collect(),
        _ => Props::new(),
    };
    for (name, value) in headers {
        let (Some(name), Ok(value)) = (
            name.as_str().strip_prefix(PROP_HEADER_PREFIX),
            value.to_str(),
        ) else {
            continue;
        };
        let value =
            serde_json::from_str(value).unwrap_or_else(|_| Value::String(value.to_string()));
        props.insert(name.to_string(), value);
    }
    if !props.contains_key(client_api::props::PROMPT)
        && let Some(text) = prompt_text(body)
    {
        props.insert(client_api::props::PROMPT.to_string(), Value::String(text));
    }
    if !props.contains_key(client_api::props::MAX_TOKENS)
        && let Some(limit) = ["max_completion_tokens", "max_tokens", "max_output_tokens"]
            .iter()
            .find_map(|key| body.get(*key).and_then(Value::as_u64))
    {
        props.insert(
            client_api::props::MAX_TOKENS.to_string(),
            Value::from(limit),
        );
    }
    props
}

/// In order, so a later turn's text starts with an earlier one's.
fn prompt_text(body: &Value) -> Option<String> {
    fn gather(value: &Value, text: &mut String) {
        match value {
            Value::String(part) => {
                text.push_str(part);
                text.push('\n');
            }
            Value::Array(items) => items.iter().for_each(|item| gather(item, text)),
            Value::Object(map) => ["role", "content", "text", "parts"]
                .iter()
                .filter_map(|key| map.get(*key))
                .for_each(|value| gather(value, text)),
            _ => {}
        }
    }
    let mut text = String::new();
    for key in [
        "system",
        "instructions",
        "systemInstruction",
        "messages",
        "contents",
        "input",
        "prompt",
    ] {
        if let Some(value) = body.get(key) {
            gather(value, &mut text);
        }
    }
    let mut end = text.len().min(client_api::props::MAX_PROMPT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text.truncate(end);
    (!text.is_empty()).then_some(text)
}

fn error_body(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({"error": {"message": message.into(), "type": "server_error"}})),
    )
        .into_response()
}

/// What the runtime says when one request's sequence can never fit: past
/// the context one sequence holds, or more KV pages than the pool has.
const OVERFLOW: &[&str] = &[
    "one sequence holds on this engine",
    "KV pool exhausted by this process alone",
    "can never fit",
];

/// A turn that failed because its prompt and budget do not fit is the
/// client's to shorten: a 400, as the OpenAI servers answer it.
fn overflow(message: &str) -> Option<Response> {
    OVERFLOW.iter().any(|m| message.contains(m)).then(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {
                "message": format!("the prompt and max_tokens do not fit this model's context: {message}"),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            }})),
        )
            .into_response()
    })
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// What a turn is launched with, kept so it can be launched again.
#[derive(Clone)]
struct Launch {
    state: GatewayState,
    identity: Identity,
    inferlet: String,
    input: Value,
    props: Props,
    session: Option<String>,
}

impl Launch {
    /// The turn, or (`continuation`) the same request again, going on after
    /// what the client was sent ([`Kept`]).
    async fn start(
        &self,
        continuation: Option<Value>,
    ) -> Result<(SessionHandle, TokenRx), SessionError> {
        let mut input = self.input.clone();
        if let Some(continuation) = continuation {
            input["continuation"] = continuation;
        }
        let turn = TurnInput {
            message: ClientMessage::LaunchProcess {
                corr_id: 1,
                inferlet: self.inferlet.clone(),
                input: input.to_string(),
                capture_outputs: true,
                props: self.props.clone(),
                session: self.session.clone(),
                cluster_props: Props::new(),
                executors: Vec::new(),
            },
            blobs: Vec::new(),
            priority: Priority::Normal,
        };
        self.state
            .sessions
            .create(self.identity.clone(), turn, Affinity::Ephemeral)
            .await
    }
}

/// What the client has been sent of a streamed reply, as the frames named it
/// under `continuation` (the built-in inferlets' `compat::continuation`): the
/// response id, the tokens the frames covered, and how many frames there
/// were. A turn whose worker is lost after output goes on from it elsewhere.
#[derive(Default)]
struct Kept {
    id: Option<String>,
    tokens: Vec<u64>,
    frames: u64,
}

impl Kept {
    /// One frame, its `continuation` taken off and kept: what the client sees.
    fn frame(&mut self, value: &str) -> Event {
        self.frames += 1;
        let Ok(mut v) = serde_json::from_str::<Value>(value) else {
            return frame(value);
        };
        if let Some(covers) = v.as_object_mut().and_then(|o| o.remove("continuation")) {
            if let Some(id) = covers.get("id").and_then(Value::as_str) {
                self.id = Some(id.to_string());
            }
            self.tokens.extend(
                covers
                    .get("tokens")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_u64),
            );
            return frame(&v.to_string());
        }
        frame(value)
    }

    /// The input's `continuation`, once a frame has named the id.
    fn continuation(&self) -> Option<Value> {
        let id = self.id.as_ref()?;
        Some(json!({"id": id, "tokens": self.tokens, "frames": self.frames}))
    }
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
    let props = props(&headers, &body);
    let launch = Launch {
        state,
        identity: identity(&headers),
        inferlet: inferlet.to_string(),
        input: json!({
            "path": uri.path(),
            "query": query,
            "body": body,
            "time": unix_now(),
        }),
        props,
        session: headers
            .get(SESSION_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string),
    };
    let (handle, mut rx) = match launch.start(None).await {
        Ok(pair) => pair,
        Err(e @ SessionError::Launch(_)) => {
            return error_body(StatusCode::BAD_REQUEST, e.to_string());
        }
        Err(e @ SessionError::Refused(_)) => {
            return error_body(StatusCode::TOO_MANY_REQUESTS, e.to_string());
        }
        Err(e) => return error_body(StatusCode::SERVICE_UNAVAILABLE, format!("admission: {e}")),
    };

    // The first event decides the response.
    loop {
        match rx.recv().await {
            Some(Tokens::Chunk(ServerMessage::Response { ok, result, .. })) => {
                if !ok {
                    if let Some(response) = overflow(&result) {
                        return response;
                    }
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
                        return Sse::new(frames(launch, value, rx, handle))
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
    if let Some(response) = overflow(value) {
        return response;
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

/// A streamed reply, frame by frame. If its worker is lost after the client
/// had output, the turn goes on elsewhere from what the client was sent
/// ([`Kept`]): the new run drops the frames the client has, so the stream
/// goes on as it stood.
fn frames(
    launch: Launch,
    first: String,
    rx: TokenRx,
    handle: SessionHandle,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let mut kept = Kept::default();
    let first = kept.frame(&first);
    let rest = futures::stream::unfold(Some((rx, handle, kept, launch)), |st| async move {
        let (mut rx, mut handle, mut kept, launch) = st?;
        loop {
            let lost = match rx.recv().await {
                Some(Tokens::Chunk(ServerMessage::ProcessEvent { event, value, .. })) => {
                    match event.as_str() {
                        "message" => {
                            let event = kept.frame(&value);
                            return Some((Ok(event), Some((rx, handle, kept, launch))));
                        }
                        "return" => return None,
                        "error" if rx.lost() => true,
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
                        _ => false,
                    }
                }
                Some(Tokens::Chunk(_)) => false,
                Some(Tokens::Eos) => return None,
                None if rx.lost() => true,
                None => {
                    return Some((
                        Ok(Event::default().event("error").data("stream aborted")),
                        None,
                    ));
                }
            };
            if !lost {
                continue;
            }
            // The worker is gone after the client had output: the request
            // runs again elsewhere, going on after what the client has.
            let again = match kept.continuation() {
                Some(continuation) => launch.start(Some(continuation)).await.ok(),
                None => None,
            };
            let Some((next_handle, next_rx)) = again else {
                let data = json!({
                    "error": {"message": crate::session::LOST_AFTER_OUTPUT, "type": "server_error"}
                })
                .to_string();
                return Some((Ok(Event::default().event("error").data(data)), None));
            };
            tracing::info!(
                frames = kept.frames,
                tokens = kept.tokens.len(),
                "continuing a reply its worker lost"
            );
            (handle, rx) = (next_handle, next_rx);
        }
    });
    futures::stream::once(async move { Ok(first) }).chain(rest)
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
    fn a_reply_keeps_what_its_frames_covered_and_the_client_never_sees_it() {
        let mut kept = Kept::default();
        assert_eq!(kept.continuation(), None, "nothing named yet");
        let first = kept.frame(r#"{"data":{"x":1},"continuation":{"id":"1a2b","tokens":[]}}"#);
        let _ = kept.frame(r#"{"event":"delta","data":{"x":2},"continuation":{"tokens":[5,6]}}"#);
        let _ = kept.frame(r#"{"data":"[DONE]","continuation":{"tokens":[7]}}"#);
        assert_eq!(
            format!("{first:?}"),
            format!("{:?}", frame(r#"{"data":{"x":1}}"#)),
            "the client's frame is the frame without its continuation"
        );
        assert_eq!(
            kept.continuation(),
            Some(json!({"id": "1a2b", "tokens": [5, 6, 7], "frames": 3}))
        );
    }

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
    fn a_sequence_that_cannot_fit_is_the_clients_400() {
        let hog = "prefill submit @0: KV pool exhausted by this process alone: it holds 3 pages and asks 9000 more of a 4096-page pool";
        assert_eq!(error_response(hog).status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            error_response("pipeline: this fire would grow its sequence to 9 KV page(s) (up to 144 tokens), past the 128 tokens (8 page(s)) one sequence holds on this engine").status(),
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            error_response("something else").status(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    #[test]
    fn a_front_end_names_the_tenant() {
        let mut headers = HeaderMap::new();
        assert_eq!(tenant(&headers), "default");
        headers.insert("x-plex-tenant", "acme".parse().unwrap());
        assert_eq!(tenant(&headers), "acme");
        headers.insert("x-plex-tenant", "a/b".parse().unwrap());
        assert_eq!(tenant(&headers), "default", "a tenant has no `/`");
    }

    #[test]
    fn props_come_from_the_body_and_the_headers() {
        let mut headers = HeaderMap::new();
        headers.insert("x-plex-prop-priority", "3".parse().unwrap());
        headers.insert("x-plex-prop-tier", "gold".parse().unwrap());
        headers.insert("x-plex-prop-tags", r#"["a","b"]"#.parse().unwrap());
        headers.insert("x-request-id", "r".parse().unwrap());
        let body = json!({"model": "m", "plex": {"priority": 1, "deadline-ms": 500}});
        let expected: Props = serde_json::from_value(json!({
            "priority": 3,
            "tier": "gold",
            "tags": ["a", "b"],
            "deadline-ms": 500,
        }))
        .unwrap();
        assert_eq!(props(&headers, &body), expected);
        assert!(props(&HeaderMap::new(), &json!({"plex": "not a dict"})).is_empty());
    }

    #[test]
    fn the_output_limit_is_a_prop_unless_the_client_set_one() {
        let limit = |body: Value| props(&HeaderMap::new(), &body).get("max-tokens").cloned();
        assert_eq!(limit(json!({"max_tokens": 64})), Some(json!(64)));
        assert_eq!(
            limit(json!({"max_completion_tokens": 32, "max_tokens": 64})),
            Some(json!(32))
        );
        assert_eq!(
            limit(json!({"max_tokens": 64, "plex": {"max-tokens": 8}})),
            Some(json!(8))
        );
        assert_eq!(limit(json!({"model": "m"})), None);
    }

    #[test]
    fn the_prompt_prop_is_the_requests_own_text() {
        let turn = |messages: Value| json!({"model": "m", "messages": messages});
        let first = turn(json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ]));
        let second = turn(json!([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "more"},
        ]));
        let text = |body: &Value| {
            props(&HeaderMap::new(), body)["prompt"]
                .as_str()
                .unwrap()
                .to_string()
        };
        assert_eq!(text(&first), "system\nbe brief\nuser\nhi\n");
        assert!(
            text(&second).starts_with(&text(&first)),
            "a later turn extends it"
        );
        assert_eq!(text(&json!({"prompt": "once upon"})), "once upon\n");
        let mut headers = HeaderMap::new();
        headers.insert("x-plex-prop-prompt", "the client's".parse().unwrap());
        assert_eq!(props(&headers, &first)["prompt"], json!("the client's"));
        assert!(!props(&HeaderMap::new(), &json!({"model": "m"})).contains_key("prompt"));
    }

    #[test]
    fn every_route_names_an_inferlet() {
        for (path, inferlet) in ROUTES {
            assert!(path.starts_with("/v1"), "{path}");
            assert!(!inferlet.is_empty());
        }
    }
}
