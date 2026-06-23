//! REST + SSE adapter: a **one-shot** turn. Lifetime = the request. POST a
//! client payload, stream the token chunks back as Server-Sent Events. A 1-turn
//! session is the degenerate case of the general multi-turn `Session` (§6) — the
//! same `create → drain TokenRx → drop(=close)` path as `ws.rs`, just bounded to
//! one turn.

use std::convert::Infallible;

use axum::{
    Json,
    extract::State,
    http::HeaderMap,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use futures::Stream;

use crate::GatewayState;
use crate::ingress::identity;
use crate::session::{Affinity, Identity, SessionHandle, TokenRx, TurnInput};
use pie_schema::gateway::{Priority, Tokens};
use pie_schema::message::ClientMessage;

/// `POST /v1/generate` — one-shot generate. Body = a [`ClientMessage`] payload;
/// response = an SSE token stream terminated by a `[DONE]` sentinel (clean) or an
/// `error` event (mid-stream abort).
pub async fn generate(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(payload): Json<ClientMessage>,
) -> Response {
    // Trust-edge identity gate (§5). Fails closed on a misconfigured edge.
    let ident: Identity = match identity::extract(&headers) {
        Ok(id) => id,
        Err(e) => {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                format!("identity: {e}"),
            )
                .into_response();
        }
    };

    // Build the turn content. ingress hands charlie the user-turn payload only;
    // `create` mints `req_id` + `session` + derives `tenant` from `ident` (seam
    // option (a)) — no gateway-minted id is ever constructed here.
    // Image ingest (when present) would go out-of-band first:
    //   let blob = state.blobs.put(kind, bytes).await?;  // → blobs.push(blob)
    let turn = TurnInput {
        message: payload,
        blobs: Vec::new(),
        priority: Priority::Normal,
    };

    // One-shot ⇒ no warm-KV to preserve across turns: ephemeral affinity → p2c
    // load-spread (§7).
    let (handle, rx) = match state
        .sessions
        .create(ident, turn, Affinity::Ephemeral)
        .await
    {
        Ok(pair) => pair,
        Err(e) => {
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                format!("admission: {e}"),
            )
                .into_response();
        }
    };

    Sse::new(token_event_stream(handle, rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Drain one turn's [`TokenRx`] into SSE events. Holds `handle` alive for the
/// turn's duration (one-shot ⇒ drop-on-end = session close). Distinguishes a
/// clean `Eos` (→ `[DONE]`) from an unexpected channel close (abort → `error`).
fn token_event_stream(
    handle: SessionHandle,
    rx: TokenRx,
) -> impl Stream<Item = Result<Event, Infallible>> {
    enum St {
        Streaming { rx: TokenRx, handle: SessionHandle },
        End,
    }

    futures::stream::unfold(St::Streaming { rx, handle }, |st| async move {
        match st {
            St::Streaming { mut rx, handle } => match rx.recv().await {
                Some(Tokens::Chunk(msg)) => {
                    let data = serde_json::to_string(&msg)
                        .unwrap_or_else(|e| format!("{{\"encode_error\":\"{e}\"}}"));
                    let ev = Event::default().data(data);
                    Some((Ok(ev), St::Streaming { rx, handle }))
                }
                // Clean end-of-turn: emit the [DONE] sentinel, then stop.
                Some(Tokens::Eos) => {
                    let _ = &handle; // dropped after End ⇒ session closes
                    Some((Ok(Event::default().data("[DONE]")), St::End))
                }
                // Channel closed without an Eos ⇒ mid-stream abort (worker drop /
                // session gone). Surface an explicit error event, then stop.
                None => Some((
                    Ok(Event::default().event("error").data("stream aborted")),
                    St::End,
                )),
            },
            St::End => None,
        }
    })
}
