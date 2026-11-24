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
use client_api::ClientMessage;
use worker_api::{Priority, Tokens};

pub async fn generate(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    Json(payload): Json<ClientMessage>,
) -> Response {
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

    let turn = TurnInput {
        message: payload,
        blobs: Vec::new(),
        priority: Priority::Normal,
        resume: None,
    };

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
                Some(Tokens::Eos) => {
                    let _ = &handle;
                    Some((Ok(Event::default().data("[DONE]")), St::End))
                }
                None => Some((
                    Ok(Event::default().event("error").data("stream aborted")),
                    St::End,
                )),
            },
            St::End => None,
        }
    })
}
