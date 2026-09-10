use axum::{
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};

use crate::GatewayState;
use crate::ingress::identity;
use crate::session::{Affinity, Identity, TokenRx, TurnInput};
use crate::worker::{MAX_CLIENT_FRAME_BYTES, MAX_CLIENT_FRAME_RECV_BYTES};
use client_api::ClientMessage;
use worker_api::{Priority, Tokens};

pub async fn ws(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let ident = match identity::extract(&headers) {
        Ok(id) => id,
        Err(e) => return (StatusCode::UNAUTHORIZED, format!("identity: {e}")).into_response(),
    };
    upgrade
        .max_message_size(MAX_CLIENT_FRAME_RECV_BYTES)
        .max_frame_size(MAX_CLIENT_FRAME_RECV_BYTES)
        .on_upgrade(move |socket| serve(socket, state, ident))
}

fn too_large(len: usize) -> Option<String> {
    (len > MAX_CLIENT_FRAME_BYTES).then(|| {
        format!(
            "request too large: {len} bytes, limit {MAX_CLIENT_FRAME_BYTES} \
             (send large inputs as a blob reference, not inline)"
        )
    })
}

enum Incoming {
    Turn(TurnInput),
    Cancel,
}

struct LiveTurns {
    streams: Vec<TokenRx>,
    cursor: usize,
}

enum TurnEvent {
    Item(Tokens),
    Aborted,
}

impl LiveTurns {
    fn new() -> Self {
        Self {
            streams: Vec::new(),
            cursor: 0,
        }
    }

    fn push(&mut self, rx: TokenRx) {
        self.streams.push(rx);
    }

    async fn next(&mut self) -> TurnEvent {
        let live = self.streams.len();
        if live == 0 {
            return std::future::pending::<TurnEvent>().await;
        }
        self.streams.rotate_left(self.cursor % live);
        self.cursor = 0;
        let (item, index, _) = futures::future::select_all(
            self.streams
                .iter_mut()
                .map(|rx| Box::pin(rx.recv()))
                .collect::<Vec<_>>(),
        )
        .await;
        self.cursor = index + 1;
        match item {
            Some(Tokens::Eos) => {
                self.streams.remove(index);
                TurnEvent::Item(Tokens::Eos)
            }
            Some(other) => TurnEvent::Item(other),
            None => {
                self.streams.remove(index);
                TurnEvent::Aborted
            }
        }
    }
}

async fn serve(socket: WebSocket, state: GatewayState, ident: Identity) {
    let (mut tx, mut rx_ws) = socket.split();

    let first = match read_first_turn(&mut rx_ws).await {
        Ok(req) => req,
        Err(Some(why)) => {
            let _ = tx.send(Message::Text(error_json(&why).into())).await;
            let _ = tx.send(Message::Close(None)).await;
            return;
        }
        Err(None) => return,
    };
    let corr = first.message.corr_id();
    let (handle, first_rx) = match state.sessions.create(ident, first, Affinity::Sticky).await {
        Ok(pair) => pair,
        Err(e) => {
            let _ = tx.send(refusal(corr, &e.to_string())).await;
            let _ = tx.send(Message::Close(None)).await;
            return;
        }
    };

    let mut live = LiveTurns::new();
    live.push(first_rx);

    loop {
        tokio::select! {
            ev = live.next() => match ev {
                TurnEvent::Item(Tokens::Chunk(msg)) => {
                    match encode(&msg) {
                        Some(bytes) => {
                            if tx.send(Message::Binary(bytes.into())).await.is_err() {
                                break;
                            }
                        }
                        None => continue,
                    }
                }
                TurnEvent::Item(Tokens::Eos) => {
                    if tx
                        .send(Message::Text(turn_done_json().into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                TurnEvent::Aborted => {
                    if tx
                        .send(Message::Text(error_json("stream aborted").into()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            },

            incoming = rx_ws.next() => match incoming {
                Some(Ok(Message::Text(t))) if too_large(t.len()).is_some() => {
                    let why = too_large(t.len()).expect("guard matched");
                    if tx.send(Message::Text(error_json(&why).into())).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Binary(b))) if too_large(b.len()).is_some() => {
                    let why = too_large(b.len()).expect("guard matched");
                    if tx.send(Message::Text(error_json(&why).into())).await.is_err() {
                        break;
                    }
                }
                Some(Ok(Message::Text(t))) => match parse_incoming(t.as_str()) {
                    Ok(Incoming::Turn(req)) => {
                        let corr = req.message.corr_id();
                        match handle.turn(req).await {
                            Ok(new_rx) => live.push(new_rx),
                            Err(e) => {
                                if tx.send(refusal(corr, &e.to_string())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(e) => {
                        let _ = tx.send(Message::Text(error_json(&e).into())).await;
                    }
                },
                Some(Ok(Message::Binary(b))) => match parse_incoming_bytes(&b) {
                    Ok(Incoming::Turn(req)) => {
                        let corr = req.message.corr_id();
                        match handle.turn(req).await {
                            Ok(new_rx) => live.push(new_rx),
                            Err(e) => {
                                if tx.send(refusal(corr, &e.to_string())).await.is_err() {
                                    break;
                                }
                            }
                        }
                    }
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(e) => {
                        let _ = tx.send(Message::Text(error_json(&e).into())).await;
                    }
                },
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(_)) => {}
                Some(Err(_)) => break,
            },
        }
    }

    handle.close().await;
}

async fn read_first_turn(
    rx_ws: &mut futures::stream::SplitStream<WebSocket>,
) -> Result<TurnInput, Option<String>> {
    loop {
        match rx_ws.next().await {
            Some(Ok(Message::Text(t))) if too_large(t.len()).is_some() => {
                return Err(too_large(t.len()));
            }
            Some(Ok(Message::Binary(b))) if too_large(b.len()).is_some() => {
                return Err(too_large(b.len()));
            }
            Some(Ok(Message::Text(t))) => match parse_incoming(t.as_str()) {
                Ok(Incoming::Turn(req)) => return Ok(req),
                _ => continue,
            },
            Some(Ok(Message::Binary(b))) => match parse_incoming_bytes(&b) {
                Ok(Incoming::Turn(req)) => return Ok(req),
                _ => continue,
            },
            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return Err(None),
            Some(Ok(_)) => continue,
        }
    }
}

fn parse_incoming(text: &str) -> Result<Incoming, String> {
    if text.trim() == "cancel" {
        return Ok(Incoming::Cancel);
    }
    let payload: ClientMessage =
        serde_json::from_str(text).map_err(|e| format!("bad client frame: {e}"))?;
    Ok(Incoming::Turn(into_turn(payload)))
}

fn parse_incoming_bytes(bytes: &[u8]) -> Result<Incoming, String> {
    let payload: ClientMessage =
        rmp_serde::from_slice(bytes).map_err(|e| format!("bad client frame: {e}"))?;
    Ok(Incoming::Turn(into_turn(payload)))
}

fn into_turn(payload: ClientMessage) -> TurnInput {
    TurnInput {
        message: payload,
        blobs: Vec::new(),
        priority: Priority::Normal,
    }
}

fn encode(msg: &client_api::ServerMessage) -> Option<Vec<u8>> {
    match rmp_serde::to_vec_named(msg) {
        Ok(bytes) => Some(bytes),
        Err(e) => {
            tracing::error!(error = %e, "ServerMessage msgpack encode failed; dropping frame");
            None
        }
    }
}

fn turn_done_json() -> String {
    "{\"type\":\"turn_done\"}".to_string()
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "type": "error", "message": msg }).to_string()
}

fn refusal(corr_id: Option<u32>, why: &str) -> Message {
    match corr_id.and_then(|corr_id| {
        encode(&client_api::ServerMessage::Response {
            corr_id,
            ok: false,
            result: why.to_string(),
        })
    }) {
        Some(bytes) => Message::Binary(bytes.into()),
        None => Message::Text(error_json(why).into()),
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::*;

    #[test]
    fn ws_every_case() {
        a_correlated_call_is_refused_under_its_own_id();
        an_uncorrelated_frame_gets_the_bare_error();
        every_call_that_expects_a_response_carries_its_id();
    }

    fn a_correlated_call_is_refused_under_its_own_id() {
        let frame = refusal(Some(7), "cluster saturated");
        let Message::Binary(bytes) = frame else {
            panic!("a refusal of a correlated call is a binary response frame");
        };
        let decoded: client_api::ServerMessage =
            rmp_serde::from_slice(&bytes).expect("the codec pie-client reads");
        match decoded {
            client_api::ServerMessage::Response {
                corr_id,
                ok,
                result,
            } => {
                assert_eq!(corr_id, 7);
                assert!(!ok);
                assert_eq!(result, "cluster saturated");
            }
            other => panic!("expected a response frame, got {other:?}"),
        }
    }

    fn an_uncorrelated_frame_gets_the_bare_error() {
        let Message::Text(text) = refusal(None, "no route") else {
            panic!("a refusal with no id to answer under is the error text frame");
        };
        let value: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["message"], "no route");
    }

    fn every_call_that_expects_a_response_carries_its_id() {
        let ping: client_api::ClientMessage =
            serde_json::from_str(r#"{"type":"ping","corr_id":3}"#).unwrap();
        assert_eq!(ping.corr_id(), Some(3));
        let signal: client_api::ClientMessage =
            serde_json::from_str(r#"{"type":"signal_process","process_id":"p","message":"m"}"#)
                .unwrap();
        assert_eq!(signal.corr_id(), None);
    }
}
