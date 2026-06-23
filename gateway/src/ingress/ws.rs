//! WebSocket adapter: a **multi-turn** interactive session. Lifetime = the
//! connection (§6). The general case; `http.rs` is its 1-turn degenerate form.
//!
//! One `select!` multiplexes the two directions on charlie's `Session` seam:
//!   * worker→user: drain the current turn's `TokenRx` to WS frames, and
//!   * user→worker: read the next client turn (`handle.turn`) or a `cancel`.
//!
//! The `&self`-control (`turn`/`cancel`) + owned-`&mut TokenRx` split is what
//! lets both arms live in one `select!` with no borrow clash. Between turns the
//! token arm is parked (`pending()`), so the loop waits only on the client.

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
use pie_schema::gateway::{Priority, Tokens};
use pie_schema::message::ClientMessage;

/// `GET /v1/ws` — upgrade to a multi-turn session. Identity is extracted from the
/// edge headers *before* the upgrade (a bad edge never gets a socket).
pub async fn ws(
    State(state): State<GatewayState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    let ident = match identity::extract(&headers) {
        Ok(id) => id,
        Err(e) => return (StatusCode::UNAUTHORIZED, format!("identity: {e}")).into_response(),
    };
    upgrade.on_upgrade(move |socket| serve(socket, state, ident))
}

/// One parsed client frame: a new turn, or a cancel of the in-flight turn.
enum Incoming {
    Turn(TurnInput),
    Cancel,
}

/// Drive one WebSocket connection across many turns.
async fn serve(socket: WebSocket, state: GatewayState, ident: Identity) {
    let (mut tx, mut rx_ws) = socket.split();

    // The first client frame opens the session (its `create` mints the ReqId).
    let first = match read_first_turn(&mut rx_ws).await {
        Some(req) => req,
        None => return, // closed / errored before any turn
    };
    // Multi-turn ⇒ sticky affinity so every turn prefers the warm-KV worker
    // (HRW; `turn()` reuses the key automatically). §7.
    let (handle, first_rx) = match state.sessions.create(ident, first, Affinity::Sticky).await {
        Ok(pair) => pair,
        Err(e) => {
            let _ = tx.send(Message::Text(error_json(&e.to_string()))).await;
            let _ = tx.send(Message::Close(None)).await;
            return;
        }
    };

    // `Some` = a turn is streaming; `None` = idle between turns (token arm parked).
    let mut cur: Option<TokenRx> = Some(first_rx);

    loop {
        tokio::select! {
            tok = next_token(&mut cur) => match tok {
                Some(Tokens::Chunk(msg)) => {
                    if tx.send(Message::Text(encode(&msg))).await.is_err() {
                        break; // user hung up
                    }
                }
                // Clean end-of-turn: tell the client, then park awaiting the next.
                Some(Tokens::Eos) => {
                    cur = None;
                    if tx.send(Message::Text(turn_done_json())).await.is_err() {
                        break;
                    }
                }
                // Channel closed without Eos ⇒ mid-stream abort: close the socket.
                None => {
                    let _ = tx.send(Message::Text(error_json("stream aborted"))).await;
                    let _ = tx.send(Message::Close(None)).await;
                    break;
                }
            },

            incoming = rx_ws.next() => match incoming {
                Some(Ok(Message::Text(t))) => match parse_incoming(t.as_str()) {
                    Ok(Incoming::Turn(req)) => match handle.turn(req).await {
                        Ok(new_rx) => cur = Some(new_rx),
                        Err(e) => {
                            let _ = tx.send(Message::Text(error_json(&e.to_string()))).await;
                            break;
                        }
                    },
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(e) => {
                        // Bad frame is non-fatal: report and keep the session.
                        let _ = tx.send(Message::Text(error_json(&e))).await;
                    }
                },
                Some(Ok(Message::Binary(b))) => match parse_incoming_bytes(&b) {
                    Ok(Incoming::Turn(req)) => match handle.turn(req).await {
                        Ok(new_rx) => cur = Some(new_rx),
                        Err(e) => {
                            let _ = tx.send(Message::Text(error_json(&e.to_string()))).await;
                            break;
                        }
                    },
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(e) => {
                        let _ = tx.send(Message::Text(error_json(&e))).await;
                    }
                },
                Some(Ok(Message::Close(_))) | None => break,
                // axum auto-replies to pings; ignore control frames.
                Some(Ok(_)) => {}
                Some(Err(_)) => break,
            },
        }
    }

    handle.close().await;
}

/// Poll the current turn's receiver, or park forever when idle so the `select!`
/// waits only on the client between turns.
async fn next_token(cur: &mut Option<TokenRx>) -> Option<Tokens> {
    match cur.as_mut() {
        Some(rx) => rx.recv().await,
        None => std::future::pending::<Option<Tokens>>().await,
    }
}

/// Read frames until the first turn-bearing one. Returns `None` if the client
/// closes (or errors) before sending any turn.
async fn read_first_turn(rx_ws: &mut futures::stream::SplitStream<WebSocket>) -> Option<TurnInput> {
    loop {
        match rx_ws.next().await {
            Some(Ok(Message::Text(t))) => match parse_incoming(t.as_str()) {
                Ok(Incoming::Turn(req)) => return Some(req),
                _ => continue, // a cancel/bad frame before any turn is meaningless
            },
            Some(Ok(Message::Binary(b))) => match parse_incoming_bytes(&b) {
                Ok(Incoming::Turn(req)) => return Some(req),
                _ => continue,
            },
            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return None,
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
        serde_json::from_slice(bytes).map_err(|e| format!("bad client frame: {e}"))?;
    Ok(Incoming::Turn(into_turn(payload)))
}

/// Wrap a client payload into per-turn content. `req_id`/`session` stay charlie's
/// to mint in `turn`/`create`. Blob ingest (if any) attaches `blobs` here.
fn into_turn(payload: ClientMessage) -> TurnInput {
    TurnInput {
        message: payload,
        blobs: Vec::new(),
        priority: Priority::Normal,
    }
}

fn encode(msg: &pie_schema::message::ServerMessage) -> String {
    serde_json::to_string(msg).unwrap_or_else(|e| error_json(&e.to_string()))
}

fn turn_done_json() -> String {
    "{\"type\":\"turn_done\"}".to_string()
}

fn error_json(msg: &str) -> String {
    serde_json::json!({ "type": "error", "message": msg }).to_string()
}
