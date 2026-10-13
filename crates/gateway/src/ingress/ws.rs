use axum::{
    extract::{
        State,
        ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade, close_code},
    },
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use futures::{SinkExt, StreamExt};

use crate::GatewayState;
use crate::ingress::identity;
use crate::session::{Affinity, Identity, SessionError, TokenRx, TurnInput};
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
    turns: Vec<Turn>,
    cursor: usize,
}

/// A turn's stream and the request it answers: once its response has gone
/// out, the request has been heard and a later failure has no one to tell.
struct Turn {
    corr: Option<u32>,
    answered: bool,
    rx: TokenRx,
}

enum TurnEvent {
    Item(Tokens),
    Aborted { corr: Option<u32>, answered: bool },
}

impl LiveTurns {
    fn new() -> Self {
        Self {
            turns: Vec::new(),
            cursor: 0,
        }
    }

    fn push(&mut self, corr: Option<u32>, rx: TokenRx) {
        self.turns.push(Turn {
            corr,
            answered: false,
            rx,
        });
    }

    async fn next(&mut self) -> TurnEvent {
        let live = self.turns.len();
        if live == 0 {
            return std::future::pending::<TurnEvent>().await;
        }
        self.turns.rotate_left(self.cursor % live);
        self.cursor = 0;
        let (item, index, _) = futures::future::select_all(
            self.turns
                .iter_mut()
                .map(|turn| Box::pin(turn.rx.recv()))
                .collect::<Vec<_>>(),
        )
        .await;
        self.cursor = index + 1;
        match item {
            Some(Tokens::Eos) => {
                self.turns.remove(index);
                TurnEvent::Item(Tokens::Eos)
            }
            Some(Tokens::Chunk(msg)) => {
                let turn = &mut self.turns[index];
                if let client_api::ServerMessage::Response { corr_id, .. } = &msg
                    && turn.corr == Some(*corr_id)
                {
                    turn.answered = true;
                }
                TurnEvent::Item(Tokens::Chunk(msg))
            }
            None => {
                let turn = self.turns.remove(index);
                TurnEvent::Aborted {
                    corr: turn.corr,
                    answered: turn.answered,
                }
            }
        }
    }
}

type Tx = futures::stream::SplitSink<WebSocket, Message>;

/// Says `why` to the request `corr` names and goes on serving. A failure
/// with no request to say it to ends the socket instead: nothing the client
/// waits on would otherwise ever hear back. `false` once the socket is done.
async fn answer(tx: &mut Tx, corr: Option<u32>, why: &str) -> bool {
    if corr.is_none() {
        end(tx, None, why).await;
        return false;
    }
    tx.send(refusal(corr, why)).await.is_ok()
}

/// Ends the socket over `why`, answered under `corr` where there is one and
/// on the close frame, so a client that reads its responses or only its
/// close hears why, and so does the server log.
async fn end(tx: &mut Tx, corr: Option<u32>, why: &str) {
    tracing::warn!("closing a client socket: {why}");
    let _ = tx.send(refusal(corr, why)).await;
    // A close frame's reason is at most 123 bytes.
    let mut cut = why.len().min(123);
    while !why.is_char_boundary(cut) {
        cut -= 1;
    }
    let _ = tx
        .send(Message::Close(Some(CloseFrame {
            code: close_code::POLICY,
            reason: why[..cut].into(),
        })))
        .await;
}

async fn serve(socket: WebSocket, state: GatewayState, ident: Identity) {
    let (mut tx, mut rx_ws) = socket.split();

    let first = match read_first_turn(&mut rx_ws).await {
        Ok(req) => req,
        Err(Some((corr, why))) => return end(&mut tx, corr, &why).await,
        Err(None) => return,
    };
    let corr = first.message.corr_id();
    let (handle, first_rx) = match state.sessions.create(ident, first, Affinity::Sticky).await {
        Ok(pair) => pair,
        Err(e) => return end(&mut tx, corr, &e.to_string()).await,
    };

    let mut live = LiveTurns::new();
    live.push(corr, first_rx);
    // A launch may wait at admission for a seat, so it waits here beside the
    // session's running turns and keeps their output flowing; every other
    // frame (an upload's chunks among them) is still dispatched in order.
    let admit = |req: TurnInput| {
        let corr = req.message.corr_id();
        let handle = &handle;
        async move { (corr, handle.turn(req).await) }
    };
    let mut admitting = futures::stream::FuturesUnordered::new();

    loop {
        tokio::select! {
            Some((corr, turn)) = admitting.next(), if !admitting.is_empty() => match turn {
                Ok(new_rx) => live.push(corr, new_rx),
                Err(e) => {
                    if !answer(&mut tx, corr, &SessionError::to_string(&e)).await {
                        break;
                    }
                }
            },
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
                TurnEvent::Aborted { corr, answered } => {
                    // A turn already answered (a launched process) has no
                    // request left to hear that its worker went away.
                    let corr = if answered { None } else { corr };
                    if !answer(&mut tx, corr, "the worker serving this turn went away").await {
                        break;
                    }
                }
            },

            incoming = rx_ws.next() => match incoming {
                Some(Ok(frame @ (Message::Text(_) | Message::Binary(_)))) => match read(&frame) {
                    Ok(Incoming::Turn(req)) if is_launch(&req) => admitting.push(admit(req)),
                    Ok(Incoming::Turn(req)) => {
                        let corr = req.message.corr_id();
                        match handle.turn(req).await {
                            Ok(new_rx) => live.push(corr, new_rx),
                            Err(e) => {
                                if !answer(&mut tx, corr, &e.to_string()).await {
                                    break;
                                }
                            }
                        }
                    }
                    Ok(Incoming::Cancel) => handle.cancel().await,
                    Err(why) => {
                        if !answer(&mut tx, corr_of(&frame), &why).await {
                            break;
                        }
                    }
                },
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(_)) => {}
                Some(Err(_)) => break,
            },
        }
    }

    drop(admitting);
    handle.close().await;
}

fn is_launch(req: &TurnInput) -> bool {
    matches!(req.message, ClientMessage::LaunchProcess { .. })
}

async fn read_first_turn(
    rx_ws: &mut futures::stream::SplitStream<WebSocket>,
) -> Result<TurnInput, Option<(Option<u32>, String)>> {
    loop {
        match rx_ws.next().await {
            Some(Ok(frame @ (Message::Text(_) | Message::Binary(_)))) => match read(&frame) {
                Ok(Incoming::Turn(req)) => return Ok(req),
                Ok(Incoming::Cancel) => continue,
                Err(why) => return Err(Some((corr_of(&frame), why))),
            },
            Some(Ok(Message::Close(_))) | None | Some(Err(_)) => return Err(None),
            Some(Ok(_)) => continue,
        }
    }
}

/// A text or binary frame as the message it carries.
fn read(frame: &Message) -> Result<Incoming, String> {
    match frame {
        Message::Text(t) => too_large(t.len()).map_or_else(|| parse_incoming(t.as_str()), Err),
        Message::Binary(b) => too_large(b.len()).map_or_else(|| parse_incoming_bytes(b), Err),
        _ => Err("not a text or binary frame".to_string()),
    }
}

/// The request id a frame carries, read on its own: a frame refused before
/// it parses whole (too large, or not a message this gateway knows) is still
/// answered under it when it has one.
fn corr_of(frame: &Message) -> Option<u32> {
    #[derive(serde::Deserialize)]
    struct Corr {
        corr_id: Option<u32>,
    }
    match frame {
        Message::Text(t) => serde_json::from_str::<Corr>(t).ok()?.corr_id,
        Message::Binary(b) => rmp_serde::from_slice::<Corr>(b).ok()?.corr_id,
        _ => None,
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
        a_frame_too_large_to_serve_is_refused_under_the_id_it_carries();
        a_frame_this_gateway_cannot_read_is_refused_under_its_id();
        a_frame_with_no_id_has_none_to_answer_under();
    }

    fn a_frame_too_large_to_serve_is_refused_under_the_id_it_carries() {
        let frame = Message::Binary(
            rmp_serde::to_vec_named(&client_api::ClientMessage::Query {
                corr_id: 9,
                subject: "x".repeat(MAX_CLIENT_FRAME_BYTES),
                record: String::new(),
            })
            .unwrap()
            .into(),
        );
        let why = read(&frame).err().expect("past the frame limit");
        assert!(why.contains("too large"), "{why}");
        assert_eq!(corr_of(&frame), Some(9));
    }

    fn a_frame_this_gateway_cannot_read_is_refused_under_its_id() {
        let frame = Message::Text(r#"{"type": "no_such_message", "corr_id": 4}"#.into());
        assert!(read(&frame).is_err());
        assert_eq!(corr_of(&frame), Some(4));
    }

    fn a_frame_with_no_id_has_none_to_answer_under() {
        for frame in [
            Message::Text("{not json".into()),
            Message::Text(r#"{"type": "no_such_message"}"#.into()),
            Message::Binary(vec![0xc1].into()),
        ] {
            assert!(read(&frame).is_err());
            assert_eq!(corr_of(&frame), None);
        }
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
