//! `pie-gateway` — Pie's client-facing **edge plane**.
//!
//! The single global host clients dial. It terminates client WebSockets, will
//! authenticate them, and routes each session to a worker chosen by the
//! controller. It is the third coordination plane alongside the data plane
//! (`pie-transport`, KV tensors) and the control plane (`pie-controller`,
//! placement decisions).
//!
//! # Plane boundaries
//!
//! The gateway is a pure control-plane **client**: it does not register as a
//! worker — it only *queries* the controller for placement (`route` → `resolve`)
//! over the same generated [`ControlApiClient`] workers use. It mediates the
//! **client API stream** (`pie_client::message`, prompts + tokens); KV and
//! activations never touch it — those stay worker↔worker on `pie-transport`.
//!
//! # Status
//!
//! Implemented. The gateway terminates client WebSockets, asks the controller
//! for placement (`route` + `resolve`), dials the placed worker's
//! `WorkerSessionApi`, opens a worker session, forwards client messages with
//! `send`, and relays worker messages back to the client with long-poll `recv`.

pub mod edge_rpc;

use std::net::SocketAddr;
use std::sync::Arc;

use crate::edge_rpc::{GatewayFrame, WorkerSessionApiClient};
use anyhow::{Context, Result, anyhow};
use futures::{SinkExt, StreamExt};
use pie_client::message::ClientMessage;
use pie_controller::{ControlApiClient, RequestId, RequestMeta, WorkerId};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex as TokioMutex, watch};
use tokio_tungstenite::accept_async;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Max edge-RPC frame — must match the worker's `edge_session` server. A `recv`
/// batch of up to 64 messages, each able to carry a 256 KiB session chunk, can
/// reach ~16 MiB; 64 MiB gives headroom over tarpc's 8 MiB default.
const EDGE_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Runtime configuration for the gateway process.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Address the client-facing WebSocket listens on (the single global host).
    pub listen: SocketAddr,
    /// Controller's tarpc control endpoint: `tcp://host:port`, a bare
    /// `host:port`, or `unix:/path`.
    pub controller: String,
}

/// Dial the controller, bind the client WebSocket, and serve until cancelled.
pub async fn run(config: GatewayConfig) -> Result<()> {
    let controller = connect_controller(&config.controller).await?;
    let listener = TcpListener::bind(config.listen)
        .await
        .with_context(|| format!("bind gateway listener on {}", config.listen))?;
    tracing::info!(
        listen = %config.listen,
        controller = %config.controller,
        "pie-gateway serving edge plane",
    );

    loop {
        let (stream, peer) = listener
            .accept()
            .await
            .context("accept client connection")?;
        let controller = controller.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, controller).await {
                tracing::warn!(?peer, error = %e, "gateway connection ended");
            }
        });
    }
}

/// Dial the controller's tarpc endpoint and spawn the client request dispatcher.
/// Control messages are tiny → tarpc's default frame cap is fine.
async fn connect_controller(addr: &str) -> Result<ControlApiClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        let conn = unix::connect(path, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlApiClient::new(cfg, conn).spawn())
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlApiClient::new(cfg, conn).spawn())
    }
}

/// Dial a worker's edge-rpc endpoint. The session path proxies `chunk_data`, so
/// the frame cap is lifted to match the worker's edge server.
async fn connect_worker(addr: &str) -> Result<WorkerSessionApiClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        let mut conn = unix::connect(path, Bincode::default);
        conn.config_mut().max_frame_length(EDGE_MAX_FRAME_BYTES);
        let conn = conn
            .await
            .with_context(|| format!("dialing worker edge-rpc at {addr}"))?;
        Ok(WorkerSessionApiClient::new(cfg, conn).spawn())
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let mut conn = tcp::connect(tcp_addr, Bincode::default);
        conn.config_mut().max_frame_length(EDGE_MAX_FRAME_BYTES);
        let conn = conn
            .await
            .with_context(|| format!("dialing worker edge-rpc at {addr}"))?;
        Ok(WorkerSessionApiClient::new(cfg, conn).spawn())
    }
}

/// Serve one client WebSocket: decode `ClientMessage` frames (msgpack — the
/// unchanged client wire format), then proxy the session to one routed worker
/// over `WorkerSessionApi` (`open`/`send`/`recv`/`close`).
async fn handle_connection(stream: TcpStream, controller: ControlApiClient) -> Result<()> {
    let ws = accept_async(stream).await.context("websocket handshake")?;
    let (tx, mut rx) = ws.split();

    let (worker_id, worker_addr) = place(&controller, 0).await?;
    let worker = connect_worker(&worker_addr).await?;
    let session = worker
        .open(tarpc::context::current())
        .await
        .context("worker open rpc")?
        .map_err(|e| anyhow!("worker rejected open: {e}"))?;

    tracing::debug!(
        ?worker_id,
        worker_addr = %worker_addr,
        session = session.0,
        "gateway proxied session opened",
    );

    let ws_tx = Arc::new(TokioMutex::new(tx));
    let (stop_tx, stop_rx) = watch::channel(false);

    let poll_task = {
        let worker = worker.clone();
        let ws_tx = Arc::clone(&ws_tx);
        tokio::spawn(async move {
            let mut stop_rx = stop_rx;
            loop {
                tokio::select! {
                    changed = stop_rx.changed() => {
                        if changed.is_err() || *stop_rx.borrow() {
                            break;
                        }
                    }
                    poll = worker.recv(tarpc::context::current(), session, 200) => {
                        let frames = match poll {
                            Ok(Ok(frames)) => frames,
                            Ok(Err(msg)) => {
                                tracing::warn!(session = session.0, error = %msg, "worker recv rejected");
                                break;
                            }
                            Err(e) => {
                                tracing::warn!(session = session.0, error = %e, "worker recv transport error");
                                break;
                            }
                        };

                        for frame in frames {
                            let encoded = match rmp_serde::to_vec_named(&frame.message) {
                                Ok(bytes) => bytes,
                                Err(e) => {
                                    tracing::warn!(session = session.0, error = %e, "encode ServerMessage failed");
                                    continue;
                                }
                            };

                            let mut guard = ws_tx.lock().await;
                            if guard.send(WsMessage::Binary(encoded.into())).await.is_err() {
                                return;
                            }
                        }
                    }
                }
            }
        })
    };

    while let Some(frame) = rx.next().await {
        let bytes = match frame.context("websocket read")? {
            WsMessage::Binary(b) => b,
            WsMessage::Close(_) => break,
            _ => continue,
        };
        let msg: ClientMessage =
            rmp_serde::from_slice(bytes.as_ref()).context("decode ClientMessage")?;

        worker
            .send(
                tarpc::context::current(),
                session,
                GatewayFrame { message: msg },
            )
            .await
            .context("worker send rpc")?
            .map_err(|e| anyhow!("worker rejected send: {e}"))?;
    }

    let _ = stop_tx.send(true);
    let _ = poll_task.await;
    let _ = worker.close(tarpc::context::current(), session).await;
    Ok(())
}

/// Ask the controller to place a request, then resolve the chosen worker's
/// dialable control address. Two round-trips today; the resolved address is the
/// natural thing for the gateway to cache per worker once the worker leg exists.
async fn place(controller: &ControlApiClient, prompt_tokens: u32) -> Result<(WorkerId, String)> {
    let placement = controller
        .route(
            tarpc::context::current(),
            RequestMeta {
                id: RequestId(0),
                prompt_tokens,
            },
        )
        .await
        .context("route rpc")?
        .map_err(|e| anyhow!("controller rejected route: {e}"))?;

    let info = controller
        .resolve(tarpc::context::current(), placement.worker)
        .await
        .context("resolve rpc")?
        .map_err(|e| anyhow!("controller rejected resolve: {e}"))?;

    Ok((placement.worker, info.control_addr))
}
