//! `pie-gateway` — Pie's client-facing **edge plane**.
//!
//! The single global host clients dial. It terminates client WebSockets, will
//! authenticate them, and routes each session to a worker it selects locally
//! from the controller's pushed routing table. It is the third coordination
//! plane alongside the data plane (`pie-transport`, KV tensors) and the control
//! plane (`pie-control`, membership + routing).
//!
//! # Plane boundaries
//!
//! The gateway is a control-plane **client** (`pie_control::ControlClient`): it
//! registers as a gateway, heartbeats for liveness, and long-polls
//! `watch_gateway` for the global [`RoutingTable`]. It does **not** ask the
//! controller per request — it caches the latest table and selects a worker
//! locally for each session. It mediates the **client API stream**
//! (`pie_client::message`, prompts + tokens); KV and activations never touch it —
//! those stay worker↔worker on `pie-transport`.
//!
//! # Status
//!
//! Implemented. The gateway registers with the controller, runs heartbeat +
//! routing-watch loops, terminates client WebSockets, selects a worker from the
//! cached routing table, dials the worker's `WorkerSessionApi`, opens a session,
//! forwards client messages with `send`, and relays worker messages back with
//! long-poll `recv`.

pub mod edge_rpc;

use std::net::SocketAddr;
use std::sync::Arc;

use crate::edge_rpc::{GatewayFrame, WorkerSessionApiClient};
use anyhow::{Context, Result, anyhow};
use futures::{SinkExt, StreamExt};
use pie_client::message::ClientMessage;
use pie_control::ControlClient;
use pie_schema::control::{Ack, GatewayId, GatewayInfo, Health, NodeId, RoutableWorker, RoutingTable};
use std::time::{Duration, Instant};
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

/// How often the gateway heartbeats the controller. Must be comfortably shorter
/// than the controller's liveness-timeout window so a healthy gateway is never
/// evicted between beats.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// Client deadline for the `watch_gateway` long-poll. Generous so the
/// server-side block (until the routing epoch advances) dominates; on expiry we
/// simply re-poll with the same `since`. Transport errors surface immediately
/// regardless of this bound.
const WATCH_DEADLINE: Duration = Duration::from_secs(300);

/// Backoff before retrying `watch_gateway` after a transport error, so a wedged
/// or restarting controller doesn't spin the loop.
const WATCH_RETRY_BACKOFF: Duration = Duration::from_secs(1);

/// Runtime configuration for the gateway process.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Address the client-facing WebSocket listens on (the single global host).
    pub listen: SocketAddr,
    /// Controller's tarpc control endpoint: `tcp://host:port`, a bare
    /// `host:port`, or `unix:/path`.
    pub controller: String,
}

/// Dial the controller, register as a gateway, start the heartbeat + routing-watch
/// loops, bind the client WebSocket, and serve until cancelled.
pub async fn run(config: GatewayConfig) -> Result<()> {
    let control = connect_controller(&config.controller).await?;

    // Register with the controller. `addr` is where this gateway is reachable —
    // the client-facing listen address.
    let info = GatewayInfo {
        addr: config.listen.to_string(),
    };
    let gateway_id = control
        .register_gateway(tarpc::context::current(), info.clone())
        .await
        .context("register_gateway rpc")?;
    tracing::info!(%gateway_id, "gateway registered with controller");

    // Latest routing table, pushed by the watch loop, read locally per session.
    let (routing_tx, routing_rx) = watch::channel(RoutingTable {
        epoch: 0,
        workers: Vec::new(),
    });

    // Liveness + re-register on `Ack::ReRegister`.
    tokio::spawn(heartbeat_loop(control.clone(), gateway_id, info));
    // Long-poll the global routing table; publish each update.
    tokio::spawn(watch_routing_loop(control, routing_tx));

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
        let routing_rx = routing_rx.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(stream, routing_rx).await {
                tracing::warn!(?peer, error = %e, "gateway connection ended");
            }
        });
    }
}

/// Heartbeat the controller on a fixed interval. On [`Ack::ReRegister`] the
/// controller has lost our soft state (restart / timeout), so we re-register and
/// adopt the new [`GatewayId`]. Transport errors are logged and retried next tick.
async fn heartbeat_loop(control: ControlClient, mut id: GatewayId, info: GatewayInfo) {
    let mut ticker = tokio::time::interval(HEARTBEAT_INTERVAL);
    loop {
        ticker.tick().await;
        match control
            .heartbeat(tarpc::context::current(), NodeId::Gateway(id))
            .await
        {
            Ok(Ack::Ok) => {}
            Ok(Ack::ReRegister) => {
                tracing::warn!(%id, "controller lost our registration; re-registering");
                match control
                    .register_gateway(tarpc::context::current(), info.clone())
                    .await
                {
                    Ok(new_id) => {
                        id = new_id;
                        tracing::info!(%id, "gateway re-registered");
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "re-register failed; retrying next tick");
                    }
                }
            }
            Err(e) => tracing::warn!(error = %e, "heartbeat transport error"),
        }
    }
}

/// Long-poll `watch_gateway`, publishing each new [`RoutingTable`] to the shared
/// channel. The controller blocks until the routing epoch advances past `since`;
/// we re-poll with the returned epoch. On a transport error we back off and retry.
async fn watch_routing_loop(control: ControlClient, routing_tx: watch::Sender<RoutingTable>) {
    let mut since: u64 = 0;
    loop {
        let mut ctx = tarpc::context::current();
        ctx.deadline = Instant::now() + WATCH_DEADLINE;
        match control.watch_gateway(ctx, since).await {
            Ok(table) => {
                since = table.epoch;
                tracing::debug!(epoch = since, workers = table.workers.len(), "routing table updated");
                // All receivers dropped → the gateway is shutting down.
                if routing_tx.send(table).is_err() {
                    break;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "watch_gateway error; retrying");
                tokio::time::sleep(WATCH_RETRY_BACKOFF).await;
            }
        }
    }
}

/// Dial the controller's tarpc endpoint and spawn the client request dispatcher.
/// Control messages are tiny → tarpc's default frame cap is fine.
async fn connect_controller(addr: &str) -> Result<ControlClient> {
    let cfg = tarpc::client::Config::default();
    if let Some(path) = addr
        .strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
    {
        let conn = unix::connect(path, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlClient::new(cfg, conn).spawn())
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let conn = tcp::connect(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("dialing controller at {addr}"))?;
        Ok(ControlClient::new(cfg, conn).spawn())
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
/// unchanged client wire format), then proxy the session to one worker —
/// selected locally from the latest pushed [`RoutingTable`] — over
/// `WorkerSessionApi` (`open`/`send`/`recv`/`close`).
async fn handle_connection(
    stream: TcpStream,
    routing_rx: watch::Receiver<RoutingTable>,
) -> Result<()> {
    let ws = accept_async(stream).await.context("websocket handshake")?;
    let (tx, mut rx) = ws.split();

    // Pick a worker locally from the cached routing table (no per-request
    // controller RPC). Scope the borrow so it's dropped before we await.
    let (worker_id, worker_addr) = {
        let table = routing_rx.borrow();
        let worker = select_worker(&table).ok_or_else(|| {
            anyhow!(
                "no healthy worker available in routing table (epoch {})",
                table.epoch
            )
        })?;
        (worker.id, worker.addr.clone())
    };
    let worker = connect_worker(&worker_addr).await?;
    let session = worker
        .open(tarpc::context::current())
        .await
        .context("worker open rpc")?
        .map_err(|e| anyhow!("worker rejected open: {e}"))?;

    tracing::debug!(
        %worker_id,
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

/// Select a worker for a new session from the cached routing table: healthy
/// workers only, least-loaded by coarse load (KV-pressure bucket, then in-flight).
/// Least-loaded to start; role/model filtering can be layered in once a session's
/// target model is known (placement happens before the first client message, so
/// only the table's coarse signals are available here). Returns `None` when no
/// healthy worker is known yet (empty/stale table).
fn select_worker(table: &RoutingTable) -> Option<&RoutableWorker> {
    table
        .workers
        .iter()
        .filter(|w| w.health == Health::Healthy)
        .min_by_key(|w| (w.coarse_load.kv_pressure_bucket, w.coarse_load.inflight))
}
