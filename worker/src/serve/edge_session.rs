//! Worker-side edge-rpc server (distributed mode).
//!
//! Serves the worker's own `crate::rpc::worker_session_api::WorkerSessionApi` on
//! the worker's public endpoint so a gateway can proxy one client session into
//! one worker runtime session over tarpc. (Single-node terminates clients
//! directly instead — see [`super::client_server`].)

use std::io;

use anyhow::{Context, Result};
use futures::{Stream, StreamExt, future};
use pie_schema::{GatewayFrame, SessionId, WorkerFrame};
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Bincode;

use crate::rpc::worker_session_api::{
    WorkerSessionApi, WorkerSessionApiRequest, WorkerSessionApiResponse,
};

#[derive(Clone)]
struct WorkerSessionServer;

fn to_client_id(session: SessionId) -> std::result::Result<pie::server::ClientId, String> {
    u32::try_from(session.0).map_err(|_| format!("invalid session id {}", session.0))
}

impl WorkerSessionApi for WorkerSessionServer {
    async fn open(self, _: tarpc::context::Context) -> std::result::Result<SessionId, String> {
        pie::server::open_session()
            .map(|id| SessionId(id as u64))
            .map_err(|e| e.to_string())
    }

    async fn send(
        self,
        _: tarpc::context::Context,
        session: SessionId,
        frame: GatewayFrame,
    ) -> std::result::Result<(), String> {
        let client_id = to_client_id(session)?;
        pie::server::send_client_message(client_id, frame.message).map_err(|e| e.to_string())
    }

    async fn recv(
        self,
        _: tarpc::context::Context,
        session: SessionId,
        max_wait_ms: u64,
    ) -> std::result::Result<Vec<WorkerFrame>, String> {
        let client_id = to_client_id(session)?;
        pie::server::recv_messages(client_id, max_wait_ms, 64)
            .await
            .map(|messages| {
                messages
                    .into_iter()
                    .map(|message| WorkerFrame { message })
                    .collect()
            })
            .map_err(|e| e.to_string())
    }

    async fn close(
        self,
        _: tarpc::context::Context,
        session: SessionId,
    ) -> std::result::Result<(), String> {
        let client_id = to_client_id(session)?;
        pie::server::close_session(client_id);
        Ok(())
    }
}

pub struct EdgeSessionServerHandle {
    /// The resolved bound endpoint in canonical `tcp://`/`unix:` form (for logs;
    /// reflects the actual ephemeral port when bound to `:0`).
    pub bound: String,
    pub task: tokio::task::JoinHandle<()>,
}

/// Max edge-RPC frame. A `recv` long-poll returns a batch of up to 64 session
/// messages, each able to carry a `CHUNK_SIZE_BYTES` (256 KiB) chunk, so a batch
/// can reach ~16 MiB — over tarpc's 8 MiB default. 64 MiB gives headroom without
/// going unbounded; both ends (worker server here, gateway client) must agree.
const EDGE_MAX_FRAME_BYTES: usize = 64 * 1024 * 1024;

/// Bind and serve the worker's edge-rpc endpoint. `listen` is `tcp://host:port`,
/// a bare `host:port`, or `unix:/path`.
pub async fn spawn(listen: &str) -> Result<EdgeSessionServerHandle> {
    let (task, bound) = if let Some(path) = listen
        .strip_prefix("unix://")
        .or_else(|| listen.strip_prefix("unix:"))
    {
        let mut incoming = unix::listen(path, Bincode::default)
            .await
            .with_context(|| format!("bind worker edge-rpc on {listen}"))?;
        incoming.config_mut().max_frame_length(EDGE_MAX_FRAME_BYTES);
        (tokio::spawn(serve_sessions(incoming)), listen.to_string())
    } else {
        let tcp_addr = listen.strip_prefix("tcp://").unwrap_or(listen);
        let mut incoming = tcp::listen(tcp_addr, Bincode::default)
            .await
            .with_context(|| format!("bind worker edge-rpc on {listen}"))?;
        incoming.config_mut().max_frame_length(EDGE_MAX_FRAME_BYTES);
        let bound = format!("tcp://{}", incoming.local_addr());
        (tokio::spawn(serve_sessions(incoming)), bound)
    };

    Ok(EdgeSessionServerHandle { bound, task })
}

/// Serve `WorkerSessionApi` over any tarpc transport stream (TCP or UDS).
async fn serve_sessions<T>(incoming: impl Stream<Item = io::Result<T>>)
where
    T: tarpc::Transport<
            tarpc::Response<WorkerSessionApiResponse>,
            tarpc::ClientMessage<WorkerSessionApiRequest>,
        > + Send
        + 'static,
{
    incoming
        .filter_map(|conn| future::ready(conn.ok()))
        .map(BaseChannel::with_defaults)
        .for_each_concurrent(None, |channel| {
            let server = WorkerSessionServer;
            channel
                .execute(server.serve())
                .for_each_concurrent(None, |request| async move {
                    tokio::spawn(request);
                })
        })
        .await;
}
