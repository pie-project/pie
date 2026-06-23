//! The tarpc `Control` server — the distributed RPC front door.
//!
//! Thin by design: every state-changing call funnels a command to the
//! single-writer actor through the shared [`Handle`]; the two `watch_*` calls are
//! the §7 long-poll read-path (bounded by `T_HANG` so a no-change watch returns
//! as a keepalive before the client's RPC deadline). Eviction is signalled out
//! of band via `heartbeat`'s [`Ack::ReRegister`], so the watch calls return their
//! view directly.

use std::io;

use futures::{Stream, StreamExt, future};
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Bincode;
use tokio::task::JoinHandle;

use pie_control::{Control, ControlRequest, ControlResponse};
use pie_schema::control::{
    Ack, GatewayId, GatewayInfo, Neighbors, NodeId, RoutingTable, WorkerId, WorkerInfo,
    WorkerStatus,
};

use crate::Handle;

/// tarpc server: a cheap clone of the shared [`Handle`] per request.
#[derive(Clone)]
struct ControlServer {
    handle: Handle,
}

impl Control for ControlServer {
    async fn register_worker(self, _: tarpc::context::Context, info: WorkerInfo) -> WorkerId {
        self.handle.register_worker(info).await
    }

    async fn register_gateway(self, _: tarpc::context::Context, info: GatewayInfo) -> GatewayId {
        self.handle.register_gateway(info).await
    }

    async fn heartbeat(self, _: tarpc::context::Context, id: NodeId) -> Ack {
        self.handle.heartbeat(id).await
    }

    async fn report_worker(self, _: tarpc::context::Context, id: WorkerId, status: WorkerStatus) {
        self.handle.report_worker(id, status).await;
    }

    async fn watch_worker(self, _: tarpc::context::Context, id: WorkerId, since: u64) -> Neighbors {
        self.handle.watch_worker_poll(id, since).await
    }

    async fn watch_gateway(self, _: tarpc::context::Context, since: u64) -> RoutingTable {
        self.handle.watch_gateway_poll(since).await
    }
}

/// Bind the control endpoint (tcp or unix) and spawn the accept loop.
pub(crate) async fn serve(listen_addr: &str, handle: Handle) -> io::Result<JoinHandle<()>> {
    let server = ControlServer { handle };
    if let Some(path) = listen_addr
        .strip_prefix("unix://")
        .or_else(|| listen_addr.strip_prefix("unix:"))
    {
        let incoming = unix::listen(path, Bincode::default).await?;
        tracing::info!(listen = %listen_addr, "controller serving Control (tarpc/uds)");
        Ok(tokio::spawn(serve_loop(incoming, server)))
    } else {
        let tcp_addr = listen_addr.strip_prefix("tcp://").unwrap_or(listen_addr);
        let incoming = tcp::listen(tcp_addr, Bincode::default).await?;
        tracing::info!(listen = %listen_addr, "controller serving Control (tarpc/tcp)");
        Ok(tokio::spawn(serve_loop(incoming, server)))
    }
}

/// Serve `Control` over any tarpc transport stream (TCP or UDS).
async fn serve_loop<T>(incoming: impl Stream<Item = io::Result<T>>, server: ControlServer)
where
    T: tarpc::Transport<tarpc::Response<ControlResponse>, tarpc::ClientMessage<ControlRequest>>
        + Send
        + 'static,
{
    incoming
        .filter_map(|conn| future::ready(conn.ok()))
        .map(BaseChannel::with_defaults)
        .for_each_concurrent(None, |channel| {
            let server = server.clone();
            channel
                .execute(server.serve())
                .for_each_concurrent(None, |request| async move {
                    tokio::spawn(request);
                })
        })
        .await;
}
