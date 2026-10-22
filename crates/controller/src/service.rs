use std::io;

use futures::{Stream, StreamExt, future};
use tarpc::serde_transport::tcp;
#[cfg(unix)]
use tarpc::serde_transport::unix;
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Bincode;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use controller_api::{
    Ack, Control, ControlRequest, ControlResponse, GatewayInfo, Neighbors, RoutingTable,
    ScaleEntry, ScaleStatus, WorkerInfo, WorkerStatus,
};
use ids::{GatewayId, NodeId, WorkerId};

use crate::Handle;

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

    async fn desire_scale(
        self,
        _: tarpc::context::Context,
        gateway: GatewayId,
        entries: Vec<ScaleEntry>,
    ) {
        self.handle.desire_scale(gateway, entries).await;
    }

    async fn scale_status(self, _: tarpc::context::Context) -> Vec<ScaleStatus> {
        self.handle.scale_status().await
    }
}

pub(crate) async fn serve(
    listen_addr: &str,
    handle: Handle,
    cancel: CancellationToken,
) -> io::Result<JoinHandle<()>> {
    let server = ControlServer { handle };
    if let Some(path) = listen_addr
        .strip_prefix("unix://")
        .or_else(|| listen_addr.strip_prefix("unix:"))
    {
        #[cfg(unix)]
        {
            let incoming = unix::listen(path, Bincode::default).await?;
            tracing::info!(listen = %listen_addr, "controller serving Control (tarpc/uds)");
            Ok(tokio::spawn(async move {
                tokio::select! {
                    _ = serve_loop(incoming, server) => {}
                    _ = cancel.cancelled() => {}
                }
            }))
        }
        #[cfg(not(unix))]
        {
            let _ = (path, server, cancel);
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "{listen_addr}: a `unix://` control address is distributed serving, which needs a unix-domain socket; this build is single-node and speaks `tcp://`"
                ),
            ))
        }
    } else {
        let tcp_addr = listen_addr.strip_prefix("tcp://").unwrap_or(listen_addr);
        let incoming = tcp::listen(tcp_addr, Bincode::default).await?;
        tracing::info!(listen = %listen_addr, "controller serving Control (tarpc/tcp)");
        Ok(tokio::spawn(async move {
            tokio::select! {
                _ = serve_loop(incoming, server) => {}
                _ = cancel.cancelled() => {}
            }
        }))
    }
}

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
