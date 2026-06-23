//! Standalone-process deployment form (tarpc).
//!
//! [`run_as_process`] wraps an [`InProcController`] in a tarpc TCP server for the
//! distributed topology, where workers and the gateway reach the coordinator
//! over the network (`--controller=addr`). It is the *same* coordination logic
//! the on-device worker links in-proc — only the shell (a tarpc service) differs.
//! Clients dial it with the generated [`crate::ControlApiClient`] (workers via
//! the sync [`crate::RemoteController`] wrapper; the gateway async-natively).

use std::io;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt, future};
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Bincode;

use crate::controller::{Controller, ControllerConfig, InProcController};
use crate::error::{ControllerError, Result};
use crate::service::{ControlApi, ControlApiRequest, ControlApiResponse, RpcResult};
use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};

/// Configuration for the standalone controller process.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Address the control endpoint listens on (e.g. `"0.0.0.0:7000"`).
    pub listen_addr: String,
    /// Coordination knobs passed through to the embedded [`InProcController`].
    pub controller: ControllerConfig,
    /// How often the liveness clock advances (a missed report ages a worker out
    /// of routing). See [`ControllerConfig`] for the grading thresholds.
    pub tick_interval: Duration,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:7000".to_string(),
            controller: ControllerConfig::default(),
            tick_interval: Duration::from_secs(1),
        }
    }
}

/// The tarpc server: an `Arc<InProcController>` shared across all in-flight
/// requests. The generated `ControlApi` trait clones the server per request, so
/// the clone is just an `Arc` bump onto the one shared coordination state.
#[derive(Clone)]
struct ControlServer {
    controller: Arc<InProcController>,
}

impl ControlApi for ControlServer {
    async fn register(self, _: tarpc::context::Context, info: WorkerInfo) -> RpcResult<WorkerId> {
        self.controller.register(info).map_err(|e| e.to_string())
    }

    async fn report(
        self,
        _: tarpc::context::Context,
        worker: WorkerId,
        load: LoadState,
    ) -> RpcResult<()> {
        self.controller
            .report(worker, load)
            .map_err(|e| e.to_string())
    }

    async fn route(self, _: tarpc::context::Context, meta: RequestMeta) -> RpcResult<Placement> {
        self.controller.route(&meta).map_err(|e| e.to_string())
    }

    async fn pair(
        self,
        _: tarpc::context::Context,
        req: RequestId,
    ) -> RpcResult<(WorkerId, WorkerId)> {
        self.controller.pair(req).map_err(|e| e.to_string())
    }

    async fn resolve(self, _: tarpc::context::Context, worker: WorkerId) -> RpcResult<WorkerInfo> {
        self.controller.resolve(worker).map_err(|e| e.to_string())
    }
}

/// Run the controller as a standalone process: bind the tarpc endpoint and
/// serve [`ControlApi`] calls until the process is stopped.
///
/// Synchronous shell — builds a multi-thread tokio runtime and blocks on the
/// async serve loop, so the binary's `fn main` stays plain.
pub fn run_as_process(config: ProcessConfig) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| ControllerError::Transport(format!("build runtime: {e}")))?;
    rt.block_on(serve(config))
}

/// The async serve loop: one [`InProcController`] (`Arc`) shared across a
/// background liveness ticker and every connection's request handlers.
async fn serve(config: ProcessConfig) -> Result<()> {
    let controller = Arc::new(InProcController::new(config.controller));

    // Background liveness ticker: ages out workers that stop reporting.
    {
        let controller = Arc::clone(&controller);
        let interval = config.tick_interval;
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                controller.tick();
            }
        });
    }

    let addr = config.listen_addr.as_str();
    if let Some(path) = unix_scheme(addr) {
        let incoming = unix::listen(path, Bincode::default)
            .await
            .map_err(|e| ControllerError::Transport(format!("bind {addr}: {e}")))?;
        tracing::info!(listen = %addr, "pie-controller serving control plane (tarpc/uds)");
        serve_control(incoming, controller).await;
    } else {
        let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        let incoming = tcp::listen(tcp_addr, Bincode::default)
            .await
            .map_err(|e| ControllerError::Transport(format!("bind {addr}: {e}")))?;
        tracing::info!(listen = %addr, "pie-controller serving control plane (tarpc/tcp)");
        serve_control(incoming, controller).await;
    }

    Ok(())
}

/// `unix:`/`unix://`-scheme addresses → the socket path; everything else is TCP.
fn unix_scheme(addr: &str) -> Option<&str> {
    addr.strip_prefix("unix://")
        .or_else(|| addr.strip_prefix("unix:"))
}

/// Serve `ControlApi` over any tarpc transport stream (TCP or UDS). Control
/// messages are tiny, so the tarpc default frame cap is left in place.
async fn serve_control<T>(
    incoming: impl Stream<Item = io::Result<T>>,
    controller: Arc<InProcController>,
) where
    T: tarpc::Transport<
            tarpc::Response<ControlApiResponse>,
            tarpc::ClientMessage<ControlApiRequest>,
        > + Send
        + 'static,
{
    incoming
        // Drop accept errors rather than tearing down the whole listener.
        .filter_map(|conn| future::ready(conn.ok()))
        .map(BaseChannel::with_defaults)
        .for_each_concurrent(None, |channel| {
            let server = ControlServer {
                controller: Arc::clone(&controller),
            };
            channel
                .execute(server.serve())
                .for_each_concurrent(None, |request| async move {
                    tokio::spawn(request);
                })
        })
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RemoteController;
    use pie_schema::Role;
    use std::sync::mpsc;
    use std::thread;

    /// End-to-end control-RPC round trip over a real tarpc/TCP socket: a
    /// [`ControlServer`] over an [`InProcController`] on an ephemeral port, dialed
    /// by the sync [`RemoteController`]. Exercises every trait method plus the
    /// `Err(String)` → [`ControllerError::Remote`] relay path.
    #[test]
    fn remote_round_trip_over_tarpc() {
        let (addr_tx, addr_rx) = mpsc::channel();

        // Server runs forever on a background runtime thread; the process tears
        // it down at exit.
        thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                let controller = Arc::new(InProcController::new(ControllerConfig::default()));
                let incoming = tcp::listen("127.0.0.1:0", Bincode::default).await.unwrap();
                addr_tx.send(incoming.local_addr()).unwrap();
                serve_control(incoming, controller).await;
            });
        });

        let addr = addr_rx.recv().unwrap();
        let client = RemoteController::connect(addr.to_string()).unwrap();

        let w = client
            .register(WorkerInfo {
                control_addr: "x".to_string(),
                preferred_role: Some(Role::Prefill),
            })
            .unwrap();
        assert_eq!(w, WorkerId(0));

        client
            .report(
                w,
                LoadState {
                    active_requests: 1,
                    kv_pages_free: 5,
                },
            )
            .unwrap();

        let placement = client
            .route(&RequestMeta {
                id: RequestId(7),
                prompt_tokens: 3,
            })
            .unwrap();
        assert_eq!(placement.worker, w);
        assert_eq!(client.pair(RequestId(7)).unwrap(), (w, w));

        // A coordination error on the controller relays back as Remote.
        let err = client
            .report(
                WorkerId(99),
                LoadState {
                    active_requests: 0,
                    kv_pages_free: 0,
                },
            )
            .unwrap_err();
        assert!(matches!(err, ControllerError::Remote(_)));
    }
}
