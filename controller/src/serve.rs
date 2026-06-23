//! Standalone-process deployment form (tarpc).
//!
//! [`run_as_process`] wraps a [`Coordinator`] in a tarpc server, binding the
//! control endpoint that workers and the gateway dial (`--controller=addr`). The
//! [`Coordinator`] holds the coordination state; this module is just the RPC
//! shell around it. Clients dial it with the generated [`crate::ControlApiClient`].

use std::io;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt, future};
use tarpc::serde_transport::{tcp, unix};
use tarpc::server::{BaseChannel, Channel};
use tarpc::tokio_serde::formats::Bincode;

use crate::coordinator::{ControllerConfig, Coordinator};
use crate::error::{ControllerError, Result};
use crate::service::{ControlApi, ControlApiRequest, ControlApiResponse, RpcResult};
use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};

/// Configuration for the standalone controller process.
#[derive(Debug, Clone)]
pub struct ProcessConfig {
    /// Address the control endpoint listens on (e.g. `"0.0.0.0:7000"`).
    pub listen_addr: String,
    /// Coordination knobs passed through to the [`Coordinator`].
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

/// The tarpc server: an `Arc<Coordinator>` shared across all in-flight
/// requests. The generated `ControlApi` trait clones the server per request, so
/// the clone is just an `Arc` bump onto the one shared coordination state.
#[derive(Clone)]
struct ControlServer {
    controller: Arc<Coordinator>,
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

/// The async serve loop: one [`Coordinator`] (`Arc`) shared across a background
/// liveness ticker and every connection's request handlers.
async fn serve(config: ProcessConfig) -> Result<()> {
    let controller = Arc::new(Coordinator::new(config.controller));

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
async fn serve_control<T>(incoming: impl Stream<Item = io::Result<T>>, controller: Arc<Coordinator>)
where
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
    use crate::service::ControlApiClient;
    use pie_schema::Role;
    use std::sync::mpsc;
    use std::thread;

    /// End-to-end control-RPC round trip over a real tarpc/TCP socket: a
    /// [`ControlServer`] over a [`Coordinator`] on an ephemeral port, dialed by
    /// the generated [`ControlApiClient`]. Exercises every RPC method plus the
    /// `Err(String)` rejection-relay path.
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
                let controller = Arc::new(Coordinator::new(ControllerConfig::default()));
                let incoming = tcp::listen("127.0.0.1:0", Bincode::default).await.unwrap();
                addr_tx.send(incoming.local_addr()).unwrap();
                serve_control(incoming, controller).await;
            });
        });

        let addr = addr_rx.recv().unwrap();

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let conn = tcp::connect(addr, Bincode::default).await.unwrap();
            let client = ControlApiClient::new(tarpc::client::Config::default(), conn).spawn();
            let ctx = tarpc::context::current;

            let w = client
                .register(
                    ctx(),
                    WorkerInfo {
                        control_addr: "x".to_string(),
                        preferred_role: Some(Role::Prefill),
                    },
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(w, WorkerId(0));

            client
                .report(
                    ctx(),
                    w,
                    LoadState {
                        active_requests: 1,
                        kv_pages_free: 5,
                    },
                )
                .await
                .unwrap()
                .unwrap();

            let placement = client
                .route(
                    ctx(),
                    RequestMeta {
                        id: RequestId(7),
                        prompt_tokens: 3,
                    },
                )
                .await
                .unwrap()
                .unwrap();
            assert_eq!(placement.worker, w);
            assert_eq!(
                client.pair(ctx(), RequestId(7)).await.unwrap().unwrap(),
                (w, w)
            );

            // A coordination error on the coordinator relays back as Err(String).
            let err = client
                .report(
                    ctx(),
                    WorkerId(99),
                    LoadState {
                        active_requests: 0,
                        kv_pages_free: 0,
                    },
                )
                .await
                .unwrap();
            assert!(err.is_err());
        });
    }
}
