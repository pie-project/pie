//! The distributed [`RemoteController`] — a sync [`Controller`] backed by the
//! generated tarpc [`ControlApiClient`].
//!
//! The worker's boot path and the embedded [`crate::InProcController`] share one
//! sync [`Controller`] trait, so single-node and worker startup never have to
//! become async. This wrapper bridges that sync surface onto the async tarpc
//! client by owning a small tokio runtime and `block_on`-ing each call.
//!
//! The gateway, which already runs inside its own async runtime, should dial the
//! controller with [`ControlApiClient`] directly rather than through this sync
//! shim — calling `block_on` from within a runtime panics.

use crate::controller::Controller;
use crate::error::{ControllerError, Result};
use crate::service::{ControlApiClient, RpcResult};
use pie_schema::{LoadState, Placement, RequestId, RequestMeta, WorkerId, WorkerInfo};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;

fn transport(e: impl std::fmt::Display) -> ControllerError {
    ControllerError::Transport(e.to_string())
}

/// Collapse a tarpc call outcome onto the crate's error idiom: a transport/RPC
/// failure becomes [`ControllerError::Transport`]; a controller-side rejection
/// (the `Err(String)` payload) becomes [`ControllerError::Remote`].
fn call_result<T>(
    outcome: std::result::Result<RpcResult<T>, tarpc::client::RpcError>,
) -> Result<T> {
    match outcome {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(msg)) => Err(ControllerError::Remote(msg)),
        Err(e) => Err(transport(e)),
    }
}

/// Distributed [`Controller`] — dials a standalone controller process over tarpc
/// and frames each trait call as a [`crate::service::ControlApi`] request.
///
/// Owns a dedicated multi-thread runtime (one worker) that keeps the tarpc
/// client's background dispatch task driven between calls; the sync trait
/// methods `block_on` it.
#[derive(Debug)]
pub struct RemoteController {
    rt: tokio::runtime::Runtime,
    client: ControlApiClient,
}

impl RemoteController {
    /// Connect to the controller's control endpoint (`tcp://host:port`, a bare
    /// `host:port`, or `unix:/path`).
    pub fn connect(addr: String) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .map_err(transport)?;

        let client = rt.block_on(async move {
            // Control messages are tiny → tarpc's default frame cap is fine.
            let cfg = tarpc::client::Config::default();
            let client = if let Some(path) = addr
                .strip_prefix("unix://")
                .or_else(|| addr.strip_prefix("unix:"))
            {
                let conn = unix::connect(path, Bincode::default)
                    .await
                    .map_err(transport)?;
                ControlApiClient::new(cfg, conn).spawn()
            } else {
                let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(&addr);
                let conn = tcp::connect(tcp_addr, Bincode::default)
                    .await
                    .map_err(transport)?;
                ControlApiClient::new(cfg, conn).spawn()
            };
            Ok::<_, ControllerError>(client)
        })?;

        Ok(Self { rt, client })
    }
}

impl Controller for RemoteController {
    fn register(&self, worker: WorkerInfo) -> Result<WorkerId> {
        let fut = self.client.register(tarpc::context::current(), worker);
        call_result(self.rt.block_on(fut))
    }

    fn report(&self, worker: WorkerId, load: LoadState) -> Result<()> {
        let fut = self.client.report(tarpc::context::current(), worker, load);
        call_result(self.rt.block_on(fut))
    }

    fn route(&self, req: &RequestMeta) -> Result<Placement> {
        let fut = self.client.route(tarpc::context::current(), *req);
        call_result(self.rt.block_on(fut))
    }

    fn pair(&self, req: RequestId) -> Result<(WorkerId, WorkerId)> {
        let fut = self.client.pair(tarpc::context::current(), req);
        call_result(self.rt.block_on(fut))
    }

    fn resolve(&self, worker: WorkerId) -> Result<WorkerInfo> {
        let fut = self.client.resolve(tarpc::context::current(), worker);
        call_result(self.rt.block_on(fut))
    }
}
