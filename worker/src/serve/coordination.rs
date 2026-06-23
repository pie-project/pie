//! Control-plane **topology** axis: `pie serve` coordination flags + the
//! resolved [`TopologyMode`].
//!
//! Kept orthogonal to the *backend* axis (which driver) and the *role-stage*
//! wiring. Two forms, selected by flags (manager-ratified naming):
//!
//! - `--single-node` — embed the controller in-proc; this one node serves every
//!   role. The default when no `--controller` is given.
//! - `--role=<prefill|decode|encode> --controller=<addr>` — join a distributed
//!   cluster coordinated by a standalone controller process.
//!
//! This module owns only flag parsing + validation → [`TopologyMode`]. Building
//! the actual coordinator (embed vs dial) from a `TopologyMode` is the caller's
//! job in the engine boot path.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, ValueEnum};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;

use crate::rpc::control_api::ControlApiClient;
use pie_schema::cluster::Role;
use pie_schema::{WorkerId, WorkerInfo};

/// clap-parsable mirror of [`pie_schema::cluster::Role`] for `--role`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum RoleArg {
    Prefill,
    Decode,
    Encode,
}

impl From<RoleArg> for Role {
    fn from(r: RoleArg) -> Self {
        match r {
            RoleArg::Prefill => Role::Prefill,
            RoleArg::Decode => Role::Decode,
            RoleArg::Encode => Role::Encode,
        }
    }
}

/// Topology-selection flags, flattened into `pie serve`'s args.
#[derive(Args, Debug, Default)]
pub struct CoordinationArgs {
    /// Embed the controller in-proc; this node serves all roles. The default
    /// topology when no `--controller` is given.
    #[arg(long, conflicts_with_all = ["role", "controller"])]
    pub single_node: bool,

    /// This worker's role in a distributed cluster. Requires `--controller`.
    #[arg(long, value_enum, requires = "controller")]
    pub role: Option<RoleArg>,

    /// Controller's control endpoint: `tcp://host:port`, a bare `host:port`, or
    /// `unix:/path/to.sock`. Requires `--role`.
    #[arg(long, requires = "role")]
    pub controller: Option<String>,
}

/// Resolved control-plane topology — the input to building the coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyMode {
    /// In-proc controller; a single node serves all roles.
    SingleNode,
    /// Joins a distributed cluster via a standalone controller process. The
    /// address is `tcp://host:port`, a bare `host:port`, or `unix:/path`.
    Distributed { role: Role, controller: String },
}

/// Validate the flag combination and resolve it to a [`TopologyMode`].
///
/// clap's `requires`/`conflicts_with` already reject bad combinations at parse
/// time; the guards here keep the resolver correct for the library-call path
/// (e.g. the pyo3 wheel) that constructs [`CoordinationArgs`] directly.
pub fn resolve(args: &CoordinationArgs) -> Result<TopologyMode> {
    match (&args.role, &args.controller) {
        (Some(role), Some(addr)) => {
            // Light validation; the real dial error surfaces at connect time.
            if !is_valid_addr(addr) {
                bail!("--controller {addr:?}: expected host:port, tcp://host:port, or unix:/path");
            }
            Ok(TopologyMode::Distributed {
                role: (*role).into(),
                controller: addr.clone(),
            })
        }
        // `--single-node` or no topology flags → embed in-proc, all roles.
        (None, None) => Ok(TopologyMode::SingleNode),
        (Some(_), None) => bail!("--role requires --controller"),
        (None, Some(_)) => bail!("--controller requires --role"),
    }
}

/// True for `unix:`-scheme addresses or anything carrying a `host:port`.
fn is_valid_addr(addr: &str) -> bool {
    addr.starts_with("unix:") || addr.strip_prefix("tcp://").unwrap_or(addr).contains(':')
}

/// Build a control address from a worker's `host`/`port` config, honoring a
/// `unix:`/`tcp://` scheme already present in `host` (so `host = "unix:/path"`
/// selects a UDS edge).
pub fn addr_from_host_port(host: &str, port: u16) -> String {
    if host.starts_with("unix:") || host.starts_with("tcp://") {
        host.to_string()
    } else {
        format!("{host}:{port}")
    }
}

/// A live handle to the cluster controller plus this worker's identity.
///
/// Held for the engine's lifetime: it keeps the in-proc controller (single-node)
/// or the dialed control connection (distributed) alive, and carries the
/// controller-assigned [`WorkerId`] + role that later boot steps gate on.
pub struct Coordinator {
    /// `Some` only in distributed mode — the live control-plane dial, kept alive
    /// for the engine's lifetime. `None` in single-node, which has no controller.
    controller: Option<DistController>,
    /// This worker's id: minted by the controller in distributed mode, or the
    /// trivial `WorkerId(0)` in single-node.
    pub worker_id: WorkerId,
    /// This worker's role. `None` in single-node, where the node serves all
    /// stages.
    pub role: Option<Role>,
}

impl Coordinator {
    /// A clone of the control-plane client for the async heartbeat loop, or
    /// `None` in single-node (no controller to report to).
    pub fn control_client(&self) -> Option<ControlApiClient> {
        self.controller.as_ref().map(|c| c.client.clone())
    }
}

/// Distributed-mode control-plane dial. Owns a small runtime that drives the
/// tarpc [`ControlApiClient`]'s background dispatch; the `register` handshake
/// runs on it synchronously at connect time (callers are in a sync context).
/// The client is cloned out for the engine's async heartbeat loop. Dropping this
/// closes the control connection.
struct DistController {
    // Drop order: the client (and its dispatch task) before the runtime driving it.
    client: ControlApiClient,
    _rt: tokio::runtime::Runtime,
}

impl DistController {
    fn connect(addr: String) -> Result<Self> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .context("build controller-dial runtime")?;
        let client = rt.block_on(async move {
            // Control messages are tiny → tarpc's default frame cap is fine.
            let cfg = tarpc::client::Config::default();
            let client = if let Some(path) = addr
                .strip_prefix("unix://")
                .or_else(|| addr.strip_prefix("unix:"))
            {
                let conn = unix::connect(path, Bincode::default)
                    .await
                    .context("dial controller")?;
                ControlApiClient::new(cfg, conn).spawn()
            } else {
                let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(&addr);
                let conn = tcp::connect(tcp_addr, Bincode::default)
                    .await
                    .context("dial controller")?;
                ControlApiClient::new(cfg, conn).spawn()
            };
            Ok::<_, anyhow::Error>(client)
        })?;
        Ok(Self { client, _rt: rt })
    }

    fn register(&self, info: WorkerInfo) -> Result<WorkerId> {
        self._rt
            .block_on(self.client.register(tarpc::context::current(), info))
            .map_err(|e| anyhow!("controller register transport: {e}"))?
            .map_err(|msg| anyhow!("controller rejected register: {msg}"))
    }
}

/// Build the coordinator for `mode` and (in distributed mode) register this
/// worker with the controller.
///
/// - **single-node:** no controller — self-assigns `WorkerId(0)` and serves all
///   roles. The worker terminates clients directly (see
///   [`super::client_server`]); `control_addr` is unused.
/// - **distributed:** dials the standalone controller process and registers with
///   the requested role; `control_addr` is where the gateway reaches this
///   worker's edge-rpc endpoint.
pub fn connect(mode: &TopologyMode, control_addr: String) -> Result<Coordinator> {
    match mode {
        TopologyMode::SingleNode => Ok(Coordinator {
            controller: None,
            worker_id: WorkerId(0),
            role: None,
        }),
        TopologyMode::Distributed { role, controller } => {
            let dist = DistController::connect(controller.clone())
                .map_err(|e| anyhow!("dialing controller at {controller}: {e}"))?;
            let worker_id = dist
                .register(WorkerInfo {
                    control_addr,
                    preferred_role: Some(*role),
                })
                .map_err(|e| anyhow!("registering with controller: {e}"))?;
            Ok(Coordinator {
                controller: Some(dist),
                worker_id,
                role: Some(*role),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_single_node() {
        assert_eq!(
            resolve(&CoordinationArgs::default()).unwrap(),
            TopologyMode::SingleNode
        );
    }

    #[test]
    fn distributed_parses_addr_and_role() {
        let args = CoordinationArgs {
            single_node: false,
            role: Some(RoleArg::Decode),
            controller: Some("127.0.0.1:7000".to_string()),
        };
        match resolve(&args).unwrap() {
            TopologyMode::Distributed { role, controller } => {
                assert_eq!(role, Role::Decode);
                assert_eq!(controller, "127.0.0.1:7000");
            }
            other => panic!("expected Distributed, got {other:?}"),
        }
    }

    #[test]
    fn bad_controller_addr_errors() {
        let args = CoordinationArgs {
            single_node: false,
            role: Some(RoleArg::Prefill),
            controller: Some("not-an-addr".to_string()),
        };
        assert!(resolve(&args).is_err());
    }

    #[test]
    fn role_without_controller_errors() {
        let args = CoordinationArgs {
            single_node: false,
            role: Some(RoleArg::Prefill),
            controller: None,
        };
        assert!(resolve(&args).is_err());
    }

    #[test]
    fn connect_single_node_assigns_id_zero() {
        let coord = connect(&TopologyMode::SingleNode, "127.0.0.1:9000".to_string()).unwrap();
        // single-node serves all roles → no specific role
        assert_eq!(coord.role, None);
        // single-node has no controller; it self-assigns the trivial id
        assert_eq!(coord.worker_id, WorkerId(0));
    }
}
