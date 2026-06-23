//! Control-plane **topology** axis: `pie serve` coordination flags + the
//! resolved [`TopologyMode`].
//!
//! Kept orthogonal to the *backend* axis (which driver) and the *role-stage*
//! wiring. Two forms, selected by flags (manager-ratified naming):
//!
//! - `--single-node` — no controller; this one node serves every role. The
//!   default when no `--controller` is given.
//! - `--role=<prefill|decode|encode> --controller=<addr>` — join a distributed
//!   cluster coordinated by a standalone controller process.
//!
//! This module owns only flag parsing + validation → [`TopologyMode`]. Building
//! the actual coordinator (stub vs dial) from a `TopologyMode` is the caller's
//! job in the engine boot path.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, ValueEnum};
use pie_control::ControlClient;
use pie_schema::DriverCapabilities;
use pie_schema::control::{Role, WorkerId, WorkerInfo};
use tarpc::serde_transport::{tcp, unix};
use tarpc::tokio_serde::formats::Bincode;

/// clap-parsable mirror of [`pie_schema::control::Role`] for `--role`.
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
    /// No controller; this node serves all roles. The default topology when no
    /// `--controller` is given.
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
    /// No controller; a single node serves all roles.
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
        // `--single-node` or no topology flags → no controller, all roles.
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
/// Held for the engine's lifetime: it keeps the dialed control connection
/// (distributed) alive, and carries the controller-assigned [`WorkerId`] + role
/// that later boot steps gate on. Single-node has no controller and never
/// registers.
pub struct Coordinator {
    /// Drop order: the client (and its dispatch task) before the runtime driving
    /// it. `None` in single-node, which has no controller.
    client: Option<ControlClient>,
    /// Runtime that owns/drives the tarpc client dispatch in distributed mode.
    _rt: Option<tokio::runtime::Runtime>,
    /// This worker's role. `None` in single-node, where the node serves all
    /// stages.
    pub role: Option<Role>,
    /// Address advertised to peers through the control-plane registry.
    control_addr: String,
    /// Controller-minted worker id. `None` until post-driver-boot registration;
    /// single-node leaves it unset.
    worker_id: Option<WorkerId>,
}

impl Coordinator {
    /// Register with the controller after the driver ready handshake provides
    /// real capabilities. Single-node has no controller, so this is a no-op.
    pub fn register_worker(&mut self, model: String, capability: DriverCapabilities) -> Result<()> {
        let Some(client) = self.client.clone() else {
            return Ok(());
        };
        let role = self.role.context("distributed coordinator missing role")?;
        let info = WorkerInfo {
            role,
            model,
            addr: self.control_addr.clone(),
            capability,
        };
        let rpc = async move {
            client
                .register_worker(tarpc::context::current(), info)
                .await
        };
        let worker_id = if tokio::runtime::Handle::try_current().is_ok() {
            let rt = self
                ._rt
                .as_ref()
                .context("distributed coordinator missing control runtime")?;
            std::thread::scope(|scope| {
                scope
                    .spawn(move || rt.block_on(rpc))
                    .join()
                    .map_err(|_| anyhow!("controller register_worker task panicked"))
            })?
        } else {
            self._rt
                .as_ref()
                .context("distributed coordinator missing control runtime")?
                .block_on(rpc)
        }
        .map_err(|e| anyhow!("controller register_worker transport: {e}"))?;
        self.worker_id = Some(worker_id);
        Ok(())
    }

    /// A clone of the control-plane client for async control loops, or `None` in
    /// single-node (no controller to report to).
    pub fn control_client(&self) -> Option<ControlClient> {
        self.client.clone()
    }

    /// Controller-minted worker id. `None` before registration and in
    /// single-node.
    pub fn worker_id(&self) -> Option<WorkerId> {
        self.worker_id
    }
}

fn dial_controller(rt: &tokio::runtime::Runtime, addr: String) -> Result<ControlClient> {
    rt.block_on(async move {
        // Control messages are tiny → tarpc's default frame cap is fine.
        let cfg = tarpc::client::Config::default();
        let client = if let Some(path) = addr
            .strip_prefix("unix://")
            .or_else(|| addr.strip_prefix("unix:"))
        {
            let conn = unix::connect(path, Bincode::default)
                .await
                .context("dial controller")?;
            ControlClient::new(cfg, conn).spawn()
        } else {
            let tcp_addr = addr.strip_prefix("tcp://").unwrap_or(&addr);
            let conn = tcp::connect(tcp_addr, Bincode::default)
                .await
                .context("dial controller")?;
            ControlClient::new(cfg, conn).spawn()
        };
        Ok::<_, anyhow::Error>(client)
    })
}

/// Build the coordinator for `mode`; distributed mode only dials the controller.
/// Registration waits until driver capabilities are known.
///
/// - **single-node:** no controller and no worker id. The worker terminates
///   clients directly (see [`super::client_server`]); `control_addr` is kept only
///   for structural parity.
/// - **distributed:** dials the standalone controller process and records the
///   requested role; `control_addr` is where peers reach this worker's edge-rpc
///   endpoint once it registers post driver-boot.
pub fn connect(mode: &TopologyMode, control_addr: String) -> Result<Coordinator> {
    match mode {
        TopologyMode::SingleNode => Ok(Coordinator {
            client: None,
            _rt: None,
            role: None,
            control_addr,
            worker_id: None,
        }),
        TopologyMode::Distributed { role, controller } => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .context("build controller-dial runtime")?;
            let client = dial_controller(&rt, controller.clone())
                .map_err(|e| anyhow!("dialing controller at {controller}: {e}"))?;
            Ok(Coordinator {
                client: Some(client),
                _rt: Some(rt),
                role: Some(*role),
                control_addr,
                worker_id: None,
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
    fn connect_single_node_has_no_id() {
        let coord = connect(&TopologyMode::SingleNode, "127.0.0.1:9000".to_string()).unwrap();
        // single-node serves all roles → no specific role
        assert_eq!(coord.role, None);
        // single-node has no controller and does not register.
        assert_eq!(coord.worker_id(), None);
    }
}
