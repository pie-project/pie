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

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use clap::{Args, ValueEnum};

use pie_controller::{
    Controller, ControllerConfig, InProcController, RemoteController, WorkerId, WorkerInfo,
};
use pie_schema::cluster::Role;

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

    /// Address of the standalone controller's control endpoint
    /// (e.g. `10.0.0.4:7000`). Requires `--role`.
    #[arg(long, requires = "role")]
    pub controller: Option<String>,
}

/// Resolved control-plane topology — the input to building the coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyMode {
    /// In-proc controller; a single node serves all roles.
    SingleNode,
    /// Joins a distributed cluster via a standalone controller process.
    Distributed { role: Role, controller: SocketAddr },
}

/// Validate the flag combination and resolve it to a [`TopologyMode`].
///
/// clap's `requires`/`conflicts_with` already reject bad combinations at parse
/// time; the guards here keep the resolver correct for the library-call path
/// (e.g. the pyo3 wheel) that constructs [`CoordinationArgs`] directly.
pub fn resolve(args: &CoordinationArgs) -> Result<TopologyMode> {
    match (&args.role, &args.controller) {
        (Some(role), Some(addr)) => {
            let controller = addr
                .parse::<SocketAddr>()
                .map_err(|e| anyhow!("--controller {addr:?}: {e}"))?;
            Ok(TopologyMode::Distributed {
                role: (*role).into(),
                controller,
            })
        }
        // `--single-node` or no topology flags → embed in-proc, all roles.
        (None, None) => Ok(TopologyMode::SingleNode),
        (Some(_), None) => bail!("--role requires --controller"),
        (None, Some(_)) => bail!("--controller requires --role"),
    }
}

/// A live handle to the cluster controller plus this worker's identity.
///
/// Held for the engine's lifetime: it keeps the in-proc controller (single-node)
/// or the dialed control connection (distributed) alive, and carries the
/// controller-assigned [`WorkerId`] + role that later boot steps gate on.
pub struct Coordinator {
    /// The control-plane seam — embedded in-proc (single-node) or a remote dial
    /// (distributed). Same trait either way.
    pub controller: Arc<dyn Controller>,
    /// This worker's controller-assigned id.
    pub worker_id: WorkerId,
    /// This worker's role. `None` in single-node, where the node serves all
    /// stages.
    pub role: Option<Role>,
}

/// Build the coordinator for `mode` and register this worker with the
/// controller. `control_addr` is where peers reach this worker's control
/// endpoint (used by the controller for pairing in the distributed case).
///
/// - **single-node:** constructs an in-proc [`InProcController`] and self-
///   registers — no network.
/// - **distributed:** dials the standalone controller process and registers with
///   the requested role.
pub fn connect(mode: &TopologyMode, control_addr: String) -> Result<Coordinator> {
    let (controller, role): (Arc<dyn Controller>, Option<Role>) = match mode {
        TopologyMode::SingleNode => (
            Arc::new(InProcController::new(ControllerConfig::default())),
            None,
        ),
        TopologyMode::Distributed { role, controller } => {
            let remote = RemoteController::connect(controller)
                .map_err(|e| anyhow!("dialing controller at {controller}: {e}"))?;
            (Arc::new(remote), Some(*role))
        }
    };

    let worker_id = controller
        .register(WorkerInfo {
            control_addr,
            preferred_role: role,
        })
        .map_err(|e| anyhow!("registering with controller: {e}"))?;

    Ok(Coordinator {
        controller,
        worker_id,
        role,
    })
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
                assert_eq!(controller.port(), 7000);
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
    fn connect_single_node_registers_in_proc() {
        let coord = connect(&TopologyMode::SingleNode, "127.0.0.1:9000".to_string()).unwrap();
        // single-node serves all roles → no specific role
        assert_eq!(coord.role, None);
        // first worker minted by the embedded controller
        assert_eq!(coord.worker_id, pie_controller::WorkerId(0));
    }
}
