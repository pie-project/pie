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

use anyhow::{Result, bail};
use clap::{Args, ValueEnum};
use pie_schema::control::Role;

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

    /// Gateway endpoint(s) this worker dials INTO (post-inversion data plane,
    /// M3): `tcp://host:port`, a bare `host:port`, or `unix:/path`. Repeat or
    /// comma-separate for full-mesh fan-in. Requires `--controller`.
    /// (Deploy-config discovery; controller-pushed discovery is a graduation.)
    #[arg(long, requires = "controller", value_delimiter = ',')]
    pub gateway: Vec<String>,
}

/// Resolved control-plane topology — the input to building the coordinator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyMode {
    /// No controller; a single node serves all roles.
    SingleNode,
    /// Joins a distributed cluster via a standalone controller process. The
    /// address is `tcp://host:port`, a bare `host:port`, or `unix:/path`.
    Distributed {
        role: Role,
        controller: String,
        /// Gateway endpoint(s) to dial INTO — the worker is the client, the
        /// gateway the listening server (M3 inversion). Deploy-config.
        gateways: Vec<String>,
    },
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
            if args.gateway.is_empty() {
                bail!("distributed mode requires at least one --gateway to dial into");
            }
            for gw in &args.gateway {
                if !is_valid_addr(gw) {
                    bail!("--gateway {gw:?}: expected host:port, tcp://host:port, or unix:/path");
                }
            }
            Ok(TopologyMode::Distributed {
                role: (*role).into(),
                controller: addr.clone(),
                gateways: args.gateway.clone(),
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

/// Resolved control-plane topology plus the worker's advertised edge address.
///
/// Carried from flag parsing ([`connect`]) into the engine boot path
/// ([`super::start_engine`]), which builds the actual control connection on the
/// engine's async runtime — dialing the controller for distributed mode, or
/// embedding it in-proc for single-node (`single-node` feature). Keeping the
/// connection out of here means no second runtime and no pre-runtime dialing.
#[derive(Debug, Clone)]
pub struct Coordinator {
    /// The resolved topology (single-node vs distributed + role/controller).
    pub mode: TopologyMode,
    /// Address advertised to peers / the gateway as this worker's edge-rpc
    /// endpoint in distributed mode. Unused by the single-node embed, which
    /// binds an ephemeral loopback port and registers that instead.
    pub control_addr: String,
}

impl Coordinator {
    /// This worker's role, or `None` in single-node (serves all stages).
    pub fn role(&self) -> Option<Role> {
        match &self.mode {
            TopologyMode::SingleNode => None,
            TopologyMode::Distributed { role, .. } => Some(*role),
        }
    }

    /// The controller endpoint to dial in distributed mode; `None` in
    /// single-node (embedded in-proc).
    pub fn controller_addr(&self) -> Option<&str> {
        match &self.mode {
            TopologyMode::SingleNode => None,
            TopologyMode::Distributed { controller, .. } => Some(controller),
        }
    }
}

/// Resolve `mode` into a [`Coordinator`]. The control connection itself is built
/// later, on the engine runtime, by [`super::start_engine`] (dial or embed) —
/// this only carries the resolved topology + advertised address.
pub fn connect(mode: &TopologyMode, control_addr: String) -> Result<Coordinator> {
    Ok(Coordinator {
        mode: mode.clone(),
        control_addr,
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
            gateway: vec!["127.0.0.1:8000".to_string()],
        };
        match resolve(&args).unwrap() {
            TopologyMode::Distributed {
                role,
                controller,
                gateways,
            } => {
                assert_eq!(role, Role::Decode);
                assert_eq!(controller, "127.0.0.1:7000");
                assert_eq!(gateways, vec!["127.0.0.1:8000".to_string()]);
            }
            other => panic!("expected Distributed, got {other:?}"),
        }
    }

    #[test]
    fn distributed_without_gateway_errors() {
        let args = CoordinationArgs {
            single_node: false,
            role: Some(RoleArg::Decode),
            controller: Some("127.0.0.1:7000".to_string()),
            gateway: vec![],
        };
        assert!(resolve(&args).is_err());
    }

    #[test]
    fn bad_controller_addr_errors() {
        let args = CoordinationArgs {
            single_node: false,
            role: Some(RoleArg::Prefill),
            controller: Some("not-an-addr".to_string()),
            gateway: vec!["127.0.0.1:8000".to_string()],
        };
        assert!(resolve(&args).is_err());
    }

    #[test]
    fn role_without_controller_errors() {
        let args = CoordinationArgs {
            single_node: false,
            role: Some(RoleArg::Prefill),
            controller: None,
            gateway: vec![],
        };
        assert!(resolve(&args).is_err());
    }

    #[test]
    fn connect_single_node_has_no_role() {
        let coord = connect(&TopologyMode::SingleNode, "127.0.0.1:9000".to_string()).unwrap();
        // single-node serves all roles → no specific role, no controller to dial.
        assert_eq!(coord.role(), None);
        assert_eq!(coord.controller_addr(), None);
    }
}
