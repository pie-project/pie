//! Control-plane **topology** axis: the resolved [`TopologyMode`] + the
//! [`Coordinator`] the engine boot path consumes.
//!
//! Kept orthogonal to the *backend* axis (which driver) and the *role-stage*
//! wiring. Two forms:
//!
//! - **single-node** — no controller; this one node serves every role and
//!   terminates clients directly (gateway-free local inference).
//! - **distributed** — `role` + `controller` + the `gateways` this worker dials
//!   INTO (M3 inversion), joining a cluster coordinated by a standalone
//!   controller process.
//!
//! This is a pure library surface: flag parsing lives in the bins (`bin/worker`,
//! `bin/pie`), which construct a [`TopologyMode`] (via [`TopologyMode::distributed`]
//! or [`TopologyMode::SingleNode`]) and hand it to [`connect`]. Building the
//! actual control connection (dial vs in-proc embed) is the engine boot path's
//! job ([`super::start_engine`]).

use anyhow::{Result, bail};
use pie_schema::control::Role;

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

impl TopologyMode {
    /// Build a validated distributed topology. A worker needs a reachable
    /// controller address and at least one gateway to dial into; the real dial
    /// error surfaces later at connect time, this just rejects obvious garbage.
    pub fn distributed(role: Role, controller: String, gateways: Vec<String>) -> Result<Self> {
        if !is_valid_addr(&controller) {
            bail!("controller {controller:?}: expected host:port, tcp://host:port, or unix:/path");
        }
        if gateways.is_empty() {
            bail!("distributed mode requires at least one gateway to dial into");
        }
        for gw in &gateways {
            if !is_valid_addr(gw) {
                bail!("gateway {gw:?}: expected host:port, tcp://host:port, or unix:/path");
            }
        }
        Ok(TopologyMode::Distributed {
            role,
            controller,
            gateways,
        })
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

/// Resolved control-plane topology plus the worker's `host:port` identity.
///
/// Carried into the engine boot path ([`super::start_engine`]), which builds the
/// actual control connection on the engine's async runtime — dialing the
/// controller for distributed mode, or (for an in-proc embedder) taking an
/// injected control link. Keeping the connection out of here means no
/// pre-runtime dialing.
#[derive(Debug, Clone)]
pub struct Coordinator {
    /// The resolved topology (single-node vs distributed + role/controller).
    pub mode: TopologyMode,
    /// The worker's `host:port`, registered as `WorkerInfo.addr`. Post-inversion
    /// this is vestigial for dispatch — the gateway routes via its dial-in
    /// registry (keyed by `WorkerId`), not by dialing this address; it stays a
    /// stable identity/display value.
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
    /// single-node.
    pub fn controller_addr(&self) -> Option<&str> {
        match &self.mode {
            TopologyMode::SingleNode => None,
            TopologyMode::Distributed { controller, .. } => Some(controller),
        }
    }
}

/// Resolve `mode` into a [`Coordinator`]. The control connection itself is built
/// later, on the engine runtime, by [`super::start_engine`] — this only carries
/// the resolved topology + advertised address.
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
    fn distributed_builds_and_validates() {
        let mode = TopologyMode::distributed(
            Role::Decode,
            "127.0.0.1:7000".to_string(),
            vec!["127.0.0.1:8000".to_string()],
        )
        .unwrap();
        match mode {
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
        assert!(
            TopologyMode::distributed(Role::Decode, "127.0.0.1:7000".to_string(), vec![]).is_err()
        );
    }

    #[test]
    fn distributed_bad_addr_errors() {
        assert!(
            TopologyMode::distributed(
                Role::Prefill,
                "not-an-addr".to_string(),
                vec!["127.0.0.1:8000".to_string()],
            )
            .is_err()
        );
    }

    #[test]
    fn connect_single_node_has_no_role() {
        let coord = connect(&TopologyMode::SingleNode, "127.0.0.1:9000".to_string()).unwrap();
        assert_eq!(coord.role(), None);
        assert_eq!(coord.controller_addr(), None);
    }
}
