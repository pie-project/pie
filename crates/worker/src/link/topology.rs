//! Control-plane topology axis: the resolved [`TopologyMode`] + the [`Coordinator`] the runtime boot path consumes. Two forms: single-node (no controller, one node serves every role) and distributed (`role` + `controller` + the `gateways` this worker dials into, joining a cluster coordinated by a standalone controller process).
//! Pure library surface: flag parsing lives in the bins, which build a [`TopologyMode`] and hand it to [`connect`]. Building the actual control connection is the runtime boot path's job ([`crate::serve::start_runtime`]).

use anyhow::{Result, bail};
use controller_api::Role;

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
        /// Gateway endpoint(s) to dial into; the worker is the client, the gateway the listening server. Optional: a static pin for fixed/local topologies. Empty means dynamic discovery from the controller; non-empty addresses stay dialed alongside anything the controller pushes.
        gateways: Vec<String>,
    },
}

impl TopologyMode {
    /// Build a validated distributed topology. Gateways are optional (empty means dynamic discovery); supplied addresses are validated and become the pinned dial-in set.
    pub fn distributed(role: Role, controller: String, gateways: Vec<String>) -> Result<Self> {
        if !is_valid_addr(&controller) {
            bail!("controller {controller:?}: expected host:port, tcp://host:port, or unix:/path");
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

/// Resolved control-plane topology plus the worker's `host:port` identity. Keeping the connection out of here means no pre-runtime dialing.
#[derive(Debug, Clone)]
pub struct Coordinator {
    pub mode: TopologyMode,
    /// The worker's `host:port`, registered as `WorkerInfo.addr`. Vestigial for dispatch (the gateway routes via its own dial-in registry); stays a stable identity/display value.
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
/// later, on the server's async runtime, by [`crate::serve::start_runtime`] — this
/// only carries the resolved topology + advertised address.
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
}
