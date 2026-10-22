use anyhow::{Result, bail};
use controller_api::Role;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyMode {
    SingleNode,
    Distributed {
        role: Role,
        controller: String,
        gateways: Vec<String>,
    },
}

impl TopologyMode {
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

fn is_valid_addr(addr: &str) -> bool {
    addr.starts_with("unix:") || addr.strip_prefix("tcp://").unwrap_or(addr).contains(':')
}

#[derive(Debug, Clone)]
pub struct Coordinator {
    pub mode: TopologyMode,
    pub control_addr: String,
}

impl Coordinator {
    pub fn role(&self) -> Option<Role> {
        match &self.mode {
            TopologyMode::SingleNode => None,
            TopologyMode::Distributed { role, .. } => Some(*role),
        }
    }

    pub fn controller_addr(&self) -> Option<&str> {
        match &self.mode {
            TopologyMode::SingleNode => None,
            TopologyMode::Distributed { controller, .. } => Some(controller),
        }
    }
}

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
