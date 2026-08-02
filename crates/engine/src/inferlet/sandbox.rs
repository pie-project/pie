//! Per-instance security policies attached to every WASM component.
//!
//! Today's two policies — filesystem and network — are the user-facing
//! capabilities exposed by `[runtime]` config. Both are compiled once at
//! bootstrap time and shared by reference into every instance.

use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use ipnet::IpNet;

/// Security policies applied to each spawned inferlet.
#[derive(Clone)]
pub struct InstancePolicy {
    pub(crate) fs: FsPolicy,
    pub(crate) network: NetworkPolicy,
}

impl InstancePolicy {
    /// Maximally restrictive policy for components instantiated for inspection.
    pub(crate) fn deny_all() -> Self {
        Self {
            fs: FsPolicy {
                allow: false,
                base_dir: PathBuf::new(),
            },
            network: NetworkPolicy::parse(false, &[]).expect("deny-all parse"),
        }
    }
}

// =============================================================================
// Filesystem
// =============================================================================

/// Filesystem capabilities granted to each inferlet.
///
/// Today this is "all-or-nothing scratch": when `allow` is true a
/// per-process directory is mounted at `/scratch` with full RW; when
/// false, no `wasi:filesystem` access at all. The base directory is
/// always supplied by the caller — Python is the source of truth.
#[derive(Debug, Clone)]
pub struct FsPolicy {
    pub allow: bool,
    /// Base directory under which per-process scratch dirs are created.
    /// Each instance gets `<base_dir>/<process_id>`.
    pub base_dir: PathBuf,
}

// =============================================================================
// Network
// =============================================================================

/// Network capabilities granted to each inferlet.
///
/// `allow == false` denies all socket operations. `allow == true` with
/// `unrestricted == true` matches the legacy `inherit_network()`
/// behavior. Otherwise each connect/bind goes through `check`.
#[derive(Debug, Clone)]
pub struct NetworkPolicy {
    pub allow: bool,
    unrestricted: bool,
    rules: Arc<[Rule]>,
}

#[derive(Debug, Clone)]
struct Rule {
    cidr: IpNet,
    port: PortFilter,
}

#[derive(Debug, Clone, Copy)]
enum PortFilter {
    Any,
    Single(u16),
    Range(u16, u16),
}

impl PortFilter {
    fn matches(&self, port: u16) -> bool {
        match *self {
            PortFilter::Any => true,
            PortFilter::Single(p) => port == p,
            PortFilter::Range(lo, hi) => port >= lo && port <= hi,
        }
    }
}

impl NetworkPolicy {
    /// Build a policy from `allow_network` + the raw `network_allowed_hosts`
    /// list. Each entry is `cidr` or `cidr:port` or `cidr:lo-hi`. The
    /// special entry `"*"` is allowed only as a sole element and means
    /// "no restriction".
    pub fn parse(allow: bool, items: &[String]) -> Result<Self> {
        if !allow {
            // Network is blocked entirely; the rule list is unused. Don't
            // require it to parse — but reject contradictions.
            if items.iter().any(|s| s != "*") && !items.is_empty() {
                tracing::warn!(
                    "[runtime] allow_network = false but network_allowed_hosts \
                     is non-empty; the allowlist is ignored."
                );
            }
            return Ok(NetworkPolicy {
                allow: false,
                unrestricted: false,
                rules: Arc::new([]),
            });
        }

        // Empty list is equivalent to allow_network = false.
        if items.is_empty() {
            return Ok(NetworkPolicy {
                allow: true,
                unrestricted: false,
                rules: Arc::new([]),
            });
        }

        // The wildcard must be alone; mixing it with rules is almost
        // certainly a config bug, not something to silently flatten.
        if items.iter().any(|s| s == "*") {
            if items.len() != 1 {
                return Err(anyhow!(
                    "network_allowed_hosts: \"*\" must be the only entry, \
                     got {} entries",
                    items.len()
                ));
            }
            return Ok(NetworkPolicy {
                allow: true,
                unrestricted: true,
                rules: Arc::new([]),
            });
        }

        let rules: Result<Vec<Rule>> = items.iter().map(|s| parse_rule(s)).collect();
        Ok(NetworkPolicy {
            allow: true,
            unrestricted: false,
            rules: rules?.into(),
        })
    }

    /// True if the policy is `allow_network = true` + `["*"]` — the legacy
    /// `inherit_network()` behavior, no per-address check needed.
    pub fn is_unrestricted(&self) -> bool {
        self.allow && self.unrestricted
    }

    /// True if the address satisfies any rule. If the policy is
    /// unrestricted, always returns true; if `allow == false`, always
    /// returns false. Otherwise checks each rule in order.
    pub fn check(&self, addr: &SocketAddr) -> bool {
        if !self.allow {
            return false;
        }
        if self.unrestricted {
            return true;
        }
        let ip = addr.ip();
        let port = addr.port();
        self.rules
            .iter()
            .any(|r| ip_in_cidr(ip, &r.cidr) && r.port.matches(port))
    }
}

/// `cidr` or `cidr:port` or `cidr:lo-hi`. A bare IP without `/<n>` is
/// accepted as a host route (`/32` for v4, `/128` for v6). Bracketed
/// IPv6 (`[::1]:443`) is also accepted; for unbracketed IPv6 use the
/// no-port form (`::1/128`).
fn parse_rule(spec: &str) -> Result<Rule> {
    // Bracketed IPv6: [addr] or [addr/cidr] or [addr]:port etc.
    if let Some(rest) = spec.strip_prefix('[') {
        let (inside, after) = rest
            .split_once(']')
            .ok_or_else(|| anyhow!("network_allowed_hosts: missing ']' in {spec:?}"))?;
        let cidr = parse_cidr(inside)
            .map_err(|e| anyhow!("network_allowed_hosts: bad CIDR/IP in {spec:?}: {e}"))?;
        let port = if after.is_empty() {
            PortFilter::Any
        } else {
            let port_str = after.strip_prefix(':').ok_or_else(|| {
                anyhow!("network_allowed_hosts: expected ':' after ']' in {spec:?}")
            })?;
            parse_port_filter(port_str)
                .map_err(|e| anyhow!("network_allowed_hosts: bad port in {spec:?}: {e}"))?
        };
        return Ok(Rule { cidr, port });
    }

    // Unbracketed: try the last-`:` split first. The LHS must parse as
    // a CIDR/IP; if it does, the RHS is unconditionally treated as a
    // port (so a typo there gives "bad port", not a confusing "bad
    // CIDR" from re-parsing the whole spec). If the LHS doesn't parse,
    // fall back to the whole string as a CIDR — handles IPv6 forms
    // like `::1/128` or `2001:db8::/32` which contain colons.
    if let Some((host, port_str)) = spec.rsplit_once(':') {
        if let Ok(cidr) = parse_cidr(host) {
            let port = parse_port_filter(port_str)
                .map_err(|e| anyhow!("network_allowed_hosts: bad port in {spec:?}: {e}"))?;
            return Ok(Rule { cidr, port });
        }
    }
    let cidr = parse_cidr(spec)
        .map_err(|e| anyhow!("network_allowed_hosts: bad CIDR/IP in {spec:?}: {e}"))?;
    Ok(Rule {
        cidr,
        port: PortFilter::Any,
    })
}

fn parse_cidr(s: &str) -> Result<IpNet> {
    if let Ok(net) = s.parse::<IpNet>() {
        return Ok(net);
    }
    // Bare IP: promote to host route.
    let ip: IpAddr = s.parse().map_err(|e| anyhow!("{e}"))?;
    Ok(match ip {
        IpAddr::V4(v4) => IpNet::V4(ipnet::Ipv4Net::new(v4, 32).unwrap()),
        IpAddr::V6(v6) => IpNet::V6(ipnet::Ipv6Net::new(v6, 128).unwrap()),
    })
}

fn parse_port_filter(s: &str) -> Result<PortFilter> {
    if let Some((lo, hi)) = s.split_once('-') {
        let lo: u16 = lo
            .parse()
            .map_err(|_| anyhow!("low port not u16: {lo:?}"))?;
        let hi: u16 = hi
            .parse()
            .map_err(|_| anyhow!("high port not u16: {hi:?}"))?;
        if lo > hi {
            return Err(anyhow!("port range {lo}-{hi} is reversed"));
        }
        Ok(PortFilter::Range(lo, hi))
    } else {
        let p: u16 = s.parse().map_err(|_| anyhow!("port not u16: {s:?}"))?;
        Ok(PortFilter::Single(p))
    }
}

fn ip_in_cidr(ip: IpAddr, net: &IpNet) -> bool {
    match (ip, net) {
        (IpAddr::V4(a), IpNet::V4(n)) => n.contains(&a),
        (IpAddr::V6(a), IpNet::V6(n)) => n.contains(&a),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sa(s: &str) -> SocketAddr {
        s.parse().unwrap()
    }

    #[test]
    fn deny_all_when_disabled() {
        let p = NetworkPolicy::parse(false, &[]).unwrap();
        assert!(!p.allow);
        assert!(!p.check(&sa("10.0.0.1:443")));
        assert!(!p.check(&sa("127.0.0.1:80")));
    }

    #[test]
    fn empty_allowlist_blocks_when_enabled() {
        let p = NetworkPolicy::parse(true, &[]).unwrap();
        assert!(p.allow);
        assert!(!p.check(&sa("10.0.0.1:443")));
    }

    #[test]
    fn star_means_unrestricted() {
        let p = NetworkPolicy::parse(true, &["*".into()]).unwrap();
        assert!(p.is_unrestricted());
        assert!(p.check(&sa("8.8.8.8:53")));
    }

    #[test]
    fn star_must_be_alone() {
        let err = NetworkPolicy::parse(true, &["*".into(), "10.0.0.0/8".into()]).unwrap_err();
        assert!(err.to_string().contains("must be the only entry"));
    }

    #[test]
    fn cidr_match() {
        let p = NetworkPolicy::parse(true, &["10.0.0.0/8".into()]).unwrap();
        assert!(p.check(&sa("10.0.0.1:443")));
        assert!(p.check(&sa("10.255.255.255:1")));
        assert!(!p.check(&sa("11.0.0.1:443")));
    }

    #[test]
    fn cidr_with_port() {
        let p = NetworkPolicy::parse(true, &["10.0.0.0/8:443".into()]).unwrap();
        assert!(p.check(&sa("10.0.0.1:443")));
        assert!(!p.check(&sa("10.0.0.1:80")));
    }

    #[test]
    fn cidr_with_port_range() {
        let p = NetworkPolicy::parse(true, &["10.0.0.0/8:1024-65535".into()]).unwrap();
        assert!(p.check(&sa("10.0.0.1:1024")));
        assert!(p.check(&sa("10.0.0.1:65535")));
        assert!(!p.check(&sa("10.0.0.1:80")));
    }

    #[test]
    fn bare_ip_is_host_route() {
        let p = NetworkPolicy::parse(true, &["127.0.0.1".into()]).unwrap();
        assert!(p.check(&sa("127.0.0.1:80")));
        assert!(!p.check(&sa("127.0.0.2:80")));
    }

    #[test]
    fn ipv6_cidr() {
        let p = NetworkPolicy::parse(true, &["::1/128".into()]).unwrap();
        assert!(p.check(&sa("[::1]:80")));
        assert!(!p.check(&sa("[::2]:80")));
    }

    #[test]
    fn ipv6_bracketed_with_port() {
        let p = NetworkPolicy::parse(true, &["[::1]:443".into()]).unwrap();
        assert!(p.check(&sa("[::1]:443")));
        assert!(!p.check(&sa("[::1]:80")));
    }

    #[test]
    fn ipv4_doesnt_match_ipv6_rule_and_vice_versa() {
        let p = NetworkPolicy::parse(true, &["::/0".into()]).unwrap();
        assert!(p.check(&sa("[::1]:80")));
        assert!(!p.check(&sa("10.0.0.1:80")));
    }

    #[test]
    fn multiple_rules_or() {
        let p = NetworkPolicy::parse(true, &["10.0.0.0/8".into(), "127.0.0.0/8".into()]).unwrap();
        assert!(p.check(&sa("10.0.0.1:443")));
        assert!(p.check(&sa("127.0.0.1:80")));
        assert!(!p.check(&sa("8.8.8.8:53")));
    }

    #[test]
    fn malformed_cidr() {
        let err = NetworkPolicy::parse(true, &["not-a-cidr".into()]).unwrap_err();
        assert!(err.to_string().contains("bad CIDR"));
    }

    #[test]
    fn malformed_port() {
        let err = NetworkPolicy::parse(true, &["10.0.0.0/8:not-a-port".into()]).unwrap_err();
        assert!(err.to_string().contains("bad port"));
    }

    #[test]
    fn reversed_port_range() {
        let err = NetworkPolicy::parse(true, &["10.0.0.0/8:9000-1000".into()]).unwrap_err();
        assert!(err.to_string().contains("reversed"));
    }
}
