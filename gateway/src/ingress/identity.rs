//! Trust-edge identity gate (§2/§5). The gateway sits behind an edge proxy that
//! has already authenticated the caller; we **trust** the edge-supplied identity
//! header and merely *extract* tenant/user for routing/quota/isolation. This is
//! NOT authentication — no token is re-verified here.
//!
//! Trust boundary: because we trust these headers, the gateway must only accept
//! connections from the edge (private bind / mTLS). Enforced at deploy, not here.

use std::net::IpAddr;

use anyhow::{Context, anyhow};
use axum::http::HeaderMap;
use pie_ids::TenantId;

use crate::session::Identity;

/// Edge-supplied verified identity claim. Convention: `tenant/user` (a forwarded
/// JWT-claims summary). The edge guarantees its presence + verification.
pub const IDENTITY_HEADER: &str = "x-pie-identity";
/// Standard client-IP forwarding header set by the edge proxy.
pub const FORWARDED_FOR_HEADER: &str = "x-forwarded-for";
/// Per-request trace id propagated from the edge.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Build an [`Identity`] from edge-supplied headers. Fails closed: a missing or
/// malformed identity header is a misconfigured edge (§2), so we reject rather
/// than serve an unattributed request.
pub fn extract(headers: &HeaderMap) -> anyhow::Result<Identity> {
    let raw = headers
        .get(IDENTITY_HEADER)
        .ok_or_else(|| anyhow!("missing `{IDENTITY_HEADER}` (edge must inject verified identity)"))?
        .to_str()
        .context("`x-pie-identity` is not valid UTF-8")?;

    let (tenant, user) = parse_identity(raw)?;

    let client_ip = headers
        .get(FORWARDED_FOR_HEADER)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_forwarded_for);

    let request_id = headers
        .get(REQUEST_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .filter(|s| !s.is_empty());

    Ok(Identity {
        tenant: TenantId(tenant),
        user,
        client_ip,
        request_id,
    })
}

/// Parse the identity claim into `(tenant, user)`. Accepts `tenant/user` or a
/// bare `user` (tenant defaults to `"default"`). Empty user is rejected.
fn parse_identity(raw: &str) -> anyhow::Result<(String, String)> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(anyhow!("empty `{IDENTITY_HEADER}`"));
    }
    let (tenant, user) = match raw.split_once('/') {
        Some((t, u)) => (t.trim(), u.trim()),
        None => ("default", raw),
    };
    if user.is_empty() {
        return Err(anyhow!("`{IDENTITY_HEADER}` has empty user component"));
    }
    Ok((tenant.to_string(), user.to_string()))
}

/// Extract the origin client IP from an `X-Forwarded-For` value. The header is a
/// comma-separated list appended hop-by-hop; the **left-most** entry is the
/// original client. Tolerates a trailing `:port`. `None` if absent/unparseable
/// (the IP is carried for tracing/quota only, so a bad value is non-fatal).
fn parse_forwarded_for(value: &str) -> Option<IpAddr> {
    let first = value.split(',').next()?.trim();
    if first.is_empty() {
        return None;
    }
    // Try as-is, then strip a trailing `:port` (host:port form).
    first.parse::<IpAddr>().ok().or_else(|| {
        first
            .rsplit_once(':')
            .and_then(|(host, _)| host.parse().ok())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::{HeaderName, HeaderValue};

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn extracts_tenant_user_ip_and_request_id() {
        let h = headers(&[
            (IDENTITY_HEADER, "acme/alice"),
            (FORWARDED_FOR_HEADER, "203.0.113.7, 10.0.0.1"),
            (REQUEST_ID_HEADER, "req-123"),
        ]);
        let id = extract(&h).unwrap();
        assert_eq!(id.tenant.0, "acme");
        assert_eq!(id.user, "alice");
        assert_eq!(id.client_ip, Some("203.0.113.7".parse().unwrap()));
        assert_eq!(id.request_id.as_deref(), Some("req-123"));
    }

    #[test]
    fn bare_user_defaults_tenant() {
        let id = extract(&headers(&[(IDENTITY_HEADER, "bob")])).unwrap();
        assert_eq!(id.tenant.0, "default");
        assert_eq!(id.user, "bob");
        assert!(id.client_ip.is_none());
    }

    #[test]
    fn missing_header_fails_closed() {
        assert!(extract(&HeaderMap::new()).is_err());
    }

    #[test]
    fn empty_user_rejected() {
        assert!(extract(&headers(&[(IDENTITY_HEADER, "acme/")])).is_err());
    }

    #[test]
    fn forwarded_for_strips_port_and_picks_leftmost() {
        assert_eq!(
            parse_forwarded_for("198.51.100.5:443, 10.0.0.1"),
            Some("198.51.100.5".parse().unwrap())
        );
        assert_eq!(parse_forwarded_for("not-an-ip"), None);
        assert_eq!(parse_forwarded_for(""), None);
    }
}
