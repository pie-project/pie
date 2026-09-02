//! Trust-edge identity gate. The gateway sits behind an edge proxy that has
//! already authenticated the caller; this only extracts tenant/user from the
//! edge-supplied header for routing/quota/isolation — it is not
//! authentication. The gateway must therefore only accept connections from
//! the edge (private bind / mTLS), enforced at deploy, not here.

use std::net::IpAddr;

use anyhow::{Context, anyhow};
use axum::http::HeaderMap;
use ids::TenantId;

use crate::session::Identity;

/// Edge-supplied verified identity claim. Convention: `tenant/user` (a forwarded
/// JWT-claims summary). The edge guarantees its presence + verification.
pub const IDENTITY_HEADER: &str = "x-pie-identity";
/// Standard client-IP forwarding header set by the edge proxy.
pub const FORWARDED_FOR_HEADER: &str = "x-forwarded-for";
/// Per-request trace id propagated from the edge.
pub const REQUEST_ID_HEADER: &str = "x-request-id";

/// Build an [`Identity`] from edge-supplied headers. Fails closed: a missing
/// or malformed identity header is a misconfigured edge, so this rejects
/// rather than serve an unattributed request.
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

/// Extract the origin client IP from an `X-Forwarded-For` value: a
/// comma-separated, hop-appended list whose left-most entry is the original
/// client. `None` if absent/unparseable (non-fatal; carried for tracing only).
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

