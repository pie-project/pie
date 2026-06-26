//! HTTP for inferlets — a thin async wrapper over the host `pie:core/http`
//! interface.
//!
//! Under component-model-async there is no guest-side bridge from a wasi:http
//! `future-incoming-response` pollable to a Rust future — the only thing that
//! provided one (wstd's reactor) reintroduces the duplicate `cabi_realloc`
//! link error. So the transport is a host-provided async func,
//! [`pie:core/http.fetch`](crate::pie::core::http::fetch): it performs **one**
//! request host-side and buffers the full response. Redirect-following and the
//! default `User-Agent` live here so the policy is testable and the host
//! primitive stays a single, simple request.
//!
//! Two entry points:
//! - [`fetch`] — high-level `GET` that follows redirects and returns the body.
//! - [`send`] — one raw [`Request`] → [`Response`], for custom method / headers
//!   / body and concurrent use (drive many `send(...)` futures with
//!   `futures::stream::FuturesUnordered`; bound a single one with
//!   `futures::select!(send(req), inferlet::sleep(timeout))`).

use crate::Result;

/// Host HTTP request/response records, re-exported for [`send`].
pub use crate::pie::core::http::{Request, Response};

/// Maximum redirect hops before giving up.
const MAX_REDIRECTS: usize = 8;

/// `User-Agent` sent by [`fetch`]. Some hosts (e.g. Wikimedia) 403 a request
/// that lacks one.
const USER_AGENT: &str = "pie-inferlet/0.1";

/// Perform a single HTTP request host-side and return the buffered response.
///
/// The host does **not** follow redirects (a 3xx comes back as a normal
/// [`Response`] with a `location` header). Only transport/host failures (DNS,
/// connect, TLS, disabled network policy) surface as `Err`; any HTTP status —
/// including 4xx/5xx — is an `Ok(Response)`.
///
/// Use directly for non-GET verbs, custom headers/body, or concurrency
/// (multiple in-flight `send` futures run concurrently on the host event loop).
pub async fn send(request: Request) -> Result<Response> {
    crate::pie::core::http::fetch(request)
        .await
        .map_err(|e| e.to_string())
}

/// GET `url` and return the response body, following up to [`MAX_REDIRECTS`]
/// redirects. Each `Location` is resolved against the current URL via
/// [`resolve_redirect`]; a default `User-Agent` is attached.
///
/// ```ignore
/// let bytes = inferlet::http::fetch("https://example.org/cat.jpg").await?;
/// ```
pub async fn fetch(url: &str) -> Result<Vec<u8>> {
    let mut current = url.to_string();
    for _ in 0..MAX_REDIRECTS {
        let req = Request {
            method: "GET".to_string(),
            url: current.clone(),
            headers: vec![("user-agent".to_string(), USER_AGENT.to_string())],
            body: None,
        };
        let resp = send(req).await?;
        let status = resp.status;
        if (300..400).contains(&status) {
            let loc = header(&resp.headers, "location")
                .ok_or_else(|| format!("redirect {status} without Location ({current})"))?;
            current = resolve_redirect(&current, &loc);
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(format!("HTTP {status} fetching {current}"));
        }
        return Ok(resp.body);
    }
    Err(format!("too many redirects (>{MAX_REDIRECTS}) fetching {url}"))
}

/// Case-insensitive lookup of a response header value.
fn header(headers: &[(String, String)], name: &str) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

/// Resolve a redirect `Location` against the request URL: absolute,
/// scheme-relative (`//host/…`), root-relative (`/path`), or path-relative.
fn resolve_redirect(base: &str, loc: &str) -> String {
    if loc.starts_with("http://") || loc.starts_with("https://") {
        loc.to_string()
    } else if let Some(rest) = loc.strip_prefix("//") {
        let scheme = base.split("://").next().unwrap_or("https");
        format!("{scheme}://{rest}")
    } else if loc.starts_with('/') {
        let (scheme, after) = base.split_once("://").unwrap_or(("https", base));
        let authority = after.split('/').next().unwrap_or(after);
        format!("{scheme}://{authority}{loc}")
    } else {
        match base.rsplit_once('/') {
            Some((prefix, _)) => format!("{prefix}/{loc}"),
            None => loc.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_redirect;

    #[test]
    fn redirect_resolution_cases() {
        let base = "https://host.example/a/b/c.jpg";
        // Absolute Location is taken verbatim.
        assert_eq!(
            resolve_redirect(base, "https://cdn.example/x.jpg"),
            "https://cdn.example/x.jpg"
        );
        // Scheme-relative inherits the base scheme.
        assert_eq!(
            resolve_redirect(base, "//cdn.example/x.jpg"),
            "https://cdn.example/x.jpg"
        );
        // Root-relative keeps scheme + authority, replaces the path.
        assert_eq!(resolve_redirect(base, "/x.jpg"), "https://host.example/x.jpg");
        // Path-relative resolves against the base's directory.
        assert_eq!(
            resolve_redirect(base, "d.jpg"),
            "https://host.example/a/b/d.jpg"
        );
    }
}
