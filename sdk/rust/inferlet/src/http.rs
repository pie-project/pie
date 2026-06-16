//! Minimal HTTP fetch helper for inferlets.
//!
//! `wstd::http::Client` (and the underlying `wasi:http`) is one-shot: it issues
//! a single request and hands back whatever comes — it does **not** follow
//! redirects, and Rust `std` has no HTTP client at all. Real media hosts almost
//! always redirect (the demo image/audio URLs 302 to a CDN, GitHub `raw` 302s)
//! and some 403 a request without a `User-Agent` (e.g. Wikimedia). So every
//! multimodal inferlet used to hand-roll the same redirect loop.
//!
//! [`fetch`] provides it once: GET a URL, follow redirects, return the bytes.

use crate::Result;
use wstd::http::{Client, Method, Request};
use wstd::io::{empty, AsyncRead};

/// Maximum redirect hops before giving up.
const MAX_REDIRECTS: usize = 8;

/// `User-Agent` sent with every request. Some hosts (e.g. Wikimedia) 403 a
/// request that lacks one.
const USER_AGENT: &str = "pie-inferlet/0.1";

/// GET `url` and return the response body, following up to [`MAX_REDIRECTS`]
/// redirects. The transport does not follow redirects itself, so each `Location`
/// is resolved against the current URL via [`resolve_redirect`].
///
/// ```ignore
/// let bytes = inferlet::http::fetch("https://example.org/cat.jpg").await?;
/// ```
pub async fn fetch(url: &str) -> Result<Vec<u8>> {
    let client = Client::new();
    let mut current = url.to_string();
    for _ in 0..MAX_REDIRECTS {
        let req = Request::builder()
            .uri(&current)
            .method(Method::GET)
            .header("User-Agent", USER_AGENT)
            .body(empty())
            .map_err(|e| e.to_string())?;
        let resp = client.send(req).await.map_err(|e| e.to_string())?;
        let status = resp.status().as_u16();
        if (300..400).contains(&status) {
            let loc = resp
                .headers()
                .get("location")
                .and_then(|v| v.to_str().ok())
                .ok_or_else(|| format!("redirect {status} without Location ({current})"))?;
            current = resolve_redirect(&current, loc);
            continue;
        }
        if !(200..300).contains(&status) {
            return Err(format!("HTTP {status} fetching {current}"));
        }
        let mut body = resp.into_body();
        let mut buf = Vec::new();
        body.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
        return Ok(buf);
    }
    Err(format!("too many redirects (>{MAX_REDIRECTS}) fetching {url}"))
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
