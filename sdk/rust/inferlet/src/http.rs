//! HTTP for inferlets — an async client over `wasi:http/client@0.3.0`.
//!
//! The transport is the standard wasi 0.3 outbound client: [`send`] builds a
//! `wasi:http/types.request`, drives it through
//! [`client::send`](crate::wasi::http::client::send) on the
//! component-model-async event loop, and buffers the full response body. The
//! request/response records below are SDK-local buffered mirrors so callers
//! keep the simple `Vec<u8>` body ergonomics; redirect-following and the
//! default `User-Agent` live here so the policy stays testable.
//!
//! Two entry points:
//! - [`fetch`] — high-level `GET` that follows redirects and returns the body.
//! - [`send`] — one raw [`Request`] → [`Response`], for custom method / headers
//!   / body and concurrent use (drive many `send(...)` futures with
//!   `futures::stream::FuturesUnordered`; bound a single one with
//!   `futures::select!(send(req), inferlet::sleep(timeout))`).
//!
//! Network policy is enforced host-side: when the engine denies network access
//! the request fails and `send` returns `Err`.

use crate::Result;

/// A buffered HTTP request. `body` is sent verbatim; `headers` are lowercased
/// field names paired with UTF-8 values.
#[derive(Clone, Debug, Default)]
pub struct Request {
    /// HTTP method (`GET`, `POST`, …); case-insensitive.
    pub method: String,
    /// Absolute request URL (`https://host/path?query`).
    pub url: String,
    /// Request headers as `(name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Optional request body.
    pub body: Option<Vec<u8>>,
}

/// A buffered HTTP response with the full body read into memory.
#[derive(Clone, Debug, Default)]
pub struct Response {
    /// HTTP status code.
    pub status: u16,
    /// Response headers as `(name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Full response body.
    pub body: Vec<u8>,
}

/// Maximum redirect hops before giving up.
const MAX_REDIRECTS: usize = 8;

/// `User-Agent` sent by [`fetch`]. Some hosts (e.g. Wikimedia) 403 a request
/// that lacks one.
const USER_AGENT: &str = "pie-inferlet/0.1";

/// Perform a single HTTP request over `wasi:http/client@0.3.0` and return the
/// buffered response.
///
/// The host does **not** follow redirects (a 3xx comes back as a normal
/// [`Response`] with a `location` header). Only transport/host failures (DNS,
/// connect, TLS, disabled network policy) surface as `Err`; any HTTP status —
/// including 4xx/5xx — is an `Ok(Response)`.
///
/// Use directly for non-GET verbs, custom headers/body, or concurrency
/// (multiple in-flight `send` futures run concurrently on the host event loop).
pub async fn send(request: Request) -> Result<Response> {
    use crate::wasi::http::client;
    use crate::wasi::http::types::{ErrorCode, Fields, Request as WitRequest, Response as WitResponse, Trailers};

    let (scheme, authority, path_q) = parse_url(&request.url)?;

    let entries: Vec<(String, Vec<u8>)> = request
        .headers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone().into_bytes()))
        .collect();
    let headers = Fields::from_list(&entries).map_err(|e| format!("invalid headers: {e:?}"))?;

    // Optional request body: a `stream<u8>` whose writer is driven concurrently
    // so the host can pull the body while `send` is in flight.
    let contents = match request.body {
        Some(ref b) if !b.is_empty() => {
            let (mut tx, rx) = crate::wit_stream::new::<u8>();
            let bytes = b.clone();
            crate::wit_bindgen::spawn_local(async move {
                let _ = tx.write_all(bytes).await;
            });
            Some(rx)
        }
        _ => None,
    };

    // Request trailers: none — resolve the future immediately.
    let (trailers_tx, trailers_rx) =
        crate::wit_future::new::<core::result::Result<Option<Trailers>, ErrorCode>>(|| Ok(None));
    crate::wit_bindgen::spawn_local(async move {
        let _ = trailers_tx.write(Ok(None)).await;
    });

    let (wit_req, _req_done) = WitRequest::new(headers, contents, trailers_rx, None);
    wit_req
        .set_method(&method_of(&request.method))
        .map_err(|_| "invalid method".to_string())?;
    wit_req
        .set_scheme(Some(&scheme))
        .map_err(|_| "invalid scheme".to_string())?;
    wit_req
        .set_authority(Some(&authority))
        .map_err(|_| "invalid authority".to_string())?;
    wit_req
        .set_path_with_query(Some(&path_q))
        .map_err(|_| "invalid path".to_string())?;

    let resp: WitResponse = client::send(wit_req)
        .await
        .map_err(|e| format!("http send failed: {e:?}"))?;

    let status = resp.get_status_code();
    let headers: Vec<(String, String)> = resp
        .get_headers()
        .copy_all()
        .into_iter()
        .map(|(k, v)| (k, String::from_utf8_lossy(&v).into_owned()))
        .collect();

    // Consume the response body stream to EOF.
    let (body_res_tx, body_res_rx) =
        crate::wit_future::new::<core::result::Result<(), ErrorCode>>(|| Ok(()));
    crate::wit_bindgen::spawn_local(async move {
        let _ = body_res_tx.write(Ok(())).await;
    });
    let (mut body_stream, _resp_trailers) = WitResponse::consume_body(resp, body_res_rx);
    let mut body = Vec::new();
    while let Some(chunk) = body_stream.next().await {
        body.push(chunk);
    }

    Ok(Response {
        status,
        headers,
        body,
    })
}

/// Map a case-insensitive method string to a `wasi:http` [`Method`].
fn method_of(m: &str) -> crate::wasi::http::types::Method {
    use crate::wasi::http::types::Method;
    match m.to_ascii_uppercase().as_str() {
        "GET" => Method::Get,
        "HEAD" => Method::Head,
        "POST" => Method::Post,
        "PUT" => Method::Put,
        "DELETE" => Method::Delete,
        "CONNECT" => Method::Connect,
        "OPTIONS" => Method::Options,
        "TRACE" => Method::Trace,
        "PATCH" => Method::Patch,
        other => Method::Other(other.to_string()),
    }
}

/// Split an absolute URL into `(scheme, authority, path-with-query)`.
fn parse_url(url: &str) -> Result<(crate::wasi::http::types::Scheme, String, String)> {
    use crate::wasi::http::types::Scheme;
    let (scheme_str, rest) = url
        .split_once("://")
        .ok_or_else(|| format!("URL missing scheme: {url}"))?;
    let scheme = match scheme_str.to_ascii_lowercase().as_str() {
        "http" => Scheme::Http,
        "https" => Scheme::Https,
        other => Scheme::Other(other.to_string()),
    };
    let (authority, path_q) = match rest.find('/') {
        Some(i) => (rest[..i].to_string(), rest[i..].to_string()),
        None => (rest.to_string(), "/".to_string()),
    };
    if authority.is_empty() {
        return Err(format!("URL missing authority: {url}"));
    }
    Ok((scheme, authority, path_q))
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
