//! `blob.rs` — the gateway's content-addressed **blob plane**.
//!
//! Large user binaries (e.g. image inputs) never ride the command path: that
//! would head-of-line-block latency-sensitive dispatch and balloon serde memory.
//! Instead the command path carries a [`BlobRef`] *reference* and the bytes flow
//! out-of-band over a separate bulk HTTP connection (design §9).
//!
//! This module owns the **origin side** of that out-of-band path:
//!
//! - [`BlobStore`] — the storage seam. Ingest bytes ⇒ a content-addressed
//!   [`BlobRef`]; fetch bytes by hash. Async + `dyn`-safe so the default
//!   in-memory tier can be swapped for a content-addressed object store
//!   (S3-API) with **no call-site change** when gateway egress becomes the
//!   bottleneck (the graduation in §9). The gateway then drops out of the data
//!   path entirely.
//! - [`GatewayOriginStore`] — the default **gateway-origin** tier: the ingesting
//!   gateway holds its own blobs in memory and serves them by hash. The
//!   `origin` baked into each [`BlobRef`] tells *any* worker — even one
//!   dispatched by a different gateway — where to `GET /blob/{hash}`, so no
//!   shared directory is needed.
//! - [`router`] — the `GET /blob/{hash}` axum route, merged onto the gateway's
//!   one shared listener (one server, one router — blob is just another route).
//!
//! Content-addressing makes integrity free: the hash *is* the address, so a
//! fetching worker re-hashes the bytes and compares against the requested hash.
//! It also yields free dedup and encode caching across multi-turn references to
//! the same blob.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::Result;
use async_trait::async_trait;
use axum::Router;
use axum::extract::{Path, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use bytes::Bytes;
use pie_worker_rpc::BlobRef;

/// A stored blob: its MIME `kind` and raw `bytes`. Returned by [`BlobStore::get`]
/// so the serving handler can echo the original content type.
#[derive(Clone, Debug)]
pub struct Blob {
    /// MIME type the blob was ingested with (e.g. `image/jpeg`).
    pub kind: String,
    /// The raw blob payload.
    pub bytes: Bytes,
}

/// Content-addressed blob storage behind the gateway's `GET /blob/{hash}` plane.
///
/// Kept `async` and object-safe (consumed as `Arc<dyn BlobStore>`) on purpose:
/// the default [`GatewayOriginStore`] satisfies `put`/`get` with a local map, but
/// the same seam fronts the deferred content-addressed object-store tier whose
/// `put`/`get` are genuine async network I/O — swapping the impl needs no change
/// at the ingress (`put`) or route (`get`) call sites (design §9).
#[async_trait]
pub trait BlobStore: Send + Sync + 'static {
    /// Ingest `bytes` of MIME `kind`: hash, store, and return a fully-stamped
    /// content-addressed [`BlobRef`] (its `origin` already baked in). Idempotent
    /// by content — re-ingesting identical bytes yields the same hash and stores
    /// once.
    async fn put(&self, kind: String, bytes: Bytes) -> Result<BlobRef>;

    /// Fetch a stored blob by its content `hash`. `Ok(None)` ⇒ this store does
    /// not hold the blob; `Err` ⇒ a backing-store failure (never, for the
    /// in-memory tier).
    async fn get(&self, hash: &str) -> Result<Option<Blob>>;
}

/// The default **gateway-origin** [`BlobStore`]: the ingesting gateway holds its
/// own blobs in memory and serves them by hash.
///
/// `origin` is this gateway's advertised blob base URL (e.g.
/// `http://10.0.0.5:8080`), stamped into every [`BlobRef`] this store mints so a
/// worker dispatched the ref — by this gateway or any other — can fetch the bytes
/// from the gateway that actually holds them.
pub struct GatewayOriginStore {
    origin: String,
    blobs: RwLock<HashMap<String, Blob>>,
}

impl GatewayOriginStore {
    /// Create a store that stamps `origin` (the gateway's advertised blob base
    /// URL) into every minted [`BlobRef`].
    pub fn new(origin: impl Into<String>) -> Self {
        Self {
            origin: origin.into(),
            blobs: RwLock::new(HashMap::new()),
        }
    }

    /// Number of distinct blobs currently held (observability / metrics).
    pub fn len(&self) -> usize {
        self.blobs.read().expect("blob store lock poisoned").len()
    }

    /// Whether the store holds no blobs.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl BlobStore for GatewayOriginStore {
    async fn put(&self, kind: String, bytes: Bytes) -> Result<BlobRef> {
        let hash = blake3::hash(&bytes).to_hex().to_string();
        let size = bytes.len() as u64;
        // Content-addressed dedup: identical bytes hash identically, stored once.
        self.blobs
            .write()
            .expect("blob store lock poisoned")
            .entry(hash.clone())
            .or_insert_with(|| Blob {
                kind: kind.clone(),
                bytes,
            });
        Ok(BlobRef {
            hash,
            size,
            kind,
            origin: self.origin.clone(),
        })
    }

    async fn get(&self, hash: &str) -> Result<Option<Blob>> {
        Ok(self
            .blobs
            .read()
            .expect("blob store lock poisoned")
            .get(hash)
            .cloned())
    }
}

/// The blob plane's axum routes: `GET /blob/{hash}` over `store`.
///
/// Returns a fully-stated `Router` for the gateway's `lib.rs` to `.merge()` onto
/// the one shared listener — blob is just another route alongside ingress, not a
/// second server.
pub fn router(store: Arc<dyn BlobStore>) -> Router {
    Router::new()
        .route("/blob/{hash}", get(serve_blob))
        .with_state(store)
}

/// `GET /blob/{hash}` — content-addressed byte read.
///
/// - `200 OK` with the raw bytes (`Content-Type` = the stored MIME,
///   `Content-Length` = byte size) when the blob is held.
/// - `404 Not Found` when this origin does not hold the hash.
/// - `502 Bad Gateway` on a backing-store error.
///
/// The hash *is* the integrity check — the fetching worker re-hashes the body and
/// compares against the requested hash (§9) — so no extra checksum is served here.
async fn serve_blob(State(store): State<Arc<dyn BlobStore>>, Path(hash): Path<String>) -> Response {
    match store.get(&hash).await {
        Ok(Some(blob)) => {
            let content_type = HeaderValue::from_str(&blob.kind)
                .unwrap_or_else(|_| HeaderValue::from_static("application/octet-stream"));
            ([(header::CONTENT_TYPE, content_type)], blob.bytes).into_response()
        }
        Ok(None) => StatusCode::NOT_FOUND.into_response(),
        Err(e) => {
            tracing::warn!(%hash, error = %e, "blob fetch failed");
            StatusCode::BAD_GATEWAY.into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use axum::http::Request;
    use tower::ServiceExt;

    const ORIGIN: &str = "http://gw.test:8080";

    #[tokio::test]
    async fn put_stamps_content_addressed_ref() {
        let store = GatewayOriginStore::new(ORIGIN);
        let bytes = Bytes::from_static(b"hello blob");
        let r = store.put("image/png".into(), bytes.clone()).await.unwrap();

        assert_eq!(r.hash, blake3::hash(&bytes).to_hex().to_string());
        assert_eq!(r.size, bytes.len() as u64);
        assert_eq!(r.kind, "image/png");
        assert_eq!(r.origin, ORIGIN);
    }

    #[tokio::test]
    async fn get_roundtrips_put_and_misses_unknown() {
        let store = GatewayOriginStore::new(ORIGIN);
        let bytes = Bytes::from_static(b"payload");
        let r = store
            .put("application/octet-stream".into(), bytes.clone())
            .await
            .unwrap();

        let got = store.get(&r.hash).await.unwrap().expect("blob present");
        assert_eq!(got.bytes, bytes);
        assert_eq!(got.kind, "application/octet-stream");
        assert!(store.get("deadbeef").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn identical_bytes_dedup() {
        let store = GatewayOriginStore::new(ORIGIN);
        let b = Bytes::from_static(b"same");
        let r1 = store.put("text/plain".into(), b.clone()).await.unwrap();
        let r2 = store.put("text/plain".into(), b.clone()).await.unwrap();

        assert_eq!(r1.hash, r2.hash);
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn route_serves_known_and_404s_unknown() {
        let store = Arc::new(GatewayOriginStore::new(ORIGIN));
        let bytes = Bytes::from_static(b"\x00\x01\x02image-bytes");
        let r = store.put("image/jpeg".into(), bytes.clone()).await.unwrap();
        let app = router(store as Arc<dyn BlobStore>);

        // Known hash -> 200, exact bytes, original content type.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/blob/{}", r.hash))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get(header::CONTENT_TYPE).unwrap(),
            "image/jpeg"
        );
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body, bytes);

        // Unknown hash -> 404.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/blob/0000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
