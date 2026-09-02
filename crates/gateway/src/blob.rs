//! The gateway's content-addressed blob plane, for large user binaries that
//! must not ride the command path (head-of-line blocking, serde memory).
//! The command path carries a [`BlobRef`], and bytes flow out-of-band over a
//! separate bulk HTTP connection.
//!
//! This module owns the origin side of that path: [`BlobStore`] is the
//! storage seam (async + `dyn`-safe so the default in-memory tier can be
//! swapped for an object store with no call-site change);
//! [`GatewayOriginStore`] is that default tier, stamping its own advertised
//! URL into every [`BlobRef`] it mints so any worker can fetch it; [`router`]
//! is the `GET /blob/{hash}` axum route.
//!
//! Content-addressing makes integrity free: the hash is the address, so a
//! fetching worker re-hashes the bytes and compares.

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
use worker_api::BlobRef;

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
/// `async` and object-safe (`Arc<dyn BlobStore>`) so a future object-store tier
/// with genuine network I/O can swap in with no call-site change.
#[async_trait]
pub trait BlobStore: Send + Sync + 'static {
    /// Ingest `bytes` of MIME `kind`: hash, store, and return a fully-stamped
    /// [`BlobRef`]. Idempotent by content.
    async fn put(&self, kind: String, bytes: Bytes) -> Result<BlobRef>;

    /// Fetch a stored blob by its content `hash`. `Ok(None)` if this store
    /// does not hold it.
    async fn get(&self, hash: &str) -> Result<Option<Blob>>;
}

/// The default gateway-origin [`BlobStore`]: the ingesting gateway holds its
/// own blobs in memory and serves them by hash. `origin` is this gateway's
/// advertised blob base URL, stamped into every minted [`BlobRef`].
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

/// The blob plane's axum routes: `GET /blob/{hash}` over `store`, for the
/// gateway's `lib.rs` to `.merge()` onto its one shared listener.
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
/// The hash is the integrity check: the fetching worker re-hashes the body
/// and compares, so no extra checksum is served here.
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

