//! Out-of-band data-plane blob fetch (design §9).
//!
//! Large binary inputs (user images/audio) never travel the gateway↔worker
//! command path — `dispatch` carries only a [`BlobRef`], and the worker pulls
//! the bytes here over plain HTTP (`GET {origin}/blob/{hash}`). Content-
//! addressed, so integrity is free: size pre-check, then verify
//! `blake3(body) == hash`.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use pie_schema::gateway::BlobRef;

/// Fetch a blob's bytes out-of-band (`GET {origin}/blob/{hash}`) and verify
/// integrity (content-addressed, §9): size pre-check then `blake3(body) == hash`.
/// 404 fails the turn; 502 gets a bounded retry.
pub async fn fetch(blob: &BlobRef) -> Result<Vec<u8>> {
    const RETRIES: usize = 3;
    const RETRY_BACKOFF: Duration = Duration::from_millis(100);
    let url = format!("{}/blob/{}", blob.origin.trim_end_matches('/'), blob.hash);

    let mut attempt = 0;
    let bytes = loop {
        let resp = reqwest::get(url.as_str())
            .await
            .with_context(|| format!("GET {url}"))?;
        match resp.status().as_u16() {
            200 => break resp.bytes().await.context("reading blob body")?,
            404 => bail!("blob {} not found at {} (404)", blob.hash, blob.origin),
            502 if attempt + 1 < RETRIES => {
                attempt += 1;
                tracing::warn!(%url, attempt, "blob origin 502; retrying");
                tokio::time::sleep(RETRY_BACKOFF).await;
            }
            other => bail!("blob fetch {url} failed: status {other}"),
        }
    };

    if bytes.len() as u64 != blob.size {
        bail!(
            "blob {} size mismatch: got {}, expected {}",
            blob.hash,
            bytes.len(),
            blob.size
        );
    }
    let got = blake3::hash(bytes.as_ref()).to_hex().to_string();
    if got != blob.hash {
        bail!("blob {} integrity check failed (got {got})", blob.hash);
    }
    Ok(bytes.to_vec())
}
