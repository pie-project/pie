use anyhow::{Context, Result, bail};
use worker_api::BlobRef;

/// The bytes of `blob` from its origin, asked for once: any answer but the
/// blob fails by the blob and the status, a 502 as any other.
pub async fn fetch(blob: &BlobRef) -> Result<Vec<u8>> {
    let url = format!("{}/blob/{}", blob.origin.trim_end_matches('/'), blob.hash);

    let client = crate::http::builder()?
        .build()
        .context("building HTTP client")?;
    let resp = client
        .get(url.as_str())
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    let bytes = match resp.status().as_u16() {
        200 => resp.bytes().await.context("reading blob body")?,
        404 => bail!("blob {} not found at {} (404)", blob.hash, blob.origin),
        other => bail!(
            "blob {} was not fetched from {}: status {other}",
            blob.hash,
            blob.origin
        ),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// A 502 from the origin fails the fetch by the blob and the status,
    /// and the origin is asked once.
    #[tokio::test]
    async fn a_502_fails_by_name_after_one_request() {
        // reqwest takes its TLS provider from the process, as `pie serve`
        // installs it; this fetch is plain HTTP.
        let _ = rustls::crypto::ring::default_provider().install_default();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let asked = Arc::new(AtomicUsize::new(0));
        let counted = asked.clone();
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                counted.fetch_add(1, Ordering::SeqCst);
                let mut request = [0u8; 1024];
                let _ = stream.read(&mut request).await;
                let _ = stream
                    .write_all(
                        b"HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    )
                    .await;
            }
        });
        let blob = BlobRef {
            hash: "0f".repeat(32),
            size: 4,
            kind: "program".to_string(),
            origin: origin.clone(),
        };
        let refused = fetch(&blob).await.unwrap_err().to_string();
        assert_eq!(
            refused,
            format!(
                "blob {} was not fetched from {origin}: status 502",
                blob.hash
            )
        );
        assert_eq!(asked.load(Ordering::SeqCst), 1, "asked once");
    }
}
