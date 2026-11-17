//! The HTTP client every outbound fetch builds from.
//!
//! reqwest's rustls verifier reads the platform's trust store through the
//! OS. On Android that is a JNI call into the app's Java context, which a
//! native binary run outside an app (`pie` under `adb shell`) never has, and
//! building any client panics there. Android also keeps its trust store as
//! files, one PEM certificate each: the system's under
//! `/system/etc/security/cacerts` and, since Android 14, the updatable
//! Conscrypt module's under `/apex/com.android.conscrypt/cacerts`. There the
//! client verifies against those files instead.

use anyhow::Result;

/// A client builder that verifies servers against the platform's roots.
pub fn builder() -> Result<reqwest::ClientBuilder> {
    platform(reqwest::Client::builder())
}

#[cfg(not(target_os = "android"))]
#[allow(clippy::unnecessary_wraps)]
fn platform(builder: reqwest::ClientBuilder) -> Result<reqwest::ClientBuilder> {
    Ok(builder)
}

#[cfg(target_os = "android")]
fn platform(builder: reqwest::ClientBuilder) -> Result<reqwest::ClientBuilder> {
    use anyhow::{Context, bail};

    /// Where Android keeps its trusted roots: the Conscrypt module's
    /// (updated with the module, Android 14 on) and the system image's.
    const STORES: [&str; 2] = [
        "/apex/com.android.conscrypt/cacerts",
        "/system/etc/security/cacerts",
    ];
    let mut roots = Vec::new();
    let mut read = Vec::new();
    for dir in STORES {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        read.push(dir);
        for entry in entries {
            let path = entry.with_context(|| format!("listing {dir}"))?.path();
            let pem = std::fs::read(&path)
                .with_context(|| format!("reading the root {}", path.display()))?;
            roots.push(reqwest::Certificate::from_pem(&pem).with_context(|| {
                format!("the root {} is not a PEM certificate", path.display())
            })?);
        }
    }
    if roots.is_empty() {
        bail!(
            "no trusted roots to verify HTTPS servers against: none of {} holds a certificate \
             this process can read",
            STORES.join(", ")
        );
    }
    tracing::debug!(roots = roots.len(), stores = ?read, "verifying against Android's trust store");
    Ok(builder.tls_certs_only(roots))
}
