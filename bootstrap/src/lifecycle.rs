//! Process lifecycle — panic hook, the boot banner, and the
//! wait-for-signal-then-drain loop behind [`Ctx::run_until_signal`].
//!
//! Ruling R1: the shutdown seam is a *future* (a closure the bin builds from its
//! role `Handle`), never a bootstrap-defined trait — so role libs take no
//! dependency on bootstrap.

use std::net::SocketAddr;

/// Route panics through `tracing` (so they land in the same structured log as
/// everything else) while preserving the default hook's output.
pub(crate) fn install_panic_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!("panic: {info}");
        default(info);
    }));
}

/// One-line boot banner to stderr, plus the resolved `/metrics` address.
pub(crate) fn banner(name: &str, version: &str, metrics: Option<SocketAddr>) {
    eprintln!("pie-{name} {version}");
    if let Some(addr) = metrics {
        tracing::info!("/metrics serving on http://{addr}/metrics");
    }
}

/// Block (async) until SIGINT/SIGTERM (Unix) or Ctrl-C (otherwise).
pub(crate) async fn wait_for_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut sigint = match signal(SignalKind::interrupt()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("install SIGINT handler: {e}");
                return;
            }
        };
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("install SIGTERM handler: {e}");
                return;
            }
        };
        tokio::select! {
            _ = sigint.recv() => tracing::info!("received SIGINT"),
            _ = sigterm.recv() => tracing::info!("received SIGTERM"),
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("received Ctrl-C");
    }
}
