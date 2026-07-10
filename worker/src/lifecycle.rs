//! Shutdown helpers for [`super::EngineHandle`].

/// Wait for SIGTERM (and SIGTERM only — SIGINT lives on
/// `tokio::signal::ctrl_c`). Returns once a SIGTERM is observed.
pub async fn wait_for_sigterm() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut stream = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("could not install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
                return;
            }
        };
        stream.recv().await;
    }

    #[cfg(windows)]
    {
        std::future::pending::<()>().await;
    }
}
