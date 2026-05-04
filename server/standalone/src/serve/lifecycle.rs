//! Shutdown helpers for [`super::EngineHandle`]: SIGTERM listener,
//! per-driver liveness watchdog, and the best-effort POSIX shmem
//! cleanup that runs after the drivers have joined.

use std::ffi::CString;
use std::time::Duration;

use crate::embedded_driver::EmbeddedDriver;

/// Wait for SIGTERM (and SIGTERM only — SIGINT lives on
/// `tokio::signal::ctrl_c`). Returns `Ok(())` once a SIGTERM is
/// observed; logs and returns on stream-init failure rather than
/// panicking, so the rest of the shutdown branches still race
/// normally.
pub async fn wait_for_sigterm() {
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

/// Per-second poll of every driver thread's liveness. Returns the
/// human-readable "reason" string when one dies — used by `select!`
/// to label the shutdown reason in stdout.
pub async fn watchdog(drivers: &[EmbeddedDriver]) -> &'static str {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.tick().await; // first tick fires immediately; skip it.
    loop {
        interval.tick().await;
        for d in drivers {
            if d.is_finished() {
                tracing::error!(
                    "driver thread {} exited unexpectedly; tearing down",
                    d.shmem_name
                );
                return "driver thread exited unexpectedly";
            }
        }
    }
}

/// Best-effort `shm_unlink`. Logs non-`ENOENT` failures (a missing
/// segment is the success case — driver cleaned up first).
pub fn unlink_shmem(name: &str) {
    let Ok(c) = CString::new(name) else {
        tracing::warn!("shmem name {name:?} contains a NUL; skipping unlink");
        return;
    };
    let rc = unsafe { libc::shm_unlink(c.as_ptr()) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ENOENT) {
            tracing::warn!("shm_unlink({name}) failed: {err}");
        }
    }
}
