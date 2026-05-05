//! Shutdown helpers for [`super::EngineHandle`]: SIGTERM listener,
//! per-driver liveness watchdog, and best-effort shmem cleanup that runs
//! after the drivers have joined.

#[cfg(unix)]
use std::ffi::CString;
use std::time::Duration;

use super::DriverHandle;

/// Wait for SIGTERM (and SIGTERM only — SIGINT lives on
/// `tokio::signal::ctrl_c`). Returns `Ok(())` once a SIGTERM is
/// observed; logs and returns on stream-init failure rather than
/// panicking, so the rest of the shutdown branches still race
/// normally.
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

/// Per-second poll of every driver's liveness. Returns the
/// human-readable "reason" string when one dies — used by `select!`
/// to label the shutdown reason in stdout. Walks both embedded
/// (thread) and subprocess (Python child) drivers uniformly via
/// [`DriverHandle::is_finished`].
///
/// On detected death this also flips the abort flag on every shmem
/// client. Without that, any `fire_batch` already in the runtime's
/// busy-spin would block for the full `PIE_SHMEM_TIMEOUT_S` (~5s)
/// before giving up — much longer than the watchdog's own 1s tick.
/// Aborting first means the in-flight call returns within microseconds,
/// the scheduler's batch task unwinds, and the shutdown sequence runs
/// without competing with stuck callers.
pub async fn watchdog(drivers: &[DriverHandle]) -> &'static str {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    interval.tick().await; // first tick fires immediately; skip it.
    loop {
        interval.tick().await;
        for d in drivers {
            if d.is_finished() {
                tracing::error!(
                    "driver {} exited unexpectedly; tearing down",
                    d.shmem_name(),
                );
                pie::device::abort_all_shmem_clients();
                return "driver exited unexpectedly";
            }
        }
    }
}

/// Best-effort `shm_unlink`. Logs non-`ENOENT` failures (a missing
/// segment is the success case — driver cleaned up first).
#[cfg(unix)]
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

/// Windows named mappings disappear when the last handle closes; there is
/// no `shm_unlink` equivalent.
#[cfg(windows)]
pub fn unlink_shmem(_name: &str) {}
