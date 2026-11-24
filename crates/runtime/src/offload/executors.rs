//! The workers a process's passes may run on: the worker policy's
//! executors for it, else those the cluster's placement named
//! (`placement/helpers`). Workers named by id are reached through the
//! [`Dialer`] the worker's link layer installs.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex, RwLock};

use anyhow::Result;
use futures::future::BoxFuture;

use super::prefill::PrefillPeer;
use crate::inferlet::process::ProcessId;

/// Reaches another worker by its cluster id, to send it passes.
pub trait Dialer: Send + Sync {
    fn dial(&self, worker: String) -> BoxFuture<'static, Result<Arc<dyn PrefillPeer>>>;
}

static DIALER: RwLock<Option<Arc<dyn Dialer>>> = RwLock::new(None);

/// The executors the cluster named, by process.
static NAMED: LazyLock<Mutex<HashMap<ProcessId, Vec<String>>>> = LazyLock::new(Mutex::default);

pub(crate) fn reset() {
    NAMED.lock().unwrap().clear();
    *DIALER.write().unwrap() = None;
}

pub fn set_dialer(dialer: Option<Arc<dyn Dialer>>) {
    *DIALER.write().unwrap() = dialer;
}

/// Whether passes can go anywhere but home.
pub fn reachable() -> bool {
    DIALER.read().unwrap().is_some()
}

pub(crate) fn set(pid: ProcessId, workers: Vec<String>) {
    if !workers.is_empty() {
        NAMED.lock().unwrap().insert(pid, workers);
    }
}

/// The cluster names `pid`'s helpers anew (empty: none, so its passes stay
/// home).
pub fn name(pid: ProcessId, workers: Vec<String>) {
    let mut named = NAMED.lock().unwrap();
    if workers.is_empty() {
        named.remove(&pid);
    } else {
        named.insert(pid, workers);
    }
}

pub(crate) fn forget(pid: ProcessId) {
    NAMED.lock().unwrap().remove(&pid);
}

/// Where `pid`'s pass may go, best first, each by the worker's id; `policy`
/// is the worker policy's executors for it. A worker that cannot be reached
/// (this one among them) is passed over.
pub(crate) async fn candidates(
    pid: ProcessId,
    policy: &[String],
) -> Vec<(String, Arc<dyn PrefillPeer>)> {
    let named = match policy {
        [] => NAMED.lock().unwrap().get(&pid).cloned().unwrap_or_default(),
        some => some.to_vec(),
    };
    if named.is_empty() {
        return Vec::new();
    }
    let Some(dialer) = DIALER.read().unwrap().clone() else {
        return Vec::new();
    };
    let mut peers = Vec::with_capacity(named.len());
    for worker in named {
        match dialer.dial(worker.clone()).await {
            Ok(peer) => peers.push((worker, peer)),
            Err(error) => {
                tracing::debug!(target: "pie::offload", %worker, %error, "an executor not reached");
            }
        }
    }
    peers
}
