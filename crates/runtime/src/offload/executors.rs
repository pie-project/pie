//! The workers a process's passes may run on: the worker policy's
//! executors for it, else those the cluster's placement named, else the
//! peer this worker's configuration dialed. Workers named by id are reached
//! through the [`Dialer`] the worker's link layer installs.

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

pub fn set_dialer(dialer: Option<Arc<dyn Dialer>>) {
    *DIALER.write().unwrap() = dialer;
}

/// Whether passes can go anywhere but home.
pub fn reachable() -> bool {
    DIALER.read().unwrap().is_some() || super::prefill::has_peer()
}

pub(crate) fn set(pid: ProcessId, workers: Vec<String>) {
    if !workers.is_empty() {
        NAMED.lock().unwrap().insert(pid, workers);
    }
}

pub(crate) fn forget(pid: ProcessId) {
    NAMED.lock().unwrap().remove(&pid);
}

/// Where `pid`'s pass may go, best first; `policy` is the worker policy's
/// executors for it. A worker that cannot be reached is passed over.
pub(crate) async fn candidates(pid: ProcessId, policy: &[String]) -> Vec<Arc<dyn PrefillPeer>> {
    let named = match policy {
        [] => NAMED.lock().unwrap().get(&pid).cloned().unwrap_or_default(),
        some => some.to_vec(),
    };
    if named.is_empty() {
        return super::prefill::configured_peer().into_iter().collect();
    }
    let Some(dialer) = DIALER.read().unwrap().clone() else {
        return Vec::new();
    };
    let mut peers = Vec::with_capacity(named.len());
    for worker in named {
        match dialer.dial(worker.clone()).await {
            Ok(peer) => peers.push(peer),
            Err(error) => {
                tracing::debug!(target: "pie::offload", %worker, %error, "an executor not reached");
            }
        }
    }
    peers
}
