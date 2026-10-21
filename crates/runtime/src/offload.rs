//! Other workers' part in this one's passes: a pass's prefill run on a peer
//! (`prefill`, streamed as it is computed by `stream`), and pages another
//! worker's cache holds (`directory`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, RwLock};

pub mod directory;
pub mod executors;
pub mod prefill;
pub mod stream;

static PREFILL_MIN_SUFFIX_TOKENS: AtomicUsize = AtomicUsize::new(0);

static HOME_KV_HANDLE: LazyLock<RwLock<Option<engine::KvHandle>>> =
    LazyLock::new(|| RwLock::new(None));

/// The configuration's shortest suffix worth a peer's prefill (`0`: the
/// default).
pub fn configure(prefill_min_suffix_tokens: usize) {
    PREFILL_MIN_SUFFIX_TOKENS.store(prefill_min_suffix_tokens, Ordering::Release);
}

/// This worker's kv as its engine exported it, for peers to read in place.
pub fn set_home_kv_handle(handle: engine::KvHandle) {
    *HOME_KV_HANDLE.write().unwrap() = Some(handle);
}

pub fn home_kv_handle() -> Option<engine::KvHandle> {
    HOME_KV_HANDLE.read().unwrap().clone()
}

/// Below this many uncached tokens a pass is not worth sending away.
pub(crate) fn prefill_min_suffix_tokens() -> usize {
    PREFILL_MIN_SUFFIX_TOKENS.load(Ordering::Acquire)
}
