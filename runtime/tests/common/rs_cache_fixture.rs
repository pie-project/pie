//! Shared fixtures for the rs_cache integration tests.
//!
//! The multi-model `ServiceArray` is gone — `context::spawn` now installs a
//! single per-process context-manager singleton. Each rs_cache config
//! (1-slot / 4-slot-replay / 0-slot-kv-only) therefore lives in its own test
//! binary (its own process → its own singleton). This module holds the
//! recording mock backend and the context helpers those binaries share.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use super::mock_device::Behavior;

/// Mock backend that records every forward request so tests can assert on the
/// rs_cache RESET flags / slot ids the runtime emits.
pub struct RecordingBehavior {
    forwards: Mutex<Vec<pie_driver_abi::ForwardRequest>>,
}

impl RecordingBehavior {
    pub fn new() -> Self {
        Self {
            forwards: Mutex::new(Vec::new()),
        }
    }

    pub fn clear(&self) {
        self.forwards.lock().unwrap().clear();
    }

    pub fn forwards(&self) -> Vec<pie_driver_abi::ForwardRequest> {
        self.forwards.lock().unwrap().clone()
    }
}

impl Behavior for RecordingBehavior {
    fn handle_fire_batch(
        &self,
        req: &pie_driver_abi::ForwardRequest,
    ) -> pie_driver_abi::ForwardResponse {
        self.forwards.lock().unwrap().push(req.clone());
        let n = req.qo_indptr.len().saturating_sub(1) as u32;
        pie_driver_abi::ForwardResponse {
            num_requests: n,
            tokens_indptr: (0..=n).collect(),
            tokens: vec![7; n as usize],
            dists_req_indptr: vec![0; (n + 1) as usize],
            dists_kv_indptr: vec![0],
            dists_ids: Vec::new(),
            dists_probs: Vec::new(),
            logits_req_indptr: vec![0; (n + 1) as usize],
            logits_byte_indptr: vec![0],
            logits_bytes: Vec::new(),
            logprobs_req_indptr: vec![0; (n + 1) as usize],
            logprobs_val_indptr: vec![0],
            logprobs_values: Vec::new(),
            entropies_indptr: vec![0; (n + 1) as usize],
            entropies: Vec::new(),
            spec_indptr: vec![0; (n + 1) as usize],
            spec_tokens: Vec::new(),
            spec_positions: Vec::new(),
            ..Default::default()
        }
    }
}

/// Spawn the singleton context manager + inference service for one rs_cache
/// config. `rs_slots` = rs_cache slots on the single mock driver; `restore_pause`
/// = restore-pause-at-utilization (use a negative value to keep suspended
/// replay sources off-GPU so fork/take exercise direct lineage replay).
pub async fn spawn_managers(rs_slots: usize, restore_pause: f64) {
    pie::context::spawn(
        4,
        vec![32],
        vec![32],
        vec![rs_slots],
        vec![false],
        restore_pause,
    );
    pie::inference::spawn(&[0], 4, 30, 0).await;
}

static LAUNCH_SEQ: AtomicU64 = AtomicU64::new(0);

fn next_launch_seq() -> u64 {
    LAUNCH_SEQ.fetch_add(1, Ordering::Relaxed)
}

/// Register a fresh process and return its pid. The context actor panics on
/// `create`/`pin` if the process isn't registered first.
pub async fn fresh_pid() -> uuid::Uuid {
    let pid = uuid::Uuid::new_v4();
    pie::context::register_process(pid, next_launch_seq())
        .await
        .unwrap();
    pid
}

/// Drain any pending fire-and-forget messages on the context actor (FIFO).
pub async fn flush_mailbox(id: pie::context::ContextId) {
    let _ = pie::context::debug_context_state(id).await;
}

/// Create a context, pin/unpin to assign a fresh rs slot, then append + commit
/// one page so it has committed rs-cache lineage available to replay.
pub async fn make_committed_rs_context() -> pie::context::ContextId {
    let id = pie::context::create(fresh_pid().await).await.unwrap();
    let pinned = pie::context::pin(id, 0).await.unwrap();
    assert_eq!(pinned.rs_flags, 1);
    pie::context::unpin(id);
    flush_mailbox(id).await;

    pie::context::append_working_page_tokens(
        id,
        vec![10, 11, 12, 13],
        vec![0, 1, 2, 3],
        vec![],
        None,
        None,
    );
    flush_mailbox(id).await;
    pie::context::reserve_working_pages(id, 1).await.unwrap();
    pie::context::commit_working_pages(id, 1).await.unwrap();
    id
}

/// Poll until a context finishes lineage replay (Active + no pending replay).
pub async fn wait_for_replay(id: pie::context::ContextId) {
    for _ in 0..100 {
        let debug = pie::context::debug_context_state(id).await;
        if debug.contains("state: Active") && debug.contains("pending_replay: false") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "replay did not complete: {}",
        pie::context::debug_context_state(id).await
    );
}
