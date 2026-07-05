//! Snapshot manifests — the thin **data + wasi:filesystem I/O** keep-core.
//!
//! A Pie "snapshot" is **not** a device KV blob — it is a CPU-resident
//! **manifest**: a materialized token log plus the unflushed buffer and a bit
//! of geometry, serialized to a file under the per-instance `/scratch` preopen.
//! Restoring a snapshot is a **token-log REPLAY** — a prefill forward pass over
//! `tokens` that rebuilds the KV. That replay is an ordinary prefill on the
//! carrier keep-core (`carrier::submit_pass` / `prefill::tokens`), so it lives
//! in the inferlet's decode loop — **not** here. This module keeps only the
//! genuinely-primitive part: the [`SnapshotData`] struct (pure serde) plus the
//! five thin `save`/`snapshot`/`open`/`take`/`delete` free functions over
//! `std::fs`.
//!
//! This is the snapshot analog of the `geometry` / `carrier` / `sampler`
//! keep-core primitives (see `ptir-sdk-minimization-audit` /
//! `ptir-snapshot-keepcore-spec`): the `Context::save/open/snapshot/take/delete`
//! facade dies; this thin data+I/O core survives, and the replay factors into
//! the inferlet's normal carrier prefill.
//!
//! **Boundary — multimodal:** [`SnapshotData`] is token-only, so a multimodal
//! context (soft-token KV that cannot be rebuilt from a token log) simply has no
//! snapshot to take — the same v1 limit the old facade enforced explicitly.
//! **Boundary — CAS-reattach:** the physical reuse-sealed-pages-by-hash path is
//! a future runtime attach-by-cas op; `cas_hashes` is kept empty for
//! forward-compat.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Snapshot manifest schema version. Bumped when the on-disk layout changes;
/// [`open`] rejects a mismatch loudly.
pub const SNAPSHOT_VERSION: u32 = 1;

static SNAPSHOT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The snapshot manifest: a materialized token log + unflushed tail + geometry.
/// Pure data (serde) — no device or `Context` state. Restore is a REPLAY of
/// `tokens` through a prefill on the carrier keep-core (done by the inferlet).
#[derive(Serialize, Deserialize, Clone)]
pub struct SnapshotData {
    /// Manifest schema version (must equal [`SNAPSHOT_VERSION`] on open).
    pub version: u32,
    /// KV tokens-per-page geometry the log was materialized under.
    pub page_size: u32,
    /// Materialized sequence length (tokens in `tokens`, excluding `buffer`).
    pub seq_len: u32,
    /// The materialized token log to replay (a single prefill rebuilds the KV).
    pub tokens: Vec<u32>,
    /// Unflushed tail — appended after the replay, not yet prefilled.
    pub buffer: Vec<u32>,
    /// A deferred `chat::system` prompt not yet folded into the buffer.
    pub pending_system: Option<String>,
    /// Reserved for the future physical CAS-reattach path; empty in v1.
    pub cas_hashes: Vec<u64>,
}

fn snapshot_path(name: &str) -> String {
    // The runtime preopens the per-instance scratch dir as `/scratch` in the
    // guest (runtime/src/instance.rs); a relative path has no matching preopen,
    // so snapshot blobs must be written there.
    format!("/scratch/{name}.pie-snapshot")
}

/// Serialize `data` to the named snapshot blob (under the `/scratch` preopen).
pub fn save(name: &str, data: &SnapshotData) -> Result<()> {
    let bytes =
        serde_json::to_vec(data).map_err(|e| format!("snapshot '{name}': serialize: {e}"))?;
    std::fs::write(snapshot_path(name), bytes)
        .map_err(|e| format!("snapshot '{name}': write: {e}"))?;
    Ok(())
}

/// Anonymous save — serialize `data` under a freshly-generated unique name and
/// return it (instance-scoped, so concurrent instances never collide).
pub fn snapshot(data: &SnapshotData) -> Result<String> {
    let name = format!(
        "anon-{}-{}",
        crate::runtime::instance_id(),
        SNAPSHOT_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    save(&name, data)?;
    Ok(name)
}

/// Read + deserialize a snapshot blob (the blob stays on disk — an implicit
/// fork). Synchronous: the replay forward pass belongs to the inferlet. A
/// missing file or version mismatch is a loud error.
pub fn open(name: &str) -> Result<SnapshotData> {
    let bytes =
        std::fs::read(snapshot_path(name)).map_err(|e| format!("snapshot '{name}': read: {e}"))?;
    let data: SnapshotData =
        serde_json::from_slice(&bytes).map_err(|e| format!("snapshot '{name}': parse: {e}"))?;
    if data.version != SNAPSHOT_VERSION {
        return Err(format!(
            "snapshot '{name}': version {} unsupported (expected {SNAPSHOT_VERSION})",
            data.version
        ));
    }
    Ok(data)
}

/// Take ownership of a snapshot: [`open`] it, then delete the blob.
pub fn take(name: &str) -> Result<SnapshotData> {
    let data = open(name)?;
    let _ = std::fs::remove_file(snapshot_path(name));
    Ok(data)
}

/// Delete a snapshot blob by name. A missing blob is a no-op.
pub fn delete(name: &str) -> Result<()> {
    let _ = std::fs::remove_file(snapshot_path(name));
    Ok(())
}
