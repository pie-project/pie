//! KV page content hashing for CAS dedup (W6/W7).
//!
//! Relocated from the retired `context::pagestore` in Phase 5 to a neutral home
//! that survives the context teardown. Pure: chains a per-page content hash
//! (tokens + positions + masks + adapter_seed) from a `prev_hash`, so equal
//! chained hashes imply equal content *and* equal prefix — the key the KV CAS
//! index (`working_set::kv_cas`) dedups on.

use std::hash::{Hash, Hasher};

use ahash::AHasher;
use pie_grammar::brle::RunMask;

/// A chained KV page content hash.
pub type PageHash = u64;

/// Compute the chained content hash of each `page_size`-token page covered by
/// `tokens`/`positions`/`masks`, starting the chain at `prev_hash`. `adapter_seed`
/// is folded in so ZO-perturbed pages with different seeds never dedup-share.
pub fn compute_page_hashes(
    page_size: usize,
    tokens: &[u32],
    positions: &[u32],
    masks: &[RunMask],
    prev_hash: PageHash,
    adapter_seed: Option<i64>,
) -> Vec<PageHash> {
    let mut hashes = Vec::new();
    let mut running = prev_hash;

    for (chunk_idx, chunk) in tokens.chunks(page_size).enumerate() {
        let start = chunk_idx * page_size;
        let end = start + chunk.len();
        let chunk_pos = &positions[start..end];
        let chunk_masks = &masks[start..end];

        let mut hasher = AHasher::default();
        chunk.hash(&mut hasher);
        for pos in chunk_pos {
            pos.hash(&mut hasher);
        }
        for mask in chunk_masks {
            mask.hash(&mut hasher);
        }
        adapter_seed.hash(&mut hasher);
        let content_hash = hasher.finish();

        let mut chain_hasher = AHasher::default();
        content_hash.hash(&mut chain_hasher);
        running.hash(&mut chain_hasher);
        let page_hash = chain_hasher.finish();

        hashes.push(page_hash);
        running = page_hash;
    }

    hashes
}
