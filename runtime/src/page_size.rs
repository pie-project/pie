//! Per-model KV page size (tokens per page).
//!
//! Relocated from the retired `context` module (Phase 5) to a neutral home that
//! survives the teardown. The page size is the unified arena's block size for
//! the model's primary driver (v1: one KV page == one arena `KvPage` block).

/// Tokens per KV page for `model_idx` (0 if the model has no registered arena).
pub fn tokens_per_page(model_idx: usize) -> u32 {
    crate::arena::try_get(model_idx, 0)
        .map(|a| a.lock().expect("arena lock poisoned").block_size())
        .unwrap_or(0)
}

/// Valid token count in the last of `num_pages` pages holding `total_kv` tokens.
/// Relocated from the retired `context::pagestore` (Phase 5).
pub fn compute_last_page_len(total_kv: u32, num_pages: u32, page_size: u32) -> u32 {
    if num_pages == 0 {
        0
    } else {
        let r = total_kv % page_size;
        if r == 0 { page_size } else { r }
    }
}
