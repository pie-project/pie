//! KV forward-prepare projection: a pure view over the store's resolved
//! physical pages, so it lives in `store/kv/` even though `pipeline/fire/`
//! is what calls it.
//!
//! Pure projection at the heart of the forward transaction's `prepare` step:
//! it takes the resolved physical pages (from charlie's `KvWorkingSet::resolve_*`
//! over bravo's arena) plus the inferlet's `kv-output` per-page valid lengths and
//! produces the `(physical_page_ids, last_page_len, active_page_idx)` triple that
//! the scheduler/`wire` build path consumes. It enforces the brief §5 forward-contract
//! validations and the v1 dense-array contiguity rule (the v1 driver ABI's
//! `kv_page_indices` / `kv_last_page_lens` express only a contiguous ordered
//! active page run) and owns the seal-eligibility split (full vs partial pages, W7).
//!
//! `PrepareError` lives here (not `pipeline::fire::kv`) because it is
//! `project_kv`'s error type and `store/` must not import upward into
//! `pipeline/`; `pipeline::fire::kv::{check_generation, check_input_nonempty}`
//! import it from here (the WIT-descriptor validation half of the fire
//! `prepare` gate).

/// Physical block id within one pool's id-space.
pub type BlockId = u32;

/// A physical KV page id the driver consumes (arena `KvPage` block = one page).
pub type PhysicalPageId = BlockId;

#[derive(Debug, PartialEq, Eq)]
pub enum PrepareError {
    /// A `kv-output` descriptor was built against a since-mutated working set
    /// (captured generation no longer matches). Rejected before any arena work.
    StaleGeneration { captured: u32, current: u32 },
    /// A per-page valid length is 0 or exceeds `page_size`.
    InvalidValidLen {
        index: u32,
        valid_len: u32,
        page_size: u32,
    },
    /// An output slot index appears more than once in one pass.
    DuplicateOutputIndex(u32),
    /// The active run `[0, active_len)` has a slot covered by neither context
    /// nor output — the v1 ABI's contiguous ordered page list can't express it.
    NonContiguousActiveRun { gap_at: u32 },
    /// Nothing to read and nothing to write.
    EmptyForward,
    /// A forward pass supplied no input rows (no text tokens, image, or audio
    /// span): the driver `qo_indptr` would collapse to `[0, 0]` and the pass
    /// would be a no-op. The old context API rejected this as "empty input".
    NoInputTokens,
}

/// One resolved KV write target (post-CoW physical page) for this pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvWrite {
    /// Relative slot index in the dense `kv-working-set` array.
    pub slot_index: u32,
    /// Resolved (post-CoW) physical page the driver writes into.
    pub page: PhysicalPageId,
    /// Tokens written into this page (`kv-output.per-page-valid-lens[i]`).
    pub valid_len: u32,
}

/// The triple the downstream submit path expects, plus seal-eligibility.
#[derive(Debug, PartialEq, Eq)]
pub struct KvProjection {
    /// Ordered active page run `[0, active_len)`.
    pub physical_page_ids: Vec<PhysicalPageId>,
    /// Valid tokens in the final active page.
    pub last_page_len: u32,
    /// Index (in `physical_page_ids`) of the last page that receives writes;
    /// `None` for a read-only forward (no `kv-output`).
    pub active_page_idx: Option<usize>,
    /// Output slot indices whose page is full (`valid_len == page_size`) and is
    /// therefore CAS-seal-eligible at commit. Partial pages stay private (W7).
    pub full_page_writes: Vec<u32>,
}

/// Project explicit KV read/write descriptors onto the driver's contiguous
/// active page run.
///
/// `context_pages` are the resolved physical pages for read slots
/// `[0, context_pages.len())` (v1 requires the read window to start at slot 0;
/// RoPE positions run from 0). `context_valid_tokens` is how many tokens are
/// valid across that window. `writes` are the resolved (post-CoW) output
/// targets. `physical_page_ids` covers the contiguous active run `[0, active_len)`
/// where `active_len = max(context_len, max_output_slot + 1)`; every slot in that
/// run must be backed by either an output target or a context page (no gaps).
pub fn project_kv(
    context_pages: &[PhysicalPageId],
    context_valid_tokens: u32,
    writes: &[KvWrite],
    page_size: u32,
) -> Result<KvProjection, PrepareError> {
    if context_pages.is_empty() && writes.is_empty() {
        return Err(PrepareError::EmptyForward);
    }

    let context_len = context_pages.len() as u32;
    let mut max_write_slot: Option<u32> = None;
    let mut seen: Vec<u32> = Vec::with_capacity(writes.len());
    for wr in writes {
        if wr.valid_len == 0 || wr.valid_len > page_size {
            return Err(PrepareError::InvalidValidLen {
                index: wr.slot_index,
                valid_len: wr.valid_len,
                page_size,
            });
        }
        if seen.contains(&wr.slot_index) {
            return Err(PrepareError::DuplicateOutputIndex(wr.slot_index));
        }
        seen.push(wr.slot_index);
        max_write_slot = Some(max_write_slot.map_or(wr.slot_index, |m| m.max(wr.slot_index)));
    }

    let active_len = context_len.max(max_write_slot.map_or(0, |m| m + 1));

    // Assemble the contiguous active page run.
    let mut physical_page_ids = Vec::with_capacity(active_len as usize);
    for slot in 0..active_len {
        if let Some(wr) = writes.iter().find(|w| w.slot_index == slot) {
            physical_page_ids.push(wr.page);
        } else if slot < context_len {
            physical_page_ids.push(context_pages[slot as usize]);
        } else {
            return Err(PrepareError::NonContiguousActiveRun { gap_at: slot });
        }
    }

    // last_page_len = valid tokens in the final active page.
    let last_slot = active_len - 1;
    let last_page_len = if let Some(wr) = writes.iter().find(|w| w.slot_index == last_slot) {
        wr.valid_len
    } else {
        // Final slot is a read-only context tail page.
        let consumed = last_slot * page_size;
        let rem = context_valid_tokens.saturating_sub(consumed);
        if rem == 0 || rem > page_size {
            page_size
        } else {
            rem
        }
    };

    let active_page_idx = max_write_slot.map(|m| m as usize);

    let full_page_writes = writes
        .iter()
        .filter(|w| w.valid_len == page_size)
        .map(|w| w.slot_index)
        .collect();

    Ok(KvProjection {
        physical_page_ids,
        last_page_len,
        active_page_idx,
        full_page_writes,
    })
}
