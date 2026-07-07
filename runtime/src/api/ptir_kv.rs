//! PTIR pipeline KV projection (§6.2 / §6.1) — the ws-alloc + arena-txn +
//! `project_kv` lifecycle behind `ptir_host`'s forward. Delta's
//! `ptir_host::submit` calls [`ptir_kv_prepare`] to get the real driver page
//! geometry (fixing the `physical_page_ids=[]` crash), threads it into
//! `submit_async`, holds the returned [`PtirKvTxn`] across the async fire, then
//! [`ptir_kv_finalize`]s it (commit → KV persists for the next fire / abort →
//! revert). The commit/seal + growing-KV lifecycle — the crash-prone part — is
//! owned here.
//!
//! Delta's shipped minimal (`8c895486`) is single-token fresh prefill
//! (`committed_tokens = 0`, one write slot, no prior context). This module
//! GENERALIZES it: prior-context `resolve_read` (`committed_tokens > 0`),
//! multi-token writes (MTP K drafts), and the working set GROWING one-or-more
//! slots per fire — real decode loops, §6.2 beam, §6.1 mtpverify.

use crate::arena::{Arena, ArenaTxn, MovePlan};
use crate::inference::forward_prepare::{project_kv, KvProjection, KvWrite, PhysicalPageId};
use crate::working_set::kv::{KvWorkingSet, WriteTxnId};

/// The open KV/arena transactions for one in-flight PTIR fire — held across
/// `submit_async` until [`ptir_kv_finalize`]. ONE handle: it bundles the S4
/// write-txn id with the owned arena txn (`commit_writes`/`abort_writes` need the
/// id, and there is no commit-all), so the caller threads a single value through
/// the async boundary.
pub struct PtirKvTxn {
    arena_txn: ArenaTxn,
    write_txn: WriteTxnId,
    /// The working set's committed token length AFTER this fire commits
    /// (`committed_tokens + new_tokens.len()`). The caller stores it on the
    /// pipeline (the growing cursor) and passes it as `committed_tokens` next fire.
    pub committed_tokens_after: u32,
}

/// Prepare the KV projection for a PTIR fire that appends `new_tokens` to a
/// persistent, per-instance `KvWorkingSet` currently holding `committed_tokens`
/// committed tokens (BOS/fresh prefill = `(0, &[bos])`; a decode = `(k, &[t])`; an
/// MTP verify = `(k, &[d0, d1, …])`). Grows the ws so the write slots exist,
/// `resolve_read`s + pins the prior context, `cow_write_slot`s the new-token
/// slots, and projects the driver geometry.
///
/// Returns `(proj, move_plans, txn)`: pass `proj.physical_page_ids` /
/// `proj.last_page_len` / `move_plans` into `submit_async`, hold `txn` across the
/// fire, then [`ptir_kv_finalize`]. `new_tokens`' VALUES are unused here — the
/// projection is pure page geometry keyed by the count; the token ids ride the
/// `ForwardRequest`. `move_plans` is empty for the single-context pipeline (fresh
/// BOS page + in-place decode appends); non-empty only under a forked/shared page.
pub fn ptir_kv_prepare(
    ws: &mut KvWorkingSet,
    committed_tokens: u32,
    new_tokens: &[u32],
    arena: &mut Arena,
    page_size: u32,
) -> Result<(KvProjection, Vec<MovePlan>, PtirKvTxn), String> {
    let n_new = new_tokens.len() as u32;
    if n_new == 0 {
        return Err("ptir_kv_prepare: new_tokens must be non-empty".to_string());
    }
    let total = committed_tokens + n_new;
    let needed_pages = total.div_ceil(page_size);

    let mut arena_txn = arena.txn_begin();

    // Grow the ws so every write slot exists (Reserved). Single-context pipeline:
    // live size == slot count (no frees).
    if ws.size() < needed_pages {
        ws.alloc(needed_pages - ws.size())
            .map_err(|e| e.to_string())?;
    }

    // Prior context: pages [0, valid_pages) for the committed tokens (pinned).
    let valid_pages = committed_tokens.div_ceil(page_size);
    let mut context_pages: Vec<PhysicalPageId> = Vec::with_capacity(valid_pages as usize);
    if valid_pages > 0 {
        let objs = ws.resolve_read(0, valid_pages).map_err(|e| e.to_string())?;
        for obj in &objs {
            arena
                .txn_pin(&mut arena_txn, *obj)
                .map_err(|e| e.to_string())?;
            context_pages.push(arena.blocks(*obj).map_err(|e| e.to_string())?[0]);
        }
    }

    // CoW-write the new-token slots [output_start, needed_pages). `offset` is the
    // in-page index of the first new token (prior valid tokens in the first output
    // page, preserved by the CoW/in-place write).
    let write_txn = ws.begin_write_txn();
    let output_start = committed_tokens / page_size;
    let offset = committed_tokens % page_size;
    let mut writes: Vec<KvWrite> = Vec::new();
    let mut move_plans: Vec<MovePlan> = Vec::new();
    for slot in output_start..needed_pages {
        let (obj, mp) = ws
            .cow_write_slot(write_txn, slot, &mut arena_txn, arena)
            .map_err(|e| e.to_string())?;
        if let Some(mp) = mp {
            move_plans.push(mp);
        }
        arena
            .txn_pin(&mut arena_txn, obj)
            .map_err(|e| e.to_string())?;
        let page = arena.blocks(obj).map_err(|e| e.to_string())?[0];
        let i = slot - output_start;
        let valid_len = (offset + n_new)
            .saturating_sub(i * page_size)
            .min(page_size);
        writes.push(KvWrite {
            slot_index: slot,
            page,
            valid_len,
        });
    }

    let proj = project_kv(&context_pages, committed_tokens, &writes, page_size)
        .map_err(|e| format!("{e:?}"))?;

    Ok((
        proj,
        move_plans,
        PtirKvTxn {
            arena_txn,
            write_txn,
            committed_tokens_after: total,
        },
    ))
}

/// Finalize a PTIR fire's KV txns after `submit_async` resolves. `success`
/// commits (arena pages + slot writes persist for the next fire); otherwise aborts
/// (revert both). Mirrors `InstanceState::finalize_forward_txn`.
pub fn ptir_kv_finalize(
    ws: &mut KvWorkingSet,
    arena: &mut Arena,
    txn: PtirKvTxn,
    success: bool,
) -> Result<(), String> {
    if success {
        arena.txn_commit(txn.arena_txn).map_err(|e| e.to_string())?;
        ws.commit_writes(txn.write_txn);
    } else {
        arena.txn_abort(txn.arena_txn);
        ws.abort_writes(txn.write_txn);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u32 = 4;

    fn arena(kv_pages: u32) -> Arena {
        Arena::new(crate::arena::ArenaConfig {
            device: 0,
            block_size: PAGE,
            kv_pages,
            rs_blocks: 0,
            scratch_blocks: 0,
            cpu_blocks: 0,
        })
    }

    fn toks(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn fresh_prefill_single_token_is_delta_minimal() {
        // Delta's shipped minimal: committed 0, one write slot, no prior context.
        let mut a = arena(8);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        let (proj, move_plans, txn) =
            ptir_kv_prepare(&mut ws, 0, &toks(1), &mut a, PAGE).unwrap();
        assert_eq!(proj.physical_page_ids.len(), 1); // NON-EMPTY (the crash fix)
        assert_eq!(proj.last_page_len, 1);
        assert!(move_plans.is_empty());
        assert_eq!(txn.committed_tokens_after, 1);
        ptir_kv_finalize(&mut ws, &mut a, txn, true).unwrap();
    }

    #[test]
    fn growing_decode_loop_single_token() {
        // BOS + 5 single-token decodes: the ws grows + persists across the page
        // boundary; every fire's projection is non-empty with correct geometry.
        let mut a = arena(8);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        let mut committed = 0u32;
        for step in 0..6u32 {
            let (proj, _mp, txn) =
                ptir_kv_prepare(&mut ws, committed, &toks(1), &mut a, PAGE).unwrap();
            assert!(!proj.physical_page_ids.is_empty(), "step {step}");
            let want_pages = (committed + 1).div_ceil(PAGE);
            assert_eq!(proj.physical_page_ids.len() as u32, want_pages, "step {step}");
            assert_eq!(
                proj.last_page_len,
                (committed + 1) - (want_pages - 1) * PAGE,
                "step {step}"
            );
            committed = txn.committed_tokens_after;
            ptir_kv_finalize(&mut ws, &mut a, txn, true).unwrap();
        }
        assert_eq!(committed, 6);
        assert_eq!(ws.size(), 2);
    }

    #[test]
    fn growing_multi_token_mtpverify() {
        // §6.1 mtpverify: K>1 tokens per fire (draft width), growing across page
        // boundaries with correct partial/full-page geometry.
        const K: u32 = 3;
        let mut a = arena(16);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        let mut committed = 0u32;
        for step in 0..4u32 {
            let (proj, _mp, txn) =
                ptir_kv_prepare(&mut ws, committed, &toks(K), &mut a, PAGE).unwrap();
            let total = committed + K;
            let want_pages = total.div_ceil(PAGE);
            assert_eq!(proj.physical_page_ids.len() as u32, want_pages, "step {step}");
            assert_eq!(proj.last_page_len, total - (want_pages - 1) * PAGE, "step {step}");
            committed = txn.committed_tokens_after;
            ptir_kv_finalize(&mut ws, &mut a, txn, true).unwrap();
        }
        assert_eq!(committed, 12);
        assert_eq!(ws.size(), 3);
    }

    #[test]
    fn abort_then_reprepare_is_consistent() {
        let mut a = arena(8);
        let mut ws = KvWorkingSet::new(PAGE, 0);
        let (_p, _m, t0) = ptir_kv_prepare(&mut ws, 0, &toks(1), &mut a, PAGE).unwrap();
        ptir_kv_finalize(&mut ws, &mut a, t0, true).unwrap();
        let (_p, _m, t1) = ptir_kv_prepare(&mut ws, 1, &toks(1), &mut a, PAGE).unwrap();
        ptir_kv_finalize(&mut ws, &mut a, t1, false).unwrap(); // abort
        let (proj, _m, t2) = ptir_kv_prepare(&mut ws, 1, &toks(1), &mut a, PAGE).unwrap();
        assert!(!proj.physical_page_ids.is_empty());
        assert_eq!(proj.last_page_len, 2);
        ptir_kv_finalize(&mut ws, &mut a, t2, true).unwrap();
    }
}
