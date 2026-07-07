//! PTIR pipeline recurrent-state (rs_cache) slot assignment (§6.1 MTP GDN) — the
//! RS sibling of `ptir_kv`. A linear-attention / GDN model (Qwen3.5 GDN,
//! Nemotron-H Mamba2 — `executor.rs_cache != nullptr`) requires every forward to
//! carry one runtime-assigned recurrent-state slot per request in
//! `ForwardRequest.rs_slot_ids`, else the executor throws "rs_cache forward
//! missing runtime-assigned slot ids" (executor.cpp:2671). The normal
//! `execute_impl` path populates this from a WIT-bound `RsWorkingSet`; the PTIR
//! pipeline has no such WIT call, so this module assigns the slot itself —
//! mirroring `execute_impl`'s rs block, the RS analog of `ptir_kv`'s KV
//! projection.
//!
//! Lifecycle: `ptir_host` allocates a persistent per-instance `RsWorkingSet` at
//! `instantiate` (alongside the `KvWorkingSet`); each fire [`ptir_rs_prepare`]
//! stages the folded recurrent-state write slot (a fresh RESET slab on the first
//! fire, a CoW-continue of the prior state after — the recurrent state GROWS
//! across the decode loop), threads it into `submit_async` as `rs_slot_ids` /
//! `rs_slot_flags`, holds the returned [`PtirRsTxn`] across the async fire, then
//! [`ptir_rs_finalize`]s it (commit → the written state persists for the next
//! fire / abort → revert).
//!
//! Driver contract (handed to charlie): `rs_slot_ids[r]` = the folded slot's
//! driver block — the GDN forward reads/writes the recurrent state THERE
//! in-forward (the `commit_len` primitive); `rs_slot_flags[r] & RS_FLAG_RESET`
//! on the fresh first fire (the driver zeroes the state before writing). The
//! buffered fold-from-slabs channel (`rs_buffer_slot_ids`) is the parked Ph7
//! extension — NOT needed for the in-forward write.

use crate::arena::{Arena, ArenaTxn, MovePlan};
use crate::working_set::rs::{RsWorkingSet, RsWritePlan};

/// The open recurrent-state / arena transaction for one in-flight PTIR fire —
/// held across `submit_async` until [`ptir_rs_finalize`]. Bundles the owned arena
/// txn with the [`RsWritePlan`] (`commit_write` needs the plan's `folded_slot` to
/// adopt), so the caller threads a single value through the async boundary — the
/// RS mirror of `ptir_kv`'s `PtirKvTxn`.
pub struct PtirRsTxn {
    arena_txn: ArenaTxn,
    plan: RsWritePlan,
}

/// Prepare the recurrent-state slot for a PTIR fire on a persistent, per-instance
/// `RsWorkingSet`. Stages the folded write target (fresh alloc + RESET on the
/// first fire, CoW-continue of the prior state after) and returns the wire
/// columns for a single-request fire:
///
/// Returns `(rs_slot_ids, rs_slot_flags, cow_move, txn)`: pass `rs_slot_ids` /
/// `rs_slot_flags` into the `ForwardRequest` (parallel to the KV geometry), issue
/// `cow_move` as a d2d copy before the fire when present (a shared-fork CoW; empty
/// for the single-context pipeline), hold `txn` across the fire, then
/// [`ptir_rs_finalize`]. Single-request (R=1) — the PTIR pipeline fires one
/// sequence per forward; the executor's `rs_slot_ids.size() == R` gate is thus
/// satisfied with one entry.
pub fn ptir_rs_prepare(
    rs_ws: &mut RsWorkingSet,
    arena: &mut Arena,
) -> Result<(Vec<u32>, Vec<u8>, Option<MovePlan>, PtirRsTxn), String> {
    let (plan, arena_txn) = rs_ws.prepare_write(arena).map_err(|e| e.to_string())?;
    let block = arena.blocks(plan.folded_slot).map_err(|e| e.to_string())?[0];
    let rs_slot_ids = vec![block];
    let flag = if plan.reset {
        pie_driver_abi::RS_FLAG_RESET
    } else {
        0u8
    };
    let rs_slot_flags = vec![flag];
    let cow_move = plan.cow_move.clone();
    Ok((
        rs_slot_ids,
        rs_slot_flags,
        cow_move,
        PtirRsTxn { arena_txn, plan },
    ))
}

/// Finalize a PTIR fire's recurrent-state txn after `submit_async` resolves.
/// `success` commits (the written folded state persists as the recurrent state
/// for the next fire); otherwise aborts (revert the staged slab / CoW copy, the
/// prior state stays visible). Mirrors [`ptir_kv_finalize`](super::ptir_kv) /
/// `execute_impl`'s finalize.
pub fn ptir_rs_finalize(
    rs_ws: &mut RsWorkingSet,
    arena: &mut Arena,
    txn: PtirRsTxn,
    success: bool,
) -> Result<(), String> {
    if success {
        rs_ws
            .commit_write(arena, txn.arena_txn, &txn.plan)
            .map_err(|e| e.to_string())?;
    } else {
        rs_ws.abort_write(arena, txn.arena_txn);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{Arena, ArenaConfig};
    use crate::working_set::rs::{RsGeometry, RsWorkingSet};

    const STATE_BLOCKS: u32 = 1;

    fn arena(rs_blocks: u32) -> Arena {
        Arena::new(ArenaConfig {
            device: 0,
            block_size: 4,
            kv_pages: 0,
            rs_blocks,
            scratch_blocks: 0,
            cpu_blocks: 0,
        })
    }

    fn ws() -> RsWorkingSet {
        RsWorkingSet::new(
            0,
            RsGeometry {
                state_size: 64,
                state_blocks: STATE_BLOCKS,
                buffer_page_tokens: 4,
                fold_granularity: 1,
            },
        )
    }

    /// Fresh first fire: allocates a real folded slot (a valid driver block, NOT
    /// 0) and flags RESET so the driver zeroes the recurrent state before the
    /// in-forward write — the fix that clears the executor's `size == R` gate.
    #[test]
    fn fresh_fire_allocates_slot_and_flags_reset() {
        let mut a = arena(8);
        let mut w = ws();
        let (ids, flags, cow_move, txn) = ptir_rs_prepare(&mut w, &mut a).unwrap();
        assert_eq!(ids.len(), 1, "one folded slot for the single-request fire");
        assert_eq!(flags, vec![pie_driver_abi::RS_FLAG_RESET], "fresh ⇒ RESET");
        assert!(cow_move.is_none(), "fresh alloc ⇒ no CoW d2d");
        assert!(w.folded_object().is_none(), "not adopted until commit");
        ptir_rs_finalize(&mut w, &mut a, txn, true).unwrap();
        assert!(w.folded_object().is_some(), "committed ⇒ folded slot adopted");
    }

    /// A continuing decode fire (folded state already committed) CoW-continues the
    /// prior state: NO reset (prior recurrent state is read in), a valid slot id,
    /// and the state persists across fires — the growing recurrent-state loop.
    #[test]
    fn continuing_fire_no_reset_and_persists() {
        let mut a = arena(8);
        let mut w = ws();
        // Fire 1 (fresh): commit so the folded state exists.
        let (ids0, flags0, _m0, t0) = ptir_rs_prepare(&mut w, &mut a).unwrap();
        assert_eq!(flags0, vec![pie_driver_abi::RS_FLAG_RESET]);
        ptir_rs_finalize(&mut w, &mut a, t0, true).unwrap();
        let slot0 = ids0[0];

        // Fire 2 (continuing): CoW the prior folded state, no reset.
        let (ids1, flags1, _m1, t1) = ptir_rs_prepare(&mut w, &mut a).unwrap();
        assert_eq!(ids1.len(), 1);
        assert_eq!(flags1, vec![0u8], "continuing fire ⇒ no RESET");
        ptir_rs_finalize(&mut w, &mut a, t1, true).unwrap();
        assert!(w.folded_object().is_some(), "state still present after fire 2");
        let _ = slot0;
    }

    /// Multi-fire decode loop: 6 fires, only the first resets; the folded state is
    /// adopted every fire (the persistent, growing recurrent state).
    #[test]
    fn decode_loop_resets_once() {
        let mut a = arena(8);
        let mut w = ws();
        for step in 0..6u32 {
            let (ids, flags, _m, txn) = ptir_rs_prepare(&mut w, &mut a).unwrap();
            assert_eq!(ids.len(), 1, "step {step}");
            let want = if step == 0 {
                pie_driver_abi::RS_FLAG_RESET
            } else {
                0u8
            };
            assert_eq!(flags, vec![want], "step {step} reset flag");
            ptir_rs_finalize(&mut w, &mut a, txn, true).unwrap();
        }
        assert!(w.folded_object().is_some());
    }

    /// Abort consistency: a fresh fire that aborts leaves NO folded state (the
    /// staged alloc is reverted), so the next fire re-resets — no drift.
    #[test]
    fn abort_reverts_to_no_folded_state() {
        let mut a = arena(8);
        let mut w = ws();
        let (_ids, _flags, _m, txn) = ptir_rs_prepare(&mut w, &mut a).unwrap();
        ptir_rs_finalize(&mut w, &mut a, txn, false).unwrap(); // abort
        assert!(
            w.folded_object().is_none(),
            "aborted fresh fire ⇒ no folded state adopted"
        );
        // The next fire is still fresh (re-resets) — no half-committed drift.
        let (_ids2, flags2, _m2, t2) = ptir_rs_prepare(&mut w, &mut a).unwrap();
        assert_eq!(flags2, vec![pie_driver_abi::RS_FLAG_RESET], "re-fresh after abort");
        ptir_rs_finalize(&mut w, &mut a, t2, true).unwrap();
    }
}
