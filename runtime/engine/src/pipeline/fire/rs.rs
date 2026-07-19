//! Fire RS (recurrent-state) preparation over the typed `RsStore`
//! (kv_refact.md). The GDN / linear-attention forward writes the advanced
//! folded state INTO the folded slot directly (the driver's in-forward
//! `commit_len` path): [`prepare_many`] classifies every folded-slot target
//! (fresh+reset / CoW-after-fork / in-place), returns the driver lowering
//! (`rs_slot_ids`, `rs_slot_flags`, pre-launch d2d copy), and the prepared
//! writes held across the async fire until [`finalize_many`].
//!
//! Complete pipeline domain API: some methods here (relaxed geometry
//! variants, per-channel introspection, the pure `instantiate`/registry
//! probe entry points, device-geometry lease internals) are not yet
//! called by the current single-model/mock-driver fire path, but are
//! exercised by this module's own unit tests and reserved for upcoming
//! wiring (multi-pass channels, device-geometry beams) — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use crate::store::rs::write::RsPreparedWrite;
use crate::store::rs::{RsStore, RsWorkingSetId};

/// The prepared RS write for one in-flight PTIR fire.
pub struct RsTxn {
    prepared: RsPreparedWrite,
}

/// Validate the recurrent-state arity against the resolved forward rows.
pub fn validate_count(
    rs_count: usize,
    qo_indptr: &[u32],
    has_recurrent_state: bool,
) -> Result<usize, String> {
    if !has_recurrent_state {
        if rs_count == 0 {
            return Ok(qo_indptr.len().saturating_sub(1));
        }
        return Err(format!(
            "pure-attention model bound {rs_count} rs-working-set(s); expected 0"
        ));
    }
    let request_count = qo_indptr
        .len()
        .checked_sub(1)
        .ok_or_else(|| "resolved qo_indptr is empty".to_string())?;
    if rs_count != request_count {
        return Err(format!(
            "resolved forward has {request_count} request row(s), but recurrent-state model bound \
             {rs_count} rs-working-set(s); expected {request_count}",
        ));
    }
    Ok(request_count)
}

/// Prepare the in-forward folded-state write. Returns
/// `(rs_slot_ids, rs_slot_flags, (copy_src, copy_dst), txns)`: thread the ids
/// and flags into the launch in request order, issue one aggregated state-copy
/// command before the launch when non-empty, hold all `txns` across the fire,
/// then [`finalize_many`].
pub fn prepare_many(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
) -> Result<(Vec<u32>, Vec<u8>, (Vec<u32>, Vec<u32>), Vec<RsTxn>), String> {
    for (index, ws) in working_sets.iter().enumerate() {
        if working_sets[..index].contains(ws) {
            return Err(format!(
                "rs-working-set at request row {index} aliases an earlier row"
            ));
        }
    }

    let mut slot_ids = Vec::with_capacity(working_sets.len());
    let mut slot_flags = Vec::with_capacity(working_sets.len());
    let mut copy_src = Vec::new();
    let mut copy_dst = Vec::new();
    let mut txns = Vec::with_capacity(working_sets.len());
    for &ws in working_sets {
        let prepared = match store.prepare_write(ws, true, None) {
            Ok(prepared) => prepared,
            Err(error) => {
                abandon_many(store, txns);
                return Err(error.to_string());
            }
        };
        let state = prepared.state().expect("write_state requested");
        slot_ids.push(state.slot.0);
        slot_flags.push(if state.reset {
            crate::driver::RS_FLAG_RESET
        } else {
            0
        });
        if let Some(src) = state.copy_from {
            copy_src.push(src.0);
            copy_dst.push(state.slot.0);
        }
        txns.push(RsTxn { prepared });
    }
    Ok((slot_ids, slot_flags, (copy_src, copy_dst), txns))
}

pub fn prepare(
    store: &mut RsStore,
    ws: RsWorkingSetId,
) -> Result<(Vec<u32>, Vec<u8>, (Vec<u32>, Vec<u32>), RsTxn), String> {
    let (ids, flags, copies, mut txns) = prepare_many(store, &[ws])?;
    Ok((
        ids,
        flags,
        copies,
        txns.pop()
            .expect("one working set produces one transaction"),
    ))
}

/// Abandon a fire's prepared RS write (guest dropped the working set while
/// the fire was in flight).
pub fn abandon(store: &mut RsStore, txn: RsTxn) {
    abandon_many(store, vec![txn]);
}

pub fn abandon_many(store: &mut RsStore, txns: Vec<RsTxn>) {
    let Some(seq) = txns.iter().map(|txn| txn.prepared.seq()).max() else {
        return;
    };
    let epoch = store.current_epoch();
    store.abort_batch(txns.into_iter().map(|txn| txn.prepared).collect(), epoch);
    store.retire_through(seq);
    store.retire_idle();
}

/// Finalize after the fire resolves: `success` adopts the folded slot for the
/// next fire; otherwise the pending slot releases and the prior folded state
/// stays visible.
pub fn finalize(store: &mut RsStore, txn: RsTxn, success: bool) -> Result<(), String> {
    finalize_many(store, vec![txn], success)
}

pub fn finalize_many(store: &mut RsStore, txns: Vec<RsTxn>, success: bool) -> Result<(), String> {
    let Some(seq) = txns.iter().map(|txn| txn.prepared.seq()).max() else {
        return Ok(());
    };
    let epoch = store.current_epoch();
    let prepared = txns.into_iter().map(|txn| txn.prepared).collect();
    let result = if success {
        store
            .commit_batch(prepared, epoch)
            .map_err(|e| e.to_string())
    } else {
        store.abort_batch(prepared, epoch);
        Ok(())
    };
    store.retire_through(seq);
    store.retire_idle();
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::rs::RsGeometry;

    fn geom() -> RsGeometry {
        RsGeometry {
            state_size: 1024,
            buffer_page_tokens: 4,
            fold_granularity: 1,
        }
    }

    #[test]
    fn first_fire_resets_then_continues_in_place() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());

        let (ids, flags, (src, _), txn) = prepare(&mut store, ws).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(flags, vec![crate::driver::RS_FLAG_RESET]);
        assert!(src.is_empty());
        finalize(&mut store, txn, true).unwrap();
        let slot = store.folded_slot(ws).unwrap().unwrap();

        let (ids, flags, (src, _), txn) = prepare(&mut store, ws).unwrap();
        assert_eq!(ids, vec![slot.0]);
        assert_eq!(flags, vec![0]);
        assert!(src.is_empty());
        finalize(&mut store, txn, true).unwrap();
    }

    #[test]
    fn forked_fire_copies_the_folded_state() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, ws).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let shared = store.folded_slot(ws).unwrap().unwrap();

        let forked = store.fork(ws).unwrap();
        let (ids, flags, (src, dst), txn) = prepare(&mut store, forked).unwrap();
        assert_eq!(src, vec![shared.0]);
        assert_eq!(dst, ids);
        assert_eq!(flags, vec![0]); // copied, not reset
        finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.folded_slot(ws).unwrap(), Some(shared));
        assert_ne!(store.folded_slot(forked).unwrap(), Some(shared));
    }

    #[test]
    fn serialized_fresh_runahead_prepares_second_fire_from_first_commit() {
        let mut store = RsStore::new(3);
        let ws = store.create_working_set(geom());

        let (first_ids, first_flags, _, first) = prepare(&mut store, ws).unwrap();
        assert_eq!(first_flags, vec![crate::driver::RS_FLAG_RESET]);
        finalize(&mut store, first, true).unwrap();

        let (second_ids, second_flags, (copy_src, _), second) = prepare(&mut store, ws).unwrap();
        assert_eq!(
            second_ids, first_ids,
            "second fire continues committed slot"
        );
        assert_eq!(second_flags, vec![0], "second fire must not RESET again");
        assert!(copy_src.is_empty());
        finalize(&mut store, second, true).unwrap();
    }

    #[test]
    fn serialized_forked_runahead_cows_once_then_continues_child() {
        let mut store = RsStore::new(4);
        let parent = store.create_working_set(geom());
        let (_, _, _, parent_write) = prepare(&mut store, parent).unwrap();
        finalize(&mut store, parent_write, true).unwrap();
        let shared = store.folded_slot(parent).unwrap().unwrap().0;
        let child = store.fork(parent).unwrap();

        let (first_ids, _, (first_src, _), first) = prepare(&mut store, child).unwrap();
        assert_eq!(first_src, vec![shared]);
        finalize(&mut store, first, true).unwrap();

        let (second_ids, second_flags, (second_src, _), second) =
            prepare(&mut store, child).unwrap();
        assert_eq!(
            second_ids, first_ids,
            "child continues its committed CoW slot"
        );
        assert_eq!(second_flags, vec![0]);
        assert!(
            second_src.is_empty(),
            "serialized successor must not CoW from stale parent again"
        );
        finalize(&mut store, second, true).unwrap();
    }

    #[test]
    fn failed_fire_keeps_prior_state() {
        let mut store = RsStore::new(2);
        let ws = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, ws).unwrap();
        finalize(&mut store, txn, false).unwrap();
        assert_eq!(store.folded_slot(ws).unwrap(), None);
        assert_eq!(store.available_slots(), 2);
    }

    #[test]
    fn two_rows_lower_distinct_slots_in_request_order() {
        let mut store = RsStore::new(4);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, first).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let (_, _, _, txn) = prepare(&mut store, second).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let first_slot = store.folded_slot(first).unwrap().unwrap().0;
        let second_slot = store.folded_slot(second).unwrap().unwrap().0;

        let (ids, flags, copies, txns) = prepare_many(&mut store, &[second, first]).unwrap();
        assert_eq!(ids, vec![second_slot, first_slot]);
        assert_ne!(ids[0], ids[1]);
        assert_eq!(flags, vec![0, 0]);
        assert_eq!(copies, (Vec::new(), Vec::new()));
        finalize_many(&mut store, txns, true).unwrap();
    }

    #[test]
    fn duplicate_parent_forks_get_independent_cow_children() {
        let mut store = RsStore::new(6);
        let parent = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, parent).unwrap();
        finalize(&mut store, txn, true).unwrap();
        let parent_slot = store.folded_slot(parent).unwrap().unwrap().0;
        let left = store.fork(parent).unwrap();
        let right = store.fork(parent).unwrap();

        let (ids, flags, (src, dst), txns) = prepare_many(&mut store, &[left, right]).unwrap();
        assert_eq!(src, vec![parent_slot, parent_slot]);
        assert_eq!(dst, ids);
        assert_ne!(ids[0], ids[1]);
        assert_eq!(flags, vec![0, 0]);
        finalize_many(&mut store, txns, true).unwrap();

        assert_eq!(store.folded_slot(parent).unwrap().unwrap().0, parent_slot);
        assert_ne!(
            store.folded_slot(left).unwrap(),
            store.folded_slot(right).unwrap()
        );
    }

    #[test]
    fn failed_multi_row_fire_rolls_back_every_transaction() {
        let mut store = RsStore::new(2);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        let (_, _, _, txns) = prepare_many(&mut store, &[first, second]).unwrap();
        finalize_many(&mut store, txns, false).unwrap();
        assert_eq!(store.folded_slot(first).unwrap(), None);
        assert_eq!(store.folded_slot(second).unwrap(), None);
        assert_eq!(store.available_slots(), 2);
    }

    #[test]
    fn reverse_order_abort_retires_every_slot_when_idle() {
        let mut store = RsStore::new(2);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        let (_, _, _, first_txn) = prepare(&mut store, first).unwrap();
        let (_, _, _, second_txn) = prepare(&mut store, second).unwrap();

        finalize(&mut store, second_txn, false).unwrap();
        finalize(&mut store, first_txn, false).unwrap();

        assert_eq!(store.available_slots(), 2);
    }

    #[test]
    fn commit_validation_failure_adopts_no_row() {
        let mut store = RsStore::new(2);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        let (_, _, _, txns) = prepare_many(&mut store, &[first, second]).unwrap();
        let epoch = store.current_epoch();
        store.release_working_set(second, epoch);

        assert!(finalize_many(&mut store, txns, true).is_err());
        assert_eq!(store.folded_slot(first).unwrap(), None);
        assert_eq!(store.available_slots(), 2);
    }

    #[test]
    fn preparation_failure_rolls_back_earlier_rows() {
        let mut store = RsStore::new(1);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        assert!(prepare_many(&mut store, &[first, second]).is_err());
        assert_eq!(store.folded_slot(first).unwrap(), None);
        assert_eq!(store.folded_slot(second).unwrap(), None);
        assert_eq!(store.available_slots(), 1);
    }

    #[test]
    fn folded_lowering_never_uses_a_buffered_slot() {
        let mut store = RsStore::new(3);
        let ws = store.create_working_set(geom());
        store.alloc_buffer(ws, 1).unwrap();
        let buffered = store.prepare_write(ws, false, Some((0, 1))).unwrap();
        let buffer_slot = buffered.buffer_targets()[0].dst().0;
        let epoch = store.current_epoch();
        store.commit(buffered, epoch).unwrap();

        let (ids, _, _, txn) = prepare(&mut store, ws).unwrap();
        assert_ne!(ids, vec![buffer_slot]);
        finalize(&mut store, txn, true).unwrap();
    }

    #[test]
    fn validates_pure_single_and_multi_request_arity() {
        assert_eq!(validate_count(0, &[], false).unwrap(), 0);
        assert_eq!(validate_count(0, &[0, 1, 2], false).unwrap(), 2);
        assert_eq!(validate_count(1, &[0, 1], true).unwrap(), 1);
        assert_eq!(validate_count(2, &[0, 1, 2], true).unwrap(), 2);
        assert!(validate_count(1, &[0, 1, 2], true).is_err());
        assert!(validate_count(1, &[0, 1], false).is_err());
    }
}
