//! Fire RS (recurrent-state) preparation over the typed `RsStore`
//! (kv_refact.md). The GDN / linear-attention forward writes the advanced
//! folded state INTO the folded slot directly (the driver's in-forward
//! `commit_len` path): [`prepare_many`] classifies every folded-slot target
//! (fresh+reset / CoW-after-fork / in-place), PUBLISHES the resulting mapping
//! under the same store lock — in guest submission order, before the launch
//! reaches the scheduler, exactly as `fire::kv::prepare` does — and returns
//! the driver lowering (`rs_slot_ids`, `rs_slot_flags`, pre-launch d2d copy)
//! plus an [`RsTxn`] receipt the fire holds until [`settle`].
//!
//! Publishing at prepare is what lets RS fires run ahead: a successor
//! prepared before its predecessor completes classifies against a mapping
//! that already carries the predecessor's decision, so it can neither RESET
//! a slot twice nor re-CoW an already-privatized one. The advanced state's
//! CONTENTS are ordered by the pipeline stream, which the pre-launch CoW copy
//! already relies on.
//!
//! The allow below covers exactly the singular test-exercised convenience
//! wrapper ([`prepare`]) around the `_many` production entries — kept rather
//! than deleted, allowed rather than silently masked.
#![allow(dead_code)]

use crate::store::rs::write::{RsPreparedWrite, RsPublished};
use crate::store::rs::{RsStore, RsWorkingSetId};

/// The published RS write for one in-flight PTIR fire, held across
/// `submit_async` until [`settle`].
pub struct RsTxn {
    published: RsPublished,
}

impl RsTxn {
    /// Submission sequence of the newest row this fire published.
    pub fn seq(&self) -> u64 {
        self.published.seq()
    }
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

/// Phase-A demand for [`prepare_many`] over these working sets: how many
/// folded slots the prepare would allocate, with no allocation or open
/// transaction. The acquisition seam sizes its RS ask from this.
pub fn demand(store: &RsStore, working_sets: &[RsWorkingSetId]) -> Result<usize, String> {
    let mut total = 0;
    for &ws in working_sets {
        total += store.write_state_demand(ws).map_err(|e| e.to_string())?;
    }
    Ok(total)
}

/// Driver lowering for published folded-state writes:
/// `(rs_slot_ids, rs_slot_flags, (copy_src, copy_dst), txn)`.
pub type PreparedRs = (Vec<u32>, Vec<u8>, (Vec<u32>, Vec<u32>), Option<RsTxn>);

/// Prepare and publish the in-forward folded-state write. Returns
/// [`PreparedRs`]: thread the ids and flags into the launch in request order,
/// issue one aggregated state-copy command before the launch when non-empty,
/// hold `txn` across the fire, then [`settle`].
pub fn prepare_many(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
) -> Result<PreparedRs, String> {
    prepare_many_impl(store, working_sets, None)
}

/// [`prepare_many`] from caller-owned reserved slots (the acquisition
/// grant), consuming exactly the required prefix of `granted`.
pub fn prepare_many_reserved(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    granted: &mut Vec<crate::store::rs::RsSlotId>,
) -> Result<PreparedRs, String> {
    prepare_many_impl(store, working_sets, Some(granted))
}

fn prepare_many_impl(
    store: &mut RsStore,
    working_sets: &[RsWorkingSetId],
    mut granted: Option<&mut Vec<crate::store::rs::RsSlotId>>,
) -> Result<PreparedRs, String> {
    for (index, ws) in working_sets.iter().enumerate() {
        if working_sets[..index].contains(ws) {
            return Err(format!(
                "rs-working-set at request row {index} aliases an earlier row"
            ));
        }
    }
    if working_sets.is_empty() {
        return Ok((Vec::new(), Vec::new(), (Vec::new(), Vec::new()), None));
    }

    let mut slot_ids = Vec::with_capacity(working_sets.len());
    let mut slot_flags = Vec::with_capacity(working_sets.len());
    let mut copy_src = Vec::new();
    let mut copy_dst = Vec::new();
    let mut prepared_rows: Vec<RsPreparedWrite> = Vec::with_capacity(working_sets.len());
    for &ws in working_sets {
        let prepared = match granted.as_deref_mut() {
            Some(granted) => store.prepare_write_reserved(ws, granted),
            None => store.prepare_write(ws, true, None),
        };
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                store.cancel_batch(prepared_rows);
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
        prepared_rows.push(prepared);
    }
    // Publish before returning: the successor fire's classification must see
    // this fire's decision without waiting for the device.
    let published = store
        .publish_batch(prepared_rows)
        .map_err(|error| error.to_string())?;
    Ok((
        slot_ids,
        slot_flags,
        (copy_src, copy_dst),
        Some(RsTxn { published }),
    ))
}

pub fn prepare(
    store: &mut RsStore,
    ws: RsWorkingSetId,
) -> Result<(Vec<u32>, Vec<u8>, (Vec<u32>, Vec<u32>), RsTxn), String> {
    let (ids, flags, copies, txn) = prepare_many(store, &[ws])?;
    Ok((
        ids,
        flags,
        copies,
        txn.expect("one working set produces one transaction"),
    ))
}

/// Settle a fire's published RS write once it resolves, successfully or not.
///
/// There is no mapping rollback: a published RS mapping is fail-stop
/// pipeline-local state (`fire::kv::abandon` says the same for KV). A failed
/// fire leaves the recurrent state undefined, the pass poisons its readers,
/// and the pre-failure state survives only through a fork taken before it —
/// the rule `RsStore` already documents for a committed fold. Settling
/// releases the store's in-flight hold so recycled slots become allocatable.
pub fn settle(store: &mut RsStore, txn: Option<RsTxn>) {
    let Some(RsTxn { published }) = txn else {
        return;
    };
    store.settle(published);
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
        settle(&mut store, Some(txn));
        let slot = store.folded_slot(ws).unwrap().unwrap();

        let (ids, flags, (src, _), txn) = prepare(&mut store, ws).unwrap();
        assert_eq!(ids, vec![slot.0]);
        assert_eq!(flags, vec![0]);
        assert!(src.is_empty());
        settle(&mut store, Some(txn));
    }

    #[test]
    fn forked_fire_copies_the_folded_state() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, ws).unwrap();
        settle(&mut store, Some(txn));
        let shared = store.folded_slot(ws).unwrap().unwrap();

        let forked = store.fork(ws).unwrap();
        let (ids, flags, (src, dst), txn) = prepare(&mut store, forked).unwrap();
        assert_eq!(src, vec![shared.0]);
        assert_eq!(dst, ids);
        assert_eq!(flags, vec![0]); // copied, not reset
        settle(&mut store, Some(txn));
        assert_eq!(store.folded_slot(ws).unwrap(), Some(shared));
        assert_ne!(store.folded_slot(forked).unwrap(), Some(shared));
    }

    // ------------------------------------------------------------------
    // Run-ahead. Each successor prepares while its predecessor is still in
    // flight — the shape the deleted `drain_rs_predecessors` forbade.
    // ------------------------------------------------------------------

    #[test]
    fn unsettled_fresh_runahead_never_lowers_a_second_reset() {
        let mut store = RsStore::new(3);
        let ws = store.create_working_set(geom());

        let (first_ids, first_flags, _, first) = prepare(&mut store, ws).unwrap();
        assert_eq!(first_flags, vec![crate::driver::RS_FLAG_RESET]);

        // No settle: the first fire is still on the device.
        let (second_ids, second_flags, (copy_src, _), second) = prepare(&mut store, ws).unwrap();
        assert_eq!(
            second_ids, first_ids,
            "second fire continues the published slot"
        );
        assert_eq!(second_flags, vec![0], "second fire must not RESET again");
        assert!(copy_src.is_empty());
        assert_eq!(store.available_slots(), 2, "one slot serves both fires");

        settle(&mut store, Some(first));
        settle(&mut store, Some(second));
    }

    #[test]
    fn unsettled_forked_runahead_cows_once_then_continues_the_child() {
        let mut store = RsStore::new(4);
        let parent = store.create_working_set(geom());
        let (_, _, _, parent_write) = prepare(&mut store, parent).unwrap();
        settle(&mut store, Some(parent_write));
        let shared = store.folded_slot(parent).unwrap().unwrap().0;
        let child = store.fork(parent).unwrap();

        let (first_ids, _, (first_src, _), first) = prepare(&mut store, child).unwrap();
        assert_eq!(first_src, vec![shared]);

        // No settle in between.
        let (second_ids, second_flags, (second_src, _), second) =
            prepare(&mut store, child).unwrap();
        assert_eq!(
            second_ids, first_ids,
            "child continues its published CoW slot"
        );
        assert_eq!(second_flags, vec![0]);
        assert!(
            second_src.is_empty(),
            "a run-ahead successor must not CoW from the stale parent again"
        );
        settle(&mut store, Some(first));
        settle(&mut store, Some(second));
    }

    #[test]
    fn a_three_slot_frame_lowers_one_slot_and_one_reset() {
        // The k=3 decode frame: three fires submitted back to back, none
        // settled, all writing one working set's folded state.
        let mut store = RsStore::new(2);
        let ws = store.create_working_set(geom());

        let mut txns = Vec::new();
        let mut lowered = Vec::new();
        for slot in 0..3 {
            let (ids, flags, (src, dst), txn) = prepare(&mut store, ws).unwrap();
            assert!(
                src.is_empty() && dst.is_empty(),
                "slot {slot} must not need a pre-launch copy"
            );
            lowered.push((ids, flags));
            txns.push(txn);
        }
        assert_eq!(
            lowered[0].1,
            vec![crate::driver::RS_FLAG_RESET],
            "wave 0 resets the cold slot"
        );
        assert_eq!(lowered[1].1, vec![0], "wave 1 continues in place");
        assert_eq!(lowered[2].1, vec![0], "wave 2 continues in place");
        assert_eq!(lowered[0].0, lowered[1].0);
        assert_eq!(lowered[1].0, lowered[2].0);
        assert_eq!(
            store.available_slots(),
            1,
            "a three-wave frame allocates exactly one folded slot"
        );
        for txn in txns {
            settle(&mut store, Some(txn));
        }
    }

    #[test]
    fn failed_fire_keeps_the_published_state() {
        let mut store = RsStore::new(2);
        let ws = store.create_working_set(geom());
        let (ids, _, _, txn) = prepare(&mut store, ws).unwrap();
        settle(&mut store, Some(txn)); // the fire failed on the device
        assert_eq!(
            store.folded_slot(ws).unwrap().map(|s| s.0),
            Some(ids[0]),
            "fail-stop: the published mapping stays authoritative"
        );
    }

    #[test]
    fn two_rows_lower_distinct_slots_in_request_order() {
        let mut store = RsStore::new(4);
        let first = store.create_working_set(geom());
        let second = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, first).unwrap();
        settle(&mut store, Some(txn));
        let (_, _, _, txn) = prepare(&mut store, second).unwrap();
        settle(&mut store, Some(txn));
        let first_slot = store.folded_slot(first).unwrap().unwrap().0;
        let second_slot = store.folded_slot(second).unwrap().unwrap().0;

        let (ids, flags, copies, txn) = prepare_many(&mut store, &[second, first]).unwrap();
        assert_eq!(ids, vec![second_slot, first_slot]);
        assert_ne!(ids[0], ids[1]);
        assert_eq!(flags, vec![0, 0]);
        assert_eq!(copies, (Vec::new(), Vec::new()));
        settle(&mut store, txn);
    }

    #[test]
    fn duplicate_parent_forks_get_independent_cow_children() {
        let mut store = RsStore::new(6);
        let parent = store.create_working_set(geom());
        let (_, _, _, txn) = prepare(&mut store, parent).unwrap();
        settle(&mut store, Some(txn));
        let parent_slot = store.folded_slot(parent).unwrap().unwrap().0;
        let left = store.fork(parent).unwrap();
        let right = store.fork(parent).unwrap();

        let (ids, flags, (src, dst), txn) = prepare_many(&mut store, &[left, right]).unwrap();
        assert_eq!(src, vec![parent_slot, parent_slot]);
        assert_eq!(dst, ids);
        assert_ne!(ids[0], ids[1]);
        assert_eq!(flags, vec![0, 0]);
        settle(&mut store, txn);

        assert_eq!(store.folded_slot(parent).unwrap().unwrap().0, parent_slot);
        assert_ne!(
            store.folded_slot(left).unwrap(),
            store.folded_slot(right).unwrap()
        );
    }

    #[test]
    fn no_working_sets_prepares_nothing() {
        let mut store = RsStore::new(2);
        let (ids, flags, copies, txn) = prepare_many(&mut store, &[]).unwrap();
        assert!(ids.is_empty() && flags.is_empty());
        assert_eq!(copies, (Vec::new(), Vec::new()));
        assert!(txn.is_none());
        assert_eq!(store.available_slots(), 2);
    }

    #[test]
    fn aliased_rows_are_rejected_before_any_allocation() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());
        assert!(prepare_many(&mut store, &[ws, ws]).is_err());
        assert_eq!(store.available_slots(), 4);
        assert_eq!(store.folded_slot(ws).unwrap(), None);
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
        let published = store.publish_prepared(buffered).unwrap();
        store.settle(published);

        let (ids, _, _, txn) = prepare(&mut store, ws).unwrap();
        assert_ne!(ids, vec![buffer_slot]);
        settle(&mut store, Some(txn));
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
