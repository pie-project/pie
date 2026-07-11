//! PTIR fire RS (recurrent-state) preparation over the typed `RsStore`
//! (kv_refact.md). The GDN / linear-attention forward writes the advanced
//! folded state INTO the folded slot directly (the driver's in-forward
//! `commit_len` path): [`ptir_rs_prepare`] classifies the folded-slot target
//! (fresh+reset / CoW-after-fork / in-place), returns the driver lowering
//! (`rs_slot_ids`, `rs_slot_flags`, pre-launch d2d copy), and the prepared
//! write held across the async fire until [`ptir_rs_finalize`].

use crate::store::rs::write::RsPreparedWrite;
use crate::store::rs::{RsStore, RsWorkingSetId};

/// The prepared RS write for one in-flight PTIR fire.
pub struct PtirRsTxn {
    prepared: RsPreparedWrite,
}

/// Prepare the in-forward folded-state write. Returns
/// `(rs_slot_ids, rs_slot_flags, (copy_src, copy_dst), txn)`: thread the ids
/// and flags into the launch, issue one `driver::copy_d2d(copy_src,
/// copy_dst)` before the launch when non-empty (shared-fork CoW), hold `txn`
/// across the fire, then [`ptir_rs_finalize`].
pub fn ptir_rs_prepare(
    store: &mut RsStore,
    ws: RsWorkingSetId,
) -> Result<(Vec<u32>, Vec<u8>, (Vec<u32>, Vec<u32>), PtirRsTxn), String> {
    let prepared = store
        .prepare_write(ws, true, None)
        .map_err(|e| e.to_string())?;
    let state = prepared.state().expect("write_state requested");
    let rs_slot_ids = vec![state.slot.0];
    let rs_slot_flags = vec![if state.reset {
        crate::driver::RS_FLAG_RESET
    } else {
        0
    }];
    let copies = match state.copy_from {
        Some(src) => (vec![src.0], vec![state.slot.0]),
        None => (Vec::new(), Vec::new()),
    };
    Ok((rs_slot_ids, rs_slot_flags, copies, PtirRsTxn { prepared }))
}

/// Abandon a fire's prepared RS write (guest dropped the working set while
/// the fire was in flight).
pub fn ptir_rs_abandon(store: &mut RsStore, txn: PtirRsTxn) {
    let seq = txn.prepared.seq();
    let epoch = store.current_epoch();
    store.abort(txn.prepared, epoch);
    store.retire_through(seq);
}

/// Finalize after the fire resolves: `success` adopts the folded slot for the
/// next fire; otherwise the pending slot releases and the prior folded state
/// stays visible.
pub fn ptir_rs_finalize(store: &mut RsStore, txn: PtirRsTxn, success: bool) -> Result<(), String> {
    let seq = txn.prepared.seq();
    let epoch = store.current_epoch();
    if success {
        store.commit(txn.prepared, epoch).map_err(|e| e.to_string())?;
    } else {
        store.abort(txn.prepared, epoch);
    }
    store.retire_through(seq);
    Ok(())
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

        let (ids, flags, (src, _), txn) = ptir_rs_prepare(&mut store, ws).unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(flags, vec![crate::driver::RS_FLAG_RESET]);
        assert!(src.is_empty());
        ptir_rs_finalize(&mut store, txn, true).unwrap();
        let slot = store.folded_slot(ws).unwrap().unwrap();

        let (ids, flags, (src, _), txn) = ptir_rs_prepare(&mut store, ws).unwrap();
        assert_eq!(ids, vec![slot.0]);
        assert_eq!(flags, vec![0]);
        assert!(src.is_empty());
        ptir_rs_finalize(&mut store, txn, true).unwrap();
    }

    #[test]
    fn forked_fire_copies_the_folded_state() {
        let mut store = RsStore::new(4);
        let ws = store.create_working_set(geom());
        let (_, _, _, txn) = ptir_rs_prepare(&mut store, ws).unwrap();
        ptir_rs_finalize(&mut store, txn, true).unwrap();
        let shared = store.folded_slot(ws).unwrap().unwrap();

        let forked = store.fork(ws).unwrap();
        let (ids, flags, (src, dst), txn) = ptir_rs_prepare(&mut store, forked).unwrap();
        assert_eq!(src, vec![shared.0]);
        assert_eq!(dst, ids);
        assert_eq!(flags, vec![0]); // copied, not reset
        ptir_rs_finalize(&mut store, txn, true).unwrap();
        assert_eq!(store.folded_slot(ws).unwrap(), Some(shared));
        assert_ne!(store.folded_slot(forked).unwrap(), Some(shared));
    }

    #[test]
    fn failed_fire_keeps_prior_state() {
        let mut store = RsStore::new(2);
        let ws = store.create_working_set(geom());
        let (_, _, _, txn) = ptir_rs_prepare(&mut store, ws).unwrap();
        ptir_rs_finalize(&mut store, txn, false).unwrap();
        assert_eq!(store.folded_slot(ws).unwrap(), None);
        assert_eq!(store.available_slots(), 2);
    }
}
