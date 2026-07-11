//! Unit tests for the RS store.

use super::write::RsBufferTarget;
use super::{RsError, RsGeometry, RsStore, RsWorkingSetId};

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: 4096,
        buffer_page_tokens: 4,
        fold_granularity: 4,
    }
}

fn store() -> RsStore {
    RsStore::new(12)
}

fn write_state(store: &mut RsStore, ws: RsWorkingSetId, epoch: u64) {
    let prepared = store.prepare_write(ws, true, None).unwrap();
    store.commit(prepared, epoch).unwrap();
}

#[test]
fn first_state_write_is_fresh_with_reset() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    let prepared = s.prepare_write(ws, true, None).unwrap();
    let state = prepared.state().unwrap();
    assert!(state.reset);
    assert!(state.copy_from.is_none());
    assert_eq!(s.available_slots(), 11);
    let slot = state.slot;
    s.commit(prepared, 1).unwrap();
    assert_eq!(s.folded_slot(ws).unwrap(), Some(slot));
}

#[test]
fn continuing_state_write_is_in_place() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws, 1);
    let slot = s.folded_slot(ws).unwrap().unwrap();
    let prepared = s.prepare_write(ws, true, None).unwrap();
    let state = prepared.state().unwrap();
    assert_eq!(state.slot, slot);
    assert!(!state.reset);
    assert!(state.copy_from.is_none());
    assert_eq!(s.available_slots(), 11); // no new allocation
    s.commit(prepared, 2).unwrap();
}

#[test]
fn forked_state_write_copies_on_write() {
    let mut s = store();
    let a = s.create_working_set(geom());
    write_state(&mut s, a, 1);
    let shared = s.folded_slot(a).unwrap().unwrap();
    let b = s.fork(a).unwrap();

    let prepared = s.prepare_write(b, true, None).unwrap();
    let state = *prepared.state().unwrap();
    assert!(!state.reset);
    assert_eq!(state.copy_from, Some(shared));
    assert_ne!(state.slot, shared);
    s.commit(prepared, 2).unwrap();

    // b owns the copy; a still reads the shared original.
    assert_eq!(s.folded_slot(b).unwrap(), Some(state.slot));
    assert_eq!(s.folded_slot(a).unwrap(), Some(shared));

    // a is now the sole owner: its next write is in place again.
    let prepared = s.prepare_write(a, true, None).unwrap();
    assert_eq!(prepared.state().unwrap().slot, shared);
    s.commit(prepared, 3).unwrap();
}

#[test]
fn buffer_writes_materialize_then_in_place_then_cow_after_fork() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 3).unwrap();

    // First write materializes all three pages.
    let prepared = s.prepare_write(ws, false, Some((0, 12))).unwrap();
    assert!(prepared
        .buffer_targets()
        .iter()
        .all(|t| matches!(t, RsBufferTarget::Fresh { .. })));
    let ids: Vec<_> = prepared.buffer_targets().iter().map(|t| t.dst()).collect();
    s.commit(prepared, 1).unwrap();
    assert_eq!(s.resolve_buffer(ws, 0, 12).unwrap(), ids);

    // Second write is in place.
    let prepared = s.prepare_write(ws, false, Some((0, 12))).unwrap();
    assert!(prepared
        .buffer_targets()
        .iter()
        .all(|t| matches!(t, RsBufferTarget::InPlace { .. })));
    s.commit(prepared, 2).unwrap();

    // After a fork the pages are shared: partial-range write CoWs its span.
    let forked = s.fork(ws).unwrap();
    let prepared = s.prepare_write(forked, false, Some((4, 4))).unwrap();
    assert_eq!(prepared.buffer_targets().len(), 1);
    assert!(matches!(
        prepared.buffer_targets()[0],
        RsBufferTarget::Cow { .. }
    ));
    assert_eq!(prepared.buffer_copy_plan().count(), 1);
    s.commit(prepared, 3).unwrap();
    // Original keeps its page; fork has the copy at the same slot index.
    assert_eq!(s.resolve_buffer(ws, 4, 4).unwrap(), vec![ids[1]]);
    assert_ne!(s.resolve_buffer(forked, 4, 4).unwrap(), vec![ids[1]]);
}

#[test]
fn fold_validates_granularity_and_capacity() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap(); // capacity 8 tokens
    assert_eq!(s.validate_fold(ws, 0), Err(RsError::FoldZero));
    assert_eq!(
        s.validate_fold(ws, 6),
        Err(RsError::FoldGranularity {
            tokens: 6,
            granularity: 4
        })
    );
    assert_eq!(
        s.validate_fold(ws, 12),
        Err(RsError::FoldExceedsBuffer {
            tokens: 12,
            capacity: 8
        })
    );
    assert_eq!(s.validate_fold(ws, 8), Ok(()));
}

#[test]
fn committed_fold_advances_the_boundary_and_drops_head_pages() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 3).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 12))).unwrap();
    s.commit(prepared, 1).unwrap();
    assert_eq!(s.available_slots(), 9);

    let prepared = s.prepare_fold(ws, 8).unwrap();
    assert_eq!(prepared.state().unwrap().fold_tokens, Some(8));
    assert_eq!(s.available_slots(), 8); // fresh folded slot taken
    s.commit(prepared, 2).unwrap();
    // Two fully covered head pages dropped; recycled once the epoch retires.
    assert_eq!(s.buffer_size(ws).unwrap(), 1);
    assert_eq!(s.available_slots(), 8);
    s.retire_through(2);
    assert_eq!(s.available_slots(), 10);
    // Folded state exists now.
    assert!(s.folded_slot(ws).unwrap().is_some());
}

#[test]
fn free_and_reorder_validate_inputs() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 3).unwrap();
    assert_eq!(
        s.free_buffer(ws, &[3], 1),
        Err(RsError::IndexOutOfRange { index: 3, size: 3 })
    );
    assert_eq!(
        s.free_buffer(ws, &[1, 1], 1),
        Err(RsError::DuplicateIndex { index: 1 })
    );
    assert_eq!(
        s.reorder_buffer(ws, &[0, 0, 2]),
        Err(RsError::BadPermutation { size: 3 })
    );
    s.free_buffer(ws, &[1], 1).unwrap();
    assert_eq!(s.buffer_size(ws).unwrap(), 2);
    s.reorder_buffer(ws, &[1, 0]).unwrap();
}

#[test]
fn release_frees_shared_slots_only_after_both_sides_drop() {
    let mut s = store();
    let a = s.create_working_set(geom());
    write_state(&mut s, a, 1);
    s.alloc_buffer(a, 2).unwrap();
    let prepared = s.prepare_write(a, false, Some((0, 8))).unwrap();
    s.commit(prepared, 1).unwrap();
    let b = s.fork(a).unwrap();

    s.release_working_set(a, 2);
    s.retire_through(2);
    // b still holds everything (1 state + 2 buffer slots in use).
    assert_eq!(s.available_slots(), 9);
    assert_eq!(s.resolve_buffer(b, 0, 8).unwrap().len(), 2);

    s.release_working_set(b, 3);
    s.retire_through(3);
    assert_eq!(s.available_slots(), 12);
}

#[test]
fn abort_releases_pending_slots_and_keeps_committed_state() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, true, Some((0, 8))).unwrap();
    assert_eq!(s.available_slots(), 9); // 1 state + 2 buffer pending
    s.abort(prepared, 5);
    s.retire_through(5);
    assert_eq!(s.available_slots(), 12);
    assert_eq!(s.folded_slot(ws).unwrap(), None);
    assert_eq!(
        s.resolve_buffer(ws, 0, 4),
        Err(RsError::UnmaterializedRead { index: 0 })
    );
}

#[test]
fn pool_exhaustion_is_typed_and_all_or_nothing() {
    let mut s = RsStore::new(2);
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    // Needs 1 state slot + 2 buffer pages; only 2 slots exist.
    let err = s.prepare_write(ws, true, Some((0, 8))).unwrap_err();
    assert_eq!(
        err,
        RsError::OutOfSlots {
            requested: 3,
            available: 2
        }
    );
    assert_eq!(s.available_slots(), 2); // nothing leaked
}

#[test]
fn unmaterialized_read_is_an_error_and_zero_length_read_is_empty() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 1).unwrap();
    assert_eq!(s.resolve_buffer(ws, 0, 0).unwrap(), Vec::new());
    assert_eq!(
        s.resolve_buffer(ws, 0, 4),
        Err(RsError::UnmaterializedRead { index: 0 })
    );
}
