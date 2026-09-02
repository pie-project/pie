//! Unit tests for the RS store.

use super::{RsError, RsGeometry, RsStore, RsWorkingSetId};

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: 4096,
        buffer_page_tokens: 4,
        fold_granularity: 4,
    }
}

/// Production uses `fold_granularity = 1`; the default `geom()` sets it to
/// the page size instead, so folds land on page boundaries.
fn geom_with_granularity(granularity: u32) -> RsGeometry {
    RsGeometry {
        fold_granularity: granularity,
        ..geom()
    }
}

fn store() -> RsStore {
    RsStore::new(12)
}

/// Publish a prepared write and settle it immediately — the shape of a fire
/// that resolved before the next one prepared.
fn settled(store: &mut RsStore, prepared: super::RsPreparedWrite) {
    let published = store.publish_prepared(prepared).unwrap();
    store.settle(published);
}

fn write_state(store: &mut RsStore, ws: RsWorkingSetId) {
    let prepared = store.prepare_write(ws, true, None).unwrap();
    settled(store, prepared);
}

// Run-ahead: each test below prepares a successor while its predecessor is
// still in flight.

#[test]
fn run_ahead_successor_never_resets_twice() {
    let mut s = store();
    let ws = s.create_working_set(geom());

    let first = s.prepare_write(ws, true, None).unwrap();
    assert!(first.state().unwrap().reset, "cold state resets once");
    let slot = first.state().unwrap().slot;
    let first = s.publish_prepared(first).unwrap();

    // The predecessor has NOT settled: it is still on the device.
    let second = s.prepare_write(ws, true, None).unwrap();
    let state = second.state().unwrap();
    assert_eq!(state.slot, slot, "successor continues the published slot");
    assert!(!state.reset, "a run-ahead successor must not RESET again");
    assert!(state.copy_from.is_none());
    assert_eq!(s.available_slots(), 11, "no second allocation");

    let second = s.publish_prepared(second).unwrap();
    s.settle(first);
    s.settle(second);
    assert_eq!(s.folded_slot(ws).unwrap(), Some(slot));
}

#[test]
fn run_ahead_successor_after_fork_cows_exactly_once() {
    let mut s = store();
    let parent = store_with_state(&mut s);
    let shared = s.folded_slot(parent).unwrap().unwrap();
    let child = s.fork(parent).unwrap();

    let first = s.prepare_write(child, true, None).unwrap();
    let private = first.state().unwrap().slot;
    assert_eq!(first.state().unwrap().copy_from, Some(shared));
    let first = s.publish_prepared(first).unwrap();

    // Still in flight — the successor must see the privatized slot.
    let second = s.prepare_write(child, true, None).unwrap();
    let state = second.state().unwrap();
    assert_eq!(state.slot, private, "successor continues the CoW slot");
    assert!(
        state.copy_from.is_none(),
        "a run-ahead successor must not re-copy from the stale parent"
    );
    assert!(!state.reset);

    let second = s.publish_prepared(second).unwrap();
    s.settle(first);
    s.settle(second);
    assert_eq!(s.folded_slot(parent).unwrap(), Some(shared));
    assert_eq!(s.folded_slot(child).unwrap(), Some(private));
}

fn store_with_state(s: &mut RsStore) -> RsWorkingSetId {
    let ws = s.create_working_set(geom());
    write_state(s, ws);
    ws
}

// Buffers, folds, lifecycle

#[test]
fn publish_batch_rejects_an_aliased_working_set() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    let a = s.prepare_write(ws, true, None).unwrap();
    let b = s.prepare_write(ws, true, None).unwrap();
    assert_eq!(
        s.publish_batch(vec![a, b]).err(),
        Some(RsError::DuplicateWorkingSet)
    );
    assert_eq!(s.folded_slot(ws).unwrap(), None);
    assert_eq!(s.available_slots(), 12);
}

// A bound of zero pins the true occupancy at exactly zero; `Occupancy::at_most`
// collapses that case at construction so `AtMost(0)` is unrepresentable.
#[test]
fn a_bound_driven_to_zero_is_exact_again() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);

    let mut prepared = s.prepare_fold(ws, 8).unwrap();
    prepared.mark_fold_len_device();
    settled(&mut s, prepared);
    assert!(!s.buffer_tokens_exact(ws), "the device fold suspended it");

    // Discarding fewer than the bound leaves it a bound: the uncertainty is
    // how much the fold took off the FRONT, and this takes from the TAIL.
    s.discard_buffered(ws, 3).unwrap();
    assert!(!s.buffer_tokens_exact(ws));
    assert_eq!(s.buffer_tokens_bound(ws).unwrap(), 5);

    // Discarding the rest drives the bound to zero, which leaves nothing for
    // the fold to have absorbed that is not already accounted for.
    s.discard_buffered(ws, 5).unwrap();
    assert!(
        s.buffer_tokens_exact(ws),
        "a bound of zero pins the count, so exactness returns without free_buffer"
    );
    assert_eq!(s.buffer_tokens(ws).unwrap(), 0);
    assert_eq!(
        s.buffer_size(ws).unwrap(),
        2,
        "capacity is untouched: this released TOKENS, not pages"
    );
}

