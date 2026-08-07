//! Unit tests for the RS store.

use super::write::RsBufferTarget;
use super::{RsBufferIntent, RsError, RsGeometry, RsStore, RsWorkingSetId};

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: 4096,
        buffer_page_tokens: 4,
        fold_granularity: 4,
    }
}

/// Production declares `fold_granularity = 1` (token-causal) while a buffer
/// page is the KV page size, so folds routinely land mid-page. The default
/// `geom()` hides that by making the two equal.
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
    settled(&mut s, prepared);
    assert_eq!(s.folded_slot(ws).unwrap(), Some(slot));
}

#[test]
fn continuing_state_write_is_in_place() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    let slot = s.folded_slot(ws).unwrap().unwrap();
    let prepared = s.prepare_write(ws, true, None).unwrap();
    let state = prepared.state().unwrap();
    assert_eq!(state.slot, slot);
    assert!(!state.reset);
    assert!(state.copy_from.is_none());
    assert_eq!(s.available_slots(), 11); // no new allocation
    settled(&mut s, prepared);
}

#[test]
fn forked_state_write_copies_on_write() {
    let mut s = store();
    let a = s.create_working_set(geom());
    write_state(&mut s, a);
    let shared = s.folded_slot(a).unwrap().unwrap();
    let b = s.fork(a).unwrap();

    let prepared = s.prepare_write(b, true, None).unwrap();
    let state = *prepared.state().unwrap();
    assert!(!state.reset);
    assert_eq!(state.copy_from, Some(shared));
    assert_ne!(state.slot, shared);
    settled(&mut s, prepared);

    // b owns the copy; a still reads the shared original.
    assert_eq!(s.folded_slot(b).unwrap(), Some(state.slot));
    assert_eq!(s.folded_slot(a).unwrap(), Some(shared));

    // a is now the sole owner: its next write is in place again.
    let prepared = s.prepare_write(a, true, None).unwrap();
    assert_eq!(prepared.state().unwrap().slot, shared);
    settled(&mut s, prepared);
}

// ----------------------------------------------------------------------
// Run-ahead: the publication seam. Every test below prepares a successor
// while its predecessor is still in flight — the exact shape the deleted
// `drain_rs_predecessors` barrier used to forbid.
// ----------------------------------------------------------------------

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

#[test]
fn run_ahead_buffer_writes_materialize_each_page_once() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();

    let first = s.prepare_write(ws, false, Some((0, 4))).unwrap();
    let page = first.buffer_targets()[0].dst();
    assert!(matches!(
        first.buffer_targets()[0],
        RsBufferTarget::Fresh { .. }
    ));
    let first = s.publish_prepared(first).unwrap();

    // Same page, predecessor still in flight: must be in place, not a second
    // materialization.
    let second = s.prepare_write(ws, false, Some((0, 4))).unwrap();
    assert!(
        matches!(second.buffer_targets()[0], RsBufferTarget::InPlace { dst, .. } if dst == page),
        "a run-ahead successor must not materialize the same page twice"
    );
    // A disjoint append is independent of both.
    let third = s.prepare_write(ws, false, Some((4, 4))).unwrap();
    assert!(matches!(
        third.buffer_targets()[0],
        RsBufferTarget::Fresh { .. }
    ));

    let second = s.publish_prepared(second).unwrap();
    let third = s.publish_prepared(third).unwrap();
    s.settle(first);
    s.settle(second);
    s.settle(third);
    assert_eq!(s.resolve_buffer(ws, 0, 8).unwrap().len(), 2);
}

#[test]
fn fork_between_run_ahead_fires_snapshots_the_submitted_mapping() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    let first = s.prepare_write(ws, true, None).unwrap();
    let slot = first.state().unwrap().slot;
    let first = s.publish_prepared(first).unwrap();

    // Fork while the fire is still on the device: it shares the slot that
    // fire is writing, which is exactly "fork ordered after that submit".
    let child = s.fork(ws).unwrap();
    assert_eq!(s.folded_slot(child).unwrap(), Some(slot));

    // Both sides now CoW away from the shared slot on their next write.
    let parent_next = s.prepare_write(ws, true, None).unwrap();
    assert_eq!(parent_next.state().unwrap().copy_from, Some(slot));
    let child_next = s.prepare_write(child, true, None).unwrap();
    assert_eq!(child_next.state().unwrap().copy_from, Some(slot));
    assert_ne!(
        parent_next.state().unwrap().slot,
        child_next.state().unwrap().slot
    );

    let parent_next = s.publish_prepared(parent_next).unwrap();
    let child_next = s.publish_prepared(child_next).unwrap();
    s.settle(first);
    s.settle(parent_next);
    s.settle(child_next);
}

#[test]
fn three_deep_run_ahead_resets_once_and_allocates_once() {
    // A k=3 frame's slots prepare back to back before any of them settles.
    let mut s = store();
    let ws = s.create_working_set(geom());
    let mut published = Vec::new();
    let mut slots = Vec::new();
    let mut resets = 0;
    for _ in 0..3 {
        let prepared = s.prepare_write(ws, true, None).unwrap();
        let state = *prepared.state().unwrap();
        resets += usize::from(state.reset);
        slots.push(state.slot);
        assert!(state.copy_from.is_none());
        published.push(s.publish_prepared(prepared).unwrap());
    }
    assert_eq!(resets, 1, "only the cold slot resets");
    assert_eq!(slots[0], slots[1]);
    assert_eq!(slots[1], slots[2], "all three waves write one folded slot");
    assert_eq!(s.available_slots(), 11, "one slot for the whole frame");
    for handle in published {
        s.settle(handle);
    }
    assert_eq!(s.folded_slot(ws).unwrap(), Some(slots[0]));
}

#[test]
fn retirement_waits_for_every_in_flight_write() {
    let mut s = store();
    let parent = store_with_state(&mut s);
    let shared = s.folded_slot(parent).unwrap().unwrap();
    let child = s.fork(parent).unwrap();

    let cow = s.prepare_write(child, true, None).unwrap();
    assert_eq!(cow.state().unwrap().copy_from, Some(shared));
    let cow = s.publish_prepared(cow).unwrap();
    let free_after_publish = s.available_slots();

    // A second write is still in flight when the shared slot loses its last
    // owner, so the displaced slot must NOT be handed back out.
    let held = s.prepare_write(child, true, None).unwrap();
    let held = s.publish_prepared(held).unwrap();
    s.release_working_set(parent, s.current_epoch());
    s.settle(cow);
    assert_eq!(
        s.available_slots(),
        free_after_publish,
        "retirement waits for the write whose epoch the free is tagged with"
    );

    s.settle(held);
    assert!(
        s.available_slots() > free_after_publish,
        "the displaced slot returns once nothing is in flight"
    );
}

#[test]
fn a_completed_free_retires_while_a_newer_write_is_outstanding() {
    let mut s = store();
    let doomed = store_with_state(&mut s);
    let other = s.create_working_set(geom());

    // Nothing is outstanding, so this free is tagged with an epoch that has
    // already completed on the device.
    s.release_working_set(doomed, s.current_epoch());

    // Now a NEWER write goes in flight. Its sequence is above the free's tag,
    // so it was prepared after the slot became unreachable and cannot be
    // touching it. Retirement must not wait for it.
    let held = s.prepare_write(other, true, None).unwrap();
    let held = s.publish_prepared(held).unwrap();
    let under_load = s.available_slots();

    s.retire_idle();
    assert!(
        s.available_slots() > under_load,
        "a free whose epoch has completed retires even under sustained load"
    );
    s.settle(held);
}

#[test]
fn a_failed_fire_keeps_its_published_mapping() {
    // Fail-stop, exactly like KV: settlement never rolls the mapping back.
    let mut s = store();
    let ws = s.create_working_set(geom());
    let prepared = s.prepare_write(ws, true, None).unwrap();
    let slot = prepared.state().unwrap().slot;
    let published = s.publish_prepared(prepared).unwrap();
    s.settle(published); // the fire failed on the device
    assert_eq!(
        s.folded_slot(ws).unwrap(),
        Some(slot),
        "a failed fire leaves pipeline-local fail-stop state, not a rollback"
    );
    // The slot stays owned, so it is not back in the pool.
    assert_eq!(s.available_slots(), 11);
}

fn store_with_state(s: &mut RsStore) -> RsWorkingSetId {
    let ws = s.create_working_set(geom());
    write_state(s, ws);
    ws
}

// ----------------------------------------------------------------------
// Buffers, folds, lifecycle
// ----------------------------------------------------------------------

#[test]
fn buffer_writes_materialize_then_in_place_then_cow_after_fork() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 3).unwrap();

    // First write materializes all three pages.
    let prepared = s.prepare_write(ws, false, Some((0, 12))).unwrap();
    assert!(
        prepared
            .buffer_targets()
            .iter()
            .all(|t| matches!(t, RsBufferTarget::Fresh { .. }))
    );
    let ids: Vec<_> = prepared.buffer_targets().iter().map(|t| t.dst()).collect();
    settled(&mut s, prepared);
    assert_eq!(s.resolve_buffer(ws, 0, 12).unwrap(), ids);

    // Second write is in place.
    let prepared = s.prepare_write(ws, false, Some((0, 12))).unwrap();
    assert!(
        prepared
            .buffer_targets()
            .iter()
            .all(|t| matches!(t, RsBufferTarget::InPlace { .. }))
    );
    settled(&mut s, prepared);

    // After a fork the pages are shared: partial-range write CoWs its span.
    let forked = s.fork(ws).unwrap();
    let prepared = s.prepare_write(forked, false, Some((4, 4))).unwrap();
    assert_eq!(prepared.buffer_targets().len(), 1);
    assert!(matches!(
        prepared.buffer_targets()[0],
        RsBufferTarget::Cow { .. }
    ));
    assert_eq!(prepared.buffer_copy_plan().count(), 1);
    settled(&mut s, prepared);
    // Original keeps its page; fork has the copy at the same slot index.
    assert_eq!(s.resolve_buffer(ws, 4, 4).unwrap(), vec![ids[1]]);
    assert_ne!(s.resolve_buffer(forked, 4, 4).unwrap(), vec![ids[1]]);
}

#[test]
fn fold_validates_granularity_and_capacity() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap(); // capacity 8 tokens
    let write = |t, b| s.validate_fold(ws, t, Some(b), RsBufferIntent::Write);
    assert_eq!(write(0, (0, 8)), Err(RsError::FoldZero));
    assert_eq!(
        write(6, (0, 8)),
        Err(RsError::FoldGranularity {
            tokens: 6,
            granularity: 4
        })
    );
    assert_eq!(
        write(12, (0, 12)),
        Err(RsError::FoldExceedsBuffer {
            tokens: 12,
            capacity: 8
        })
    );
    assert_eq!(write(8, (0, 8)), Ok(()));
}

#[test]
fn a_fold_may_not_reach_past_the_tokens_that_exist() {
    // Two pages of four hold capacity for eight, but only four tokens are
    // live. Bounding on capacity would accept a fold of eight and gather
    // four slab tokens that were never written.
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s
        .prepare_general(ws, false, None, Some((0, 4)), RsBufferIntent::Write)
        .unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws), Ok(4));
    // A commit replays only what is buffered.
    assert_eq!(
        s.validate_fold(ws, 8, None, RsBufferIntent::Replay),
        Err(RsError::FoldExceedsBuffer {
            tokens: 8,
            capacity: 4
        })
    );
    assert_eq!(s.validate_fold(ws, 4, None, RsBufferIntent::Replay), Ok(()));
    // A write extends the live space by its own new tokens.
    assert_eq!(
        s.validate_fold(ws, 8, Some((4, 4)), RsBufferIntent::Write),
        Ok(())
    );
}

#[test]
fn published_fold_advances_the_boundary_and_drops_head_pages() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 3).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 12))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.available_slots(), 9);

    let prepared = s.prepare_fold(ws, 8).unwrap();
    assert_eq!(prepared.state().unwrap().fold_tokens, Some(8));
    assert_eq!(s.available_slots(), 8); // fresh folded slot taken
    let published = s.publish_prepared(prepared).unwrap();
    // The boundary moves at publish; the dropped pages are recycled but
    // still held while the fold is in flight.
    assert_eq!(s.buffer_size(ws).unwrap(), 1);
    assert_eq!(s.available_slots(), 8);
    s.settle(published);
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
    write_state(&mut s, a);
    s.alloc_buffer(a, 2).unwrap();
    let prepared = s.prepare_write(a, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);
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
fn cancel_releases_pending_slots_and_leaves_the_mapping_untouched() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, true, Some((0, 8))).unwrap();
    assert_eq!(s.available_slots(), 9); // 1 state + 2 buffer pending
    s.cancel_prepared(prepared);
    assert_eq!(s.available_slots(), 12);
    assert_eq!(s.folded_slot(ws).unwrap(), None);
    assert_eq!(
        s.resolve_buffer(ws, 0, 4),
        Err(RsError::UnmaterializedRead { index: 0 })
    );
}

#[test]
fn publish_batch_rejects_a_released_working_set_and_cancels_every_row() {
    let mut s = store();
    let first = s.create_working_set(geom());
    let second = s.create_working_set(geom());
    let a = s.prepare_write(first, true, None).unwrap();
    let b = s.prepare_write(second, true, None).unwrap();
    s.release_working_set(second, s.current_epoch());

    assert_eq!(
        s.publish_batch(vec![a, b]).err(),
        Some(RsError::UnknownWorkingSet)
    );
    assert_eq!(
        s.folded_slot(first).unwrap(),
        None,
        "no row is adopted when the batch fails validation"
    );
    assert_eq!(s.available_slots(), 12, "both allocations returned");
}

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

#[test]
fn publish_batch_adopts_every_row_of_one_fire() {
    let mut s = store();
    let first = s.create_working_set(geom());
    let second = s.create_working_set(geom());
    let a = s.prepare_write(first, true, None).unwrap();
    let b = s.prepare_write(second, true, None).unwrap();
    let (a_slot, b_slot) = (a.state().unwrap().slot, b.state().unwrap().slot);
    let (published, folds) = s.publish_batch(vec![a, b]).unwrap();
    s.commit_folds(folds);
    assert_eq!(published.rows(), 2);
    assert_eq!(s.folded_slot(first).unwrap(), Some(a_slot));
    assert_eq!(s.folded_slot(second).unwrap(), Some(b_slot));
    s.settle(published);
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

/// Reserving a buffer page does not buffer a token.
///
/// The fire classifier asks "is the buffer empty?" to decide whether an append
/// is the legal empty-buffer case or the unimplemented read path. It used to
/// ask the PAGE count, but a guest must reserve a page before it can write into
/// one — so a brand-new buffer claimed a full page of occupancy and every
/// speculative window was refused before it ran.
#[test]
fn reserving_a_buffer_page_does_not_buffer_a_token() {
    let mut s = store();
    let ws = s.create_working_set(geom());

    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        0,
        "a new working set is empty"
    );

    s.alloc_buffer(ws, 2).unwrap();
    assert_eq!(s.buffer_size(ws).unwrap(), 2, "two pages reserved");
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        0,
        "reserved but unwritten: still zero buffered tokens"
    );

    let prepared = s.prepare_write(ws, false, Some((0, 3))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        3,
        "exactly the tokens written, not the 4-token page they sit in"
    );

    // Rewriting the same span must not double-count.
    let prepared = s.prepare_write(ws, false, Some((0, 3))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        3,
        "a rewrite is not an append"
    );

    // A disjoint append advances the fill to its far edge.
    let prepared = s.prepare_write(ws, false, Some((4, 4))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 8);
}

/// Folding absorbs buffered tokens into the folded prefix, so they stop being
/// buffered — which is what lets the next window append onto an empty buffer.
#[test]
fn folding_drains_the_buffered_token_count() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();

    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 8);

    let prepared = s.prepare_fold(ws, 4).unwrap();
    settled(&mut s, prepared);
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        4,
        "half the buffer folded away"
    );

    let prepared = s.prepare_fold(ws, 4).unwrap();
    settled(&mut s, prepared);
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        0,
        "folding everything empties the buffer"
    );
}

/// Freeing every buffered page empties the buffer — the reset a speculative
/// loop performs between windows.
#[test]
fn freeing_every_page_empties_the_buffered_token_count() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 8);

    s.free_buffer(ws, &[0, 1], 0).unwrap();
    assert_eq!(s.buffer_size(ws).unwrap(), 0);
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        0,
        "no pages left, so nothing is buffered"
    );

    // And a fresh reservation is once again genuinely empty.
    s.alloc_buffer(ws, 1).unwrap();
    assert_eq!(s.buffer_tokens(ws).unwrap(), 0);
}

/// A fork shares the buffer, so it must share the occupancy too — otherwise
/// the child would classify its first fire as an empty-buffer append.
#[test]
fn fork_inherits_the_buffered_token_count() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 5))).unwrap();
    settled(&mut s, prepared);

    let child = s.fork(ws).unwrap();
    assert_eq!(s.buffer_tokens(child).unwrap(), 5);
}

/// A fold that lands MID-PAGE cannot release the page it half-consumed, so the
/// surviving tokens keep their physical offsets. `buffer_head` is what records
/// that, and every later span has to be resolved through it — otherwise the
/// next append overwrites live tokens and the next replay re-scans tokens that
/// are already inside the folded state.
#[test]
fn a_partial_fold_moves_the_buffer_head_rather_than_compacting() {
    let mut s = store();
    let ws = s.create_working_set(geom_with_granularity(1));
    s.alloc_buffer(ws, 2).unwrap();

    // 4 tokens into page 0 (page = 4 tokens).
    let prepared = s.prepare_write(ws, false, Some((0, 4))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 4);
    assert_eq!(s.buffer_head(ws).unwrap(), 0);

    // Fold 2 of them. Page 0 is only half covered, so it survives.
    let prepared = s.prepare_fold(ws, 2).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 2, "two tokens still buffered");
    assert_eq!(s.buffer_size(ws).unwrap(), 2, "no page could be released");
    assert_eq!(
        s.buffer_head(ws).unwrap(),
        2,
        "logical token 0 now sits at physical offset 2"
    );

    // Appending onto that buffer starts at LOGICAL 2 == PHYSICAL 4, which is
    // page 1. Resolving it as physical 2 would overwrite a live token.
    let prepared = s.prepare_write(ws, false, Some((2, 2))).unwrap();
    let pages: Vec<u32> = prepared
        .buffer_targets()
        .iter()
        .map(|target| match *target {
            RsBufferTarget::Fresh { index, .. }
            | RsBufferTarget::InPlace { index, .. }
            | RsBufferTarget::Cow { index, .. } => index,
        })
        .collect();
    assert_eq!(
        pages,
        vec![1],
        "the append lands on page 1, not back inside page 0"
    );
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 4);
}

/// Folding whole pages away rebases the head back down: dropping page 0 moves
/// every survivor one page earlier.
#[test]
fn folding_a_whole_page_rebases_the_head() {
    let mut s = store();
    let ws = s.create_working_set(geom_with_granularity(1));
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);

    let prepared = s.prepare_fold(ws, 6).unwrap();
    settled(&mut s, prepared);
    assert_eq!(
        s.buffer_size(ws).unwrap(),
        1,
        "page 0 fully covered, dropped"
    );
    assert_eq!(
        s.buffer_head(ws).unwrap(),
        2,
        "6 folded - 4 dropped with the page = 2 into the surviving page"
    );
    assert_eq!(s.buffer_tokens(ws).unwrap(), 2);
}

/// A fold's span names the pages it REPLAYS on the way into the folded state.
/// Those tokens leave the buffer; counting the span as a write would re-add
/// what the fold just removed.
#[test]
fn a_fold_does_not_count_its_replay_span_as_newly_buffered() {
    let mut s = store();
    let ws = s.create_working_set(geom_with_granularity(1));
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 4))).unwrap();
    settled(&mut s, prepared);

    // Fold 3 of 4 — more than half, which is what exposes the double count.
    let prepared = s.prepare_write(ws, true, Some((0, 3))).unwrap();
    let prepared = {
        let _ = prepared;
        s.prepare_fold(ws, 3).unwrap()
    };
    settled(&mut s, prepared);
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        1,
        "one token left, not the 3 the replay span named"
    );
}

/// The mtp-native-verify shape: every window allocates fresh pages, buffers a
/// window, folds a prefix and frees everything. Without a rebase the head
/// ratchets up until a fresh single-page window no longer fits its own tokens.
#[test]
fn emptying_the_buffer_rebases_the_head_so_the_next_window_starts_at_zero() {
    let mut s = store();
    let ws = s.create_working_set(geom_with_granularity(1));
    for _ in 0..4 {
        s.alloc_buffer(ws, 1).unwrap();
        let prepared = s.prepare_write(ws, false, Some((0, 3))).unwrap();
        settled(&mut s, prepared);
        let prepared = s.prepare_fold(ws, 2).unwrap();
        settled(&mut s, prepared);
        let size = s.buffer_size(ws).unwrap();
        s.free_buffer(ws, &(0..size).collect::<Vec<_>>(), 0)
            .unwrap();
        assert_eq!(
            s.buffer_head(ws).unwrap(),
            0,
            "an emptied buffer must not carry a head into the next window"
        );
    }
}

/// A fold whose length lived on the device makes the boundary INDETERMINATE.
///
/// The host planned against an upper bound, so after the fire it knows only
/// that `F` moved somewhere into `[F, F+b]`. Guessing either way is
/// unrecoverable: guessing MORE drops pages that are still live, guessing LESS
/// replays tokens already absorbed into the state — a double fold. So the
/// store advances nothing, retains every page, and REFUSES to report an exact
/// occupancy until the guest empties the buffer itself.
#[test]
fn a_device_fold_length_suspends_the_boundary_rather_than_guessing() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 8);
    assert!(s.buffer_tokens_exact(ws));

    let mut prepared = s.prepare_fold(ws, 8).unwrap();
    prepared.mark_fold_len_device();
    assert!(prepared.fold_len_is_bound());
    settled(&mut s, prepared);

    // Every page is retained: the boundary may not have moved at all.
    assert_eq!(s.buffer_size(ws).unwrap(), 2);
    assert!(!s.buffer_tokens_exact(ws));
    assert_eq!(
        s.buffer_tokens_bound(ws).unwrap(),
        8,
        "the pre-fold occupancy is now only an upper bound"
    );
    assert!(
        matches!(
            s.buffer_tokens(ws),
            Err(RsError::BufferOccupancyIndeterminate { bound: 8 })
        ),
        "an exact read must refuse rather than return the bound"
    );
}

/// ...and freeing the buffer restores exactness, which is what makes the
/// suspension recoverable instead of terminal. This is the shape a speculative
/// loop already has: buffer a window, commit its accepted prefix, free the
/// window, start the next one.
#[test]
fn freeing_the_buffer_restores_an_exact_occupancy_after_a_device_fold() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);

    let mut prepared = s.prepare_fold(ws, 8).unwrap();
    prepared.mark_fold_len_device();
    settled(&mut s, prepared);
    assert!(s.buffer_tokens(ws).is_err());

    s.free_buffer(ws, &[0, 1], 0).unwrap();
    assert!(s.buffer_tokens_exact(ws));
    assert_eq!(
        s.buffer_tokens(ws).unwrap(),
        0,
        "no pages left, so the occupancy is knowable again"
    );

    s.alloc_buffer(ws, 1).unwrap();
    assert_eq!(s.buffer_tokens(ws).unwrap(), 0);
}

/// A bound of zero is not a bound: it pins the true occupancy at exactly zero,
/// because the live buffer is somewhere in `0..=n` and `n` is zero.
///
/// `Occupancy::at_most` collapses that case at construction, so `AtMost(0)` is
/// unrepresentable. This is the invariant the type exists to hold: the old
/// pair of fields could spell it, and three separate call sites had to
/// remember to normalize it away by hand. Discarding the whole bound from the
/// TAIL is the path that reaches zero without the guest ever freeing a page,
/// so it is the one that would regress silently.
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

/// A fork shares the buffer, so it inherits the indeterminacy too — otherwise/// the child would read a stale exact occupancy and plan a second fold over
/// tokens the parent's device fold may already have absorbed.
#[test]
fn a_fork_inherits_an_indeterminate_buffer() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 2).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 8))).unwrap();
    settled(&mut s, prepared);
    let mut prepared = s.prepare_fold(ws, 8).unwrap();
    prepared.mark_fold_len_device();
    settled(&mut s, prepared);

    let child = s.fork(ws).unwrap();
    assert!(!s.buffer_tokens_exact(child));
    assert!(s.buffer_tokens(child).is_err());
}

/// `publish_batch` must NOT move the fold boundary.
///
/// Every wire array a fire carries describes the buffer as that fire's own
/// rows are laid out: extended row `j` is physical `buffer_head + j`, the
/// write CSR lists the pages that span holds, and the read CSR starts at the
/// head. All of it is the PRE-fold frame. Advancing the boundary inside
/// `publish_batch` therefore handed a fire whose rows straddle the boundary a
/// head — and a page list — from the far side of it.
///
/// It stayed invisible for as long as it did because the only folds that
/// existed either moved nothing (`n == 0`) or emptied the buffer (`n == b+t`,
/// which rebases the head to 0 anyway). An INTERIOR fold is neither, and it
/// read the tokens the fold had just absorbed instead of the ones that
/// survived it.
#[test]
fn publishing_a_fold_leaves_the_boundary_for_commit_folds() {
    let mut s = store();
    let ws = s.create_working_set(geom_with_granularity(1));
    s.alloc_buffer(ws, 2).unwrap();

    // Four tokens in, boundary still at 0.
    let prepared = s.prepare_write(ws, false, Some((0, 4))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_head(ws).unwrap(), 0);

    // A fire that writes its own tokens AND folds a prefix of the result.
    let prepared = s.prepare_fold(ws, 2).unwrap();
    let (published, folds) = s.publish_batch(vec![prepared]).unwrap();

    assert_eq!(
        s.buffer_head(ws).unwrap(),
        0,
        "the wire arrays are still being built against the pre-fold buffer"
    );
    assert_eq!(s.buffer_tokens(ws).unwrap(), 4, "and against its occupancy");

    s.commit_folds(folds);
    assert_eq!(
        s.buffer_head(ws).unwrap(),
        2,
        "only now does logical token 0 move to physical offset 2"
    );
    assert_eq!(s.buffer_tokens(ws).unwrap(), 2);
    s.settle(published);
}

/// `discard` is the twin of a fold on the other end of the buffer: a fold
/// moves the folded boundary right and cannot be undone, this moves the live
/// end left and costs nothing. It releases CONTENT; `free_buffer` releases
/// CAPACITY. Having only the second is why a speculative window needed two
/// fires — dropping a rejected tail meant emptying the buffer, which forced
/// the accepted prefix to be folded away first.
#[test]
fn discarding_buffered_tokens_releases_content_but_not_capacity() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    s.alloc_buffer(ws, 3).unwrap();
    let prepared = s.prepare_write(ws, false, Some((0, 10))).unwrap();
    settled(&mut s, prepared);
    assert_eq!(s.buffer_tokens(ws).unwrap(), 10);

    s.discard_buffered(ws, 4).unwrap();
    assert_eq!(s.buffer_tokens(ws).unwrap(), 6);
    assert_eq!(s.buffer_size(ws).unwrap(), 3); // pages untouched

    // It may take the whole live buffer, and not one token more.
    assert_eq!(
        s.discard_buffered(ws, 7),
        Err(RsError::DiscardExceedsBuffer {
            count: 7,
            buffered: 6
        })
    );
    s.discard_buffered(ws, 6).unwrap();
    assert_eq!(s.buffer_tokens(ws).unwrap(), 0);
    assert_eq!(s.buffer_size(ws).unwrap(), 3);
}
