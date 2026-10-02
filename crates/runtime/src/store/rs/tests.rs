use super::{RsError, RsGeometry, RsStore, RsWorkingSetId};

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: 4096,
        buffer_page_tokens: 4,
        fold_granularity: 4,
    }
}

fn geom_with_granularity(granularity: u32) -> RsGeometry {
    RsGeometry {
        fold_granularity: granularity,
        ..geom()
    }
}

fn store() -> RsStore {
    RsStore::new(12)
}

fn settled(store: &mut RsStore, prepared: super::RsPreparedWrite) {
    let published = store.publish_prepared(prepared).unwrap();
    store.settle(published);
}

fn write_state(store: &mut RsStore, ws: RsWorkingSetId) {
    let prepared = store.prepare_write(ws, true, None).unwrap();
    settled(store, prepared);
}

#[test]
fn tests_every_case() {
    run_ahead_successor_never_resets_twice();
    run_ahead_successor_after_fork_cows_exactly_once();
    publish_batch_rejects_an_aliased_working_set();
    a_bound_driven_to_zero_is_exact_again();
    an_indexed_state_survives_its_owner_and_forks_copy_on_write();
    snapshots_yield_slots_oldest_first_under_pressure();
}

fn run_ahead_successor_never_resets_twice() {
    let mut s = store();
    let ws = s.create_working_set(geom());

    let first = s.prepare_write(ws, true, None).unwrap();
    assert!(first.state().unwrap().reset, "cold state resets once");
    let slot = first.state().unwrap().slot;
    let first = s.publish_prepared(first).unwrap();

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

fn run_ahead_successor_after_fork_cows_exactly_once() {
    let mut s = store();
    let parent = store_with_state(&mut s);
    let shared = s.folded_slot(parent).unwrap().unwrap();
    let child = s.fork(parent).unwrap();

    let first = s.prepare_write(child, true, None).unwrap();
    let private = first.state().unwrap().slot;
    assert_eq!(first.state().unwrap().copy_from, Some(shared));
    let first = s.publish_prepared(first).unwrap();

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

    s.discard_buffered(ws, 3).unwrap();
    assert!(!s.buffer_tokens_exact(ws));
    assert_eq!(s.buffer_tokens_bound(ws).unwrap(), 5);

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

fn an_indexed_state_survives_its_owner_and_forks_copy_on_write() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    let slot = s.folded_slot(ws).unwrap().unwrap();

    s.update_index(b"k".to_vec(), ws).unwrap();
    s.release_working_set(ws, s.current_epoch());
    s.retire_idle();
    assert_eq!(s.available_slots(), 11, "the snapshot keeps the slot alive");

    let hit = s.from_index(b"k").unwrap().expect("indexed");
    assert_eq!(s.folded_slot(hit).unwrap(), Some(slot));
    let write = s.prepare_write(hit, true, None).unwrap();
    let state = write.state().unwrap();
    assert_ne!(state.slot, slot, "a hit never writes the shared slot");
    assert_eq!(state.copy_from, Some(slot));
    assert!(!state.reset);
    settled(&mut s, write);

    let again = s.from_index(b"k").unwrap().expect("still indexed");
    assert_eq!(s.folded_slot(again).unwrap(), Some(slot));
    assert!(s.from_index(b"missing").unwrap().is_none());
    assert!(s.remove_index(b"k"));
    assert!(!s.remove_index(b"k"));
}

fn snapshots_yield_slots_oldest_first_under_pressure() {
    let mut s = RsStore::new(8);
    let mut keys = Vec::new();
    for i in 0..2u8 {
        let ws = s.create_working_set(geom());
        write_state(&mut s, ws);
        s.update_index(vec![i], ws).unwrap();
        s.release_working_set(ws, s.current_epoch());
        keys.push(vec![i]);
    }
    s.retire_idle();
    assert_eq!(s.snapshot_count(), 2);
    assert_eq!(s.available_slots(), 6);

    assert!(s.from_index(&keys[0]).unwrap().is_some());
    let granted = s.reserve_slots(7).expect("evicts one snapshot");
    assert_eq!(granted.len(), 7);
    assert_eq!(s.snapshot_count(), 1);
    assert!(s.from_index(&keys[1]).unwrap().is_none(), "the oldest went first");
    assert!(s.from_index(&keys[0]).unwrap().is_some());
}
