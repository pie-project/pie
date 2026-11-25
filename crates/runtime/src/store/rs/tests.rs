use super::{RsError, RsGeometry, RsStore, RsUnits, RsWorkingSetId};

fn slots(slots: u32) -> RsUnits {
    RsUnits::of(slots, 0)
}

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: 4096,
        buffer_page_tokens: 4,
    }
}

fn store() -> RsStore {
    RsStore::new(12, 12)
}

fn settled(store: &mut RsStore, prepared: super::RsPreparedWrite) {
    let published = store.publish_prepared(prepared).unwrap();
    store.settle(published);
}

fn write_state(store: &mut RsStore, ws: RsWorkingSetId) {
    let prepared = store
        .prepare_group(ws, true, false, None, None, None)
        .unwrap();
    settled(store, prepared);
}

#[test]
fn tests_every_case() {
    run_ahead_successor_never_resets_twice();
    run_ahead_successor_after_fork_cows_exactly_once();
    publish_batch_rejects_an_aliased_working_set();
    a_fold_drops_the_buffer_pages_behind_it();
    a_truncate_takes_back_the_buffer_past_its_cut();
    a_state_others_read_is_folded_into_a_copy();
    a_checkpoint_is_cache_and_goes_when_idle();
    new_weights_drop_every_checkpoint();
    idle_snapshots_go_in_retention_order();
    a_checkpoint_past_a_step_goes_first();
    a_checkpoint_resumes_a_fresh_working_set();
    a_snapshot_resumes_at_its_row();
}

// A checkpoint's rows are held by its cache entry: they count as cache only
// once no working set holds them too, and the hold goes with the checkpoint.

fn idle_snapshots_go_in_retention_order() {
    use crate::inferlet::process::ProcessId;
    let mut s = store();
    let (low, high) = (ProcessId::from_u128(1), ProcessId::from_u128(2));
    for (key, owner) in [([1; 32], high), ([2; 32], low)] {
        let ws = s.create_working_set(geom());
        write_state(&mut s, ws);
        assert!(s.checkpoint(ws, key, Some(owner)).unwrap());
        s.release_working_set(ws);
    }
    let rank = |owner: Option<ProcessId>| i64::from(owner == Some(high));
    assert_eq!(s.reclaim_cache(rank), 1, "one working-set at a time");
    assert!(
        !s.has_checkpoint(&[2; 32]),
        "the lower-ranked, though newer"
    );
    assert!(s.has_checkpoint(&[1; 32]), "the higher-ranked stays");
    // No lower tier for a checkpoint: cached state yields to the next
    // demand, whatever its key.
    assert_eq!(s.reclaim_cache(rank), 1);
    assert!(!s.has_checkpoint(&[1; 32]));
    assert_eq!(s.reclaim_cache(rank), 0);
}

/// A reload in place drops every checkpoint, whatever its key: none of
/// them may be resumed under the new weights.
fn new_weights_drop_every_checkpoint() {
    use crate::inferlet::process::ProcessId;
    let mut s = store();
    for (key, owner) in [([6; 32], 6), ([7; 32], 7)] {
        let ws = s.create_working_set(geom());
        write_state(&mut s, ws);
        assert!(s.checkpoint(ws, key, Some(ProcessId::from_u128(owner))).unwrap());
        s.release_working_set(ws);
    }
    assert!(s.cache_only() != RsUnits::default());
    assert_eq!(s.drop_cache(), 2);
    assert!(!s.holds_checkpoints());
    assert!(!s.has_checkpoint(&[6; 32]) && !s.has_checkpoint(&[7; 32]));
    assert_eq!(s.cache_only(), RsUnits::default());
    assert_eq!(s.drop_cache(), 0);
}

fn a_checkpoint_past_a_step_goes_first() {
    use crate::inferlet::process::ProcessId;
    use crate::policy::retention::Keys;
    struct Upto(u64);
    impl Keys for Upto {
        fn key_at(&self, _: Option<ProcessId>, at: u64) -> i64 {
            i64::from(at < self.0)
        }
    }
    let mut s = store();
    let owner = ProcessId::from_u128(3);
    for (key, frozen) in [([4; 32], 32), ([5; 32], 64)] {
        let ws = s.create_working_set(geom());
        write_state(&mut s, ws);
        s.fold(ws, frozen);
        assert!(s.checkpoint(ws, key, Some(owner)).unwrap());
        s.release_working_set(ws);
    }
    assert_eq!(s.reclaim_cache(Upto(48)), 1);
    assert!(!s.has_checkpoint(&[5; 32]), "past the step, though newer");
    assert!(s.has_checkpoint(&[4; 32]));
}

fn a_checkpoint_is_cache_and_goes_when_idle() {
    let mut s = store();
    let (k, o, r) = ([1; 32], [2; 32], [3; 32]);
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    assert!(s.checkpoint(ws, k, None).unwrap());
    let slot = s.folded_slot(ws).unwrap();

    write_state(&mut s, ws);
    assert_ne!(
        s.folded_slot(ws).unwrap(),
        slot,
        "the owner's next write copies"
    );
    let adopted = s.create_working_set(geom());
    assert!(s.adopt_checkpoint(adopted, &k).unwrap());
    assert_eq!(s.folded_slot(adopted).unwrap(), slot);
    assert_eq!(
        s.reclaim_cache(|_| 0),
        0,
        "an adopter still holds the state"
    );

    let other = s.create_working_set(geom());
    write_state(&mut s, other);
    assert!(s.checkpoint(other, o, None).unwrap());
    assert_eq!(s.group_demand(other, true, false, None), Ok(slots(1)));
    // Cached state yields to running work, whatever its key.
    assert_eq!(s.reclaim_sharing(&[other]), 1);
    assert_eq!(
        s.group_demand(other, true, false, None),
        Ok(slots(0)),
        "writes in place"
    );
    assert!(!s.has_checkpoint(&o));

    s.release_working_set(adopted);
    let busy = s
        .prepare_group(other, true, false, None, None, None)
        .unwrap();
    let busy = s.publish_prepared(busy).unwrap();
    assert_eq!(s.reclaim_cache(|_| 0), 1);
    assert!(!s.has_checkpoint(&k));
    assert_eq!(
        s.available().slots,
        10,
        "a write the checkpoint never fed does not hold its slot back"
    );

    let third = s.create_working_set(geom());
    write_state(&mut s, third);
    let read = s.folded_slot(third).unwrap().unwrap();
    assert!(s.checkpoint(third, r, None).unwrap());
    s.release_working_set(third);
    s.note_reads(*busy.seqs().last().unwrap(), [read]);
    let before = s.available().slots;
    assert_eq!(s.reclaim_cache(|_| 0), 1);
    assert_eq!(
        s.available().slots,
        before,
        "a fire reading a dropped checkpoint's slot holds it"
    );
    s.settle(busy);
    assert_eq!(s.available().slots, before + 1);
}

fn run_ahead_successor_never_resets_twice() {
    let mut s = store();
    let ws = s.create_working_set(geom());

    let first = s.prepare_group(ws, true, false, None, None, None).unwrap();
    assert!(first.state().unwrap().reset, "cold state resets once");
    let slot = first.state().unwrap().slot;
    let first = s.publish_prepared(first).unwrap();

    let second = s.prepare_group(ws, true, false, None, None, None).unwrap();
    let state = second.state().unwrap();
    assert_eq!(state.slot, slot, "successor continues the published slot");
    assert!(!state.reset, "a run-ahead successor must not RESET again");
    assert!(state.copy_from.is_none());
    assert_eq!(s.available().slots, 11, "no second allocation");

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

    let first = s
        .prepare_group(child, true, false, None, None, None)
        .unwrap();
    let private = first.state().unwrap().slot;
    assert_eq!(first.state().unwrap().copy_from, Some(shared));
    let first = s.publish_prepared(first).unwrap();

    let second = s
        .prepare_group(child, true, false, None, None, None)
        .unwrap();
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
    let a = s.prepare_group(ws, true, false, None, None, None).unwrap();
    let b = s.prepare_group(ws, true, false, None, None, None).unwrap();
    assert_eq!(
        s.publish_batch(vec![a, b]).err(),
        Some(RsError::DuplicateWorkingSet)
    );
    assert_eq!(s.folded_slot(ws).unwrap(), None);
    assert_eq!(s.available().slots, 12);
}

fn a_fold_drops_the_buffer_pages_behind_it() {
    let mut s = store();
    let ws = store_with_state(&mut s);
    let prepared = s
        .prepare_group(ws, false, false, None, Some(0..8), None)
        .unwrap();
    settled(&mut s, prepared);
    assert_eq!((s.buffer_size(ws), s.base(ws)), (Ok(2), Ok(0)));

    s.fold(ws, 5);
    assert_eq!((s.frozen(ws), s.base(ws)), (Ok(5), Ok(4)));
    assert_eq!(s.buffer_size(ws), Ok(1), "the page of slots 0..4 went");
    assert_eq!(
        s.prepare_group(ws, false, false, None, Some(3..6), None)
            .err(),
        Some(RsError::BelowFrozen { slot: 3, frozen: 5 })
    );
    s.fold(ws, 3);
    assert_eq!(
        (s.frozen(ws), s.base(ws), s.buffer_size(ws)),
        (Ok(8), Ok(8), Ok(0))
    );
}

// A stop's take-back on a set with a recurrent state: the buffer pages
// wholly past the cut go back, the page the cut falls in stays; a cut below
// what the state holds is refused, the buffer untouched.
fn a_truncate_takes_back_the_buffer_past_its_cut() {
    let mut s = store();
    let ws = store_with_state(&mut s);
    let prepared = s
        .prepare_group(ws, false, false, None, Some(0..12), None)
        .unwrap();
    settled(&mut s, prepared);
    s.fold(ws, 2);
    let held = s.held([ws]);
    assert_eq!(s.buffer_size(ws), Ok(3));
    s.truncate(ws, 5).unwrap();
    assert_eq!(s.buffer_size(ws), Ok(2), "slots 8..12 went, 4..8 holds row 4");
    assert_eq!(s.held([ws]).pages + 1, held.pages, "and its page went back");
    assert_eq!(
        s.truncate(ws, 1),
        Err(RsError::FoldedPastCut {
            committed: 2,
            cut: 1
        })
    );
    assert_eq!(s.buffer_size(ws), Ok(2));
    s.truncate(ws, 2).unwrap();
    assert_eq!(s.buffer_size(ws), Ok(1), "the page the state ends in stays");
    // A state that buffers no row (no rs list) is refused by name.
    let mut bare = RsStore::new(4, 0);
    let ws = bare.create_working_set(geom());
    assert_eq!(bare.truncate(ws, 0), Err(RsError::Unbuffered));
}

fn a_state_others_read_is_folded_into_a_copy() {
    let mut s = store();
    let ws = store_with_state(&mut s);
    let held = s.folded_slot(ws).unwrap().unwrap();
    let prepared = s
        .prepare_group(ws, true, true, Some(4), None, None)
        .unwrap();
    let state = *prepared.state().unwrap();
    assert_ne!(state.slot, held);
    assert_eq!(state.copy_from, Some(held));
    settled(&mut s, prepared);
    assert_eq!(s.folded_slot(ws).unwrap(), Some(state.slot));
    assert_eq!(s.frozen(ws), Ok(4));
}

fn a_checkpoint_resumes_a_fresh_working_set() {
    let mut s = store();
    let key = [7u8; 32];
    let ws = s.create_working_set(geom());
    let prepared = s.prepare_group(ws, true, false, None, None, None).unwrap();
    let published = s.publish_prepared(prepared).unwrap();
    assert_eq!(
        s.checkpoint(ws, key, None),
        Err(RsError::CheckpointUnsettled),
        "a fire still writes it"
    );
    s.settle(published);
    assert!(s.checkpoint(ws, key, None).unwrap());
    assert!(s.has_checkpoint(&key));
    let free = s.available().slots;

    let resumed = s.create_working_set(geom());
    assert!(s.adopt_checkpoint(resumed, &key).unwrap());
    assert_eq!(s.folded_slot(resumed).unwrap(), s.folded_slot(ws).unwrap());
    assert!(
        !s.adopt_checkpoint(resumed, &key).unwrap(),
        "it holds state now"
    );
    let prepared = s
        .prepare_group(resumed, true, false, None, None, None)
        .unwrap();
    assert!(prepared.allocated.len() == 1, "its first write copies");
    s.cancel_prepared(prepared);

    s.forget_state(resumed).unwrap();
    assert_eq!(s.folded_slot(resumed).unwrap(), None);
    s.release_working_set(ws);
    assert_eq!(s.available().slots, free, "the checkpoint holds the slot");
    assert!(!s.has_checkpoint(&[8u8; 32]));
}

fn a_snapshot_resumes_at_its_row() {
    let mut s = store();
    let free = s.available().slots;
    let ws = s.create_working_set(geom());
    let mut prepared = s
        .prepare_group(ws, true, false, Some(10), None, None)
        .unwrap();
    s.prepare_snapshots(&mut prepared, &[4], None).unwrap();
    let published = s.publish_prepared(prepared).unwrap();
    let [snap] = published.snapshots() else {
        panic!("one snapshot a row")
    };
    assert_eq!((snap.ws, snap.row), (ws, 4));
    s.settle(published.clone());
    s.keep(snap.snapshot, [3; 32], None);
    assert!(s.has_checkpoint(&[3; 32]));

    let resumed = s.create_working_set(geom());
    assert!(s.adopt_checkpoint(resumed, &[3; 32]).unwrap());
    assert_eq!(s.folded_slot(resumed).unwrap(), Some(snap.slot));
    assert_eq!(s.frozen(resumed), Ok(4), "the state as of its row");
    assert_eq!(s.frozen(ws), Ok(10));

    for id in [ws, resumed] {
        s.release_working_set(id);
    }
    s.retire_idle();
    assert_eq!(
        s.available().slots,
        free - 1,
        "the checkpoint holds its slot"
    );
}

// A fold the device counts: from its fire's submit until it settles the
// record stands ahead of the frozen length, and only the engine, and only
// where the record stands, knows by how much.

#[test]
fn device_fold_gate_every_case() {
    a_device_counted_fold_gates_every_copy_until_it_settles();
    two_device_counted_folds_gate_until_the_last_settles();
    a_device_counted_fold_that_failed_refuses_by_its_failure();
    a_released_set_has_nothing_left_to_wait_for();
}

fn folded_at(s: &mut RsStore, frozen: u32) -> RsWorkingSetId {
    let ws = s.create_working_set(geom());
    write_state(s, ws);
    s.fold(ws, frozen);
    ws
}

fn a_device_counted_fold_gates_every_copy_until_it_settles() {
    let mut s = store();
    let ws = folded_at(&mut s, 4);
    s.begin_device_fold(ws);
    let unsettled = || RsError::DeviceFoldUnsettled;
    assert_eq!(s.fork(ws), Err(unsettled()));
    assert_eq!(s.export(ws).map(|set| set.frozen), Err(unsettled()));
    assert_eq!(s.checkpoint(ws, [1; 32], None), Err(unsettled()));
    assert!(!s.has_checkpoint(&[1; 32]));

    s.end_device_fold(ws, Some(3), None);
    let child = s.fork(ws).expect("the fold settled");
    assert_eq!(s.frozen(child), Ok(7), "the child takes what the fold left");
    assert_eq!(s.export(ws).map(|set| set.frozen), Ok(7));
    assert_eq!(s.checkpoint(ws, [1; 32], None), Ok(true));
    let resumed = s.create_working_set(geom());
    assert!(s.adopt_checkpoint(resumed, &[1; 32]).unwrap());
    assert_eq!(s.frozen(resumed), Ok(7));
}

fn two_device_counted_folds_gate_until_the_last_settles() {
    let mut s = store();
    let ws = folded_at(&mut s, 4);
    s.begin_device_fold(ws);
    s.begin_device_fold(ws);
    s.end_device_fold(ws, Some(1), None);
    assert_eq!(
        s.fork(ws),
        Err(RsError::DeviceFoldUnsettled),
        "another is on its way"
    );
    s.end_device_fold(ws, Some(2), None);
    let child = s.fork(ws).unwrap();
    assert_eq!(s.frozen(child), Ok(7));
}

fn a_device_counted_fold_that_failed_refuses_by_its_failure() {
    let mut s = store();
    let ws = folded_at(&mut s, 4);
    s.begin_device_fold(ws);
    s.end_device_fold(ws, None, Some("pipeline: forward failed: device lost"));
    let failed = || RsError::DeviceFoldFailed("pipeline: forward failed: device lost".to_string());
    assert_eq!(s.fork(ws), Err(failed()));
    assert_eq!(s.export(ws).map(|set| set.frozen), Err(failed()));
    assert_eq!(s.checkpoint(ws, [2; 32], None), Err(failed()));
    assert!(
        failed().to_string().contains("forward failed: device lost"),
        "by the fire's own failure: {}",
        failed()
    );
    assert_eq!(s.frozen(ws), Ok(4), "nothing was counted");

    // A fold that settles after it does not make the length known again.
    s.begin_device_fold(ws);
    s.end_device_fold(ws, Some(1), None);
    assert_eq!(s.fork(ws), Err(failed()));

    // With the record gone there is nothing to be behind.
    s.forget_state(ws).unwrap();
    assert!(s.fork(ws).is_ok());
}

fn a_released_set_has_nothing_left_to_wait_for() {
    use futures::FutureExt;
    let store = std::sync::Mutex::new(store());
    let ws = folded_at(&mut store.lock().unwrap(), 4);
    store.lock().unwrap().begin_device_fold(ws);
    let sets = [ws];
    let mut waiting = Box::pin(super::device_folds_settled(&store, &sets));
    assert!(waiting.as_mut().now_or_never().is_none());
    store.lock().unwrap().release_working_set(ws);
    assert!(waiting.as_mut().now_or_never().is_some());
}

#[test]
fn a_wait_on_a_device_counted_fold_ends_when_it_settles_however_it_ended() {
    use futures::FutureExt;
    for failure in [None, Some("pipeline: forward failed: device lost")] {
        let store = std::sync::Mutex::new(store());
        let (ws, other) = {
            let mut s = store.lock().unwrap();
            (folded_at(&mut s, 4), folded_at(&mut s, 4))
        };
        assert!(
            super::device_folds_settled(&store, &[ws, other])
                .now_or_never()
                .is_some(),
            "no fold the device counts is on its way: nothing waits"
        );
        store.lock().unwrap().begin_device_fold(ws);
        let sets = [other, ws];
        let mut waiting = Box::pin(super::device_folds_settled(&store, &sets));
        assert!(waiting.as_mut().now_or_never().is_none());
        assert!(
            super::device_folds_settled(&store, &[other])
                .now_or_never()
                .is_some(),
            "another set's fold delays nothing"
        );
        store
            .lock()
            .unwrap()
            .end_device_fold(ws, failure.is_none().then_some(3), failure);
        assert!(waiting.as_mut().now_or_never().is_some());
        let s = store.lock().unwrap();
        match failure {
            None => assert_eq!(s.frozen(ws), Ok(7), "the waiter reads what it left"),
            Some(failure) => assert_eq!(
                s.device_fold_gate(ws),
                Err(RsError::DeviceFoldFailed(failure.to_string()))
            ),
        }
    }
}
