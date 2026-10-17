use super::{RsError, RsGeometry, RsStore, RsUnits, RsWorkingSetId};

fn slots(slots: u32) -> RsUnits {
    RsUnits { slots, pages: 0 }
}

fn pages(pages: u32) -> RsUnits {
    RsUnits { slots: 0, pages }
}

fn geom() -> RsGeometry {
    RsGeometry {
        state_size: 4096,
        buffer_page_tokens: 4,
    }
}

fn store() -> RsStore {
    RsStore::new(12, 12, 0, 0)
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
    a_state_others_read_is_folded_into_a_copy();
    a_checkpoint_is_cache_and_goes_when_idle();
    idle_snapshots_go_in_retention_order();
    a_checkpoint_past_a_step_goes_first();
    a_suspend_parks_private_slots_on_host_and_restore_brings_them_back();
    a_window_is_a_ring_of_pages_anchored_at_its_commit();
    a_checkpoint_resumes_a_fresh_working_set();
}

// A windowed model's state is a ring: a fire takes the pages it writes,
// the pages behind the committed position's window retire, a write moves the
// head back without touching a page, a fork shares the ring and copies
// only the page it overwrites, and a suspend parks the ring on the host.
fn a_window_is_a_ring_of_pages_anchored_at_its_commit() {
    use super::{RingAdvance, RingGeometry};
    use std::collections::HashSet;
    let mut s = RsStore::new_ring(
        RingGeometry {
            tokens: 32,
            page_size: 16,
        },
        8,
        4,
    );
    let ws = s.create_working_set(geom());
    let order: Vec<u32> = (0..16).collect();
    let advance = |fold, end| RingAdvance {
        positioned: true,
        settled: 0,
        fold,
        commit_end: end,
        head: Some(end),
    };
    let fire = |s: &mut RsStore, ws, pages: &[u32], advance| {
        let prepared = s.prepare_ring(ws, pages, None).unwrap();
        let copies: Vec<_> = prepared.buffer_copy_plan().collect();
        let published = s.publish_prepared(prepared).unwrap();
        s.ring_advance(ws, advance, &order);
        s.settle(published);
        copies
    };
    assert_eq!(s.ring_demand(ws, &[0, 1, 2, 3, 4, 5]), Ok(pages(6)));
    fire(&mut s, ws, &[0, 1, 2, 3, 4, 5], advance(Some(u32::MAX), 96));
    assert_eq!(s.available().pages, 5, "four pages behind 96 - 32 retired");
    assert_eq!(s.ring_positions(ws), Ok(Some((96, 96))));
    let held = s.ring_translation(ws, 0..7).unwrap();
    assert_eq!(
        &held[..4],
        &[0, 0, 0, 0],
        "behind the ring reads the null page"
    );
    assert!(held[4] != 0 && held[5] != 0 && held[6] == 0);

    // A speculative window: buffered, then written again and partly committed.
    assert_eq!(s.ring_demand(ws, &[6]), Ok(pages(1)));
    fire(&mut s, ws, &[6], advance(Some(0), 104));
    assert_eq!(
        s.ring_demand(ws, &[6]),
        Ok(pages(0)),
        "the page is the ring's already"
    );
    fire(&mut s, ws, &[6], advance(Some(5), 104));
    assert_eq!(s.ring_positions(ws), Ok(Some((101, 104))));
    assert_eq!(
        s.ring_demand(ws, &[3]),
        Ok(pages(1)),
        "a page behind the window is given back"
    );

    let child = s.fork(ws).unwrap();
    assert_eq!(
        s.ring_demand(child, &[6]),
        Ok(pages(1)),
        "a shared page copies on write"
    );
    let copies = fire(&mut s, child, &[6], advance(Some(u32::MAX), 112));
    assert_eq!(copies.len(), 1);
    assert_eq!(s.available().pages, 3);
    s.release_working_set(child);
    s.retire_idle();
    assert_eq!(s.available().pages, 4);

    let set: HashSet<_> = [ws].into();
    let txn = s.prepare_suspend(&set).expect("three private pages");
    assert!(txn.is_ring() && txn.copy_plan().pages.len() == 3);
    assert_eq!(s.commit_suspend(txn), pages(3));
    assert_eq!((s.held([ws]), s.swapped_units(&set)), (pages(0), pages(3)));
    assert_eq!(s.available().pages, 7);
    let mut granted = s.reserve(pages(3)).unwrap();
    let txn = s.prepare_restore(&set, &mut granted).unwrap();
    assert_eq!(s.commit_restore(txn), 3);
    assert_eq!(s.swapped_units(&set), pages(0));
    assert_eq!(
        s.ring_demand(ws, &[7]),
        Ok(pages(1)),
        "the ring reads on from its pages"
    );

    let device = |settled, end| RingAdvance {
        positioned: true,
        settled,
        fold: None,
        commit_end: end,
        head: Some(end),
    };
    fire(&mut s, ws, &[7], device(0, 128));
    assert_eq!(s.ring_positions(ws), Ok(Some((101, 128))));
    fire(&mut s, ws, &[8], device(128, 144));
    assert_eq!(
        s.ring_positions(ws),
        Ok(Some((128, 144))),
        "a device-decided fold is settled by the next fire's first token"
    );

    let lease = RingAdvance {
        positioned: false,
        settled: 0,
        fold: None,
        commit_end: 0,
        head: None,
    };
    fire(&mut s, ws, &[2], lease);
    assert_eq!(s.ring_positions(ws), Ok(Some((128, 144))));
    assert_ne!(
        s.ring_translation(ws, [2]).unwrap(),
        [0],
        "a device-geometry fire keeps the page its lease names"
    );
}

fn a_suspend_parks_private_slots_on_host_and_restore_brings_them_back() {
    let mut s = RsStore::new(2, 2, 2, 2);
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    let other = s.create_working_set(geom());
    write_state(&mut s, other);
    let shared = s.fork(other).unwrap();
    let set = |ws| std::collections::HashSet::from([ws]);

    let txn = s.prepare_suspend(&set(ws)).expect("a private slot moves");
    assert_eq!(txn.copy_plan().slots, vec![(0, 0)]);
    assert_eq!(s.commit_suspend(txn), slots(1));
    assert_eq!(s.available().slots, 1);
    assert!(
        !s.checkpoint(ws, [1; 32], None).unwrap(),
        "a suspended state is not checkpointed"
    );
    let child = s.fork(ws).unwrap();
    assert!(
        s.prepare_suspend(&set(shared)).is_none(),
        "a slot a fork shares stays on device"
    );

    let both = std::collections::HashSet::from([ws, child]);
    assert_eq!(
        s.swapped_units(&both),
        slots(1),
        "the fork shares the parked slot"
    );
    let mut granted = s.reserve(slots(1)).unwrap();
    let txn = s.prepare_restore(&both, &mut granted).unwrap();
    assert_eq!(txn.copy_plan().slots, vec![(0, 0)]);
    assert_eq!(s.commit_restore(txn), 1);
    assert_eq!(
        (s.swapped_units(&both), s.host_available()),
        (slots(0), RsUnits { slots: 2, pages: 2 })
    );
    assert_eq!(s.folded_slot(ws), s.folded_slot(child));
    s.release_working_set(child);
    write_state(&mut s, ws);

    assert!(s.checkpoint(ws, [1; 32], None).unwrap());
    assert_eq!(
        s.suspendable(&set(ws)),
        slots(1),
        "a checkpoint is no sharer"
    );
    let txn = s.prepare_suspend(&set(ws)).expect("the checkpoint yields");
    assert!(!s.has_checkpoint(&[1; 32]));
    assert_eq!(s.commit_suspend(txn), slots(1));
}

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
    assert_eq!(s.discard_cache(high), 1);
    assert_eq!(s.reclaim_cache(rank), 0);
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
    assert_eq!(s.yield_checkpoints(&[other]), 1);
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
    assert!(
        !s.checkpoint(ws, key, None).unwrap(),
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
