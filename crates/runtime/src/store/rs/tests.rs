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
    index_snapshots_state_and_is_reclaimed_when_idle();
    a_suspend_parks_private_slots_on_host_and_restore_brings_them_back();
    a_window_is_a_ring_of_pages_anchored_at_its_commit();
}

// A windowed model's state is a ring: a fire takes the pages it writes,
// the pages behind the committed position's window retire, a discard moves
// the head back without touching a page, a fork shares the ring and copies
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
    assert_eq!(s.ring_demand(ws, &[0, 1, 2, 3, 4, 5]), Ok(6));
    fire(&mut s, ws, &[0, 1, 2, 3, 4, 5], advance(Some(u32::MAX), 96));
    assert_eq!(s.available_slots(), 5, "four pages behind 96 - 32 retired");
    assert_eq!(s.ring_positions(ws), Ok(Some((96, 96))));
    let held = s.ring_translation(ws, 0..7).unwrap();
    assert_eq!(
        &held[..4],
        &[0, 0, 0, 0],
        "behind the ring reads the null page"
    );
    assert!(held[4] != 0 && held[5] != 0 && held[6] == 0);

    // A speculative window: buffered, then taken back, then partly committed.
    assert_eq!(s.ring_demand(ws, &[6]), Ok(1));
    fire(&mut s, ws, &[6], advance(Some(0), 104));
    assert_eq!(s.buffer_tokens(ws), Ok(8));
    assert_eq!(s.discard_buffered(ws, 8), Ok(()));
    assert_eq!(
        s.ring_demand(ws, &[6]),
        Ok(0),
        "the page is the ring's already"
    );
    fire(&mut s, ws, &[6], advance(Some(5), 104));
    assert_eq!(s.discard_buffered(ws, 3), Ok(()));
    assert_eq!(s.ring_positions(ws), Ok(Some((101, 101))));
    assert_eq!(
        s.ring_demand(ws, &[3]),
        Ok(1),
        "a page behind the window is given back"
    );

    let child = s.fork(ws).unwrap();
    assert_eq!(
        s.ring_demand(child, &[6]),
        Ok(1),
        "a shared page copies on write"
    );
    let copies = fire(&mut s, child, &[6], advance(Some(u32::MAX), 112));
    assert_eq!(copies.len(), 1);
    assert_eq!(s.available_slots(), 3);
    s.release_working_set(child);
    s.retire_idle();
    assert_eq!(s.available_slots(), 4);

    let set: HashSet<_> = [ws].into();
    let txn = s.prepare_suspend(&set).expect("three private pages");
    assert!(txn.is_ring() && txn.slot_count() == 3);
    assert_eq!(s.commit_suspend(txn), 3);
    assert_eq!((s.held_slots([ws]), s.swapped_slots(&set)), (0, 3));
    assert_eq!(s.available_slots(), 7);
    let mut granted = s.reserve_slots(3).unwrap();
    let txn = s.prepare_restore(&set, &mut granted).unwrap();
    assert_eq!(s.commit_restore(txn), 3);
    assert_eq!(s.swapped_slots(&set), 0);
    assert_eq!(
        s.ring_demand(ws, &[7]),
        Ok(1),
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
    let mut s = RsStore::new_with_host(2, 2);
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    let other = s.create_working_set(geom());
    write_state(&mut s, other);
    let shared = s.fork(other).unwrap();
    let set = |ws| std::collections::HashSet::from([ws]);

    let txn = s.prepare_suspend(&set(ws)).expect("a private slot moves");
    assert_eq!(txn.copy_plan(), (vec![0], vec![0]));
    assert_eq!(s.commit_suspend(txn), 1);
    assert_eq!(s.available_slots(), 1);
    assert_eq!(
        s.update_index(b"k".to_vec(), ws),
        Err(RsError::Suspended),
        "a suspended state is not indexed"
    );
    let child = s.fork(ws).unwrap();
    assert!(
        s.prepare_suspend(&set(shared)).is_none(),
        "a slot a fork shares stays on device"
    );

    let both = std::collections::HashSet::from([ws, child]);
    assert_eq!(s.swapped_slots(&both), 1, "the fork shares the parked slot");
    let mut granted = s.reserve_slots(1).unwrap();
    let txn = s.prepare_restore(&both, &mut granted).unwrap();
    assert_eq!(txn.copy_plan(), (vec![0], vec![0]));
    assert_eq!(s.commit_restore(txn), 1);
    assert_eq!((s.swapped_slots(&both), s.host_available()), (0, 2));
    assert_eq!(s.folded_slot(ws), s.folded_slot(child));
    s.release_working_set(child);
    write_state(&mut s, ws);

    s.update_index(b"k".to_vec(), ws).unwrap();
    assert_eq!(s.suspendable_slots(&set(ws)), 1, "an index is no sharer");
    let txn = s.prepare_suspend(&set(ws)).expect("the index yields");
    assert_eq!(s.from_index(b"k"), Ok(None));
    assert_eq!(s.commit_suspend(txn), 1);
}

fn index_snapshots_state_and_is_reclaimed_when_idle() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    let pending = s.prepare_write(ws, true, None).unwrap();
    let published = s.publish_prepared(pending).unwrap();
    assert_eq!(s.update_index(b"k".to_vec(), ws), Ok(0));
    assert_eq!(
        s.from_index(b"k"),
        Ok(None),
        "unreadable until its write settles"
    );
    s.settle(published);
    let slot = s.folded_slot(ws).unwrap();

    write_state(&mut s, ws);
    assert_ne!(
        s.folded_slot(ws).unwrap(),
        slot,
        "the owner's next write copies"
    );
    let adopted = s.from_index(b"k").unwrap().unwrap();
    assert_eq!(s.folded_slot(adopted).unwrap(), slot);
    assert_eq!(
        s.drop_unused_indexes(),
        0,
        "an adopter still holds the state"
    );

    let other = s.create_working_set(geom());
    write_state(&mut s, other);
    assert_eq!(s.update_index(b"o".to_vec(), other), Ok(0));
    assert_eq!(s.write_demand(other, true, None), Ok(1));
    assert_eq!(s.yield_indexes(&[other]), 1);
    assert_eq!(s.write_demand(other, true, None), Ok(0), "writes in place");
    assert_eq!(s.from_index(b"o"), Ok(None));

    s.release_working_set(adopted);
    let busy = s.prepare_write(other, true, None).unwrap();
    let busy = s.publish_prepared(busy).unwrap();
    assert_eq!(s.drop_unused_indexes(), 1);
    assert_eq!(s.from_index(b"k"), Ok(None));
    assert_eq!(
        s.available_slots(),
        10,
        "a write the snapshot never fed does not hold its slot back"
    );

    let third = s.create_working_set(geom());
    write_state(&mut s, third);
    let read = s.folded_slot(third).unwrap().unwrap();
    assert_eq!(s.update_index(b"r".to_vec(), third), Ok(0));
    s.release_working_set(third);
    s.note_reads(*busy.seqs().last().unwrap(), [read]);
    let before = s.available_slots();
    assert_eq!(s.drop_unused_indexes(), 1);
    assert_eq!(
        s.available_slots(),
        before,
        "a fire reading a dropped snapshot's slot holds it"
    );
    s.settle(busy);
    assert_eq!(s.available_slots(), before + 1);
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
