use std::collections::HashSet;

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
    a_state_others_read_is_folded_into_a_copy();
    a_checkpoint_is_cache_and_goes_when_idle();
    idle_snapshots_go_in_retention_order();
    a_checkpoint_past_a_step_goes_first();
    a_suspend_short_of_host_rows_moves_and_drops_nothing();
    a_window_is_a_list_of_slots_trimmed_behind_its_commit();
    a_checkpoint_resumes_a_fresh_working_set();
    a_snapshot_resumes_at_its_row();
    a_checkpoint_holds_its_rows_as_a_cache_entry();
}

// A checkpoint's rows are held by its cache entry: they count as cache only
// once no working set holds them too, and the hold goes with the checkpoint.
fn a_checkpoint_holds_its_rows_as_a_cache_entry() {
    let mut s = store();
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    assert!(s.checkpoint(ws, [9; 32], None).unwrap());
    assert_eq!(s.cache_only(), slots(0), "the owner reads it still");
    write_state(&mut s, ws);
    assert_eq!(s.cache_only(), slots(1), "the owner's write copied away");
    assert_eq!(s.reclaim_cache(|_| 0), 1);
    assert_eq!(s.cache_only(), slots(0), "and its hold went with it");
}

// A windowed model's state is a window list: a fire takes the slots it
// writes, the slots behind the committed position's window retire, a write
// moves the head back without touching a slot, a fork shares the list and
// copies only the slot it overwrites.
fn a_window_is_a_list_of_slots_trimmed_behind_its_commit() {
    use super::{WindowAdvance, WindowGeometry};
    let mut s = RsStore::new_window(
        WindowGeometry {
            window: 32,
            tokens: 16,
        },
        8,
    );
    let ws = s.create_working_set(geom());
    let win = |n| RsUnits::of_windows([n, 0, 0, 0]);
    // The rows of slots `first..=last` (16 a slot).
    let rows = |slots: &[u32]| slots[0] * 16..(slots[slots.len() - 1] + 1) * 16;
    let advance = |fold, end| WindowAdvance {
        positioned: true,
        settled: 0,
        reach: 0,
        fold,
        commit_end: end,
        head: Some(end),
    };
    let fire = |s: &mut RsStore, ws, slots: &[u32], advance| {
        let prepared = s.prepare_window(ws, rows(slots), None).unwrap();
        let copies: Vec<_> = prepared.buffer_copy_plan().collect();
        let published = s.publish_prepared(prepared).unwrap();
        s.window_advance(ws, advance);
        s.settle(published);
        copies
    };
    assert_eq!(s.window_demand(ws, rows(&[0, 1, 2, 3, 4, 5])), Ok(win(6)));
    fire(&mut s, ws, &[0, 1, 2, 3, 4, 5], advance(Some(u32::MAX), 96));
    assert_eq!(s.available().pages, 5, "four pages behind 96 - 32 retired");
    assert_eq!(s.window_positions(ws), Ok(Some((96, 96))));
    let held = s.window_tables(ws, 7 * 16).unwrap().remove(0);
    assert_eq!(
        &held[..4],
        &[0, 0, 0, 0],
        "behind the window reads the null page"
    );
    assert!(held[4] != 0 && held[5] != 0 && held[6] == 0);

    // A speculative window: buffered, then written again and partly committed.
    assert_eq!(s.window_demand(ws, rows(&[6])), Ok(win(1)));
    fire(&mut s, ws, &[6], advance(Some(0), 104));
    assert_eq!(
        s.window_demand(ws, rows(&[6])),
        Ok(win(0)),
        "the slot is the list's already"
    );
    fire(&mut s, ws, &[6], advance(Some(5), 104));
    assert_eq!(s.window_positions(ws), Ok(Some((101, 104))));
    assert_eq!(
        s.window_demand(ws, rows(&[3])),
        Ok(win(1)),
        "a page behind the window is given back"
    );

    let child = s.fork(ws).unwrap();
    assert_eq!(
        s.window_demand(child, rows(&[6])),
        Ok(win(1)),
        "a shared page copies on write"
    );
    let copies = fire(&mut s, child, &[6], advance(Some(u32::MAX), 112));
    assert_eq!(copies.len(), 1);
    assert_eq!(s.available().pages, 3);
    s.release_working_set(child);
    s.retire_idle();
    assert_eq!(s.available().pages, 4);

    assert_eq!(
        s.window_demand(ws, rows(&[7])),
        Ok(win(1)),
        "the window reads on from its slots"
    );

    let device = |settled, end| WindowAdvance {
        positioned: true,
        settled,
        reach: settled,
        fold: None,
        commit_end: end,
        head: Some(end),
    };
    fire(&mut s, ws, &[7], device(0, 128));
    assert_eq!(s.window_positions(ws), Ok(Some((101, 128))));
    fire(&mut s, ws, &[8], device(128, 144));
    assert_eq!(
        s.window_positions(ws),
        Ok(Some((128, 144))),
        "a device-decided fold is settled by the next fire's first token"
    );

    let lease = WindowAdvance {
        positioned: false,
        settled: 0,
        reach: 32,
        fold: None,
        commit_end: 0,
        head: None,
    };
    fire(&mut s, ws, &[2], lease);
    assert_eq!(s.window_positions(ws), Ok(Some((128, 144))));
    assert_ne!(
        s.window_tables(ws, 48).unwrap()[0][2],
        0,
        "a device-geometry fire keeps the page its lease names"
    );

    let decode = WindowAdvance {
        reach: 144,
        ..lease
    };
    fire(&mut s, ws, &[9], decode);
    assert_eq!(
        s.window_positions(ws),
        Ok(Some((128, 144))),
        "a device-geometry fire commits nothing"
    );
    assert_eq!(
        s.window_tables(ws, 128).unwrap()[0][6],
        0,
        "and retires what lies wholly behind the window of the first token it writes"
    );
    assert_ne!(s.window_tables(ws, 128).unwrap()[0][7], 0);
}

fn a_suspend_short_of_host_rows_moves_and_drops_nothing() {
    let mut s = RsStore::new(10, 0);
    let set = |ws| HashSet::from([ws]);
    let ws = s.create_working_set(geom());
    write_state(&mut s, ws);
    assert!(s.checkpoint(ws, [7; 32], None).unwrap());
    let before = s.available();
    assert!(s.prepare_suspend(&set(ws)).is_none());
    assert_eq!(
        s.suspend_short(&set(ws)),
        slots(1),
        "a host row for its slot"
    );
    assert!(s.has_checkpoint(&[7; 32]), "nothing dropped");
    assert_eq!(s.available(), before, "nothing moved");
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
    assert_eq!(
        s.reclaim_droppable(rank),
        0,
        "the runtime leaves what a policy keyed above 0"
    );
    assert!(s.has_checkpoint(&[1; 32]));
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
    let kept = |_: Option<crate::inferlet::process::ProcessId>| 1;
    assert_eq!(
        s.reclaim_sharing(&[other], kept),
        0,
        "a checkpoint a policy keeps is not the runtime's to drop"
    );
    assert!(s.has_checkpoint(&o));
    assert_eq!(s.reclaim_sharing(&[other], |_| 0), 1);
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

#[test]
fn state_on_demand_hands_out_only_what_is_committed_and_plans_over_the_rest() {
    let mut store = RsStore::new(32, 32);
    store.commit_on_demand();
    let one = slots(1);
    assert!(store.reserve(one).is_none());
    assert_eq!(
        store.free().slots,
        32,
        "the planner plans over a whole pool"
    );
    assert!(store.note_short(one));
    assert_eq!(store.next_extent(16), Some(16));
    store.set_committed(16);
    assert!(store.reserve(one).is_some());
    assert_eq!(store.next_extent(16), Some(32), "one short of the step");
}

#[test]
fn a_fixed_store_stands_whole_and_asks_for_nothing() {
    let mut store = RsStore::new(4, 4);
    let all = slots(4);
    assert!(!store.note_short(all));
    assert_eq!(store.next_extent(16), None);
    assert!(store.reserve(all).is_some());
}

#[test]
fn a_target_past_the_highest_number_out_cuts_nothing_off() {
    let mut store = RsStore::new(32, 32);
    store.commit_on_demand();
    store.set_committed(32);
    let low = store.reserve(slots(7)).expect("seven stand free");
    let _high = store.reserve(slots(1)).expect("one stands free");
    store.release_reservation(low);
    let (reach, _) = store.demand();
    assert_eq!(reach, 8, "holes below do not lower how far it reaches");
}

#[test]
fn a_waiter_on_a_full_target_is_told_no() {
    let mut store = RsStore::new(32, 32);
    store.commit_on_demand();
    store.set_target(0);
    assert!(!store.note_short(slots(1)));
    assert_eq!(store.free().slots, 0);
}

#[test]
fn a_state_commit_that_lands_after_a_cut_leaves_the_cut_standing() {
    let mut store = RsStore::new(32, 32);
    store.commit_on_demand();
    store.set_target(4);
    store.set_committed(16);
    assert_eq!(store.reserve(slots(5)).map(|_| ()), None);
    assert_eq!(store.mapped(), 16, "the tail gives it back");
}
