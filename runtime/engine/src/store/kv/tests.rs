//! Unit tests for the KV mapping trie, hashes, pool, and KvStore protocol.

use std::collections::HashSet;

use super::hash::{self, Hash256};
use super::page_table::{
    KvPageBacking, KvPageTable, KvTableError, PhysicalKvPageId, PublishedPage, TriePageLocation,
    WorkingSetId,
};
use super::write::{PageCommit, PreparedTarget};
use super::{KvStore, KvStoreError};
use crate::store::pool::Pool;

fn h(seed: u32) -> Hash256 {
    let mut out = [0u8; 32];
    out[..4].copy_from_slice(&seed.to_le_bytes());
    out
}

fn page(id: u32) -> PublishedPage {
    PublishedPage {
        id: PhysicalKvPageId(id),
        token_hashes: Vec::new(),
        page_hash: Some(h(id)),
    }
}

fn pages(range: std::ops::Range<u32>) -> Vec<PublishedPage> {
    range.map(page).collect()
}

/// Reserve + publish `range` as one batch.
fn publish(table: &mut KvPageTable, ws: WorkingSetId, range: std::ops::Range<u32>) {
    let count = (range.end - range.start) as u64;
    table.reserve(ws, count).unwrap();
    table.publish_appended(ws, pages(range)).unwrap();
}

fn ids(table: &KvPageTable, ws: WorkingSetId) -> Vec<u32> {
    table.flatten(ws).unwrap().iter().map(|p| p.0).collect()
}

fn sorted(mut v: Vec<PhysicalKvPageId>) -> Vec<u32> {
    v.sort();
    v.into_iter().map(|p| p.0).collect()
}

fn sorted_backings(mut v: Vec<KvPageBacking>) -> Vec<u32> {
    v.sort_by_key(|backing| match backing {
        KvPageBacking::Resident(page) => page.0,
        KvPageBacking::Swapped(slot) => slot.0,
    });
    v.into_iter()
        .map(|backing| match backing {
            KvPageBacking::Resident(page) => page.0,
            KvPageBacking::Swapped(slot) => slot.0,
        })
        .collect()
}

#[test]
fn offloaded_prefix_adoption_publishes_pages_hashes_and_cas() {
    let mut store = KvStore::new(8, h(99));
    let ws = store.create_working_set();
    let tokens = (0..32).collect::<Vec<u32>>();
    let reserved = store.reserve_device_pages(2).unwrap();
    let reserved_ids = reserved.iter().map(|page| page.0).collect::<Vec<_>>();

    assert_eq!(
        store
            .adopt_offloaded_prefix(ws, &tokens, reserved, 16)
            .unwrap(),
        2
    );
    assert_eq!(
        store
            .flat_table(ws)
            .unwrap()
            .1
            .iter()
            .map(|page| page.0)
            .collect::<Vec<_>>(),
        reserved_ids
    );
    assert_eq!(store.committed_token_len(ws, 16).unwrap(), 32);
    let boundary = store.chain_state(ws).unwrap().unwrap();
    assert!(store.lookup_cached_page(&boundary).is_some());
}

#[test]
fn invalid_offloaded_prefix_releases_reserved_pages() {
    let mut store = KvStore::new(4, h(100));
    let ws = store.create_working_set();
    let reserved = store.reserve_device_pages(1).unwrap();
    assert_eq!(store.available_pages(), 3);
    assert!(
        store
            .adopt_offloaded_prefix(ws, &[1, 2, 3], reserved, 16)
            .is_err()
    );
    assert_eq!(store.available_pages(), 4);
}

/// A WorkingSet with two owned nodes: N1 = ids 0..5 shared-then-released via a
/// throwaway fork, N2 = ids 5..10.
fn two_node_ws(table: &mut KvPageTable) -> WorkingSetId {
    let ws = table.create_working_set();
    publish(table, ws, 0..5);
    let block = table.fork(ws).unwrap(); // forces the next publish into a child
    publish(table, ws, 5..10);
    table.release_working_set(block);
    ws
}

// ----------------------------------------------------------------------
// Basic mapping
// ----------------------------------------------------------------------

#[test]
fn publish_lookup_flatten_roundtrip() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    publish(&mut t, ws, 0..5);
    assert_eq!(ids(&t, ws), vec![0, 1, 2, 3, 4]);
    for i in 0..5 {
        assert_eq!(t.lookup(ws, i).unwrap(), PhysicalKvPageId(i as u32));
    }
    assert_eq!(t.page_len(ws).unwrap(), 5);
    assert_eq!(t.mapped_len(ws).unwrap(), 5);
}

#[test]
fn single_state_runahead_reuses_the_published_mapping() {
    let mut store = KvStore::new(2, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 1, 1);

    let first = store.prepare_write(ws, &[0]).unwrap();
    let (first_seq, first_intents) = store.publish_prepared(first, &[pc(19)]).unwrap();
    let follower = store.prepare_write(ws, &[0]).unwrap();
    assert!(matches!(
        follower.targets(),
        [PreparedTarget::InPlace { .. }]
    ));
    let (follow_seq, follow_intents) = store.publish_prepared(follower, &[pc(21)]).unwrap();
    store.settle(first_intents, true);
    store.retire_through(first_seq);
    store.settle(follow_intents, true);
    store.retire_through(follow_seq);
    assert_eq!(store.available_pages(), 1);
}

#[test]
fn reserve_is_logical_and_does_not_shift_committed_indexes() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    publish(&mut t, ws, 0..3);
    let range = t.reserve(ws, 2).unwrap();
    assert_eq!(range, 3..5);
    assert_eq!(t.page_len(ws).unwrap(), 5);
    assert_eq!(t.mapped_len(ws).unwrap(), 3);
    // Committed indexes are stable across a pending reservation.
    assert_eq!(t.lookup(ws, 0).unwrap(), PhysicalKvPageId(0));
    assert_eq!(
        t.lookup(ws, 3),
        Err(KvTableError::Unwritten {
            index: 3,
            mapped_len: 3
        })
    );
    assert_eq!(
        t.lookup(ws, 5),
        Err(KvTableError::IndexOutOfRange {
            index: 5,
            page_len: 5
        })
    );
    t.publish_appended(ws, pages(3..5)).unwrap();
    assert_eq!(ids(&t, ws), vec![0, 1, 2, 3, 4]);
}

#[test]
fn publish_without_reservation_is_rejected() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    assert!(matches!(
        t.publish_appended(ws, pages(0..2)),
        Err(KvTableError::PublishExceedsReservation { .. })
    ));
}

#[test]
fn private_terminal_extends_in_place() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    publish(&mut t, ws, 0..3);
    publish(&mut t, ws, 3..5);
    assert_eq!(t.node_count(), 1); // extended, no new node
    assert_eq!(ids(&t, ws), vec![0, 1, 2, 3, 4]);
}

// ----------------------------------------------------------------------
// Fork
// ----------------------------------------------------------------------

#[test]
fn fork_shares_prefix_and_diverges_into_children() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    publish(&mut t, a, 5..7);
    publish(&mut t, b, 100..102);
    assert_eq!(ids(&t, a), vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(ids(&t, b), vec![0, 1, 2, 3, 4, 100, 101]);
    // Shared root plus one fresh child per branch; no copies.
    assert_eq!(t.node_count(), 3);
    let root_of_a = t.node_parent(t.terminal(a).unwrap().unwrap()).unwrap();
    let root_of_b = t.node_parent(t.terminal(b).unwrap().unwrap()).unwrap();
    assert_eq!(root_of_a, root_of_b);
}

#[test]
fn fork_blocks_in_place_extension_of_shared_terminal() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..3);
    let b = t.fork(a).unwrap();
    publish(&mut t, a, 3..5); // must go to a child, b still sees 3 pages
    assert_eq!(t.node_count(), 2);
    assert_eq!(ids(&t, b), vec![0, 1, 2]);
    assert_eq!(ids(&t, a), vec![0, 1, 2, 3, 4]);
}

// ----------------------------------------------------------------------
// Slice
// ----------------------------------------------------------------------

#[test]
fn slice_ending_at_node_boundary_points_directly_at_the_node() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let nodes_before = t.node_count();
    let c = t.slice(a, 0..5).unwrap();
    assert_eq!(t.node_count(), nodes_before); // no new node
    assert_eq!(ids(&t, c), vec![0, 1, 2, 3, 4]);
    let n1 = t.node_parent(t.terminal(a).unwrap().unwrap()).unwrap();
    assert_eq!(t.terminal(c).unwrap().unwrap(), n1);
}

#[test]
fn multi_node_slice_needs_one_prefix_selection_and_no_chain() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let nodes_before = t.node_count();
    // [2..8) starts inside N1 and ends inside N2: the front cut is implicit
    // (page_len truncation); only the end boundary materializes a selection.
    let c = t.slice(a, 2..8).unwrap();
    assert_eq!(t.node_count(), nodes_before + 1);
    assert_eq!(ids(&t, c), vec![2, 3, 4, 5, 6, 7]);
    for (i, expected) in (2..8).enumerate() {
        assert_eq!(t.lookup(c, i as u64).unwrap(), PhysicalKvPageId(expected));
    }
    let s = t.terminal(c).unwrap().unwrap();
    assert!(t.node_is_selection(s));
    assert_eq!(t.node_len(s), 3); // prefix [0..3) of N2
    assert_eq!(t.node_parent(s).unwrap(), t.terminal(a).unwrap().unwrap());
    // The parent is untouched.
    assert_eq!(ids(&t, a), (0..10).collect::<Vec<_>>());
}

#[test]
fn slice_of_slice_composes_runs_as_a_sibling_under_the_owner() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let n2 = t.terminal(a).unwrap().unwrap();
    let c = t.slice(a, 2..8).unwrap(); // selection of N2, runs [0..3)
    let d = t.slice(c, 0..4).unwrap(); // ends inside that selection
    let s2 = t.terminal(d).unwrap().unwrap();
    assert!(t.node_is_selection(s2));
    // Composed against the owned owner N2, inserted as its child — selections
    // never chain.
    assert_eq!(t.node_parent(s2).unwrap(), n2);
    assert_eq!(ids(&t, d), vec![2, 3, 4, 5]);
}

// ----------------------------------------------------------------------
// Discard
// ----------------------------------------------------------------------

#[test]
fn private_interior_discard_drains_across_node_boundaries() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let freed = t.discard(a, &[3..7]).unwrap();
    assert_eq!(sorted_backings(freed), vec![3, 4, 5, 6]);
    assert_eq!(ids(&t, a), vec![0, 1, 2, 7, 8, 9]);
    assert_eq!(t.mapped_len(a).unwrap(), 6);
    assert_eq!(t.page_len(a).unwrap(), 6);
}

#[test]
fn shared_discard_within_terminal_becomes_a_multi_run_selection() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    let freed = t.discard(b, &[1..3]).unwrap();
    assert!(freed.is_empty()); // a still holds every page
    assert_eq!(ids(&t, b), vec![0, 3, 4]);
    assert_eq!(ids(&t, a), vec![0, 1, 2, 3, 4]); // untouched
    let s = t.terminal(b).unwrap().unwrap();
    assert!(t.node_is_selection(s));
    assert_eq!(t.node_parent(s).unwrap(), t.terminal(a).unwrap().unwrap());
}

#[test]
fn shared_tail_discard_above_terminal_moves_the_terminal_up() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let b = t.fork(a).unwrap();
    let freed = t.discard(b, &[3..10]).unwrap();
    assert!(freed.is_empty());
    assert_eq!(ids(&t, b), vec![0, 1, 2]);
    assert_eq!(ids(&t, a), (0..10).collect::<Vec<_>>());
    let s = t.terminal(b).unwrap().unwrap();
    assert!(t.node_is_selection(s)); // prefix [0..3) of N1
    assert_eq!(t.node_len(s), 3);
}

#[test]
fn shared_front_discard_is_pure_truncation() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let b = t.fork(a).unwrap();
    let nodes_before = t.node_count();
    let freed = t.discard(b, &[0..7]).unwrap();
    assert!(freed.is_empty());
    assert_eq!(t.node_count(), nodes_before); // no structural node
    assert_eq!(ids(&t, b), vec![7, 8, 9]);
    assert_eq!(t.mapped_len(b).unwrap(), 3);
    assert_eq!(ids(&t, a), (0..10).collect::<Vec<_>>());
}

#[test]
fn shared_interior_discard_is_rejected_atomically() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let b = t.fork(a).unwrap();
    // Interior range above the terminal with a shared suffix below: rerouting
    // N2 under a selection would violate the growth-boundary invariant.
    assert_eq!(
        t.discard(b, &[2..7]),
        Err(KvTableError::SharedInteriorDiscard)
    );
    // Atomicity: a legal range in the same call must not have been applied.
    assert_eq!(
        t.discard(b, &[8..9, 2..7]),
        Err(KvTableError::SharedInteriorDiscard)
    );
    assert_eq!(ids(&t, b), (0..10).collect::<Vec<_>>());
    assert_eq!(t.page_len(b).unwrap(), 10);
}

#[test]
fn discard_of_reserved_unpublished_space_is_logical_only() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..3);
    t.reserve(a, 4).unwrap();
    let freed = t.discard(a, &[5..7]).unwrap();
    assert!(freed.is_empty());
    assert_eq!(t.page_len(a).unwrap(), 5);
    assert_eq!(t.mapped_len(a).unwrap(), 3);
}

// ----------------------------------------------------------------------
// Lifetime: reachability, cache roots, pins, compaction
// ----------------------------------------------------------------------

#[test]
fn release_reclaims_exclusive_suffix_but_keeps_shared_prefix() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let c = t.fork(a).unwrap();
    publish(&mut t, a, 5..10); // a's exclusive suffix
    let freed = t.release_working_set(a);
    assert_eq!(sorted(freed), vec![5, 6, 7, 8, 9]);
    assert_eq!(ids(&t, c), vec![0, 1, 2, 3, 4]);
    let freed = t.release_working_set(c);
    assert_eq!(sorted(freed), vec![0, 1, 2, 3, 4]);
    assert_eq!(t.node_count(), 0);
}

#[test]
fn cache_root_lease_keeps_an_otherwise_unused_path_alive() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let root = t.terminal(a).unwrap().unwrap();
    t.lease_cache_root(root);
    let freed = t.release_working_set(a);
    assert!(freed.is_empty());
    assert_eq!(t.node_count(), 1);
    let freed = t.release_cache_root(root);
    assert_eq!(sorted_backings(freed), vec![0, 1, 2, 3, 4]);
    assert_eq!(t.node_count(), 0);
}

#[test]
fn drop_unused_cache_leases_reclaims_lease_only_prefixes() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let shared_root = t.terminal(a).unwrap().unwrap();
    t.lease_cache_root(shared_root); // in use: a's terminal IS this node

    let b = t.create_working_set();
    publish(&mut t, b, 5..8);
    let orphan_root = t.terminal(b).unwrap().unwrap();
    t.lease_cache_root(orphan_root);
    t.release_working_set(b); // now retained ONLY by the lease

    let (dropped, freed) = t.drop_unused_cache_leases();
    assert_eq!(dropped, 1);
    assert_eq!(sorted_backings(freed), vec![5, 6, 7]); // orphan lease reclaimed
    assert_eq!(ids(&t, a), vec![0, 1, 2, 3, 4]); // in-use lease kept
    assert_eq!(t.drop_unused_cache_leases().0, 0); // idempotent
}

#[test]
fn exclusive_footprint_counts_only_the_private_suffix() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    publish(&mut t, a, 5..10); // a's private suffix
    publish(&mut t, b, 10..12); // b's private suffix

    assert_eq!(t.exclusive_footprint(a).unwrap(), 5);
    assert_eq!(t.exclusive_footprint(b).unwrap(), 2);

    // A pinned terminal (in-flight fire) is not reclaimable by preemption.
    let term_a = t.terminal(a).unwrap().unwrap();
    t.pin(term_a);
    assert_eq!(t.exclusive_footprint(a).unwrap(), 0);
    t.unpin(term_a);
    assert_eq!(t.exclusive_footprint(a).unwrap(), 5);

    // Releasing b makes the shared prefix a's alone.
    let freed = t.release_working_set(b);
    assert_eq!(sorted(freed), vec![10, 11]);
    assert_eq!(t.exclusive_footprint(a).unwrap(), 10);
}

#[test]
fn pin_blocks_in_place_extension() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..3);
    let terminal = t.terminal(a).unwrap().unwrap();
    t.pin(terminal);
    publish(&mut t, a, 3..5); // must not mutate the pinned snapshot's node
    assert_eq!(t.node_count(), 2);
    assert!(t.unpin(terminal).is_empty());
    assert_eq!(ids(&t, a), vec![0, 1, 2, 3, 4]);
}

#[test]
fn owner_compaction_frees_excluded_slots_once_the_selection_is_sole_consumer() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    let freed = t.discard(b, &[0..2]).unwrap();
    assert!(freed.is_empty()); // a still consumes every page
    assert_eq!(ids(&t, b), vec![2, 3, 4]);
    // a leaves: b's selection becomes the sole consumer; compaction takes
    // ownership of the selected entries and frees the excluded slots.
    let freed = t.release_working_set(a);
    assert_eq!(sorted(freed), vec![0, 1]);
    assert_eq!(ids(&t, b), vec![2, 3, 4]); // mapping unchanged
    assert_eq!(t.lookup(b, 0).unwrap(), PhysicalKvPageId(2));
}

#[test]
fn liveness_counts_anchor_multiplicity_without_scanning() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..3);
    let root = t.terminal(a).unwrap().unwrap();
    let b = t.fork(a).unwrap();
    t.lease_cache_root(root);
    t.lease_cache_root(root);
    t.pin(root);
    t.pin(root);
    t.assert_liveness_consistent();

    assert!(t.release_working_set(a).is_empty());
    assert!(t.unpin(root).is_empty());
    assert!(t.release_cache_root(root).is_empty());
    t.assert_liveness_consistent();
    assert!(t.release_working_set(b).is_empty());
    assert!(t.unpin(root).is_empty());
    let freed = t.release_cache_root(root);
    assert_eq!(sorted_backings(freed), vec![0, 1, 2]);
    assert_eq!(t.node_count(), 0);
    t.assert_liveness_consistent();
}

#[test]
fn selection_rewrite_recompacts_its_owner() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    assert!(t.discard(b, &[0..2]).unwrap().is_empty());
    assert_eq!(sorted(t.release_working_set(a)), vec![0, 1]);
    let freed = t.discard(b, &[0..1]).unwrap();
    assert_eq!(sorted_backings(freed), vec![2]);
    assert_eq!(ids(&t, b), vec![3, 4]);
    t.assert_liveness_consistent();
}

#[test]
fn owner_compaction_retries_after_swap_location_unpins() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let owner = t.terminal(a).unwrap().unwrap();
    let b = t.fork(a).unwrap();
    assert!(t.discard(b, &[0..2]).unwrap().is_empty());
    let location = TriePageLocation {
        node: owner,
        local: 0,
    };
    t.pin_swap_locations([location]);
    assert!(t.release_working_set(a).is_empty());
    let freed = t.unpin_swap_locations([location]);
    assert_eq!(sorted_backings(freed), vec![0, 1]);
    assert_eq!(ids(&t, b), vec![2, 3, 4]);
    t.assert_liveness_consistent();
}

#[test]
fn adoption_rejects_zero_visible_extent_with_hidden_backings() {
    let mut t = KvPageTable::new();
    let parent = two_node_ws(&mut t);
    let target = t.fork(parent).unwrap();
    assert!(t.discard(target, &[0..3]).unwrap().is_empty());
    assert!(t.release_working_set(parent).is_empty());
    assert_eq!(
        sorted_backings(t.discard(target, &[0..7]).unwrap()),
        vec![3, 4, 5, 6, 7, 8, 9]
    );
    assert_eq!(t.mapped_len(target).unwrap(), 0);

    let source = t.create_working_set();
    publish(&mut t, source, 10..11);
    let node = t.terminal(source).unwrap().unwrap();
    assert!(matches!(
        t.adopt_path_prefix(target, node, 0),
        Err(KvTableError::BadRange { .. })
    ));
    assert_eq!(sorted(t.release_working_set(target)), vec![0, 1, 2]);
    t.assert_liveness_consistent();
}

#[test]
fn failed_tail_replacement_preserves_mapping_and_backings() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    publish(&mut t, ws, 0..2);
    assert_eq!(
        t.replace_tail(ws, 1, pages(10..12)),
        Err(KvTableError::PublishExceedsReservation {
            count: 2,
            mapped_len: 1,
            page_len: 2,
        })
    );
    assert_eq!(ids(&t, ws), vec![0, 1]);
    t.assert_liveness_consistent();
}

// ----------------------------------------------------------------------
// Hashes
// ----------------------------------------------------------------------

#[test]
fn path_hash_is_independent_of_node_boundaries() {
    let mut t = KvPageTable::new();
    // x: one node holding pages 0..4.
    let x = t.create_working_set();
    publish(&mut t, x, 0..4);
    // y: the same page-hash sequence split across two nodes.
    let y = t.create_working_set();
    publish(&mut t, y, 0..2);
    let blocker = t.fork(y).unwrap();
    publish(&mut t, y, 2..4);
    let hx = t.terminal_path_hash(x).unwrap();
    let hy = t.terminal_path_hash(y).unwrap();
    assert!(hx.is_some());
    assert_eq!(hx, hy);
    t.release_working_set(blocker);
}

#[test]
fn path_hash_invalidates_on_in_place_extension() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    publish(&mut t, ws, 0..3);
    let h3 = t.terminal_path_hash(ws).unwrap().unwrap();
    publish(&mut t, ws, 3..5); // in place: same node, longer contribution
    let h5 = t.terminal_path_hash(ws).unwrap().unwrap();
    assert_ne!(h3, h5);
    let expected = hash::fold_path_hash(None, &(0..5).map(h).collect::<Vec<_>>()).unwrap();
    assert_eq!(h5, expected);
}

#[test]
fn path_hash_is_none_while_any_contributing_page_hash_is_pending() {
    let mut t = KvPageTable::new();
    let ws = t.create_working_set();
    t.reserve(ws, 1).unwrap();
    t.publish_appended(
        ws,
        vec![PublishedPage {
            id: PhysicalKvPageId(0),
            token_hashes: Vec::new(),
            page_hash: None,
        }],
    )
    .unwrap();
    assert_eq!(t.terminal_path_hash(ws).unwrap(), None);
}

#[test]
fn sliced_child_shares_page_hashes_through_the_selection() {
    let mut t = KvPageTable::new();
    let a = two_node_ws(&mut t);
    let c = t.slice(a, 2..8).unwrap();
    assert_eq!(t.page_hash_at(c, 0).unwrap(), Some(h(2)));
    assert_eq!(t.page_hash_at(c, 5).unwrap(), Some(h(7)));
}

#[test]
fn token_slot_hash_chains_causally() {
    let domain = h(999);
    let s0 = hash::chain_token_slot_hash(&domain, None, 10, 0);
    let s1 = hash::chain_token_slot_hash(&domain, Some(&s0), 11, 1);
    // Same token/position with a different causal prefix must differ.
    let s0_alt = hash::chain_token_slot_hash(&domain, None, 12, 0);
    let s1_alt = hash::chain_token_slot_hash(&domain, Some(&s0_alt), 11, 1);
    assert_ne!(s1, s1_alt);
    // And a different domain changes everything.
    let other_domain = h(1000);
    assert_ne!(s0, hash::chain_token_slot_hash(&other_domain, None, 10, 0));
}

// ----------------------------------------------------------------------
// KvStore: prepare / commit / abort
// ----------------------------------------------------------------------

fn pc(seed: u32) -> PageCommit {
    PageCommit {
        token_hashes: Vec::new(),
        page_hash: Some(h(seed)),
    }
}

fn publish_prepared(
    store: &mut KvStore,
    prepared: super::write::KvPreparedWrite,
    commits: &[PageCommit],
) {
    let (seq, intents) = store.publish_prepared(prepared, commits).unwrap();
    store.settle(intents, true);
    store.retire_through(seq);
}

/// Prepare+commit `n` fresh pages onto `ws`, returning the committed ids.
fn commit_fresh(
    store: &mut KvStore,
    ws: WorkingSetId,
    n: u64,
    epoch: u64,
) -> Vec<PhysicalKvPageId> {
    let start = store.page_len(ws).unwrap();
    store.reserve(ws, n).unwrap();
    let indexes: Vec<u64> = (start..start + n).collect();
    let prepared = store.prepare_write(ws, &indexes).unwrap();
    let ids: Vec<PhysicalKvPageId> = prepared.targets().iter().map(|t| t.dst()).collect();
    let commits: Vec<PageCommit> = (0..n as u32).map(|i| pc(1000 + i)).collect();
    let _ = epoch;
    publish_prepared(store, prepared, &commits);
    ids
}

#[test]
fn store_explicit_index_roundtrip_remove_preserves_loaded_working_set() {
    let mut store = KvStore::new(4, h(42));
    let source = store.create_working_set();
    let expected = commit_fresh(&mut store, source, 2, 1);

    assert_eq!(store.update_index(b"prompt".to_vec(), source).unwrap(), 0);
    store.release_working_set(source, store.current_epoch());
    store.retire_idle();
    assert_eq!(
        store.available_pages(),
        2,
        "the explicit index root retains its pages"
    );

    let loaded = store.from_index(b"prompt").unwrap().unwrap();
    assert_eq!(
        (0..2)
            .map(|index| store.lookup(loaded, index).unwrap())
            .collect::<Vec<_>>(),
        expected
    );

    assert_eq!(store.remove_index(b"prompt").unwrap(), (true, 0));
    assert!(store.from_index(b"prompt").unwrap().is_none());
    assert_eq!(store.lookup(loaded, 1).unwrap(), expected[1]);
    let private = store.prepare_write(loaded, &[1]).unwrap();
    assert!(matches!(
        private.targets(),
        [PreparedTarget::InPlace { index: 1, dst }] if *dst == expected[1]
    ));
    store.cancel_prepared(private);

    store.release_working_set(loaded, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.available_pages(), 4);
}

#[test]
fn store_explicit_index_rejects_bad_keys_and_unmapped_tail() {
    let mut store = KvStore::new(4, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 1).unwrap();

    assert!(matches!(
        store.update_index(Vec::new(), ws),
        Err(KvStoreError::BadIndexKey { .. })
    ));
    assert!(matches!(
        store.update_index(vec![0; super::MAX_INDEX_KEY_BYTES + 1], ws),
        Err(KvStoreError::BadIndexKey { .. })
    ));
    assert!(matches!(
        store.update_index(b"tail".to_vec(), ws),
        Err(KvStoreError::Table(KvTableError::UnmappedTail {
            mapped_len: 0,
            page_len: 1
        }))
    ));
}

#[test]
fn store_explicit_index_root_participates_in_cow() {
    let mut store = KvStore::new(6, h(42));
    let ws = store.create_working_set();
    let ids = commit_fresh(&mut store, ws, 2, 1);
    store.update_index(b"shared".to_vec(), ws).unwrap();

    let shared = store.prepare_write(ws, &[1]).unwrap();
    assert!(
        shared
            .targets()
            .iter()
            .any(|target| matches!(target, PreparedTarget::Cow { src, .. } if *src == ids[1]))
    );
    store.cancel_prepared(shared);

    assert_eq!(store.remove_index(b"shared").unwrap().0, true);
    let private = store.prepare_write(ws, &[1]).unwrap();
    assert!(matches!(
        private.targets(),
        [PreparedTarget::InPlace { index: 1, dst }] if *dst == ids[1]
    ));
    store.cancel_prepared(private);
}

#[test]
fn store_pressure_evicts_an_unowned_explicit_index() {
    let mut store = KvStore::new(4, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 2, 1);
    store.update_index(b"evict".to_vec(), ws).unwrap();
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();

    assert_eq!(store.drop_unused_cache_leases(store.current_epoch()), 2);
    store.retire_idle();
    assert!(store.from_index(b"evict").unwrap().is_none());
    assert_eq!(store.available_pages(), 4);
}

#[test]
fn store_index_replacement_keeps_existing_loaded_view() {
    let mut store = KvStore::new(4, h(42));
    let first = store.create_working_set();
    let first_id = commit_fresh(&mut store, first, 1, 1)[0];
    store.update_index(b"session".to_vec(), first).unwrap();
    let loaded_first = store.from_index(b"session").unwrap().unwrap();

    let second = store.create_working_set();
    let second_id = commit_fresh(&mut store, second, 1, 1)[0];
    store.update_index(b"session".to_vec(), second).unwrap();
    let loaded_second = store.from_index(b"session").unwrap().unwrap();

    assert_eq!(store.lookup(loaded_first, 0).unwrap(), first_id);
    assert_eq!(store.lookup(loaded_second, 0).unwrap(), second_id);
    assert_ne!(first_id, second_id);
}

#[test]
fn store_index_capacity_evicts_oldest_key() {
    let mut store = KvStore::new(4, h(42));
    store.max_indexes = 1;
    let first = store.create_working_set();
    commit_fresh(&mut store, first, 1, 1);
    let second = store.create_working_set();
    commit_fresh(&mut store, second, 1, 1);

    store.update_index(b"first".to_vec(), first).unwrap();
    store.update_index(b"second".to_vec(), second).unwrap();

    assert!(store.from_index(b"first").unwrap().is_none());
    assert!(store.from_index(b"second").unwrap().is_some());
}

#[test]
fn standing_translation_publishes_immutable_mapping_snapshots() {
    let mut store = KvStore::new(4, h(42));
    let ws = store.create_working_set();
    let translation = store.translation(ws).unwrap();
    let (v0, empty) = translation.snapshot().unwrap();
    assert_eq!(v0, 0);
    assert!(empty.is_empty());

    let ids = commit_fresh(&mut store, ws, 2, 1);
    let (v1, mapped) = translation.snapshot().unwrap();
    assert!(v1 > v0);
    assert_eq!(
        mapped.as_ref(),
        ids.iter().map(|page| page.0).collect::<Vec<_>>()
    );
    assert!(empty.is_empty(), "the prior snapshot remains immutable");

    store.discard(ws, &[1..2], store.current_epoch()).unwrap();
    let (v2, shortened) = translation.snapshot().unwrap();
    assert!(v2 > v1);
    assert_eq!(shortened.as_ref(), &[ids[0].0]);
    assert_eq!(mapped.len(), 2, "an in-flight reader keeps the old table");
}

#[test]
fn ensure_backed_materializes_only_the_missing_logical_tail() {
    let mut store = KvStore::new(3, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 3).unwrap();
    let translation = store.translation(ws).unwrap();

    assert_eq!(store.ensure_backed(ws, 2).unwrap(), 2);
    assert_eq!(store.available_pages(), 1);
    assert_eq!(translation.snapshot().unwrap().1.len(), 2);
    assert_eq!(store.ensure_backed(ws, 2).unwrap(), 0);
    assert_eq!(store.available_pages(), 1);

    assert_eq!(store.ensure_backed(ws, 3).unwrap(), 1);
    assert_eq!(translation.snapshot().unwrap().1.len(), 3);
    assert!(matches!(
        store.ensure_backed(ws, 4),
        Err(KvStoreError::BadWriteSet { .. })
    ));
}

#[test]
fn demand_and_reserved_backing_consume_exactly_once() {
    let mut store = KvStore::new(3, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 3).unwrap();
    assert_eq!(store.backing_demand(ws, 2).unwrap(), 2);

    let mut granted = store.reserve_device_pages(3).unwrap();
    assert_eq!(
        store.ensure_backed_reserved(ws, 2, &mut granted).unwrap(),
        2
    );
    assert_eq!(granted.len(), 1);
    assert_eq!(store.mapped_len(ws).unwrap(), 2);
    store.release_device_reservation(granted);
    assert_eq!(store.available_pages(), 1);
}

#[test]
fn write_demand_and_reserved_prepare_leave_surplus_owned() {
    let mut store = KvStore::new(3, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 2).unwrap();
    assert_eq!(store.write_demand(ws, &[0, 1]).unwrap(), 2);

    let mut granted = store.reserve_device_pages(3).unwrap();
    let prepared = store
        .prepare_write_reserved(ws, &[0, 1], &mut granted)
        .unwrap();
    assert_eq!(prepared.targets().len(), 2);
    assert_eq!(granted.len(), 1);
    store.cancel_prepared(prepared);
    store.release_device_reservation(granted);
    store.retire_idle();
    assert_eq!(store.available_pages(), 3);
}

#[test]
fn store_fresh_append_roundtrip() {
    let mut store = KvStore::new(8, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 3).unwrap();
    let prepared = store.prepare_write(ws, &[0, 1, 2]).unwrap();
    assert_eq!(prepared.targets().len(), 3);
    assert!(
        prepared
            .targets()
            .iter()
            .all(|t| matches!(t, PreparedTarget::Fresh { .. }))
    );
    assert_eq!(prepared.copy_plan().count(), 0);
    assert_eq!(store.available_pages(), 5);
    let ids: Vec<PhysicalKvPageId> = prepared.targets().iter().map(|t| t.dst()).collect();
    publish_prepared(&mut store, prepared, &[pc(0), pc(1), pc(2)]);
    assert_eq!(store.mapped_len(ws).unwrap(), 3);
    for (i, id) in ids.iter().enumerate() {
        assert_eq!(store.lookup(ws, i as u64).unwrap(), *id);
    }
}

#[test]
fn store_in_place_write_on_private_page() {
    let mut store = KvStore::new(8, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 3, 1);
    let before = store.terminal_path_hash(ws).unwrap().unwrap();
    let id_before = store.lookup(ws, 2).unwrap();

    let prepared = store.prepare_write(ws, &[2]).unwrap();
    assert_eq!(
        prepared.targets(),
        &[PreparedTarget::InPlace {
            index: 2,
            dst: id_before
        }]
    );
    assert_eq!(store.available_pages(), 5); // no allocation
    publish_prepared(&mut store, prepared, &[pc(77)]);

    assert_eq!(store.lookup(ws, 2).unwrap(), id_before); // id stable
    assert_eq!(store.table().page_hash_at(ws, 2).unwrap(), Some(h(77)));
    let after = store.terminal_path_hash(ws).unwrap().unwrap();
    assert_ne!(before, after); // path hash invalidated and recomputed
}

#[test]
fn store_cow_on_shared_tail() {
    let mut store = KvStore::new(16, h(42));
    let a = store.create_working_set();
    let a_ids = commit_fresh(&mut store, a, 3, 1);
    let b = store.fork(a).unwrap();
    store.reserve(b, 2).unwrap();

    // b writes the shared tail page plus two fresh pages.
    let prepared = store.prepare_write(b, &[2, 3, 4]).unwrap();
    let kinds: Vec<bool> = prepared
        .targets()
        .iter()
        .map(|t| matches!(t, PreparedTarget::Cow { .. }))
        .collect();
    assert_eq!(kinds, vec![true, false, false]);
    let copy_plan: Vec<_> = prepared.copy_plan().collect();
    assert_eq!(copy_plan.len(), 1);
    assert_eq!(copy_plan[0].0, a_ids[2]); // preserved cells come from a's tail
    let new_ids: Vec<PhysicalKvPageId> = prepared.targets().iter().map(|t| t.dst()).collect();

    publish_prepared(&mut store, prepared, &[pc(10), pc(11), pc(12)]);
    // b: shared prefix + private rebased tail.
    assert_eq!(store.lookup(b, 0).unwrap(), a_ids[0]);
    assert_eq!(store.lookup(b, 1).unwrap(), a_ids[1]);
    assert_eq!(store.lookup(b, 2).unwrap(), new_ids[0]);
    assert_eq!(store.lookup(b, 3).unwrap(), new_ids[1]);
    assert_eq!(store.lookup(b, 4).unwrap(), new_ids[2]);
    // a is untouched.
    assert_eq!(store.mapped_len(a).unwrap(), 3);
    assert_eq!(store.lookup(a, 2).unwrap(), a_ids[2]);
}

#[test]
fn store_shared_write_inside_the_tail_joins_the_rebase() {
    let mut store = KvStore::new(16, h(42));
    let a = store.create_working_set();
    let a_ids = commit_fresh(&mut store, a, 3, 1);
    let b = store.fork(a).unwrap();

    // Writing only index 1 rebases [1, mapped): page 2 is copied along even
    // though unwritten, because the mapping edit is a growth-boundary rebase.
    let prepared = store.prepare_write(b, &[1]).unwrap();
    assert_eq!(prepared.targets().len(), 2);
    assert!(
        prepared
            .targets()
            .iter()
            .all(|t| matches!(t, PreparedTarget::Cow { .. }))
    );
    publish_prepared(&mut store, prepared, &[pc(20), pc(21)]);
    assert_eq!(store.lookup(b, 0).unwrap(), a_ids[0]);
    assert_ne!(store.lookup(b, 1).unwrap(), a_ids[1]);
    assert_ne!(store.lookup(b, 2).unwrap(), a_ids[2]);
    assert_eq!(store.mapped_len(b).unwrap(), 3);
}

#[test]
fn store_prepare_oom_is_typed_and_leak_free() {
    let mut store = KvStore::new(2, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 3).unwrap();
    let err = store.prepare_write(ws, &[0, 1, 2]).unwrap_err();
    assert_eq!(
        err,
        KvStoreError::OutOfPages {
            requested: 3,
            available: 2
        }
    );
    assert_eq!(store.available_pages(), 2); // nothing leaked, nothing pinned
}

#[test]
fn store_cancel_prepared_recycles_when_store_becomes_idle() {
    let mut store = KvStore::new(4, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 2).unwrap();
    let prepared = store.prepare_write(ws, &[0, 1]).unwrap();
    assert_eq!(store.available_pages(), 2);
    store.cancel_prepared(prepared);
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.mapped_len(ws).unwrap(), 0); // committed state unchanged
}

#[test]
fn later_settlement_does_not_retire_older_inflight_pages() {
    let mut store = KvStore::new(4, h(42));
    let first_ws = store.create_working_set();
    let second_ws = store.create_working_set();
    store.reserve(first_ws, 1).unwrap();
    store.reserve(second_ws, 1).unwrap();
    let first = store.prepare_write(first_ws, &[0]).unwrap();
    let second = store.prepare_write(second_ws, &[0]).unwrap();
    let (_first_seq, first_intents) = store.publish_prepared(first, &[pc(1)]).unwrap();
    let (_second_seq, second_intents) = store.publish_prepared(second, &[pc(2)]).unwrap();
    store
        .discard(first_ws, &[0..1], store.current_epoch())
        .unwrap();

    store.settle(second_intents, true);
    assert_eq!(
        store.available_pages(),
        2,
        "a later fire cannot retire a page still protected by an older fire"
    );
    store.settle(first_intents, true);
    assert_eq!(store.available_pages(), 3);
}

#[test]
fn store_publish_mismatch_cancels_prepared_write() {
    let mut store = KvStore::new(4, h(42));
    let ws = store.create_working_set();
    store.reserve(ws, 2).unwrap();
    let prepared = store.prepare_write(ws, &[0, 1]).unwrap();
    let seq = prepared.seq();
    let err = store.publish_prepared(prepared, &[pc(0)]).unwrap_err();
    assert_eq!(err, KvStoreError::CommitMismatch);
    store.retire_through(seq);
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.mapped_len(ws).unwrap(), 0);
}

#[test]
fn store_flat_table_versions_bump_only_on_mapping_change() {
    let mut store = KvStore::new(8, h(42));
    let ws = store.create_working_set();
    let ids = commit_fresh(&mut store, ws, 3, 1);
    let (v1, flat) = store.flat_table(ws).unwrap();
    assert_eq!(flat, ids.as_slice());
    let (v1_again, _) = store.flat_table(ws).unwrap();
    assert_eq!(v1, v1_again); // stable while the mapping is unchanged
    store.discard(ws, &[0..1], 2).unwrap();
    let (v2, flat) = store.flat_table(ws).unwrap();
    assert_ne!(v1, v2);
    assert_eq!(flat, &ids[1..]);
}

#[test]
fn store_suspend_restore_roundtrip_remaps_without_exposing_stale_ids() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let ws = store.create_working_set();
    let original = commit_fresh(&mut store, ws, 3, 1);
    let working_sets = HashSet::from([ws]);
    let (before_version, _) = store.flat_table(ws).unwrap();

    let txn = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    assert_eq!(txn.page_count(), 3);
    assert_eq!(store.host_swap_available(), 1);
    assert_eq!(store.available_pages(), 1);
    store.commit_suspend(txn).unwrap();

    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.host_swap_available(), 1);
    assert!(matches!(
        store.lookup(ws, 0),
        Err(KvStoreError::Table(KvTableError::NonResident { index: 0 }))
    ));
    assert!(store.flat_table(ws).is_err());

    let granted = store.reserve_device_pages(3).unwrap();
    let restored_ids = granted.clone();
    let txn = store.prepare_restore(&working_sets, granted).unwrap();
    assert_eq!(txn.page_count(), 3);
    store.commit_restore(txn).unwrap();

    let (after_version, flat) = store.flat_table(ws).unwrap();
    let flat = flat.to_vec();
    assert!(after_version > before_version);
    let mut flat_ids: Vec<_> = flat.iter().map(|id| id.0).collect();
    let mut granted_ids: Vec<_> = restored_ids.iter().map(|id| id.0).collect();
    flat_ids.sort_unstable();
    granted_ids.sort_unstable();
    assert_eq!(flat_ids, granted_ids);
    assert_eq!(store.host_swap_available(), 4);
    assert_eq!(store.available_pages(), 1);
    assert_eq!(original.len(), flat.len());
}

#[test]
fn store_suspend_and_restore_abort_are_leak_free() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let ws = store.create_working_set();
    let original = commit_fresh(&mut store, ws, 2, 1);
    let working_sets = HashSet::from([ws]);

    let suspend = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    store.abort_suspend(suspend);
    assert_eq!(store.host_swap_available(), 4);
    assert_eq!(store.available_pages(), 2);
    assert_eq!(store.flat_table(ws).unwrap().1, original.as_slice());

    let suspend = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    store.commit_suspend(suspend).unwrap();
    let granted = store.reserve_device_pages(2).unwrap();
    let restore = store.prepare_restore(&working_sets, granted).unwrap();
    store.abort_restore(restore);
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.host_swap_available(), 2);
    assert!(store.flat_table(ws).is_err());
}

#[test]
fn host_swap_reservation_is_all_or_nothing() {
    let mut store = KvStore::new_with_swap(4, 1, h(42));
    let ws = store.create_working_set();
    let original = commit_fresh(&mut store, ws, 2, 1);
    let error = store.prepare_suspend(&HashSet::from([ws])).unwrap_err();
    assert_eq!(
        error,
        KvStoreError::HostSwapFull {
            requested: 2,
            available: 1
        }
    );
    assert_eq!(store.host_swap_available(), 1);
    assert_eq!(store.flat_table(ws).unwrap().1, original.as_slice());
}

#[test]
fn store_suspend_preserves_pages_shared_with_an_external_working_set() {
    let mut store = KvStore::new_with_swap(8, 8, h(42));
    let a = store.create_working_set();
    let original = commit_fresh(&mut store, a, 3, 1);
    let b = store.fork(a).unwrap();
    store.reserve(b, 1).unwrap();
    let prepared = store.prepare_write(b, &[3]).unwrap();
    publish_prepared(&mut store, prepared, &[pc(9)]);

    let txn = match store.prepare_suspend(&HashSet::from([b])).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    assert_eq!(txn.page_count(), 1);
    store.commit_suspend(txn).unwrap();

    assert_eq!(store.flat_table(a).unwrap().1, original.as_slice());
    assert!(store.flat_table(b).is_err());
    assert_eq!(store.backing_counts(), (3, 1));
}

#[test]
fn post_drain_reclaimability_excludes_shared_and_cache_anchored_pages() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let a = store.create_working_set();
    commit_fresh(&mut store, a, 2, 1);
    let b = store.fork(a).unwrap();
    assert_eq!(
        store
            .post_drain_reclaimable_page_count(&HashSet::from([b]))
            .unwrap(),
        0
    );

    let epoch = store.current_epoch();
    store.release_working_set(b, epoch);
    let terminal = store.terminal(a).unwrap().unwrap();
    store.lease_cache_root(terminal);
    assert_eq!(
        store
            .post_drain_reclaimable_page_count(&HashSet::from([a]))
            .unwrap(),
        0
    );
}

#[test]
fn releasing_a_swapped_working_set_returns_host_slots() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 3, 1);
    let working_sets = HashSet::from([ws]);
    let txn = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    store.commit_suspend(txn).unwrap();
    assert_eq!(store.host_swap_available(), 1);

    let epoch = store.current_epoch();
    store.release_working_set(ws, epoch);
    store.retire_idle();
    assert_eq!(store.host_swap_available(), 4);
    assert_eq!(store.available_pages(), 4);
}

#[test]
fn teardown_during_residency_transactions_reclaims_pinned_orphans() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 2, 1);
    let working_sets = HashSet::from([ws]);
    let suspend = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    let epoch = store.current_epoch();
    store.release_working_set(ws, epoch);
    store.abort_suspend(suspend);
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.host_swap_available(), 4);

    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 2, 2);
    let working_sets = HashSet::from([ws]);
    let suspend = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    store.commit_suspend(suspend).unwrap();
    let granted = store.reserve_device_pages(2).unwrap();
    let restore = store.prepare_restore(&working_sets, granted).unwrap();
    let epoch = store.current_epoch();
    store.release_working_set(ws, epoch);
    store.abort_restore(restore);
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.host_swap_available(), 4);
}

#[test]
fn teardown_during_successful_restore_retires_reclaimed_pages() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 2, 1);
    let working_sets = HashSet::from([ws]);
    let suspend = match store.prepare_suspend(&working_sets).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    store.commit_suspend(suspend).unwrap();
    let granted = store.reserve_device_pages(2).unwrap();
    let restore = store.prepare_restore(&working_sets, granted).unwrap();
    let epoch = store.current_epoch();
    store.release_working_set(ws, epoch);
    store.commit_restore(restore).unwrap();
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.host_swap_available(), 4);
}

#[test]
fn cas_adoption_rejects_a_path_pinned_by_suspend() {
    let mut store = KvStore::new_with_swap(4, 4, h(42));
    let source = store.create_working_set();
    store.reserve(source, 1).unwrap();
    let prepared = store.prepare_write(source, &[0]).unwrap();
    let key = h(77);
    publish_prepared(
        &mut store,
        prepared,
        &[PageCommit {
            token_hashes: vec![Some(key)],
            page_hash: Some(h(88)),
        }],
    );
    let target = store.create_working_set();
    store.reserve(target, 2).unwrap();
    let suspend = match store.prepare_suspend(&HashSet::from([source])).unwrap() {
        super::KvSuspendPrepare::Prepared(txn) => txn,
        other => panic!("expected suspend transaction, got {other:?}"),
    };
    assert_eq!(store.adopt_cached_prefix(target, &key, 1).unwrap(), None);
    store.abort_suspend(suspend);
    assert_eq!(store.adopt_cached_prefix(target, &key, 1).unwrap(), Some(1));
    assert_eq!(store.mapped_len(target).unwrap(), 1);
    assert_eq!(store.page_len(target).unwrap(), 2);
}

#[test]
fn store_decode_appends_extend_in_place_across_fires() {
    let mut store = KvStore::new(16, h(42));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 3, 1);
    commit_fresh(&mut store, ws, 1, 2);
    commit_fresh(&mut store, ws, 1, 3);
    // Sole-user decode must not grow a node chain per fire.
    assert_eq!(store.table().node_count(), 1);
    assert_eq!(store.mapped_len(ws).unwrap(), 5);
}

// ----------------------------------------------------------------------
// Pool
// ----------------------------------------------------------------------

#[test]
fn pool_recycles_only_after_epoch_retires() {
    let mut pool: Pool<PhysicalKvPageId> = Pool::new(4);
    let taken = pool.try_alloc_n(3).unwrap();
    assert_eq!(pool.available(), 1);
    pool.recycle_after_epoch(taken, 5);
    assert_eq!(pool.available(), 1);
    assert_eq!(pool.pending_recycle(), 3);
    assert!(pool.try_alloc_n(2).is_none());
    pool.retire_through(4);
    assert!(pool.try_alloc_n(2).is_none()); // epoch 5 not yet retired
    pool.retire_through(5);
    assert_eq!(pool.available(), 4);
    assert!(pool.try_alloc_n(4).is_some());
}
