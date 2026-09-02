//! Unit tests for the KV mapping trie, hashes, pool, and KvStore protocol.
#![allow(
    clippy::single_range_in_vec_init,
    reason = "`discard`/`page_token_hashes` take `&[Range<u64>]` — a genuinely \
              PLURAL selection (see `discard(b, &[8..9, 2..7])` below), so a \
              one-element slice literal is a one-range selection, not a range \
              that was meant to be collected"
)]

use std::collections::HashSet;

use super::hash::Hash256;
use super::page_table::{
    KvPageBacking, KvPageTable, NoReclaim, PhysicalKvPageId, PublishedPage,
    ReclaimQuote, WorkingSetId,
};
use super::write::{PageCommit, PreparedTarget};
use super::KvStore;

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

// ----------------------------------------------------------------------
// Slice
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Discard
// ----------------------------------------------------------------------

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
fn held_pages_is_a_durable_fact_where_a_reclaim_quote_is_not() {
    let group = |ws| HashSet::from([ws]);
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    publish(&mut t, a, 5..10); // a's private suffix
    publish(&mut t, b, 10..12); // b's private suffix

    // Holdings include the shared prefix; the quote counts only the private
    // suffix that could actually be freed.
    assert_eq!(t.held_pages(&group(a)).unwrap(), 10);
    assert_eq!(t.held_pages(&group(b)).unwrap(), 7);
    assert_eq!(
        t.reclaim_quotes(&[group(a)], u32::MAX),
        vec![ReclaimQuote::Pages(5)]
    );

    // A pin collapses the reclaim quote to zero, but holdings must not move.
    let term_a = t.terminal(a).unwrap().unwrap();
    t.pin(term_a);
    assert_eq!(
        t.reclaim_quotes(&[group(a)], u32::MAX),
        vec![ReclaimQuote::Nothing(NoReclaim::Pinned)]
    );
    assert_eq!(
        t.held_pages(&group(a)).unwrap(),
        10,
        "an in-flight pin must not shrink what the process is holding"
    );
    t.unpin(term_a);

    // Quoted together the prefix counts once, exactly as for quotes.
    assert_eq!(t.held_pages(&HashSet::from([a, b])).unwrap(), 12);
    assert_eq!(t.held_pages(&HashSet::new()).unwrap(), 0);
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
    store.settle(seq, intents, true);
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

    let loaded = store
        .from_index(b"prompt", Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        (0..2)
            .map(|index| store.lookup(loaded, index).unwrap())
            .collect::<Vec<_>>(),
        expected
    );

    assert_eq!(store.remove_index(b"prompt").unwrap(), (true, 0));
    assert!(
        store
            .from_index(b"prompt", Default::default())
            .unwrap()
            .is_none()
    );
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

// ----------------------------------------------------------------------
// Pool
// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Lock-free page_len mirror
// ----------------------------------------------------------------------

