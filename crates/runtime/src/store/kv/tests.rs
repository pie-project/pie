use std::collections::HashSet;

use super::KvStore;
use super::hash::Hash256;
use super::page_table::{
    KvPageBacking, KvPageTable, PhysicalKvPageId, PublishedPage, ReclaimQuote,
    WorkingSetId,
};
use super::write::{KvPreparedWrite, PageCommit};
use crate::inferlet::process::ProcessId;
use crate::policy::retention::Keys;

/// Keys that let every entry go: the runtime's to drop as its room is wanted.
fn floor(_: Option<ProcessId>) -> i64 {
    crate::policy::retention::DEFAULT_KEY
}

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

#[test]
fn tests_every_case() {
    publish_lookup_flatten_roundtrip();
    fork_shares_prefix_and_diverges_into_children();
    release_reclaims_exclusive_suffix_but_keeps_shared_prefix();
    a_page_seals_under_a_key_that_names_its_prefix();
    a_working_set_maps_in_what_another_sealed();
    a_fire_that_never_settled_leaves_the_log_cut();
    idle_cache_goes_to_the_host_and_comes_back_on_a_hit();
    a_keep_the_policy_states_is_honoured_and_unkeyed_idle_cache_waits_for_a_demand();
    a_beams_finish_waits_out_a_suspend_and_names_each_page_once();
    new_weights_find_no_page_the_old_ones_sealed();
}

/// A beam search's finish while its set is suspending waits for the move
/// to end, and never names a page the move gives back (a relist under a
/// suspend would leave the chosen pages named by the new tail and the
/// suspend's pinned old one, and the suspend's commit give them back while
/// the set still reads them).
fn a_beams_finish_waits_out_a_suspend_and_names_each_page_once() {
    let mut store = KvStore::new_with_swap(16, 16, 0, h(61));
    let total = store.free_committed();
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 4, 1);
    let sets = HashSet::from([ws]);
    let txn = store.prepare_suspend(&sets, floor).unwrap().expect("private pages suspend");
    assert!(matches!(
        store.relist_beam_needs(ws, 1, &[2, 0]).unwrap(),
        super::RelistNeeds::MoveEnd(_)
    ));
    assert!(!store.relist_beam_with(ws, 1, &[2, 0], &[]).unwrap(), "nothing while it moves");
    assert_eq!(store.commit_suspend(txn).unwrap(), 4);
    store.retire_idle();
    let mut granted = store.reserve_device_pages(4).unwrap();
    let txn = store.prepare_restore(&sets, &mut granted).unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 4);
    let before: Vec<PhysicalKvPageId> = (0..4).map(|i| store.lookup(ws, i).unwrap()).collect();
    assert!(matches!(
        store.relist_beam_needs(ws, 1, &[2, 0]).unwrap(),
        super::RelistNeeds::Copies(ref shared) if shared.is_empty()
    ));
    assert!(store.relist_beam_with(ws, 1, &[2, 0], &[]).unwrap());
    assert_eq!(store.mapped_len(ws).unwrap(), 3);
    let after: Vec<PhysicalKvPageId> = (0..3).map(|i| store.lookup(ws, i).unwrap()).collect();
    assert_eq!(after, [before[0], before[3], before[1]]);
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.free_committed(), total, "every page back once");
}

/// A reload in place rekeys the cache: what the old weights sealed is
/// dropped, and the same tokens key otherwise under the new ones.
fn new_weights_find_no_page_the_old_ones_sealed() {
    let mut store = KvStore::new(16, h(41));
    let tokens: Vec<u32> = (0..40).map(|t| 1300 + t).collect();
    let owner = ProcessId::from_u128(41);
    sealed_and_gone(&mut store, &tokens, owner);
    let reader = store.create_shared_working_set();
    let old = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&old).len(), 2);
    assert_eq!(store.live_working_sets(), 1);
    store.release_working_set(reader, store.current_epoch());
    assert_eq!(store.live_working_sets(), 0);

    assert_eq!(store.rekey(h(42)), 2, "the two kept pages go back");
    assert_eq!(store.available_pages(), 16);
    assert!(store.cached_working_sets().is_empty());
    assert!(store.cached_run(&old).is_empty());
    let reader = store.create_shared_working_set();
    let new = keys_ahead(&store, reader, 0, &tokens);
    assert_ne!(
        new, old,
        "the same tokens key otherwise under the new weights"
    );
    assert!(store.cached_run(&new).is_empty());
    store.release_working_set(reader, store.current_epoch());

    sealed_and_gone(&mut store, &tokens, owner);
    assert_eq!(store.cached_run(&new).len(), 2);
    assert!(store.cached_run(&old).is_empty());
}

/// Pages that came from another worker, whose keys this worker computes from
/// the same tokens under the same domain, are found where a pass looks, below
/// the device, and come back when their entry is dropped.
fn pages_another_worker_sealed_are_found_in_the_host_tier() {
    use super::hash::{chain_slots, page_hash};
    let domain = h(7);
    let mut store = KvStore::new_with_swap(8, 4, 0, domain);
    store.set_page_size(4);
    let tokens: Vec<u32> = (100..112).collect();
    let slots = chain_slots(&domain, None, 0, &tokens);
    let token_hashes: Vec<Vec<Option<Hash256>>> = slots
        .chunks(4)
        .map(|page| page.iter().map(|h| Some(*h)).collect())
        .collect();
    let keys: Vec<Hash256> = token_hashes.iter().map(|page| page_hash(page)).collect();

    let host = store.alloc_host_slots(3).expect("three host slots");
    assert!(store.alloc_host_slots(2).is_none(), "one slot is left");
    assert!(store.import_cached(&[], token_hashes, keys.clone(), host));

    let run = store.cached_run_any_tier(&keys);
    assert_eq!(run.len(), 3, "the whole run is found");
    assert_eq!(store.below_device(&run), 3, "all of it in the host tier");
    assert!(
        store.cached_run(&keys).is_empty(),
        "none of it on the device"
    );
    // A node is found by its first page: a run that starts past it, as no
    // pass's does, finds nothing.
    assert!(store.cached_run_any_tier(&keys[1..]).is_empty());

    // More pages extend the first import, grafted below its last page.
    let mut store = KvStore::new_with_swap(8, 4, 0, domain);
    store.set_page_size(4);
    let tokens: Vec<u32> = (100..116).collect();
    let slots = chain_slots(&domain, None, 0, &tokens);
    let token_hashes: Vec<Vec<Option<Hash256>>> = slots
        .chunks(4)
        .map(|page| page.iter().map(|h| Some(*h)).collect())
        .collect();
    let keys: Vec<Hash256> = token_hashes.iter().map(|page| page_hash(page)).collect();
    let head = store.alloc_host_slots(2).unwrap();
    assert!(store.import_cached(&[], token_hashes[..2].to_vec(), keys[..2].to_vec(), head));
    let lost = store.alloc_host_slots(1).unwrap();
    assert!(
        !store.import_cached(
            &keys[2..3],
            token_hashes[3..].to_vec(),
            keys[3..].to_vec(),
            lost
        ),
        "a prefix not held here refuses the import"
    );
    let tail = store
        .alloc_host_slots(2)
        .expect("the refused import gave its slot back");
    assert!(store.import_cached(
        &keys[..2],
        token_hashes[2..].to_vec(),
        keys[2..].to_vec(),
        tail
    ));
    assert_eq!(
        store.cached_run_any_tier(&keys).len(),
        4,
        "one run across both"
    );
    let mut runs = store.cached_runs();
    runs.sort_by_key(Vec::len);
    assert_eq!(
        runs,
        vec![keys[..2].to_vec(), keys.clone()],
        "each entry from token 0"
    );
}

/// Retention entries as `(owner, upto, key)`; a position no entry covers is 0.
#[derive(Clone, Default)]
struct Steps {
    entries: Vec<(ProcessId, Option<u64>, i64)>,
}

impl Keys for Steps {
    fn key_at(&self, owner: Option<ProcessId>, at: u64) -> i64 {
        self.entries
            .iter()
            .filter(|(of, upto, _)| Some(*of) == owner && upto.is_none_or(|upto| at < upto))
            .map(|&(_, _, key)| key)
            .max()
            .unwrap_or(0)
    }

    fn steps(&self, owner: ProcessId) -> Vec<u64> {
        self.entries
            .iter()
            .filter(|(of, _, _)| *of == owner)
            .filter_map(|&(_, upto, _)| upto)
            .collect()
    }
}

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

fn fork_shares_prefix_and_diverges_into_children() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    publish(&mut t, a, 5..7);
    publish(&mut t, b, 100..102);
    assert_eq!(ids(&t, a), vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(ids(&t, b), vec![0, 1, 2, 3, 4, 100, 101]);
    assert_eq!(t.node_count(), 3);
    let root_of_a = t.node_parent(t.terminal(a).unwrap().unwrap()).unwrap();
    let root_of_b = t.node_parent(t.terminal(b).unwrap().unwrap()).unwrap();
    assert_eq!(root_of_a, root_of_b);
}

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

fn pc(seed: u32) -> PageCommit {
    PageCommit {
        token_hashes: Vec::new(),
        page_hash: Some(h(seed)),
    }
}

fn prepare_write(store: &mut KvStore, ws: WorkingSetId, indexes: &[u64]) -> KvPreparedWrite {
    let need = store.write_demand(ws, indexes).unwrap();
    let mut granted = store.reserve_device_pages(need).unwrap();
    store
        .prepare_write_reserved(ws, indexes, &mut granted)
        .unwrap()
}

fn publish_prepared(store: &mut KvStore, prepared: KvPreparedWrite, commits: &[PageCommit]) {
    let seq = store.publish_prepared(prepared, commits).unwrap();
    store.settle(seq);
}

fn commit_fresh(
    store: &mut KvStore,
    ws: WorkingSetId,
    n: u64,
    epoch: u64,
) -> Vec<PhysicalKvPageId> {
    let start = store.page_len(ws).unwrap();
    store.reserve(ws, n).unwrap();
    let indexes: Vec<u64> = (start..start + n).collect();
    let prepared = prepare_write(store, ws, &indexes);
    let ids: Vec<PhysicalKvPageId> = prepared.targets().iter().map(|t| t.dst()).collect();
    let commits: Vec<PageCommit> = (0..n as u32).map(|i| pc(1000 + i)).collect();
    let _ = epoch;
    publish_prepared(store, prepared, &commits);
    ids
}

fn prompt(seed: u32, pages: u64) -> Vec<u32> {
    (0..pages as u32 * 16).map(|t| seed * 10_000 + t).collect()
}

fn sealed(store: &mut KvStore, ws: WorkingSetId, tokens: &[u32]) {
    commit_opaque(store, ws, tokens.len() as u64 / 16);
    seal(store, ws, 0, tokens);
}

fn kept_and_gone(store: &mut KvStore, seed: u32, pages: u64, owner: ProcessId) -> Vec<u32> {
    let tokens = prompt(seed, pages);
    let ws = store.create_shared_working_set();
    sealed(store, ws, &tokens);
    store.keep_checkpoint(ws, pages, Some(owner));
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    tokens
}

fn reader(store: &mut KvStore, tokens: &[u32], pages: u64) -> WorkingSetId {
    let ws = store.create_shared_working_set();
    let keys = keys_ahead(store, ws, 0, tokens);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, pages).unwrap(), pages);
    ws
}

fn on_device(store: &mut KvStore, tokens: &[u32]) -> usize {
    let probe = store.create_shared_working_set();
    let keys = keys_ahead(store, probe, 0, tokens);
    store.release_working_set(probe, store.current_epoch());
    let run = store.cached_run(&keys);
    store.table.resident_run(&run)
}

fn reclaim_spares_the_prefixes_a_live_working_set_kept() {
    let mut store = KvStore::new(8, h(9));
    let (running, idle) = (ProcessId::from_u128(1), ProcessId::from_u128(2));
    let tokens = prompt(5, 4);
    let ws = store.create_shared_working_set();
    sealed(&mut store, ws, &tokens);
    for k in 1..=4 {
        store.keep_checkpoint(ws, k, Some(running));
    }
    commit_fresh(&mut store, ws, 1, 1); // decoding goes on past the prompt
    std::thread::sleep(std::time::Duration::from_millis(2));
    kept_and_gone(&mut store, 6, 2, idle);
    assert_eq!(store.available_pages(), 1);
    let rank = |_: Option<ProcessId>| 0;
    let entries = |store: &KvStore, owner| {
        store
            .cached_working_sets()
            .into_iter()
            .find(|set| set.owner == owner)
            .map_or(0, |set| set.entries)
    };

    // The running process's entries are the older, but freeing pages
    // means taking the idle one, and only it.
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 2);
    assert!(store.owns_cache(running) && !store.owns_cache(idle));
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 0);
    assert_eq!(entries(&store, running), 4);

    // Once it ends they are the cache's alone, and go.
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 4);
    assert!(store.cached_working_sets().is_empty());

    // A reader of a shorter prefix spares only what it reads: the longer
    // entries go, and the pages only they held with them.
    let tokens = prompt(7, 4);
    let ws = store.create_shared_working_set();
    sealed(&mut store, ws, &tokens);
    for k in 1..=4 {
        store.keep_checkpoint(ws, k, Some(idle));
    }
    store.release_working_set(ws, store.current_epoch());
    let reader = reader(&mut store, &tokens, 2);
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 2);
    assert_eq!(entries(&store, idle), 2);
    assert_eq!(on_device(&mut store, &tokens), 2);
    store.release_working_set(reader, store.current_epoch());
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 2);
    assert_eq!(store.available_pages(), 8);
}

fn seal(store: &mut KvStore, ws: WorkingSetId, start: u64, tokens: &[u32]) -> u32 {
    match store.slot_hashes(ws, start, tokens, 16).unwrap() {
        Some(slots) => store.seal_rows(ws, start, &slots, 16).unwrap(),
        None => 0,
    }
}

fn keys_ahead(store: &KvStore, ws: WorkingSetId, start: u64, tokens: &[u32]) -> Vec<Hash256> {
    let slots = store.slot_hashes(ws, start, tokens, 16).unwrap().unwrap();
    store.page_keys(ws, start, &slots, 16).unwrap().unwrap()
}

fn commit_opaque(store: &mut KvStore, ws: WorkingSetId, n: u64) {
    let start = store.page_len(ws).unwrap();
    store.reserve(ws, n).unwrap();
    let indexes: Vec<u64> = (start..start + n).collect();
    let prepared = prepare_write(store, ws, &indexes);
    let commits: Vec<PageCommit> = (0..n)
        .map(|_| PageCommit {
            token_hashes: Vec::new(),
            page_hash: None,
        })
        .collect();
    publish_prepared(store, prepared, &commits);
}

fn a_page_seals_under_a_key_that_names_its_prefix() {
    let mut store = KvStore::new(16, h(15));
    let tokens: Vec<u32> = (0..40).map(|t| 500 + t).collect();
    let fresh = |store: &mut KvStore| {
        let ws = store.create_shared_working_set();
        commit_opaque(store, ws, 3);
        ws
    };
    let keys = |store: &KvStore, ws| {
        (0..3)
            .map(|index| store.page_hash_at(ws, index).unwrap())
            .collect::<Vec<_>>()
    };
    let whole = fresh(&mut store);
    assert_eq!(seal(&mut store, whole, 0, &tokens), 2);
    let [first, second, partial] = keys(&store, whole)[..] else {
        unreachable!()
    };
    assert!(first.is_some() && second.is_some() && partial.is_none());
    assert_eq!(store.sealed_in(whole, 0..3).unwrap(), [0, 1]);

    let cut = fresh(&mut store);
    assert_eq!(seal(&mut store, cut, 0, &tokens[..20]), 1);
    assert_eq!(seal(&mut store, cut, 20, &tokens[20..]), 1);
    assert_eq!(
        keys(&store, cut),
        keys(&store, whole),
        "cut fires seal the same keys"
    );

    let other = fresh(&mut store);
    let mut changed = tokens.clone();
    changed[3] += 1;
    seal(&mut store, other, 0, &changed);
    assert_ne!(
        keys(&store, other)[1],
        keys(&store, whole)[1],
        "a page names its prefix"
    );

    // A slot before the rows that is not hashed keys nothing after it.
    let opaque = fresh(&mut store);
    assert_eq!(seal(&mut store, opaque, 16, &tokens[16..32]), 0);
}

fn sealed_and_gone(store: &mut KvStore, tokens: &[u32], owner: ProcessId) {
    store.set_keeps(true);
    let ws = store.create_shared_working_set();
    store.set_owner(ws, owner).unwrap();
    commit_opaque(store, ws, (tokens.len() as u64).div_ceil(16));
    seal(store, ws, 0, tokens);
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
}

fn a_working_set_maps_in_what_another_sealed() {
    let mut store = KvStore::new(16, h(20));
    let tokens: Vec<u32> = (0..64).map(|t| 400 + t).collect();
    sealed_and_gone(&mut store, &tokens, ProcessId::from_u128(8));
    let held = 16 - store.available_pages() as u64;
    assert_eq!(held, 4);

    let ws = store.create_shared_working_set();
    store.reserve(ws, 5).unwrap();
    let keys = keys_ahead(&store, ws, 0, &tokens);
    let run = store.cached_run(&keys);
    assert_eq!(run.len(), 4);
    assert_eq!(
        store.attach_run(ws, &run, 3).unwrap(),
        3,
        "short of the limit"
    );
    assert_eq!(store.mapped_len(ws).unwrap(), 3);
    assert_eq!(store.page_len(ws).unwrap(), 5, "the reservation stays");
    assert_eq!(store.available_pages(), 12, "nothing allocated");
    let theirs: Vec<_> = run[..3]
        .iter()
        .zip(&keys)
        .map(|(&at, &key)| store.sealed_page_at(at, key).unwrap())
        .collect();
    assert_eq!(store.flat_table(ws).unwrap(), theirs.as_slice());
    // What follows chains on what was mapped in, as on the sealer's.
    let keys_after = keys_ahead(&store, ws, 48, &tokens[48..]);
    assert_eq!(keys_after, keys);
    // Mapping freezes nothing: the set has folded what its passes state.
    assert_eq!(store.folded(ws).unwrap(), 0);
    assert!(store.folded_exact(ws).unwrap());
    // A write into a mapped-in page copies it, and every mapped page
    // after it.
    assert!(!store.privately_writable(ws, 2).unwrap());
    assert_eq!(store.write_demand(ws, &[2]).unwrap(), 1);
    assert_eq!(store.write_demand(ws, &[1]).unwrap(), 2);
    assert_eq!(store.write_demand(ws, &[3]).unwrap(), 1, "a fresh page");
}

fn decoded(store: &mut KvStore, tokens: &[u32], prefilled: usize, logged: usize) -> WorkingSetId {
    store.set_keeps(true);
    store.set_page_size(16);
    let ws = store.create_shared_working_set();
    store.set_owner(ws, ProcessId::from_u128(11)).unwrap();
    commit_opaque(store, ws, (tokens.len() as u64).div_ceil(16));
    let seq = store.log_begin(ws, 0).unwrap();
    store
        .log_end(ws, seq, Some((0, &tokens[..prefilled])))
        .unwrap();
    store.fold(ws, prefilled as u64, false).unwrap();
    for at in prefilled..logged {
        let seq = store.log_begin(ws, at as u64).unwrap();
        store
            .log_end(ws, seq, Some((at as u64, &tokens[at..=at])))
            .unwrap();
    }
    ws
}

fn a_fire_that_never_settled_leaves_the_log_cut() {
    let mut store = KvStore::new(16, h(26));
    let tokens: Vec<u32> = (0..48).map(|t| 1000 + t).collect();
    let ws = decoded(&mut store, &tokens, 20, 40);
    store.log_begin(ws, 16).unwrap();
    assert_eq!(store.fold(ws, 40, true).unwrap(), 0);
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&keys).len(), 1);
}

fn idle_cache_goes_to_the_host_and_comes_back_on_a_hit() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(31));
    let owner = ProcessId::from_u128(21);
    let tokens = kept_and_gone(&mut store, 12, 3, owner);
    let txn = store
        .prepare_demote(8, |_| 1)
        .unwrap()
        .expect("three cache-only pages");
    assert_eq!(txn.page_count(), 3);
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, |_| 0),
        0,
        "a group in a move is not dropped"
    );
    assert_eq!(store.commit_suspend(txn).unwrap(), 3);
    store.retire_idle();
    assert_eq!(store.available_pages(), 8);
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, |_| 0),
        0,
        "nothing on the device to free"
    );

    // A hit: the run in any tier comes up, held until it is mapped in.
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    let run = store.cached_run_any_tier(&keys);
    assert_eq!((run.len(), store.below_device(&run)), (3, 3));
    let mut granted = store.reserve_device_pages(3).unwrap();
    let txn = store
        .prepare_promote(&run, &mut granted)
        .unwrap()
        .expect("pages below the device");
    let pin = store.commit_promote(txn).unwrap();
    assert_eq!(on_device(&mut store, &tokens), 3);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(reader, &run, 3).unwrap(), 3);
    store.release_reader(pin);
    store.release_working_set(reader, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, |_| 0), 3);
    assert_eq!(store.available_pages(), 8);
}

#[test]
/// A kept disk tier's journal gives its pages back on the next run as cache
/// on disk, found where a pass looks; pages that follow on from nothing
/// held are given up, and so is every page an entry drops.
fn a_kept_disk_tier_takes_back_the_pages_its_journal_names() {
    use super::hash::{chain_slots, page_hash};
    use super::journal::{Header, Journal, PageRecord};
    let domain = h(51);
    let pages_of = |tokens: &[u32]| -> Vec<Vec<Option<Hash256>>> {
        chain_slots(&domain, None, 0, tokens)
            .chunks(4)
            .map(|page| page.iter().map(|h| Some(*h)).collect())
            .collect()
    };
    let run: Vec<u32> = (100..112).collect();
    let mut branch = run[..8].to_vec();
    branch.extend(200..204);
    let (run_tokens, branch_tokens) = (pages_of(&run), pages_of(&branch));
    let keys: Vec<Hash256> = run_tokens.iter().map(|page| page_hash(page)).collect();
    let branch_keys: Vec<Hash256> = branch_tokens.iter().map(|page| page_hash(page)).collect();
    let record = |tokens: &[Vec<Option<Hash256>>], keys: &[Hash256], i: usize| PageRecord {
        page: keys[i],
        parent: i.checked_sub(1).map(|p| keys[p]),
        tokens: tokens[i].clone(),
        retention: 0,
    };

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g0.journal");
    let header = Header {
        domain: KvStore::new_with_swap(8, 0, 8, domain).domain,
        page_size: 4,
        slots: 8,
    };
    {
        let (mut journal, _) = Journal::open(&path, header, true).unwrap();
        for (slot, i) in [(5, 0), (2, 1), (7, 2)] {
            journal.put(slot, &record(&run_tokens, &keys, i)).unwrap();
        }
        journal
            .put(0, &record(&branch_tokens, &branch_keys, 2))
            .unwrap();
        let orphan = PageRecord {
            parent: Some(h(99)),
            ..record(&run_tokens, &keys, 1)
        };
        journal
            .put(
                3,
                &PageRecord {
                    page: h(98),
                    ..orphan
                },
            )
            .unwrap();
        let short = PageRecord {
            page: h(97),
            parent: None,
            tokens: vec![None],
            retention: 0,
        };
        journal.put(4, &short).unwrap();
        journal.sync().unwrap();
    }

    let store_for = || {
        let mut store = KvStore::new_with_swap(8, 0, 8, domain);
        store.set_page_size(4);
        store.set_keeps(true);
        store
    };
    let mut store = store_for();
    assert_eq!(store.keep_disk(&path, true).unwrap(), 4);
    let found = store.cached_run_any_tier(&keys);
    assert_eq!(found.len(), 3, "the whole run is found");
    assert_eq!(store.below_device(&found), 3, "all of it on disk");
    assert_eq!(
        store.cached_run_any_tier(&branch_keys).len(),
        3,
        "and its branch"
    );
    assert_eq!(
        store.disk_pool.available(),
        4,
        "the orphan's and the short page's slots are free again"
    );
    store.drop_entries_off_device(usize::MAX, &|_| 0, None);
    assert_eq!(
        store.cached_run_any_tier(&keys).len(),
        3,
        "what is kept on disk is no off-device leftover"
    );
    drop(store);

    // The next run takes the same back: the orphans were given up.
    let mut store = store_for();
    assert_eq!(store.keep_disk(&path, true).unwrap(), 4);
    assert_eq!(store.disk_pool.available(), 4);
    assert_eq!(
        store.disk_swap_available(floor),
        8,
        "swap may take every copy's slot"
    );
    store.yield_disk(8, &floor);
    assert_eq!(store.disk_pool.available(), 8, "every slot given back");
    assert_eq!(store.persisted_pages(), 0);
    drop(store);
    let mut store = store_for();
    assert_eq!(
        store.keep_disk(&path, true).unwrap(),
        0,
        "copies given up are given up in the journal"
    );
    drop(store);

    // Slot files the engine took afresh hold nothing a journal names.
    {
        let (mut journal, _) = Journal::open(&path, header, true).unwrap();
        journal.put(5, &record(&run_tokens, &keys, 0)).unwrap();
    }
    assert_eq!(store_for().keep_disk(&path, false).unwrap(), 0);
}

#[test]
/// A run persisted from the device keeps its pages there; when the device
/// wants their room they settle on their copies with nothing copied, a hit
/// brings them back with the copies kept, a restart finds them, and swap
/// takes their slots when it must.
fn a_persisted_run_settles_on_disk_and_outlives_hits_and_restarts() {
    use super::hash::{chain_slots, page_hash};
    let seed = h(52);
    let tokens: Vec<Vec<Option<Hash256>>> =
        chain_slots(&seed, None, 0, &(300..312).collect::<Vec<_>>())
            .chunks(4)
            .map(|page| page.iter().map(|h| Some(*h)).collect())
            .collect();
    let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g0.journal");
    let store_for = || {
        let mut store = KvStore::new_with_swap(8, 0, 8, seed);
        store.set_page_size(4);
        store.set_keeps(true);
        store
    };

    let mut store = store_for();
    assert_eq!(store.keep_disk(&path, true).unwrap(), 0);
    let gpu = store.pool.try_alloc_n(3).unwrap();
    assert!(store.import_cached_at(
        &[],
        tokens.clone(),
        keys.clone(),
        gpu.into_iter().map(KvPageBacking::Resident).collect(),
    ));

    let txn = store
        .prepare_persist(&keys, 0)
        .expect("three pages to copy");
    assert_eq!(txn.copy_plan().0.len(), 3);
    assert!(
        store.prepare_persist(&keys, 0).is_none(),
        "pages in a copy are copied once"
    );
    assert_eq!(store.commit_persist(txn), 3);
    assert_eq!(store.persisted_pages(), 3);
    assert_eq!(store.disk_pool.available(), 5);
    assert!(store.prepare_persist(&keys, 0).is_none(), "copied already");
    let run = store.cached_run_any_tier(&keys);
    assert_eq!(store.below_device(&run), 0, "the pages stay on the device");

    let epoch = store.current_epoch();
    assert_eq!(
        store.reclaim_cache(epoch, 1, Steps::default()),
        1,
        "settled as far as is short"
    );
    assert_eq!(store.reclaim_cache(epoch, 3, Steps::default()), 2);
    let run = store.cached_run_any_tier(&keys);
    assert_eq!(run.len(), 3, "the entry stays");
    assert_eq!(store.below_device(&run), 3, "settled on its copies");
    assert_eq!(store.disk_pool.available(), 5, "with nothing copied");
    assert_eq!(
        store.reclaim_cache(epoch, 1, Steps::default()),
        0,
        "what is on disk alone frees no device page"
    );

    let mut granted = store.pool.try_alloc_n(3).unwrap();
    let txn = store
        .prepare_promote(&run, &mut granted)
        .unwrap()
        .expect("a promotion");
    let pin = store.commit_promote(txn).unwrap();
    store.release_reader(pin);
    let run = store.cached_run_any_tier(&keys);
    assert_eq!(store.below_device(&run), 0, "a hit brings it back");
    assert_eq!(store.persisted_pages(), 3, "and keeps its copies");
    assert_eq!(store.disk_pool.available(), 5);
    drop(store);

    let mut store = store_for();
    assert_eq!(
        store.keep_disk(&path, true).unwrap(),
        3,
        "a restart finds it"
    );
    assert_eq!(store.make_swap_room(8, &floor), 8, "swap wins");
    assert_eq!(store.persisted_pages(), 0);
    drop(store);
    assert_eq!(store_for().keep_disk(&path, true).unwrap(), 0);
}

/// How long idle cache is kept is the policy's (KEEP_CACHE moved to PLEX):
/// a demand takes the lowest key first, so a key the policy states keeps
/// cache past the rest. With no policy, idle cache has no expiry: it stays
/// whatever time passes, and only a demand for its room (the planner's
/// reclaim) takes it, lowest key then least recently used first.
fn a_keep_the_policy_states_is_honoured_and_unkeyed_idle_cache_waits_for_a_demand() {
    let mut store = KvStore::new(8, h(43));
    let (kept, unkept) = (ProcessId::from_u128(47), ProcessId::from_u128(48));
    let kept_tokens = kept_and_gone(&mut store, 22, 3, kept);
    let unkept_tokens = kept_and_gone(&mut store, 23, 3, unkept);
    // No demand: nothing is reclaimed, however long the cache sits.
    for _ in 0..1000 {
        store.retire_idle();
    }
    assert_eq!(on_device(&mut store, &kept_tokens), 3);
    assert_eq!(on_device(&mut store, &unkept_tokens), 3);
    // The policy keys one higher: a demand reclaims the lower first.
    let keep = move |owner: Option<ProcessId>| i64::from(owner == Some(kept));
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, keep), 3);
    assert_eq!(on_device(&mut store, &unkept_tokens), 0, "the lower key goes first");
    assert_eq!(on_device(&mut store, &kept_tokens), 3, "the higher stays");
    // No lower tier to move to: cached KV yields to the next demand,
    // whatever its key.
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, keep), 3);
    assert_eq!(on_device(&mut store, &kept_tokens), 0);
}

/// A process waiting for admission is told when its prompt's prefix joins
/// the store's cache, and again when it is evicted; a change on a path its
/// prompt does not need looks it up not at all.
#[test]
fn a_waiting_prompt_finds_its_cached_prefix_and_loses_it_when_evicted() {
    use crate::store::registry;
    use crate::store::waiting::tests::{found, waits};
    use futures::FutureExt;

    let model = registry::register_model(16, &[16], &[1]);
    let stores = registry::all_for_model(model);
    let tokens: Vec<u32> = (0..64).map(|t| 700 + t).collect();
    let waiter = ProcessId::from_u128(71);
    waits(
        &stores,
        waiter,
        &crate::policy::share::test_tenant(),
        &tokens,
    );
    assert_eq!(found(&stores, waiter), (0, Some(0)), "nothing cached yet");

    let other: Vec<u32> = (0..64).map(|t| 900 + t).collect();
    registry::with_kv_lock(&stores[0].kv, "test", |kv| {
        sealed_and_gone(kv, &other, ProcessId::from_u128(72))
    });
    assert_eq!(
        found(&stores, waiter),
        (0, Some(0)),
        "another prompt's prefix joins: no lookup"
    );

    registry::with_kv_lock(&stores[0].kv, "test", |kv| {
        sealed_and_gone(kv, &tokens, ProcessId::from_u128(70))
    });
    assert_eq!(
        found(&stores, waiter),
        (1, Some(64)),
        "its four pages join, before it runs"
    );

    let changed = registry::with_kv_lock(&stores[0].kv, "test", |kv| kv.changed());
    let mut told = Box::pin(changed.notified());
    told.as_mut().enable();
    let dropped = registry::with_kv_lock(&stores[0].kv, "test", |kv| {
        kv.discard_cache(ProcessId::from_u128(70), crate::policy::By::Host)
    });
    assert!(dropped > 0, "its prefix is evicted");
    assert!(
        told.now_or_never().is_some(),
        "the store says its cache changed"
    );
    assert_eq!(
        found(&stores, waiter),
        (1, Some(0)),
        "and the waiter, looked up again, finds none"
    );
    crate::store::waiting::left(waiter);
}

#[test]
/// With no disk tier (and the host short, or none), a suspend whose pages
/// need the disk is refused by name, as on a full disk: nothing goes
/// anywhere else, and the pages stay on the device.
fn a_suspend_that_needs_the_disk_with_no_disk_tier_is_refused_by_name() {
    for host in [0, 2] {
        let mut store = KvStore::new_with_swap(8, host, 0, h(64));
        let ws = store.create_working_set();
        commit_fresh(&mut store, ws, 3, 1);
        match store.prepare_suspend(&HashSet::from([ws]), floor) {
            Err(error @ super::KvStoreError::HostSwapFull { requested: 3, available }) => {
                assert_eq!(available, host as usize);
                assert!(error.to_string().contains("swap exhausted"), "{error}");
            }
            other => panic!("refused for want of a disk tier: {other:?}"),
        }
        assert_eq!(store.swapped_page_count(&HashSet::from([ws])).unwrap(), 0);
        assert_eq!(store.disk_swap_capacity(), 0);
    }
}

/// No runtime responder answers tier pressure (A1): under `noop` (no rows,
/// so every tier and least recently used among equals), a device full of
/// idle cache is pressure the stores answer themselves. A process short of
/// pages waits in the planner, which moves the idle cache down to the host
/// (not dropped: its tiers name the host) and serves it there.
#[tokio::test(flavor = "current_thread")]
async fn under_noop_device_pressure_moves_idle_cache_down_and_the_waiter_is_served() {
    use crate::planner::{Acquired, Demand, RegistryPool, ResidencyPlanner};
    use crate::scheduler::worker::test_lane::TestLane;
    use crate::store::registry;
    use std::sync::Arc;

    let _lock = crate::policy::schedule::TEST_LOCK.write().await;
    crate::policy::retention::install(None);
    crate::policy::schedule::install(None);
    let model = registry::register_model_with_swap(
        16,
        &[8],
        &[8],
        &[0],
        &[registry::StatePool::default()],
        &[0],
    );
    let stores = registry::get(model, 0);
    // An exit left eight pages of idle cache: the device has none free.
    let owner = ProcessId::from_u128(0xa1);
    registry::with_kv_lock(&stores.kv, "test", |kv| {
        kept_and_gone(kv, 0xa1, 8, owner);
        assert_eq!(kv.reclaimable_cache_pages(), 8, "the device is all idle cache");
    });
    let mut lane = TestLane::install(0);
    let planner = Arc::new(ResidencyPlanner::new(Arc::new(RegistryPool::new(model, 0, true))));
    let pid = ProcessId::new_v4();
    planner.register(pid);
    planner.note_admitted(pid);
    let p = planner.clone();
    let ask = crate::rt::spawn(async move {
        p.acquire(
            pid,
            pid,
            Demand {
                kv_pages: 4,
                rs: Default::default(),
            },
        )
        .await
    });
    for _ in 0..400 {
        crate::rt::yield_now().await;
        for copy in lane.take(&mut |_| Ok(())) {
            copy.land();
        }
        if ask.is_finished() {
            break;
        }
        planner.pages_freed();
    }
    assert!(ask.is_finished(), "the waiter is served, with no responder");
    let granted = ask.await.unwrap().expect("no refusal");
    assert!(matches!(granted, Acquired::Granted(_)), "its pages granted");
    registry::with_kv_lock(&stores.kv, "test", |kv| {
        assert!(kv.host_cache_pages() > 0, "the idle cache moved down to the host");
        assert!(
            kv.cached_working_sets().iter().any(|set| set.owner == owner),
            "and is kept there, not dropped"
        );
    });
}

#[test]
/// A unified device (no host tier) with a default home: the disk tier the
/// engine derives when budget_disk is unstated ([`engine::disk::derived_budget`]:
/// every seat's kv at the full context, within the free disk) takes a
/// suspend under kv pressure whole, so the process is suspended to disk,
/// not restarted, and reads back whole.
fn a_unified_load_with_defaults_suspends_kv_to_disk() {
    const SEATS: u64 = 1;
    const PAGES_A_SEAT: u64 = 3;
    let page = ::engine::disk::ALIGN as u64;
    let derived = ::engine::disk::derived_budget(0, 0, SEATS * PAGES_A_SEAT, page, 0, u64::MAX);
    let disk = (derived / page) as u32;
    assert_eq!(disk, 3, "every seat's kv at the full context");
    let mut store = KvStore::new_with_swap(4, 0, disk, h(65));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, PAGES_A_SEAT, 1);
    let sets = HashSet::from([ws]);
    let txn = store
        .prepare_suspend(&sets, floor)
        .expect("a disk tier takes it, not a refusal")
        .expect("three resident pages suspend");
    let tiers: Vec<_> = txn
        .copy_plans()
        .iter()
        .map(|(tier, gpu, _)| (*tier, gpu.len()))
        .collect();
    use ::engine::transfer::MemoryDomain::LocalDisk;
    assert_eq!(tiers, [(LocalDisk, 3)], "straight to the disk");
    assert_eq!(store.commit_suspend(txn).unwrap(), 3);
    assert_eq!(store.swapped_page_count(&sets).unwrap(), 3);
    store.retire_idle();
    let mut granted = store.reserve_device_pages(3).unwrap();
    let txn = store.prepare_restore(&sets, &mut granted).unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 3, "read back whole");
}
