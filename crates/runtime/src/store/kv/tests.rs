use std::collections::HashSet;

use super::KvStore;
use super::hash::Hash256;
use super::page_table::{
    KvPageBacking, KvPageTable, NoReclaim, PhysicalKvPageId, PublishedPage, ReclaimQuote,
    WorkingSetId,
};
use super::write::{KvPreparedWrite, PageCommit, PreparedTarget};
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

fn sorted_backings(mut v: Vec<KvPageBacking>) -> Vec<u32> {
    v.sort_by_key(|backing| match backing {
        KvPageBacking::Resident(page) => page.0,
        KvPageBacking::Swapped(slot) => slot.0,
        KvPageBacking::OnDisk(slot) => slot.0,
    });
    v.into_iter()
        .map(|backing| match backing {
            KvPageBacking::Resident(page) => page.0,
            KvPageBacking::Swapped(slot) => slot.0,
            KvPageBacking::OnDisk(slot) => slot.0,
        })
        .collect()
}

#[test]
fn tests_every_case() {
    publish_lookup_flatten_roundtrip();
    fork_shares_prefix_and_diverges_into_children();
    release_reclaims_exclusive_suffix_but_keeps_shared_prefix();
    held_pages_is_a_durable_fact_where_a_reclaim_quote_is_not();
    an_unused_cache_lease_counts_the_pages_dropping_it_frees();
    a_dropped_entry_leaves_its_readers_whole();
    store_reclaims_cache_lowest_rank_first_and_spares_what_live_ones_read();
    reclaim_spares_the_prefixes_a_live_working_set_kept();
    a_suspend_past_host_swap_spills_to_disk_and_restores_whole();
    a_suspend_takes_shared_pages_to_the_host_in_place();
    an_entry_left_on_the_host_goes_when_swap_needs_room();
    a_move_under_way_keeps_only_the_pages_it_holds();
    a_page_seals_under_a_key_that_names_its_prefix();
    keys_computed_ahead_find_what_another_sealed();
    keys_from_row_zero_are_hashed_without_the_store();
    a_working_set_that_goes_leaves_its_sealed_prefix();
    a_working_set_maps_in_what_another_sealed();
    a_pass_maps_in_more_after_what_it_mapped_in();
    a_page_no_one_else_reads_is_written_in_place();
    a_lookup_takes_the_copy_on_the_device();
    a_lookup_follows_one_copy_to_its_children();
    what_a_decode_folds_seals();
    a_fire_that_never_settled_leaves_the_log_cut();
    a_hit_refreshes_only_the_entries_it_used_whole();
    what_decode_wrote_is_reclaimed_before_the_prompt_where_a_row_keys_it_below();
    a_sweep_suspends_only_the_readers_of_what_it_moves();
    idle_cache_goes_to_the_host_and_comes_back_on_a_hit();
    a_suspend_short_of_host_room_drops_host_cache_first();
    a_tail_past_its_step_goes_before_another_head();
    a_demote_leaves_what_a_higher_key_covers();
    idle_cache_the_tiers_keep_off_the_host_is_dropped_not_demoted();
    a_keep_the_policy_states_is_honoured_and_unkeyed_idle_cache_waits_for_a_demand();
    idle_cache_comes_back_past_the_fires_in_flight();
    among_equal_keys_the_least_recently_used_goes_first_answer_or_prompt();
    pages_another_worker_sealed_are_found_in_the_host_tier();
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

fn an_unused_cache_lease_counts_the_pages_dropping_it_frees() {
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let c = t.fork(a).unwrap();
    publish(&mut t, a, 5..10); // a's private suffix
    let term_a = t.terminal(a).unwrap().unwrap();
    t.lease_cache_root(term_a);
    assert_eq!(
        t.unused_cache_lease_pages(),
        0,
        "a's own working set still holds it"
    );

    // The process ends; its pages stay behind the lease, its suffix only.
    assert!(t.release_working_set(a).is_empty());
    assert_eq!(
        t.unused_cache_lease_pages(),
        5,
        "c still reads the shared prefix"
    );

    // c ends too: the whole chain now hangs off the lease.
    assert!(t.release_working_set(c).is_empty());
    assert_eq!(t.unused_cache_lease_pages(), 10);

    // Counting freed nothing, and the count is what dropping frees.
    let (dropped, freed) = t.drop_unused_cache_leases();
    assert_eq!(dropped, 1);
    assert_eq!(sorted_backings(freed), (0..10).collect::<Vec<_>>());
    assert_eq!(t.unused_cache_lease_pages(), 0);
}

fn held_pages_is_a_durable_fact_where_a_reclaim_quote_is_not() {
    let group = |ws| HashSet::from([ws]);
    let mut t = KvPageTable::new();
    let a = t.create_working_set();
    publish(&mut t, a, 0..5);
    let b = t.fork(a).unwrap();
    publish(&mut t, a, 5..10); // a's private suffix
    publish(&mut t, b, 10..12); // b's private suffix

    assert_eq!(t.held_pages(&group(a)).unwrap(), 10);
    assert_eq!(t.held_pages(&group(b)).unwrap(), 7);
    assert_eq!(
        t.reclaim_quotes(&[group(a)], u32::MAX),
        vec![ReclaimQuote::Pages(5)]
    );

    let term_a = t.terminal(a).unwrap().unwrap();
    t.pin(term_a);
    assert_eq!(
        t.reclaim_quotes(&[group(a)], u32::MAX),
        vec![ReclaimQuote::Nothing(NoReclaim::AllShared)]
    );
    assert_eq!(
        t.held_pages(&group(a)).unwrap(),
        10,
        "an in-flight pin must not shrink what the process is holding"
    );
    t.unpin(term_a);

    assert_eq!(t.held_pages(&HashSet::from([a, b])).unwrap(), 12);
    assert_eq!(t.held_pages(&HashSet::new()).unwrap(), 0);
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

fn store_reclaims_cache_lowest_rank_first_and_spares_what_live_ones_read() {
    let mut store = KvStore::new(8, h(7));
    let (old, new, kept) = (
        ProcessId::from_u128(1),
        ProcessId::from_u128(2),
        ProcessId::from_u128(3),
    );
    let old_tokens = kept_and_gone(&mut store, 1, 2, old);
    std::thread::sleep(std::time::Duration::from_millis(2));
    let new_tokens = kept_and_gone(&mut store, 2, 2, new);
    kept_and_gone(&mut store, 3, 2, kept);
    assert_eq!(store.available_pages(), 2);
    let rank = |owner: Option<ProcessId>| i64::from(owner == Some(kept));

    // Equal ranks: the one used longest ago goes, and alone.
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 2);
    assert!(!store.owns_cache(old) && store.owns_cache(new));
    assert_eq!(on_device(&mut store, &old_tokens), 0);

    // A live working set reads `new`, so dropping it would free nothing:
    // the higher-ranked one goes instead.
    let reader = reader(&mut store, &new_tokens, 2);
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 2);
    assert!(store.owns_cache(new) && !store.owns_cache(kept));
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 0);

    // Once nothing reads it, it can go too.
    store.release_working_set(reader, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, rank), 2);
    assert!(store.cached_working_sets().is_empty());

    // An owner's whole cached working-set goes when the policy drops it.
    kept_and_gone(&mut store, 4, 1, new);
    assert_eq!(store.discard_cache(new, crate::policy::By::Policy), 1);
    assert!(store.cached_working_sets().is_empty());
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

fn a_dropped_entry_leaves_its_readers_whole() {
    let mut store = KvStore::new(4, h(42));
    let owner = ProcessId::from_u128(1);
    let tokens = kept_and_gone(&mut store, 8, 2, owner);
    assert_eq!(store.available_pages(), 2, "the entry holds its pages");

    let loaded = reader(&mut store, &tokens, 2);
    let expected: Vec<_> = (0..2)
        .map(|index| store.lookup(loaded, index).unwrap())
        .collect();
    assert_eq!(store.discard_cache(owner, crate::policy::By::Policy), 0);
    assert!(store.cached_working_sets().is_empty());
    assert_eq!(store.lookup(loaded, 1).unwrap(), expected[1]);
    // Its own now, but sealed: a write copies it.
    let private = prepare_write(&mut store, loaded, &[1]);
    assert!(matches!(
        private.targets(),
        [PreparedTarget::Cow { index: 1, .. }]
    ));
    store.cancel_prepared(private);

    store.release_working_set(loaded, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.available_pages(), 4);
}

fn a_suspend_past_host_swap_spills_to_disk_and_restores_whole() {
    let mut store = KvStore::new_with_swap(4, 1, 2, h(42));
    let ws = store.create_shared_working_set();
    let before = commit_fresh(&mut store, ws, 3, 1);
    let sets = HashSet::from([ws]);
    let Some(txn) = store.prepare_suspend(&sets, floor).unwrap() else {
        panic!("three private resident pages suspend");
    };
    let tiers: Vec<_> = txn
        .copy_plans()
        .iter()
        .map(|(tier, gpu, _)| (*tier, gpu.len()))
        .collect();
    use ::engine::transfer::MemoryDomain::{HostPinned, LocalDisk};
    assert_eq!(tiers, [(HostPinned, 1), (LocalDisk, 2)]);
    assert_eq!(store.commit_suspend(txn).unwrap(), 3);
    assert_eq!(
        (store.host_swap_available(), store.disk_swap_available(floor)),
        (0, 0)
    );
    store.retire_idle();

    let mut granted = store.reserve_device_pages(3).unwrap();
    let txn = store.prepare_restore(&sets, &mut granted).unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 3);
    assert_eq!(
        (store.host_swap_available(), store.disk_swap_available(floor)),
        (1, 2)
    );
    assert_eq!(
        (0..3).filter(|&i| store.lookup(ws, i).is_ok()).count(),
        before.len()
    );
}

fn a_suspend_takes_shared_pages_to_the_host_in_place() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(11));
    let epoch = |store: &KvStore| store.current_epoch();
    // A keeps a 3-page prompt and decodes on; B and C map it in and decode
    // a page each.
    let tokens = prompt(9, 3);
    let a = store.create_shared_working_set();
    sealed(&mut store, a, &tokens);
    store.keep_checkpoint(a, 3, Some(ProcessId::from_u128(1)));
    commit_fresh(&mut store, a, 1, 1);
    let b = reader(&mut store, &tokens, 3);
    commit_fresh(&mut store, b, 1, 1);
    let c = reader(&mut store, &tokens, 3);
    commit_fresh(&mut store, c, 1, 1);
    assert_eq!(store.available_pages(), 2);
    let suspend = |store: &mut KvStore, ws| {
        let Some(txn) = store.prepare_suspend(&HashSet::from([ws]), floor).unwrap() else {
            panic!("a suspend with pages to move");
        };
        store.commit_suspend(txn).unwrap()
    };
    let swapped = |store: &KvStore, ws| store.swapped_page_count(&HashSet::from([ws])).unwrap();

    // While A runs, B and C take only their own pages.
    assert!(matches!(
        store.reclaim_quotes(&[HashSet::from([b])], u32::MAX)[..],
        [ReclaimQuote::Pages(1)]
    ));
    assert_eq!(suspend(&mut store, b), 1);
    assert_eq!(suspend(&mut store, c), 1);
    // A goes last and takes the prompt, once, for all of them; it is not
    // mapped in while it is on the host.
    assert!(matches!(
        store.reclaim_quotes(&[HashSet::from([a])], u32::MAX)[..],
        [ReclaimQuote::Pages(4)]
    ));
    assert_eq!(suspend(&mut store, a), 4);
    store.retire_idle();
    assert_eq!(
        (store.available_pages(), store.host_swap_available()),
        (8, 2)
    );
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert_eq!((swapped(&store, b), swapped(&store, c)), (4, 4));

    // Restoring B brings the prompt back for every reader.
    let mut granted = store.reserve_device_pages(4).unwrap();
    let txn = store
        .prepare_restore(&HashSet::from([b]), &mut granted)
        .unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 4);
    assert_eq!((swapped(&store, a), swapped(&store, c)), (1, 1));
    assert_eq!(on_device(&mut store, &tokens), 3, "back on the device");

    // B ends: only suspended A and C read the prompt now. A sweep takes it,
    // but for what the planner keeps, and a restore waits for the move.
    store.release_working_set(b, epoch(&store));
    store.retire_idle();
    assert!(
        store
            .prepare_sweep(&HashSet::from([c]), usize::MAX, &[], usize::MAX, floor)
            .unwrap()
            .is_none()
    );
    let sweep = store
        .prepare_sweep(&HashSet::new(), usize::MAX, &[], usize::MAX, floor)
        .unwrap()
        .unwrap();
    let mut granted = store.reserve_device_pages(4).unwrap();
    assert!(matches!(
        store.prepare_restore(&HashSet::from([c]), &mut granted),
        Err(super::KvStoreError::MoveInFlight)
    ));
    assert_eq!(store.commit_suspend(sweep).unwrap(), 3);
    store.retire_idle();
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert_eq!((swapped(&store, a), swapped(&store, c)), (4, 4));
    let txn = store
        .prepare_restore(&HashSet::from([c]), &mut granted)
        .unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 4);
    assert_eq!(swapped(&store, a), 1);
}

fn an_entry_left_on_the_host_goes_when_swap_needs_room() {
    let mut store = KvStore::new_with_swap(5, 3, 0, h(12));
    let tokens = prompt(10, 3);
    let w = store.create_shared_working_set();
    sealed(&mut store, w, &tokens);
    store.keep_checkpoint(w, 3, Some(ProcessId::from_u128(1)));
    let suspend =
        |store: &mut KvStore, ws| match store.prepare_suspend(&HashSet::from([ws]), floor) {
            Ok(Some(txn)) => store.commit_suspend(txn).unwrap(),
            other => panic!("a suspend with room: {other:?}"),
        };
    assert_eq!(suspend(&mut store, w), 3);
    store.release_working_set(w, store.current_epoch());
    assert_eq!(store.host_swap_available(), 0);

    let v = store.create_shared_working_set();
    commit_fresh(&mut store, v, 2, 1);
    assert_eq!(suspend(&mut store, v), 2);
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert!(store.cached_working_sets().is_empty());
    assert_eq!(store.host_swap_available(), 1);
}

fn a_move_under_way_keeps_only_the_pages_it_holds() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(13));
    let tokens = kept_and_gone(&mut store, 11, 3, ProcessId::from_u128(1));
    let a = reader(&mut store, &tokens, 3);
    commit_fresh(&mut store, a, 1, 1);
    let b = reader(&mut store, &tokens, 3);
    commit_fresh(&mut store, b, 1, 1);
    let quote = |store: &KvStore| store.reclaim_quotes(&[HashSet::from([b])], u32::MAX);
    // B runs beside A: only its own page goes.
    assert_eq!(quote(&store), [ReclaimQuote::Pages(1)]);
    let Some(txn) = store.prepare_suspend(&HashSet::from([a]), floor).unwrap() else {
        panic!("A has its own page to move");
    };
    assert_eq!(store.commit_suspend(txn).unwrap(), 1);
    // A is suspended: B would take the prompt too.
    assert_eq!(quote(&store), [ReclaimQuote::Pages(4)]);
    // A's restore under way holds its whole path; B still gives its own page.
    let mut granted = store.reserve_device_pages(1).unwrap();
    let restore = store
        .prepare_restore(&HashSet::from([a]), &mut granted)
        .unwrap();
    assert_eq!(quote(&store), [ReclaimQuote::Pages(1)]);
    let Some(txn) = store.prepare_suspend(&HashSet::from([b]), floor).unwrap() else {
        panic!("B's own page still goes");
    };
    assert_eq!(txn.page_count(), 1);
    assert_eq!(store.commit_suspend(txn).unwrap(), 1);
    assert_eq!(store.commit_restore(restore).unwrap(), 1);
    assert_eq!(
        store.swapped_page_count(&HashSet::from([a])).unwrap(),
        0,
        "the prompt stayed for A"
    );
}

#[test]
fn a_sweep_takes_readers_in_order_and_a_kept_head_last() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(14));
    let a = store.create_shared_working_set();
    commit_fresh(&mut store, a, 2, 1);
    let b = store.create_shared_working_set();
    commit_fresh(&mut store, b, 2, 1);
    store.mark_suspended(&HashSet::from([a, b]));
    let page = u64::from(store.page_size.max(1));
    let sweep = store
        .prepare_sweep(&HashSet::new(), usize::MAX, &[(b, page), (a, 0)], 1, floor)
        .unwrap()
        .expect("something to move");
    assert_eq!(sweep.page_count(), 1, "no more than was wanted");
    assert!(store.table.in_move(&HashSet::from([b])).unwrap());
    assert!(!store.table.in_move(&HashSet::from([a])).unwrap());
    store.commit_suspend(sweep).unwrap();
    let moved = store.swapped_page_count(&HashSet::from([b])).unwrap();
    assert_eq!(moved, 1, "the head b keeps stayed");
}

#[test]
fn a_sweep_suspends_only_the_readers_of_what_it_moves() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(14));
    let a = store.create_shared_working_set();
    commit_fresh(&mut store, a, 2, 1);
    let b = store.create_shared_working_set();
    commit_fresh(&mut store, b, 2, 1);
    // Suspended with nothing moved, as a kv-less eviction leaves them.
    store.mark_suspended(&HashSet::from([a, b]));
    let sweep = store
        .prepare_sweep(&HashSet::new(), 2, &[], usize::MAX, floor)
        .unwrap()
        .expect("room for two pages");
    assert_eq!(sweep.page_count(), 2);
    let moved = |store: &KvStore, ws| store.swapped_page_count(&HashSet::from([ws])).unwrap();
    // A whole reader moves; the other, whose pages stay, restores meanwhile.
    assert!(store.table.in_move(&HashSet::from([a])).unwrap());
    assert!(!store.table.in_move(&HashSet::from([b])).unwrap());
    let mut granted = Vec::new();
    let restore = store
        .prepare_restore(&HashSet::from([b]), &mut granted)
        .unwrap();
    store.commit_restore(restore).unwrap();
    store.commit_suspend(sweep).unwrap();
    assert_eq!((moved(&store, a), moved(&store, b)), (2, 0));
    // It runs: the next sweep leaves it alone.
    assert!(
        store
            .prepare_sweep(&HashSet::new(), usize::MAX, &[], usize::MAX, floor)
            .unwrap()
            .is_none()
    );
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

fn keys_from_row_zero_are_hashed_without_the_store() {
    let mut store = KvStore::new(16, h(15));
    let ws = store.create_shared_working_set();
    let tokens: Vec<u32> = (0..40).map(|t| 900 + t).collect();
    let slots = store.slot_hashes(ws, 0, &tokens, 16).unwrap().unwrap();
    let keys = store.page_keys(ws, 0, &slots, 16).unwrap().unwrap();
    assert_eq!(keys.len(), 2);
    assert_eq!(KvStore::fresh_page_keys(&slots, 16), keys);
    assert!(KvStore::fresh_page_keys(&slots[..15], 16).is_empty());
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

fn keys_computed_ahead_find_what_another_sealed() {
    let mut store = KvStore::new(16, h(17));
    let tokens: Vec<u32> = (0..48).map(|t| 900 + t).collect();
    let sealer = store.create_shared_working_set();
    commit_opaque(&mut store, sealer, 3);
    seal(&mut store, sealer, 0, &tokens[..20]);
    seal(&mut store, sealer, 20, &tokens[20..]);

    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(keys.len(), 3);
    assert_eq!(store.cached_run(&keys).len(), 3);
    let mut other = tokens.clone();
    other[40] += 1;
    let keys = keys_ahead(&store, reader, 0, &other);
    assert_eq!(store.cached_run(&keys).len(), 2, "the third page differs");

    // Continuing a working set that holds sealed pages: its own keys first.
    assert_eq!(
        store
            .cached_run(&keys_ahead(&store, sealer, 48, &[1, 2, 3]))
            .len(),
        3
    );
    let keys = keys_ahead(&store, sealer, 20, &tokens[20..]);
    assert_eq!(
        store.cached_run(&keys).len(),
        3,
        "a key chained from a held slot"
    );
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

fn a_working_set_that_goes_leaves_its_sealed_prefix() {
    let mut store = KvStore::new(16, h(19));
    let tokens: Vec<u32> = (0..40).map(|t| 300 + t).collect();
    let owner = ProcessId::from_u128(7);
    sealed_and_gone(&mut store, &tokens, owner);
    assert_eq!(
        store.available_pages(),
        14,
        "two sealed pages kept, the open one freed"
    );
    let sets = store.cached_working_sets();
    assert_eq!(sets.len(), 1);
    assert_eq!((sets[0].owner, sets[0].pages), (owner, 2));
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&keys).len(), 2);

    assert_eq!(store.discard_cache(owner, crate::policy::By::Host), 2);
    assert_eq!(store.cached_run(&keys).len(), 0);
    assert_eq!(store.available_pages(), 16);

    // A store that keeps nothing frees it all.
    let mut store = KvStore::new(16, h(19));
    sealed_and_gone(&mut store, &tokens, owner);
    store.set_keeps(false);
    let ws = store.create_shared_working_set();
    commit_opaque(&mut store, ws, 3);
    seal(&mut store, ws, 0, &tokens);
    store.release_working_set(ws, store.current_epoch());
    assert_eq!(
        store.cached_working_sets()[0].entries,
        1,
        "only the first one's"
    );
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

fn a_pass_maps_in_more_after_what_it_mapped_in() {
    let mut store = KvStore::new(16, h(21));
    let tokens: Vec<u32> = (0..64).map(|t| 500 + t).collect();
    sealed_and_gone(&mut store, &tokens, ProcessId::from_u128(9));
    let ws = store.create_shared_working_set();
    let keys = keys_ahead(&store, ws, 0, &tokens);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 1).unwrap(), 1);
    let keys = keys_ahead(&store, ws, 16, &tokens[16..]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 4).unwrap(), 3);
    assert_eq!(store.mapped_len(ws).unwrap(), 4);

    // Its own first page, computed, is not the path's: nothing maps in.
    let own = store.create_shared_working_set();
    commit_opaque(&mut store, own, 1);
    seal(&mut store, own, 0, &tokens[..16]);
    let keys = keys_ahead(&store, own, 16, &tokens[16..]);
    let run = store.cached_run(&keys);
    assert_eq!(run.len(), 4);
    assert_eq!(store.attach_run(own, &run, 4).unwrap(), 0);
}

fn a_page_no_one_else_reads_is_written_in_place() {
    let mut store = KvStore::new(16, h(23));
    store.set_keeps(true);
    let tokens: Vec<u32> = (0..40).map(|t| 700 + t).collect();
    let live = store.create_shared_working_set();
    commit_opaque(&mut store, live, 4);
    seal(&mut store, live, 0, &tokens);
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens[..32]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(reader, &run, 2).unwrap(), 2);
    assert!(!store.privately_writable(live, 1).unwrap(), "read by both");
    assert!(store.privately_writable(live, 2).unwrap(), "its open page");
    assert!(
        store.privately_writable(live, 3).unwrap(),
        "its unwritten page"
    );
    assert_eq!(
        store.write_demand(live, &[2, 3]).unwrap(),
        0,
        "written in place"
    );
}

fn a_lookup_takes_the_copy_on_the_device() {
    let mut store = KvStore::new(16, h(24));
    let tokens: Vec<u32> = (0..48).map(|t| 800 + t).collect();
    let copy = |store: &mut KvStore| {
        let ws = store.create_shared_working_set();
        commit_opaque(store, ws, 3);
        seal(store, ws, 0, &tokens);
        ws
    };
    copy(&mut store);
    copy(&mut store);
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    let run = store.cached_run(&keys);
    assert_eq!(run.len(), 3);
    store.table().pin_swap_locations([run[0]]);
    let other = store.cached_run(&keys);
    assert_eq!(other.len(), 3, "the other copy, whole");
    assert_ne!(other[0].node, run[0].node);
}

fn a_lookup_follows_one_copy_to_its_children() {
    let mut store = KvStore::new(32, h(77));
    store.set_keeps(true);
    let opening = prompt(77, 2);
    let mut history = opening.clone();
    history.extend(prompt(78, 2));
    let first = store.create_shared_working_set();
    sealed(&mut store, first, &opening);
    let agent = reader(&mut store, &opening, 2);
    commit_opaque(&mut store, agent, 2);
    seal(&mut store, agent, 32, &history[32..]);
    let second = store.create_shared_working_set();
    sealed(&mut store, second, &opening);
    for ws in [first, agent, second] {
        store.release_working_set(ws, store.current_epoch());
    }
    store.retire_idle();
    let next = store.create_shared_working_set();
    let keys = keys_ahead(&store, next, 0, &history);
    let run = store.cached_run(&keys);
    assert_eq!(run.len(), 4);
    assert_eq!(store.attach_run(next, &run, 4).unwrap(), 4);
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

/// What a decode folds seals once the set goes, not as it folds: the rows a
/// program ran ahead past its stop are taken back unsealed.
fn what_a_decode_folds_seals() {
    let mut store = KvStore::new(16, h(25));
    let tokens: Vec<u32> = (0..48).map(|t| 900 + t).collect();
    let ws = decoded(&mut store, &tokens, 20, 40);
    assert_eq!(store.sealed_in(ws, 0..3).unwrap(), [0]);
    assert_eq!(store.fold(ws, 40, true).unwrap(), 0, "frozen, not sealed");
    assert_eq!(store.folded(ws).unwrap(), 40);
    assert_eq!(store.sealed_in(ws, 0..3).unwrap(), [0]);
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    // A reader of the set's scope: only a shared set's release seals, and
    // only a reader sharing it hashes to the keys it left.
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(
        store.cached_run(&keys).len(),
        2,
        "the decoded page too, not the partial one"
    );
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

fn a_hit_refreshes_only_the_entries_it_used_whole() {
    let mut store = KvStore::new(16, h(27));
    let tokens: Vec<u32> = (0..48).map(|t| 1100 + t).collect();
    let (short, long) = (ProcessId::from_u128(12), ProcessId::from_u128(13));
    sealed_and_gone(&mut store, &tokens[..32], short);
    // `long` maps in the two pages, seals a third, and goes.
    let ws = store.create_shared_working_set();
    store.set_owner(ws, long).unwrap();
    let keys = keys_ahead(&store, ws, 0, &tokens[..32]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 2).unwrap(), 2);
    commit_opaque(&mut store, ws, 1);
    store.set_page_size(16);
    store.log_rows(ws, 0, &tokens).unwrap();
    assert_eq!(store.fold(ws, 48, false).unwrap(), 1);
    store.release_working_set(ws, store.current_epoch());
    let last_use = |store: &KvStore, owner| {
        store
            .cached_working_sets()
            .into_iter()
            .find(|set| set.owner == owner)
            .unwrap()
            .last_use
    };
    assert!(last_use(&store, long) >= last_use(&store, short));
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens[..32]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(reader, &run, 2).unwrap(), 2);
    assert!(
        last_use(&store, short) > last_use(&store, long),
        "used whole"
    );
}

/// A16: what a decode wrote goes before the prompt only where a row says
/// so (`default`'s tail keyed below its prompt); unkeyed, the set's cache
/// is one group by its keys, and goes together.
fn what_decode_wrote_is_reclaimed_before_the_prompt_where_a_row_keys_it_below() {
    struct TailBelow;
    impl Keys for TailBelow {
        fn key_at(&self, _: Option<ProcessId>, at: u64) -> i64 {
            if at < 16 { 0 } else { -1 }
        }
    }
    let mut store = KvStore::new(16, h(28));
    let tokens: Vec<u32> = (0..48).map(|t| 1200 + t).collect();
    let ws = decoded(&mut store, &tokens, 20, 40);
    store.fold(ws, 40, true).unwrap();
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.cached_working_sets()[0].entries, 2);
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&keys).len(), 2);
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, TailBelow), 1);
    assert_eq!(store.cached_run(&keys).len(), 1, "the prompt's page stays");
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

fn a_suspend_short_of_host_room_drops_host_cache_first() {
    let mut store = KvStore::new_with_swap(8, 3, 4, h(32));
    kept_and_gone(&mut store, 13, 3, ProcessId::from_u128(22));
    let txn = store.prepare_demote(8, |_| 1).unwrap().unwrap();
    store.commit_suspend(txn).unwrap();
    assert_eq!(store.host_swap_available(), 0);
    let running = store.create_shared_working_set();
    commit_fresh(&mut store, running, 2, 1);
    let Some(txn) = store.prepare_suspend(&HashSet::from([running]), floor).unwrap() else {
        panic!("two pages to move");
    };
    use ::engine::transfer::MemoryDomain::HostPinned;
    let tiers: Vec<_> = txn
        .copy_plans()
        .iter()
        .map(|(tier, gpu, _)| (*tier, gpu.len()))
        .collect();
    assert_eq!(tiers, [(HostPinned, 2)], "the host, not the disk");
    assert!(store.cached_working_sets().is_empty());
    store.commit_suspend(txn).unwrap();
}

fn a_tail_past_its_step_goes_before_another_head() {
    let mut store = KvStore::new(16, h(34));
    store.set_page_size(16);
    let (a, b) = (ProcessId::from_u128(41), ProcessId::from_u128(42));
    let (a_tokens, b_tokens) = (prompt(15, 4), prompt(16, 4));
    sealed_and_gone(&mut store, &a_tokens, a);
    sealed_and_gone(&mut store, &b_tokens, b);
    assert_eq!(store.available_pages(), 8);
    // T-LRU: A's first 40 tokens (three pages) at the default, the rest of
    // both below it.
    let keys = Steps {
        entries: vec![(a, None, -1), (a, Some(40), 0), (b, None, -1)],
    };
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, keys.clone()),
        1,
        "A's tail"
    );
    assert_eq!(on_device(&mut store, &a_tokens), 3);
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, keys.clone()),
        4,
        "B whole"
    );
    assert_eq!(on_device(&mut store, &b_tokens), 0);
    assert_eq!(on_device(&mut store, &a_tokens), 3, "A's head stays");
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, keys.clone()), 3);
    assert_eq!(store.available_pages(), 16);
}

fn a_demote_leaves_what_a_higher_key_covers() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(35));
    store.set_page_size(16);
    let a = ProcessId::from_u128(43);
    let tokens = prompt(17, 4);
    sealed_and_gone(&mut store, &tokens, a);
    // KVFlow: the fixed prompt (two pages) above the suffix, both above the
    // floor (the suffix below it would be dropped, not demoted).
    let keys = Steps {
        entries: vec![(a, None, 1), (a, Some(32), 10)],
    };
    let txn = store
        .prepare_demote(8, keys.clone())
        .unwrap()
        .expect("the tail goes to the host");
    assert_eq!(
        txn.page_count(),
        2,
        "only the tail: the head's key is higher"
    );
    store.commit_suspend(txn).unwrap();
    assert_eq!(on_device(&mut store, &tokens), 2, "the head stays");
}

/// Rows of one key and one tier set for every owner, as a test's table.
struct Rows(i64, u8);

impl crate::policy::retention::Keys for Rows {
    fn key_at(&self, _owner: Option<ProcessId>, _at: u64) -> i64 {
        self.0
    }

    fn tiers_at(&self, _owner: Option<ProcessId>, _at: u64) -> u8 {
        self.1
    }
}

/// Whether idle cache can move down a tier when the device wants its room
/// is the tiers row's say, whatever its key: a row without the host keeps
/// it there to be dropped; every tier (no row) moves it to the host.
fn idle_cache_the_tiers_keep_off_the_host_is_dropped_not_demoted() {
    use crate::policy::retention::{ALL_TIERS, DEVICE};
    let mut store = KvStore::new_with_swap(8, 8, 8, h(37));
    let owner = ProcessId::from_u128(45);
    let tokens = kept_and_gone(&mut store, 18, 3, owner);
    for key in [-1, 0, 100] {
        assert!(
            store.prepare_demote(8, Rows(key, DEVICE)).unwrap().is_none(),
            "nothing moves at key {key}: its row keeps it off the host"
        );
    }
    assert_eq!(store.host_swap_available(), 8, "the host is untouched");
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, Rows(100, DEVICE)),
        3,
        "it is dropped, however high its key"
    );
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert!(store.cached_working_sets().is_empty());

    let tokens = kept_and_gone(&mut store, 19, 3, owner);
    let txn = store
        .prepare_demote(8, Rows(0, ALL_TIERS))
        .unwrap()
        .expect("at the default key, with every tier, it moves");
    use ::engine::transfer::MemoryDomain::HostPinned;
    let tiers: Vec<_> = txn
        .copy_plans()
        .iter()
        .map(|(tier, gpu, _)| (*tier, gpu.len()))
        .collect();
    assert_eq!(tiers, [(HostPinned, 3)], "host first");
    store.commit_suspend(txn).unwrap();
    store.retire_idle();
    assert_eq!(on_device(&mut store, &tokens), 0);
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(
        store.cached_run_any_tier(&keys).len(),
        3,
        "kept on the host"
    );
}

/// A16: what a decode wrote has no order of its own over a prompt. Among
/// equal retention keys the least recently used goes first, the engine's
/// own tie: the older prompt before the newer answer. (`default` keys a
/// decoded tail below its prompt with a row.)
fn among_equal_keys_the_least_recently_used_goes_first_answer_or_prompt() {
    let mut store = KvStore::new(16, h(36));
    let other: Vec<u32> = (0..32).map(|t| 1700 + t).collect();
    let ws = decoded(&mut store, &other, 32, 32);
    store.set_owner(ws, ProcessId::from_u128(44)).unwrap();
    store.release_working_set(ws, store.current_epoch());
    let tokens: Vec<u32> = (0..48).map(|t| 1500 + t).collect();
    let ws = decoded(&mut store, &tokens, 20, 40);
    store.fold(ws, 40, true).unwrap();
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    assert!(store.reclaim_cache(store.current_epoch(), 1, |_| 0) >= 1);
    assert_eq!(on_device(&mut store, &other), 0, "the older prompt goes first");
    assert_eq!(on_device(&mut store, &tokens), 2, "the newer answer stays");
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

/// Moves idle cache down under `key`, as far as it goes, and brings the run
/// `keys` back up: its pages are on the device, with their copies on disk.
fn demoted_and_back(store: &mut KvStore, keys: &[Hash256], key: i64) {
    let txn = store.prepare_demote(8, |_| key).unwrap().expect("pages to move");
    let pages = txn.page_count();
    assert_eq!(store.commit_suspend(txn).unwrap(), pages);
    store.retire_idle();
    let run = store.cached_run_any_tier(keys);
    assert_eq!(store.below_device(&run), pages);
    promoted(store, &run);
}

fn promoted(store: &mut KvStore, run: &[super::TriePageLocation]) {
    let mut granted = store.reserve_device_pages(store.below_device(run)).unwrap();
    let txn = store
        .prepare_promote(run, &mut granted)
        .unwrap()
        .expect("a promotion");
    let pin = store.commit_promote(txn).unwrap();
    store.release_reader(pin);
}

#[test]
/// A run a hit brought back from disk is on the device with its copies kept;
/// when the device wants their room its pages settle on them with nothing
/// copied, a restart finds them, and swap takes their slots when it must.
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
        .prepare_demote(8, |_| 0)
        .unwrap()
        .expect("three pages to copy");
    assert_eq!(txn.page_count(), 3);
    assert!(
        store.prepare_demote(8, |_| 0).unwrap().is_none(),
        "pages in a copy are copied once"
    );
    assert_eq!(store.commit_suspend(txn).unwrap(), 3);
    store.retire_idle();
    assert_eq!(store.persisted_pages(), 3);
    assert_eq!(store.disk_pool.available(), 5);
    let run = store.cached_run_any_tier(&keys);
    promoted(&mut store, &run);
    assert_eq!(store.persisted_pages(), 3, "a hit keeps the copies");
    assert!(
        store.prepare_demote(8, |_| 0).unwrap().is_none(),
        "copied already"
    );
    let run = store.cached_run_any_tier(&keys);
    assert_eq!(store.below_device(&run), 0, "the pages are on the device");

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

    promoted(&mut store, &run);
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

#[test]
/// A prefix the disk keeps is a run of its own, and its copies go with its
/// pages.
fn a_persisted_prefix_is_a_run() {
    use super::hash::{chain_slots, page_hash};
    let seed = h(53);
    let tokens: Vec<Vec<Option<Hash256>>> =
        chain_slots(&seed, None, 0, &(400..412).collect::<Vec<_>>())
            .chunks(4)
            .map(|page| page.iter().map(|h| Some(*h)).collect())
            .collect();
    let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
    let dir = tempfile::tempdir().unwrap();
    let mut store = KvStore::new_with_swap(8, 0, 2, seed);
    store.set_page_size(4);
    store.set_keeps(true);
    store
        .keep_disk(&dir.path().join("g0.journal"), true)
        .unwrap();
    let gpu = store.pool.try_alloc_n(3).unwrap();
    assert!(store.import_cached_at(
        &[],
        tokens,
        keys.clone(),
        gpu.into_iter().map(KvPageBacking::Resident).collect(),
    ));
    let txn = store
        .prepare_demote(8, |_| 0)
        .unwrap()
        .expect("the disk takes two");
    assert_eq!(store.commit_suspend(txn).unwrap(), 2);
    store.retire_idle();
    assert!(
        store.prepare_demote(8, |_| 0).unwrap().is_none(),
        "the disk is full"
    );
    let mut runs = store.cached_runs();
    runs.sort_by_key(Vec::len);
    assert_eq!(
        runs,
        vec![keys[..2].to_vec(), keys.clone()],
        "the prefix is a run"
    );
    let mut persisted = store.persisted_keys();
    persisted.sort_unstable();
    let mut expected = keys[..2].to_vec();
    expected.sort_unstable();
    assert_eq!(persisted, expected);

    let ids: Vec<u64> = store.cache.keys().copied().collect();
    let epoch = store.current_epoch();
    store.drop_entries(ids, crate::policy::By::Host, epoch);
    assert_eq!(
        store.persisted_pages(),
        0,
        "and the copies with their pages"
    );
}

#[test]
/// Swap takes the copies kept under the lowest retention key first; the key
/// a run was demoted under is kept across a restart.
fn swap_takes_the_lowest_retention_key_first_and_a_restart_keeps_the_keys() {
    use super::hash::{chain_slots, page_hash};
    let seed = h(54);
    let run_of = |first: u32| {
        let tokens: Vec<Vec<Option<Hash256>>> =
            chain_slots(&seed, None, 0, &(first..first + 8).collect::<Vec<_>>())
                .chunks(4)
                .map(|page| page.iter().map(|h| Some(*h)).collect())
                .collect();
        let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
        (tokens, keys)
    };
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("g0.journal");
    let store_for = || {
        let mut store = KvStore::new_with_swap(8, 0, 4, seed);
        store.set_page_size(4);
        store.set_keeps(true);
        store
    };
    let (low, high) = (run_of(500), run_of(600));
    let mut store = store_for();
    store.keep_disk(&path, true).unwrap();
    for ((tokens, keys), key) in [(&low, -1), (&high, 0)] {
        let gpu = store.pool.try_alloc_n(2).unwrap();
        assert!(store.import_cached_at(
            &[],
            tokens.clone(),
            keys.clone(),
            gpu.into_iter().map(KvPageBacking::Resident).collect(),
        ));
        demoted_and_back(&mut store, keys, key);
    }
    assert_eq!(store.disk_pool.available(), 0);
    assert_eq!(store.make_swap_room(2, &floor), 2);
    assert_eq!(store.persisted_pages(), 2);
    let kept: HashSet<Hash256> = store.persisted_keys().into_iter().collect();
    assert!(
        high.1.iter().all(|key| kept.contains(key)),
        "the higher key's copies stay"
    );

    // The low run goes down again, under a higher key.
    demoted_and_back(&mut store, &low.1, 20);
    drop(store);

    let mut store = store_for();
    assert_eq!(store.keep_disk(&path, true).unwrap(), 4);
    let retention = |keys: &[Hash256]| {
        let persisted = store.persisted.as_ref().unwrap();
        keys.iter()
            .map(|key| persisted.retention[&persisted.by_key[key]])
            .collect::<Vec<_>>()
    };
    assert_eq!(retention(&low.1), [20, 20]);
    assert_eq!(retention(&high.1), [0, 0]);
}

/// A peer's eviction moves pages under a parked set, and a peer's restore puts
/// them back elsewhere: the map a parked set wakes with is the table's.
#[test]
fn a_parked_set_wakes_with_the_pages_its_peers_moved() {
    let mut s = KvStore::new_with_swap(8, 8, 0, h(40));
    let a = s.create_shared_working_set();
    commit_fresh(&mut s, a, 3, 1);
    let b = s.fork(a).unwrap();
    s.set_parked(a, true);
    s.set_parked(b, true);

    let txn = s
        .prepare_suspend(&HashSet::from([a]), floor)
        .unwrap()
        .expect("no running reader keeps them");
    assert_eq!(s.commit_suspend(txn).unwrap(), 3);
    s.retire_idle();
    let _elsewhere = s.reserve_device_pages(5).unwrap();
    let mut granted = s.reserve_device_pages(3).unwrap();
    let txn = s
        .prepare_restore(&HashSet::from([a]), &mut granted)
        .unwrap();
    assert_eq!(s.commit_restore(txn).unwrap(), 3);

    assert_eq!(s.swapped_page_count(&HashSet::from([b])).unwrap(), 0);
    s.set_parked(b, false);
    assert_eq!(
        s.flat_table(b).unwrap().to_vec(),
        s.table.flatten(b).unwrap()
    );
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

/// Idle cache dropped while fires are in flight goes back once they retire.
fn idle_cache_comes_back_past_the_fires_in_flight() {
    let mut store = KvStore::new(8, h(41));
    let owner = ProcessId::from_u128(46);
    kept_and_gone(&mut store, 20, 3, owner);
    assert_eq!(store.available_pages(), 5);
    // Fires of others in flight, none reading the cache.
    let busy = store.begin_fire();
    let other = store.begin_fire();
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, |_| 0),
        3,
        "dropped"
    );
    assert_eq!(store.available_pages(), 5, "fires in flight");
    store.settle(busy);
    store.settle(other);
    store.retire_idle();
    assert_eq!(store.available_pages(), 8, "back once they retire");

    // A reader holds it in a fire now in flight, and lets go.
    let tokens = kept_and_gone(&mut store, 21, 3, owner);
    let ws = reader(&mut store, &tokens, 3);
    let read = store.begin_fire();
    store.release_working_set(ws, read);
    store.retire_idle();
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), 1, |_| 0),
        3,
        "dropped"
    );
    store.retire_idle();
    assert_eq!(
        store.available_pages(),
        5,
        "the fire that read it is in flight"
    );
    store.settle(read);
    store.retire_idle();
    assert_eq!(store.available_pages(), 8, "and back once it retires");
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

/// The reports of one process that exits with `prompt` prompt rows and
/// `decoded` more, and one that stays live.
fn an_exit_and_a_live_one(
    exited: ProcessId,
    live: ProcessId,
    prompt: u64,
    decoded: u64,
) -> Vec<crate::policy::Event> {
    use crate::policy::{Arrival, Event, Outcome};
    vec![
        Event::Arrived(exited, Arrival::default()),
        Event::Arrived(live, Arrival::default()),
        Event::Started(exited),
        Event::Started(live),
        Event::Fired(exited, prompt, 1),
        Event::Fired(exited, decoded, decoded),
        Event::Fired(live, prompt, 1),
        Event::Exited(exited, Outcome::Completed),
    ]
}

/// A built-in policy, run over `events` and enacted.
fn run_policy(name: &str, events: Vec<crate::policy::Event>) {
    let policy = builtins::policy(name).expect("a built-in policy");
    crate::policy::port::testing::enact(name, policy.component, events);
}

/// Under the shipped `default` policy the cache an exited process left is
/// kept on the device alone: when the device wants its room it is dropped,
/// not demoted, while a live process's set still suspends to the host; a
/// resting handle's rows are the column's own. Its prompt outranks its
/// decoded tail (A16). Under `noop` the same cache moves down a tier.
#[test]
fn the_default_policy_drops_what_an_exit_left_and_noop_demotes_it() {
    use crate::policy::retention::{DEVICE, Installed, install};
    let _lock = crate::policy::schedule::TEST_LOCK.blocking_write();
    install(None);
    let (exited, live, handle) = (
        ProcessId::from_u128(0xdef1),
        ProcessId::from_u128(0xdef2),
        ProcessId::from_u128(0xdef3),
    );
    let mut events = an_exit_and_a_live_one(exited, live, 32, 16);
    events.push(crate::policy::Event::HandleRested(
        handle,
        crate::policy::Resting {
            tenant: "t".into(),
            session: None,
            props: client_api::Props::default(),
            tokens: 32,
            folded: 0,
        },
    ));
    run_policy("default", events);
    assert_eq!(Installed.tiers_at(Some(exited), 0), DEVICE, "kept on the device alone");
    assert_eq!(Installed.key_at(Some(exited), 31), 0, "the prompt");
    assert_eq!(Installed.key_at(Some(exited), 32), -1, "the decoded tail goes first");
    // The column's own, as for an owner nothing names.
    let own = Installed.tiers_at(Some(ProcessId::from_u128(0xdeff)), 0);
    assert_eq!(Installed.tiers_at(Some(live), 0), own, "a live set: no row");
    assert_eq!(Installed.key_at(Some(live), 0), 0);
    assert_eq!(Installed.tiers_at(Some(handle), 0), own, "a resting handle: no row");
    assert_eq!(Installed.key_at(Some(handle), 0), 0);

    // Device pressure: the exit's cache finds no tier below it and is
    // dropped; the host is left to the live set, which still suspends.
    let mut store = KvStore::new_with_swap(8, 8, 0, h(0xdef));
    let tokens = kept_and_gone(&mut store, 70, 3, exited);
    assert!(
        store.prepare_demote(8, Installed).unwrap().is_none(),
        "nothing moves: its row keeps it off the host"
    );
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, Installed), 3, "dropped");
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert!(store.cached_working_sets().is_empty());
    assert_eq!(store.host_swap_available(), 8, "nothing went to the host");
    let ws = store.create_shared_working_set();
    store.set_owner(ws, live).unwrap();
    commit_fresh(&mut store, ws, 2, 1);
    let txn = store
        .prepare_suspend(&HashSet::from([ws]), Installed)
        .unwrap()
        .expect("a live set suspends");
    assert_eq!(store.commit_suspend(txn).unwrap(), 2);
    assert_eq!(store.host_swap_available(), 6, "on the host");

    // `noop` writes no rows: the columns' own defaults demote the same
    // cache to the host.
    install(None);
    let (exited, live) = (ProcessId::from_u128(0xdef4), ProcessId::from_u128(0xdef5));
    run_policy("noop", an_exit_and_a_live_one(exited, live, 32, 16));
    assert_eq!(Installed.tiers_at(Some(exited), 0), own, "no row: the column's own");
    let mut store = KvStore::new_with_swap(8, 8, 0, h(0xdee));
    let tokens = kept_and_gone(&mut store, 71, 3, exited);
    let txn = store
        .prepare_demote(8, Installed)
        .unwrap()
        .expect("every tier: it moves down");
    store.commit_suspend(txn).unwrap();
    store.retire_idle();
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert_eq!(store.host_swap_available(), 5, "kept on the host");
    install(None);
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
