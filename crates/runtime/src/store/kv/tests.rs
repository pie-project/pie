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
    a_prefix_page_every_reader_left_on_the_device_goes_down_with_them();
    an_entry_left_on_the_host_goes_when_swap_needs_room();
    a_move_under_way_keeps_only_the_pages_it_holds();
    a_page_seals_under_a_key_that_names_its_prefix();
    keys_computed_ahead_find_what_another_sealed();
    keys_from_row_zero_are_hashed_without_the_store();
    a_working_set_that_goes_leaves_its_sealed_prefix();
    a_working_set_maps_in_what_another_sealed();
    a_pass_maps_in_more_after_what_it_mapped_in();
    a_prefix_computed_twice_maps_whole_from_either_copy();
    a_page_no_one_else_reads_is_written_in_place();
    a_lookup_takes_the_copy_on_the_device();
    what_a_decode_folds_seals();
    a_truncated_set_ends_where_it_was_cut();
    a_set_never_ends_inside_a_sealed_page();
    a_fire_that_never_settled_leaves_the_log_cut();
    a_hit_refreshes_only_the_entries_it_used_whole();
    what_decode_wrote_is_reclaimed_before_the_prompt();
    a_sweep_suspends_only_the_readers_of_what_it_moves();
    idle_cache_goes_to_the_host_and_comes_back_on_a_hit();
    a_set_moved_down_says_it_is_off_the_device();
    an_exited_owner_holds_its_kv_until_its_working_set_is_released();
    a_suspend_short_of_host_room_drops_host_cache_first();
    a_tail_past_its_step_goes_before_another_head();
    a_demote_leaves_what_a_higher_key_covers();
    idle_cache_the_tiers_keep_off_the_host_is_dropped_not_demoted();
    on_the_host_the_lower_set_goes_first();
    a_higher_set_moving_down_takes_the_room_of_a_lower_one_on_the_host();
    the_lowest_set_that_finds_no_room_below_goes_before_a_higher_one_moves();
    a_demote_keeps_off_the_disk_what_its_row_keeps_off_it();
    a_keep_the_policy_states_is_honoured_and_unkeyed_idle_cache_waits_for_a_demand();
    idle_cache_last_read_long_ago_comes_back_past_the_fires_in_flight();
    an_answer_goes_before_an_older_prompt();
    pages_another_worker_sealed_are_found_in_the_host_tier();
    a_beams_finish_waits_out_a_suspend_and_names_each_page_once();
    a_beams_finish_copies_the_chosen_pages_another_reads();
    a_cut_waits_out_a_move();
}

/// A beam search's finish while its set is suspending waits for the move
/// to end, and never names a page the move gives back (the M1's beam line:
/// a relist under a suspend left the chosen pages named by the new tail and
/// the suspend's pinned old one, and the suspend's commit gave them back
/// while the set still read them).
fn a_beams_finish_waits_out_a_suspend_and_names_each_page_once() {
    let mut store = KvStore::new_with_swap(16, 16, 0, h(61));
    let total = store.free_committed();
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 4, 1);
    let sets = HashSet::from([ws]);
    let txn = store.prepare_suspend(&sets).unwrap().expect("private pages suspend");
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

/// A beam search's finish whose chosen pages another set still reads (a
/// fork's path through the old tail) names copies of them: the other set
/// reads the pages it read, and every page goes back once.
fn a_beams_finish_copies_the_chosen_pages_another_reads() {
    let mut store = KvStore::new_with_swap(16, 16, 0, h(62));
    let total = store.free_committed();
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 4, 1);
    let other = store.fork(ws).unwrap();
    let read: Vec<PhysicalKvPageId> = (0..4).map(|i| store.lookup(other, i).unwrap()).collect();
    let super::RelistNeeds::Copies(shared) = store.relist_beam_needs(ws, 1, &[2, 0]).unwrap() else {
        panic!("out of any move");
    };
    assert_eq!(shared, [read[3], read[1]], "both chosen pages are the fork's too");
    // The engine copies them onto pages the planner granted.
    let copies = store.reserve_device_pages(shared.len()).unwrap();
    let pairs: Vec<_> = shared.iter().copied().zip(copies.iter().copied()).collect();
    let counted = super::RELIST_COPIES.load(std::sync::atomic::Ordering::Relaxed);
    assert!(store.relist_beam_with(ws, 1, &[2, 0], &pairs).unwrap());
    assert!(super::RELIST_COPIES.load(std::sync::atomic::Ordering::Relaxed) >= counted + 2);
    let after: Vec<PhysicalKvPageId> = (0..3).map(|i| store.lookup(ws, i).unwrap()).collect();
    assert_eq!(after, [read[0], copies[0], copies[1]], "the copies, named once");
    let still: Vec<PhysicalKvPageId> = (0..4).map(|i| store.lookup(other, i).unwrap()).collect();
    assert_eq!(still, read, "the fork reads what it read");
    store.release_working_set(ws, store.current_epoch());
    store.release_working_set(other, store.current_epoch());
    store.drop_entries_off_device();
    store.retire_idle();
    let cached = total - store.free_committed();
    assert!(
        cached <= 4,
        "what stays is cache alone, the copies and the originals each back once: {cached}"
    );
}

/// A cut of a set in a move is refused until the move ends: cut under it,
/// what the cut freed went back again as the move committed.
fn a_cut_waits_out_a_move() {
    let mut store = KvStore::new_with_swap(16, 16, 0, h(63));
    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 4, 1);
    let sets = HashSet::from([ws]);
    let txn = store.prepare_suspend(&sets).unwrap().expect("private pages suspend");
    assert!(matches!(store.truncate(ws, 0), Err(super::KvStoreError::MoveInFlight)));
    assert_eq!(store.mapped_len(ws).unwrap(), 4, "nothing cut while it moves");
    store.abort_suspend(txn);
    store.truncate(ws, 0).unwrap();
    assert_eq!(store.mapped_len(ws).unwrap(), 0);
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

/// The L40S at 3 GiB (disk-only): a prompt-prefix page seven forks share
/// stayed on the device after every one of them was evicted (each
/// suspend left it, another reader or a key holding it then), and nothing
/// moved it: its chunk held a head's need for good. Swept with its
/// readers' sets, it goes down a tier once, every reader's refs pointing
/// there, the chunk comes free, and a restore reads it back.
fn a_prefix_page_every_reader_left_on_the_device_goes_down_with_them() {
    let mut store = KvStore::new_with_swap(12, 0, 12, h(13));
    let owner = ProcessId::from_u128(7);
    let tokens = prompt(13, 1);
    let first = store.create_shared_working_set();
    sealed(&mut store, first, &tokens);
    store.keep_checkpoint(first, 1, Some(owner));
    let mut forks = vec![first];
    for _ in 1..7 {
        forks.push(reader(&mut store, &tokens, 1));
    }
    // Every reader evicted with the page left where it lay (the field's
    // state: each suspend found it held then, and moved nothing of it).
    store.discard_cache(owner, crate::policy::By::Policy);
    store.mark_suspended(&forks.iter().copied().collect());
    store.retire_idle();
    assert_eq!(on_device(&mut store, &tokens), 1, "left on the device, every reader evicted");
    // The move down of its readers' parked state takes it.
    let order: Vec<_> = forks.iter().map(|&ws| (ws, 0)).collect();
    let sweep = store
        .prepare_sweep(&HashSet::new(), usize::MAX, &order, usize::MAX, floor)
        .unwrap()
        .expect("the shared page moves");
    assert_eq!(store.commit_suspend(sweep).unwrap(), 1, "once, for every reader");
    store.retire_idle();
    assert_eq!(on_device(&mut store, &tokens), 0, "its chunk comes free");
    assert_eq!(store.available_pages(), 12);
    for &ws in &forks {
        assert_eq!(store.swapped_page_count(&HashSet::from([ws])).unwrap(), 1, "refs intact");
    }
    let mut granted = store.reserve_device_pages(1).unwrap();
    let txn = store.prepare_restore(&HashSet::from([forks[3]]), &mut granted).unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 1, "read back");
    assert_eq!(on_device(&mut store, &tokens), 1);

    // Evicted one at a time, each finding the page read by the others still
    // running: the last one's suspend takes it down.
    let mut store = KvStore::new_with_swap(12, 0, 12, h(14));
    let tokens = prompt(14, 1);
    let first = store.create_shared_working_set();
    sealed(&mut store, first, &tokens);
    store.keep_checkpoint(first, 1, Some(owner));
    let forks: Vec<_> = std::iter::once(first)
        .chain((1..7).map(|_| reader(&mut store, &tokens, 1)))
        .collect();
    store.discard_cache(owner, crate::policy::By::Policy);
    for &ws in &forks[..6] {
        assert!(
            store
                .prepare_suspend(&HashSet::from([ws]), floor)
                .unwrap()
                .is_none(),
            "read by the rest"
        );
        store.mark_suspended(&HashSet::from([ws]));
    }
    let last = store
        .prepare_suspend(&HashSet::from([forks[6]]), floor)
        .unwrap()
        .expect("the last reader's suspend takes the page");
    assert_eq!(store.commit_suspend(last).unwrap(), 1);
    store.retire_idle();
    assert_eq!(on_device(&mut store, &tokens), 0, "never left on the device");
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
    let reader = store.create_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(
        store.cached_run(&keys).len(),
        2,
        "the decoded page too, not the partial one"
    );
}

/// A set cut at row 37 after a decode ran on to 52: the pages wholly past
/// it go back to the pool, its folded length and log end there, a fork
/// takes nothing past it, and what it writes next is sealed under the
/// tokens it holds, never the ones taken back.
fn a_truncated_set_ends_where_it_was_cut() {
    let mut store = KvStore::new(16, h(29));
    let accepted: Vec<u32> = (0..48).map(|t| 1300 + t).collect();
    let mut ran: Vec<u32> = accepted[..37].to_vec();
    ran.extend((37..52).map(|t| 7000 + t));
    let ws = decoded(&mut store, &ran, 20, 52);
    store.fold(ws, 52, true).unwrap();
    assert_eq!(store.mapped_len(ws).unwrap(), 4);
    let free = store.free_committed();
    store.truncate(ws, 37).unwrap();
    store.retire_idle();
    assert_eq!(store.mapped_len(ws).unwrap(), 3, "page 3 (48..52) leaves");
    assert_eq!(
        store.free_committed(),
        free + 1,
        "and goes back to the pool"
    );
    assert_eq!(store.folded(ws).unwrap(), 37);
    assert_eq!(store.logged(ws).unwrap(), &ran[..37]);
    let child = store.fork(ws).unwrap();
    assert_eq!(
        store.folded(child).unwrap(),
        37,
        "a fork takes no row past it"
    );
    assert_eq!(store.mapped_len(child).unwrap(), 3);
    store.release_working_set(child, store.current_epoch());
    // It goes on from 37 with the tokens the program took.
    let seq = store.log_begin(ws, 37).unwrap();
    store
        .log_end(ws, seq, Some((37, &accepted[37..48])))
        .unwrap();
    assert_eq!(
        store.fold(ws, 48, false).unwrap(),
        1,
        "page 2 seals: the fork sealed page 1, whole below the cut, as it went"
    );
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    let reader = store.create_shared_working_set();
    let keys = keys_ahead(&store, reader, 0, &accepted);
    assert_eq!(
        store.cached_run(&keys).len(),
        3,
        "sealed under what it holds"
    );
    let keys = keys_ahead(&store, reader, 0, &ran[..48]);
    assert_eq!(
        store.cached_run(&keys).len(),
        2,
        "nothing under the rows taken back"
    );
}

/// A cut inside a page a fold sealed is refused by name: the page is frozen
/// whole under the key of the rows past the cut.
fn a_set_never_ends_inside_a_sealed_page() {
    let mut store = KvStore::new(16, h(30));
    let tokens: Vec<u32> = (0..48).map(|t| 1400 + t).collect();
    let ws = decoded(&mut store, &tokens, 32, 32);
    assert_eq!(store.sealed_in(ws, 0..2).unwrap(), [0, 1]);
    let refused = store.truncate(ws, 20).unwrap_err().to_string();
    assert!(refused.contains("sealed page"), "{refused}");
    assert_eq!(store.folded(ws).unwrap(), 32, "nothing moved");
    store.truncate(ws, 16).unwrap();
    assert_eq!(store.folded(ws).unwrap(), 16, "a page boundary is a cut");
    assert_eq!(store.mapped_len(ws).unwrap(), 1);
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

fn what_decode_wrote_is_reclaimed_before_the_prompt() {
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
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, |_| 0), 1);
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

/// What a set holds where: device pressure has nothing to gain from
/// discarding a set wholly below the device.
fn a_set_moved_down_says_it_is_off_the_device() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(33));
    let owner = ProcessId::from_u128(23);
    kept_and_gone(&mut store, 14, 3, owner);
    let set = |store: &KvStore| store.cached_working_sets()[0];
    assert!(set(&store).on_device && !set(&store).on_host);
    let txn = store.prepare_demote(8, |_| 1).unwrap().expect("three cache-only pages");
    store.commit_suspend(txn).unwrap();
    store.retire_idle();
    assert!(!set(&store).on_device && set(&store).on_host);
}

/// A process's last fire settles after it exits, and only then is what it
/// leaves cached: until its working set is released the store still holds
/// its KV, so its rows are not reaped as a gone set's (a keep the policy
/// stated would be lost, and the cache it becomes keyed by default).
fn an_exited_owner_holds_its_kv_until_its_working_set_is_released() {
    let mut store = KvStore::new(8, h(35));
    store.set_keeps(true);
    let (owner, other) = (ProcessId::from_u128(25), ProcessId::from_u128(26));
    let tokens: Vec<u32> = (0..32).map(|t| 500 + t).collect();
    let ws = store.create_shared_working_set();
    store.set_owner(ws, owner).unwrap();
    commit_opaque(&mut store, ws, 2);
    seal(&mut store, ws, 0, &tokens);
    assert!(!store.owns_cache(owner), "nothing cached yet");
    assert!(store.holds_kv_of(owner), "its working set is still here");
    assert!(!store.holds_kv_of(other));
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    assert!(store.owns_cache(owner) && store.holds_kv_of(owner), "now cached");
    assert_eq!(store.discard_cache(owner, crate::policy::By::Policy), 2, "both pages");
    assert!(!store.holds_kv_of(owner), "gone");
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

/// Conform's "the higher working set survives": A kept at 20 and B at 10,
/// both demoted to a full host; a suspend short of host room drops B, the
/// lower, and keeps A there: the host holds the highest keys first, as the
/// device does, and drops only as much as the suspend needs.
fn on_the_host_the_lower_set_goes_first() {
    let mut store = KvStore::new_with_swap(16, 8, 0, h(43));
    store.set_page_size(16);
    let (a, b) = (ProcessId::from_u128(53), ProcessId::from_u128(54));
    let (a_tokens, b_tokens) = (prompt(25, 4), prompt(26, 4));
    sealed_and_gone(&mut store, &a_tokens, a);
    sealed_and_gone(&mut store, &b_tokens, b);
    let keys = move |owner: Option<ProcessId>| if owner == Some(a) { 20 } else { 10 };
    for _ in 0..2 {
        let txn = store
            .prepare_demote(16, keys)
            .unwrap()
            .expect("a set to the host");
        store.commit_suspend(txn).unwrap();
    }
    assert_eq!(store.host_swap_available(), 0, "both on the host");
    let running = store.create_working_set();
    commit_fresh(&mut store, running, 4, 1);
    let txn = store
        .prepare_suspend(&HashSet::from([running]), keys)
        .unwrap()
        .expect("four pages to park");
    store.commit_suspend(txn).unwrap();
    let below = |store: &mut KvStore, tokens: &[u32]| {
        let probe = store.create_shared_working_set();
        let found = keys_ahead(store, probe, 0, tokens);
        store.release_working_set(probe, store.current_epoch());
        store.cached_run_any_tier(&found).len()
    };
    assert_eq!(below(&mut store, &b_tokens), 0, "the lower one is gone");
    assert_eq!(
        below(&mut store, &a_tokens),
        4,
        "the higher one stays on the host"
    );
}

/// Each tier holds the highest keys first: B (10) fills the host, and A
/// (20) moving down off the device takes B's room there; keyed alike,
/// nothing on the host is displaced and A finds no room.
fn a_higher_set_moving_down_takes_the_room_of_a_lower_one_on_the_host() {
    for (a_key, displaced) in [(20, true), (10, false)] {
        let mut store = KvStore::new_with_swap(16, 4, 0, h(47));
        store.set_page_size(16);
        let (a, b) = (ProcessId::from_u128(55), ProcessId::from_u128(56));
        let (a_tokens, b_tokens) = (prompt(27, 4), prompt(28, 4));
        sealed_and_gone(&mut store, &b_tokens, b);
        let keys = move |owner: Option<ProcessId>| if owner == Some(a) { a_key } else { 10 };
        let txn = store.prepare_demote(16, keys).unwrap().expect("B to the host");
        store.commit_suspend(txn).unwrap();
        assert_eq!(store.host_swap_available(), 0, "the host is full");
        sealed_and_gone(&mut store, &a_tokens, a);
        let below = |store: &mut KvStore, tokens: &[u32]| {
            let probe = store.create_shared_working_set();
            let found = keys_ahead(store, probe, 0, tokens);
            store.release_working_set(probe, store.current_epoch());
            store.cached_run_any_tier(&found).len()
        };
        match store.prepare_demote(16, keys).unwrap() {
            Some(txn) => {
                assert!(displaced, "keyed alike, nothing is displaced");
                store.commit_suspend(txn).unwrap();
                store.retire_idle();
                assert_eq!(on_device(&mut store, &a_tokens), 0, "A moved down");
                assert_eq!(below(&mut store, &a_tokens), 4, "A on the host");
                assert_eq!(below(&mut store, &b_tokens), 0, "B gave its room");
            }
            None => {
                assert!(!displaced, "A keyed higher takes B's room");
                assert_eq!(below(&mut store, &b_tokens), 4, "B stays");
            }
        }
    }
}

/// The conform shape after earlier turns: the host full of fillers (15),
/// B (10) idle on the device, A (20) too. B finds no room below (it may
/// displace nothing keyed higher), so no demote moves A or anything else
/// ahead of it: B is dropped first, by the reclaim that follows.
fn the_lowest_set_that_finds_no_room_below_goes_before_a_higher_one_moves() {
    let mut store = KvStore::new_with_swap(16, 4, 0, h(49));
    store.set_page_size(16);
    let (a, b, f) = (ProcessId::from_u128(57), ProcessId::from_u128(58), ProcessId::from_u128(59));
    let (a_tokens, b_tokens, f_tokens) = (prompt(29, 4), prompt(30, 4), prompt(31, 4));
    let keys = move |owner: Option<ProcessId>| match owner {
        Some(o) if o == a => 20,
        Some(o) if o == b => 10,
        _ => 15,
    };
    sealed_and_gone(&mut store, &f_tokens, f);
    let txn = store.prepare_demote(16, keys).unwrap().expect("the filler to the host");
    store.commit_suspend(txn).unwrap();
    assert_eq!(store.host_swap_available(), 0, "the host is full");
    sealed_and_gone(&mut store, &b_tokens, b);
    sealed_and_gone(&mut store, &a_tokens, a);
    assert!(
        store.prepare_demote(16, keys).unwrap().is_none(),
        "B, the lowest, finds no room: nothing above it moves"
    );
    assert_eq!(store.reclaim_droppable(store.current_epoch(), keys), 4, "B goes");
    assert_eq!(on_device(&mut store, &b_tokens), 0);
    assert_eq!(on_device(&mut store, &a_tokens), 4, "A stays");
}

/// A row naming the device and the host keeps idle cache off the disk: a
/// demote takes the host's room and no more, however much room the disk
/// has (conform's keep: fillers spilled to disk, so the host never filled
/// and the lower set on it was never displaced).
fn a_demote_keeps_off_the_disk_what_its_row_keeps_off_it() {
    use crate::policy::retention::{DEVICE, HOST};
    let mut store = KvStore::new_with_swap(16, 4, 8, h(53));
    store.set_page_size(16);
    let owner = ProcessId::from_u128(61);
    let tokens = prompt(33, 8);
    sealed_and_gone(&mut store, &tokens, owner);
    let txn = store
        .prepare_demote(16, Rows(0, DEVICE | HOST))
        .unwrap()
        .expect("the host's room");
    assert_eq!(txn.page_count(), 4, "the host's four slots, none of the disk's");
    store.commit_suspend(txn).unwrap();
    assert_eq!(store.host_swap_available(), 0);
    assert_eq!(store.disk_swap_available(), 8, "the disk untouched");
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
        store.reclaim_droppable(store.current_epoch(), 1, Rows(100, DEVICE)),
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

fn an_answer_goes_before_an_older_prompt() {
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
    assert_eq!(store.reclaim_cache(store.current_epoch(), 1, |_| 0), 1);
    assert_eq!(on_device(&mut store, &other), 2, "the older prompt stays");
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

#[test]
/// An owner's ask to keep a prefix on disk goes out with the runs it left,
/// a prefix persisted is a run of its own, and the ask goes with the last
/// of its owner's entries.
fn a_persist_hint_rides_its_owner_s_runs_and_a_persisted_prefix_is_a_run() {
    use super::hash::{chain_slots, page_hash};
    let seed = h(53);
    let tokens: Vec<Vec<Option<Hash256>>> =
        chain_slots(&seed, None, 0, &(400..412).collect::<Vec<_>>())
            .chunks(4)
            .map(|page| page.iter().map(|h| Some(*h)).collect())
            .collect();
    let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
    let dir = tempfile::tempdir().unwrap();
    let mut store = KvStore::new_with_swap(8, 0, 8, seed);
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
    let owner = ProcessId::from_u128(7);
    for entry in store.cache.values_mut() {
        entry.owner = Some(owner);
    }
    store.owners.insert(owner, store.cache.len());

    assert!(store.persist_hints().is_empty());
    store.set_persist_hint(owner, Some(8));
    assert_eq!(store.persist_hints(), vec![(keys[2], 8)]);

    let txn = store
        .prepare_persist(&keys[..2], 0)
        .expect("two pages to copy");
    assert_eq!(store.commit_persist(txn), 2);
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
    assert!(
        store.hints.is_empty(),
        "the ask goes with its owner's last entry"
    );
    assert_eq!(
        store.persisted_pages(),
        0,
        "and the copies with their pages"
    );
}

#[test]
/// Swap takes the copies kept under the lowest cluster retention key first;
/// a run asked again under a higher key keeps it, across a restart too.
fn swap_takes_the_lowest_retention_key_first_and_a_raised_key_is_kept() {
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
        let txn = store.prepare_persist(keys, key).expect("two pages");
        assert_eq!(store.commit_persist(txn), 2);
    }
    assert_eq!(store.disk_pool.available(), 0);
    assert_eq!(store.make_swap_room(2, &floor), 2);
    assert_eq!(store.persisted_pages(), 2);
    let kept: HashSet<Hash256> = store.persisted_keys().into_iter().collect();
    assert!(
        high.1.iter().all(|key| kept.contains(key)),
        "the higher key's copies stay"
    );

    // Asked again under a higher key, the low run keeps it: persisted anew
    // where its copies went, and raised where they stayed.
    let txn = store.prepare_persist(&low.1, 20).expect("its copies went");
    assert_eq!(store.commit_persist(txn), 2);
    assert!(
        store.prepare_persist(&high.1, 30).is_none(),
        "copied already"
    );
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
    assert_eq!(retention(&high.1), [30, 30], "raised in the journal");
}

#[test]
/// A process that maps in a run another left, and asks a prefix of it kept
/// on disk, has that ask go out once it is gone.
fn a_persist_hint_of_a_process_that_mapped_in_a_cached_run_goes_out() {
    let mut store = KvStore::new(16, h(54));
    let tokens: Vec<u32> = (0..64).map(|t| 600 + t).collect();
    sealed_and_gone(&mut store, &tokens, ProcessId::from_u128(1));

    let asker = ProcessId::from_u128(2);
    let ws = store.create_shared_working_set();
    store.set_owner(ws, asker).unwrap();
    store.set_persist_hint(asker, Some(32));
    store.reserve(ws, 4).unwrap();
    let keys = keys_ahead(&store, ws, 0, &tokens);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 4).unwrap(), 4);
    store.release_working_set(ws, store.current_epoch());
    assert_eq!(store.persist_hints(), vec![(keys[3], 32)]);
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
    assert_eq!(store.reclaim_droppable(store.current_epoch(), 1, keep), 3);
    assert_eq!(on_device(&mut store, &unkept_tokens), 0, "the lower key goes first");
    assert_eq!(on_device(&mut store, &kept_tokens), 3, "the higher stays");
    // No lower tier to move to: cached KV yields to the next demand,
    // whatever its key.
    assert_eq!(store.reclaim_droppable(store.current_epoch(), 1, keep), 3);
    assert_eq!(on_device(&mut store, &kept_tokens), 0);
}

/// Idle cache dropped while fires are in flight goes back once the fires
/// that could read it have retired, not after every fire handed out since:
/// what a set last read before the fires in flight began is free at once,
/// and what a reader read in one of them waits for that one.
fn idle_cache_last_read_long_ago_comes_back_past_the_fires_in_flight() {
    let mut store = KvStore::new(8, h(41));
    let owner = ProcessId::from_u128(46);
    kept_and_gone(&mut store, 20, 3, owner);
    assert_eq!(store.available_pages(), 5);
    // Fires of others in flight, none reading the cache.
    let busy = store.begin_fire();
    let other = store.begin_fire();
    assert_eq!(
        store.reclaim_droppable(store.current_epoch(), 1, |_| 0),
        3,
        "dropped"
    );
    assert_eq!(
        store.available_pages(),
        8,
        "back at once: no fire in flight read it"
    );
    store.settle(busy);
    store.settle(other);

    // A reader holds it in a fire now in flight, and lets go.
    let tokens = kept_and_gone(&mut store, 21, 3, owner);
    let ws = reader(&mut store, &tokens, 3);
    let read = store.begin_fire();
    store.release_working_set(ws, read);
    store.retire_idle();
    assert_eq!(
        store.reclaim_droppable(store.current_epoch(), 1, |_| 0),
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

/// Passes that raced each computed a prompt's first page; a later prompt
/// computed its second page on one of those copies. Whichever copy of the
/// first page the lookup takes, the prompt maps whole: the path to its
/// second page holds the same keys.
fn a_prefix_computed_twice_maps_whole_from_either_copy() {
    let mut store = KvStore::new(64, h(27));
    store.set_keeps(true);
    let prompt = |tail: u32| -> Vec<u32> {
        (0..16)
            .map(|t| 2000 + t)
            .chain((0..17).map(|t| tail * 100 + t))
            .collect()
    };
    let computed = |store: &mut KvStore, tail: u32| {
        let ws = store.create_shared_working_set();
        commit_opaque(store, ws, 3);
        seal(store, ws, 0, &prompt(tail));
    };
    let look = |store: &mut KvStore, tail: u32| {
        let ws = store.create_shared_working_set();
        let keys = keys_ahead(store, ws, 0, &prompt(tail));
        let run = store.cached_run(&keys);
        let mapped = store.attach_run(ws, &run, 2).unwrap();
        (ws, run.len(), mapped)
    };
    computed(&mut store, 1);
    computed(&mut store, 2);
    let (ws, found, mapped) = look(&mut store, 3);
    assert_eq!((found, mapped), (1, 1), "the shared page, from one copy");
    commit_opaque(&mut store, ws, 2);
    seal(&mut store, ws, 16, &prompt(3)[16..]);
    // Another pass computes the shared page again: the newest copy is the
    // one a lookup of the first page takes.
    computed(&mut store, 4);
    let (ws, found, mapped) = look(&mut store, 3);
    assert_eq!((found, mapped), (2, 2), "the whole prompt, on its path");
    let theirs = store.flat_table(ws).unwrap().to_vec();
    let (again, _, _) = look(&mut store, 3);
    assert_eq!(store.flat_table(again).unwrap(), theirs.as_slice());
}

/// A working set released with no fire in flight frees its pages at
/// once: no fire can read them, so none waits for a next fire's settle
/// (with nothing firing, they held their chunks for the kv shape while a
/// head short of whole chunks waited on them).
#[test]
fn a_release_with_nothing_in_flight_frees_its_pages_at_once() {
    let mut store = KvStore::new(8, h(11));
    let ws = store.create_shared_working_set();
    commit_opaque(&mut store, ws, 3);
    let before = store.available_pages();
    store.release_working_set(ws, store.current_epoch());
    assert_eq!(store.available_pages(), before + 3, "free now, no retire asked");
}

/// Keys that name their revision: what a reclaim reads under them it may
/// keep until they, the table or the entries move.
#[derive(Clone, Copy)]
struct Revised(u64);

impl Keys for Revised {
    fn key_at(&self, _: Option<ProcessId>, _: u64) -> i64 {
        crate::policy::retention::DEFAULT_KEY
    }

    fn revision(&self) -> Option<u64> {
        Some(self.0)
    }
}

#[test]
/// A reclaim frees what is short and no more: a store of 256 pages freed a
/// 64th of itself, times up to eight for a slow walk, whatever was short.
/// The order it walked stands for the next while nothing it reads moves,
/// and is walked again once something does.
fn a_reclaim_frees_what_is_short_and_keeps_its_order_until_it_moves() {
    let mut store = KvStore::new(256, h(60));
    let owners: Vec<ProcessId> = (1..=6).map(ProcessId::from_u128).collect();
    for (seed, &owner) in (60..).zip(&owners) {
        kept_and_gone(&mut store, seed, 1, owner);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let keys = Revised(1);
    let free = store.available_pages();
    let epoch = store.current_epoch();
    assert_eq!(store.reclaim_cache(epoch, 0, keys), 0, "nothing short, nothing goes");
    assert_eq!(store.reclaim_cache(epoch, 1, keys), 1);
    assert_eq!(store.available_pages(), free + 1);
    assert!(!store.owns_cache(owners[0]), "the least recently used goes");
    assert!(owners[1..].iter().all(|&owner| store.owns_cache(owner)));
    let kept = store.order.as_ref().expect("the walk is kept").groups.clone();
    assert_eq!(kept, store.reclaim_order(&keys), "as a walk now finds it");

    assert_eq!(store.reclaim_cache(epoch, 3, keys), 3);
    assert_eq!(store.available_pages(), free + 4);
    assert!(owners[..4].iter().all(|&owner| !store.owns_cache(owner)));
    assert!(owners[4..].iter().all(|&owner| store.owns_cache(owner)));
    let kept = store.order.as_ref().expect("the walk is kept").groups.clone();
    assert_eq!(kept.len(), 2);
    assert_eq!(kept, store.reclaim_order(&keys));

    // An entry joining moves the order: the walk takes it in.
    let late = ProcessId::from_u128(7);
    kept_and_gone(&mut store, 70, 1, late);
    assert_eq!(store.reclaim_cache(epoch, 2, keys), 2);
    assert!(store.owns_cache(late), "the most recently used goes last");
    assert_eq!(store.reclaim_cache(epoch, 1, keys), 1);
    assert!(store.cached_working_sets().is_empty());

    // Keys of no revision keep nothing.
    kept_and_gone(&mut store, 71, 1, late);
    assert_eq!(store.reclaim_cache(epoch, 1, floor), 1);
    assert!(store.order.is_none());
}

#[test]
/// A suspend short of host room drops idle cache below the device lowest
/// key first, then least recently used, only until its room is covered;
/// cache a key keeps above the floor stays, and the suspend is refused for
/// want of room rather than have it dropped.
fn a_suspend_drops_cache_off_the_device_until_covered_and_never_what_a_key_keeps() {
    let mut store = KvStore::new_with_swap(8, 4, 0, h(61));
    let (kept, old, new) = (
        ProcessId::from_u128(1),
        ProcessId::from_u128(2),
        ProcessId::from_u128(3),
    );
    let keys = move |owner: Option<ProcessId>| if owner == Some(kept) { 5 } else { 0 };
    for (seed, owner) in [(80, kept), (81, old), (82, new)] {
        let ws = store.create_working_set();
        sealed(&mut store, ws, &prompt(seed, 1));
        store.keep_checkpoint(ws, 1, Some(owner));
        let Some(txn) = store.prepare_suspend(&HashSet::from([ws]), keys).unwrap() else {
            panic!("a page to move");
        };
        assert_eq!(store.commit_suspend(txn).unwrap(), 1);
        store.release_working_set(ws, store.current_epoch());
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    assert_eq!(store.host_swap_available(), 1);

    // Two pages to park, one slot free: one entry goes, the older at the
    // lower key, and nothing past it.
    let v = store.create_working_set();
    commit_fresh(&mut store, v, 2, 1);
    let Some(txn) = store.prepare_suspend(&HashSet::from([v]), keys).unwrap() else {
        panic!("its pages to move");
    };
    assert_eq!(store.commit_suspend(txn).unwrap(), 2);
    assert!(!store.owns_cache(old), "the oldest the keys let go");
    assert!(store.owns_cache(new), "the room was covered");
    assert!(store.owns_cache(kept));
    assert_eq!(store.droppable_below_device(keys), (1, 0), "the kept page is no room");

    // Two more, none free: the last droppable entry goes, and the kept one
    // stays with the room it holds.
    let w = store.create_working_set();
    commit_fresh(&mut store, w, 2, 1);
    match store.prepare_suspend(&HashSet::from([w]), keys) {
        Err(super::KvStoreError::HostSwapFull {
            requested: 2,
            available: 1,
        }) => {}
        other => panic!("refused for want of room: {other:?}"),
    }
    assert!(!store.owns_cache(new));
    assert!(store.owns_cache(kept), "a key above the floor keeps it");
    assert_eq!(store.droppable_below_device(keys), (0, 0));
}

#[test]
/// A suspend gives up the disk tier's copies only where their keys let
/// them go: a copy kept above the floor is no room, and the suspend is
/// refused for want of it rather than take it.
fn a_suspend_takes_no_disk_copy_a_key_keeps() {
    use super::hash::{chain_slots, page_hash};
    let seed = h(62);
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
    let mut store = KvStore::new_with_swap(8, 0, 4, seed);
    store.set_page_size(4);
    store.set_keeps(true);
    store.keep_disk(&dir.path().join("g0.journal"), true).unwrap();
    let (low, high) = (run_of(700), run_of(800));
    for ((tokens, keys), key) in [(&low, 0), (&high, 9)] {
        let gpu = store.pool.try_alloc_n(2).unwrap();
        assert!(store.import_cached_at(
            &[],
            tokens.clone(),
            keys.clone(),
            gpu.into_iter().map(KvPageBacking::Resident).collect(),
        ));
        let txn = store.prepare_persist(keys, key).expect("two pages");
        assert_eq!(store.commit_persist(txn), 2);
    }
    assert_eq!(store.disk_pool.available(), 0);
    assert_eq!(store.disk_swap_available(floor), 2, "the kept copies are no room");

    let ws = store.create_working_set();
    commit_fresh(&mut store, ws, 3, 1);
    match store.prepare_suspend(&HashSet::from([ws]), floor) {
        Err(super::KvStoreError::HostSwapFull {
            requested: 3,
            available: 2,
        }) => {}
        other => panic!("refused for want of room: {other:?}"),
    }
    let kept: HashSet<Hash256> = store.persisted_keys().into_iter().collect();
    assert!(high.1.iter().all(|key| kept.contains(key)), "the kept copies stay");
    assert!(low.1.iter().all(|key| !kept.contains(key)), "the floor's went");
    assert_eq!(store.make_swap_room(4, &floor), 2, "and nothing more yields");
}

#[test]
/// What a reclaim frees while a fire in flight still reads it waits on that
/// fire's retirement: it is returning, and counts toward the shortfall, so
/// a reclaim for the same shortfall frees nothing more. Past it, only what
/// is still short goes; the fire's settle brings it back.
fn a_reclaim_counts_what_it_freed_as_returning_until_its_fire_retires() {
    let mut store = KvStore::new(8, h(63));
    let (early, late) = (ProcessId::from_u128(1), ProcessId::from_u128(2));
    let a = store.create_working_set();
    sealed(&mut store, a, &prompt(90, 2));
    store.keep_checkpoint(a, 2, Some(early));
    std::thread::sleep(std::time::Duration::from_millis(2));
    kept_and_gone(&mut store, 91, 2, late);
    let fire = store.begin_fire();
    store.release_working_set(a, store.current_epoch());
    let free = store.available_pages();
    let epoch = store.current_epoch();

    assert_eq!(store.reclaim_cache(epoch, 2, floor), 2);
    assert!(!store.owns_cache(early), "the least recently used went");
    assert_eq!(store.available_pages(), free, "the fire in flight still reads it");
    assert_eq!(store.reclaim_returning(), 2);
    assert_eq!(
        store.reclaim_cache(epoch, 2, floor),
        0,
        "the same shortfall again: what returns covers it"
    );
    assert!(store.owns_cache(late), "nothing more went");

    store.settle(fire);
    assert_eq!(store.reclaim_returning(), 0);
    assert_eq!(store.available_pages(), free + 2, "back once the fire retired");
    assert_eq!(store.reclaim_cache(store.current_epoch(), 2, floor), 2);
    assert!(!store.owns_cache(late), "a new shortfall reclaims again");
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

/// The directory hears of a shared set's sealed pages, never a private
/// set's: their keys are unmatchable, and their count would tell of it.
#[test]
fn only_a_shared_sets_sealed_keys_are_published() {
    let mut store = KvStore::new(16, h(43));
    let tokens: Vec<u32> = (0..32).map(|t| 4100 + t).collect();
    let shared = store.create_shared_working_set();
    commit_opaque(&mut store, shared, 2);
    assert_eq!(seal(&mut store, shared, 0, &tokens), 2);
    let private = store.create_working_set();
    commit_opaque(&mut store, private, 2);
    assert_eq!(seal(&mut store, private, 0, &tokens), 2, "it seals as any set does");
    let published: HashSet<Hash256> = store.sealed_key_list(&store.node_ids()).into_iter().collect();
    let keys = |store: &KvStore, ws| keys_ahead(store, ws, 0, &tokens);
    let shared_keys = keys(&store, shared);
    let private_keys = keys(&store, private);
    assert!(shared_keys.iter().all(|key| published.contains(key)));
    assert!(private_keys.iter().all(|key| !published.contains(key)));
    assert_eq!(published.len(), 2, "the shared set's two pages, nothing more");
}
