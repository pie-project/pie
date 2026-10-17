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
    a_working_set_that_goes_leaves_its_sealed_prefix();
    a_working_set_maps_in_what_another_sealed();
    a_pass_maps_in_more_after_what_it_mapped_in();
    a_page_no_one_else_reads_is_written_in_place();
    a_lookup_takes_the_copy_on_the_device();
    what_a_decode_folds_seals();
    a_fire_that_never_settled_leaves_the_log_cut();
    a_hit_refreshes_only_the_entries_it_used_whole();
    what_decode_wrote_is_reclaimed_before_the_prompt();
    a_sweep_suspends_only_the_readers_of_what_it_moves();
    idle_cache_goes_to_the_host_and_comes_back_on_a_hit();
    a_suspend_short_of_host_room_drops_host_cache_first();
    a_tail_past_its_step_goes_before_another_head();
    a_demote_leaves_what_a_higher_key_covers();
    an_answer_goes_before_an_older_prompt();
}

/// Retention entries as `(owner, upto, key)`; a position no entry covers is 0.
#[derive(Clone, Default)]
struct Steps {
    entries: Vec<(ProcessId, Option<u64>, i64)>,
    floor: Option<i64>,
}

impl Steps {
    fn floor(floor: i64) -> Self {
        Self {
            floor: Some(floor),
            ..Self::default()
        }
    }
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

    fn floor(&self) -> Option<i64> {
        self.floor
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
    let ws = store.create_working_set();
    sealed(store, ws, &tokens);
    store.keep_checkpoint(ws, pages, Some(owner));
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    tokens
}

fn reader(store: &mut KvStore, tokens: &[u32], pages: u64) -> WorkingSetId {
    let ws = store.create_working_set();
    let keys = keys_ahead(store, ws, 0, tokens);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, pages, 16).unwrap(), pages);
    ws
}

fn on_device(store: &mut KvStore, tokens: &[u32]) -> usize {
    let probe = store.create_working_set();
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
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 2);
    assert!(!store.owns_cache(old) && store.owns_cache(new));
    assert_eq!(on_device(&mut store, &old_tokens), 0);

    // A live working set reads `new`, so dropping it would free nothing:
    // the higher-ranked one goes instead.
    let reader = reader(&mut store, &new_tokens, 2);
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 2);
    assert!(store.owns_cache(new) && !store.owns_cache(kept));
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 0);

    // Once nothing reads it, it can go too.
    store.release_working_set(reader, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 2);
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
    let ws = store.create_working_set();
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
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 2);
    assert!(store.owns_cache(running) && !store.owns_cache(idle));
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 0);
    assert_eq!(entries(&store, running), 4);

    // Once it ends they are the cache's alone, and go.
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.available_pages(), 4);
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 4);
    assert!(store.cached_working_sets().is_empty());

    // A reader of a shorter prefix spares only what it reads: the longer
    // entries go, and the pages only they held with them.
    let tokens = prompt(7, 4);
    let ws = store.create_working_set();
    sealed(&mut store, ws, &tokens);
    for k in 1..=4 {
        store.keep_checkpoint(ws, k, Some(idle));
    }
    store.release_working_set(ws, store.current_epoch());
    let reader = reader(&mut store, &tokens, 2);
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 2);
    assert_eq!(entries(&store, idle), 2);
    assert_eq!(on_device(&mut store, &tokens), 2);
    store.release_working_set(reader, store.current_epoch());
    assert_eq!(store.reclaim_cache(store.current_epoch(), rank), 2);
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
    let ws = store.create_working_set();
    let before = commit_fresh(&mut store, ws, 3, 1);
    let sets = HashSet::from([ws]);
    let Some(txn) = store.prepare_suspend(&sets).unwrap() else {
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
        (store.host_swap_available(), store.disk_swap_available()),
        (0, 0)
    );
    store.retire_idle();

    let mut granted = store.reserve_device_pages(3).unwrap();
    let txn = store.prepare_restore(&sets, &mut granted).unwrap();
    assert_eq!(store.commit_restore(txn).unwrap(), 3);
    assert_eq!(
        (store.host_swap_available(), store.disk_swap_available()),
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
    let a = store.create_working_set();
    sealed(&mut store, a, &tokens);
    store.keep_checkpoint(a, 3, Some(ProcessId::from_u128(1)));
    commit_fresh(&mut store, a, 1, 1);
    let b = reader(&mut store, &tokens, 3);
    commit_fresh(&mut store, b, 1, 1);
    let c = reader(&mut store, &tokens, 3);
    commit_fresh(&mut store, c, 1, 1);
    assert_eq!(store.available_pages(), 2);
    let suspend = |store: &mut KvStore, ws| {
        let Some(txn) = store.prepare_suspend(&HashSet::from([ws])).unwrap() else {
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
            .prepare_sweep(&HashSet::from([c]), usize::MAX)
            .unwrap()
            .is_none()
    );
    let sweep = store
        .prepare_sweep(&HashSet::new(), usize::MAX)
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
    let w = store.create_working_set();
    sealed(&mut store, w, &tokens);
    store.keep_checkpoint(w, 3, Some(ProcessId::from_u128(1)));
    let suspend = |store: &mut KvStore, ws| match store.prepare_suspend(&HashSet::from([ws])) {
        Ok(Some(txn)) => store.commit_suspend(txn).unwrap(),
        other => panic!("a suspend with room: {other:?}"),
    };
    assert_eq!(suspend(&mut store, w), 3);
    store.release_working_set(w, store.current_epoch());
    assert_eq!(store.host_swap_available(), 0);

    let v = store.create_working_set();
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
    let Some(txn) = store.prepare_suspend(&HashSet::from([a])).unwrap() else {
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
    let Some(txn) = store.prepare_suspend(&HashSet::from([b])).unwrap() else {
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

fn a_sweep_suspends_only_the_readers_of_what_it_moves() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(14));
    let a = store.create_working_set();
    commit_fresh(&mut store, a, 2, 1);
    let b = store.create_working_set();
    commit_fresh(&mut store, b, 2, 1);
    // Suspended with nothing moved, as a kv-less eviction leaves them.
    store.mark_suspended(&HashSet::from([a, b]));
    let sweep = store
        .prepare_sweep(&HashSet::new(), 2)
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
            .prepare_sweep(&HashSet::new(), usize::MAX)
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
        let ws = store.create_working_set();
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
    let sealer = store.create_working_set();
    commit_opaque(&mut store, sealer, 3);
    seal(&mut store, sealer, 0, &tokens[..20]);
    seal(&mut store, sealer, 20, &tokens[20..]);

    let reader = store.create_working_set();
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
    let ws = store.create_working_set();
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
    let reader = store.create_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&keys).len(), 2);

    assert_eq!(store.discard_cache(owner, crate::policy::By::Host), 2);
    assert_eq!(store.cached_run(&keys).len(), 0);
    assert_eq!(store.available_pages(), 16);

    // A store that keeps nothing frees it all.
    let mut store = KvStore::new(16, h(19));
    sealed_and_gone(&mut store, &tokens, owner);
    store.set_keeps(false);
    let ws = store.create_working_set();
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

    let ws = store.create_working_set();
    store.reserve(ws, 5).unwrap();
    let keys = keys_ahead(&store, ws, 0, &tokens);
    let run = store.cached_run(&keys);
    assert_eq!(run.len(), 4);
    assert_eq!(
        store.attach_run(ws, &run, 3, 16).unwrap(),
        3,
        "short of the limit"
    );
    assert_eq!(store.mapped_len(ws).unwrap(), 3);
    assert_eq!(store.page_len(ws).unwrap(), 5, "the reservation stays");
    assert_eq!(store.supplied_tokens(ws).unwrap(), 48);
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
    // A write into a mapped-in page copies it.
    assert!(!store.privately_writable(ws, 2).unwrap());
}

fn a_pass_maps_in_more_after_what_it_mapped_in() {
    let mut store = KvStore::new(16, h(21));
    let tokens: Vec<u32> = (0..64).map(|t| 500 + t).collect();
    sealed_and_gone(&mut store, &tokens, ProcessId::from_u128(9));
    let ws = store.create_working_set();
    let keys = keys_ahead(&store, ws, 0, &tokens);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 1, 16).unwrap(), 1);
    let keys = keys_ahead(&store, ws, 16, &tokens[16..]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 4, 16).unwrap(), 3);
    assert_eq!(store.supplied_tokens(ws).unwrap(), 64);

    // Its own first page, computed, is not the path's: nothing maps in.
    let own = store.create_working_set();
    commit_opaque(&mut store, own, 1);
    seal(&mut store, own, 0, &tokens[..16]);
    let keys = keys_ahead(&store, own, 16, &tokens[16..]);
    let run = store.cached_run(&keys);
    assert_eq!(run.len(), 4);
    assert_eq!(store.attach_run(own, &run, 4, 16).unwrap(), 0);
}

fn a_page_no_one_else_reads_is_written_in_place() {
    let mut store = KvStore::new(16, h(23));
    store.set_keeps(true);
    let tokens: Vec<u32> = (0..40).map(|t| 700 + t).collect();
    let live = store.create_working_set();
    commit_opaque(&mut store, live, 4);
    seal(&mut store, live, 0, &tokens);
    let reader = store.create_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens[..32]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(reader, &run, 2, 16).unwrap(), 2);
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
        let ws = store.create_working_set();
        commit_opaque(store, ws, 3);
        seal(store, ws, 0, &tokens);
        ws
    };
    copy(&mut store);
    copy(&mut store);
    let reader = store.create_working_set();
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
    let ws = store.create_working_set();
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

fn what_a_decode_folds_seals() {
    let mut store = KvStore::new(16, h(25));
    let tokens: Vec<u32> = (0..48).map(|t| 900 + t).collect();
    let ws = decoded(&mut store, &tokens, 20, 40);
    assert_eq!(store.sealed_in(ws, 0..3).unwrap(), [0]);
    assert_eq!(
        store.fold(ws, 40, true).unwrap(),
        1,
        "the partial page stays open"
    );
    assert_eq!(store.sealed_in(ws, 0..3).unwrap(), [0, 1]);
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

fn a_fire_that_never_settled_leaves_the_log_cut() {
    let mut store = KvStore::new(16, h(26));
    let tokens: Vec<u32> = (0..48).map(|t| 1000 + t).collect();
    let ws = decoded(&mut store, &tokens, 20, 40);
    store.log_begin(ws, 16).unwrap();
    assert_eq!(store.fold(ws, 40, true).unwrap(), 0);
    store.release_working_set(ws, store.current_epoch());
    store.retire_idle();
    let reader = store.create_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&keys).len(), 1);
}

fn a_hit_refreshes_only_the_entries_it_used_whole() {
    let mut store = KvStore::new(16, h(27));
    let tokens: Vec<u32> = (0..48).map(|t| 1100 + t).collect();
    let (short, long) = (ProcessId::from_u128(12), ProcessId::from_u128(13));
    sealed_and_gone(&mut store, &tokens[..32], short);
    // `long` maps in the two pages, seals a third, and goes.
    let ws = store.create_working_set();
    store.set_owner(ws, long).unwrap();
    let keys = keys_ahead(&store, ws, 0, &tokens[..32]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(ws, &run, 2, 16).unwrap(), 2);
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
    let reader = store.create_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens[..32]);
    let run = store.cached_run(&keys);
    assert_eq!(store.attach_run(reader, &run, 2, 16).unwrap(), 2);
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
    let reader = store.create_working_set();
    let keys = keys_ahead(&store, reader, 0, &tokens);
    assert_eq!(store.cached_run(&keys).len(), 2);
    assert_eq!(store.reclaim_cache(store.current_epoch(), |_| 0), 1);
    assert_eq!(store.cached_run(&keys).len(), 1, "the prompt's page stays");
}

fn idle_cache_goes_to_the_host_and_comes_back_on_a_hit() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(31));
    let owner = ProcessId::from_u128(21);
    let tokens = kept_and_gone(&mut store, 12, 3, owner);
    assert!(
        store.prepare_demote(8, Steps::floor(1)).unwrap().is_none(),
        "below the floor: dropped, not kept"
    );
    let txn = store
        .prepare_demote(8, |_| 0)
        .unwrap()
        .expect("three cache-only pages");
    assert_eq!(txn.page_count(), 3);
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), |_| 0),
        0,
        "a group in a move is not dropped"
    );
    assert_eq!(store.commit_suspend(txn).unwrap(), 3);
    store.retire_idle();
    assert_eq!(store.available_pages(), 8);
    assert_eq!(on_device(&mut store, &tokens), 0);
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), |_| 0),
        0,
        "nothing on the device to free"
    );

    // A hit: the run in any tier comes up, held until it is mapped in.
    let reader = store.create_working_set();
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
    assert_eq!(store.attach_run(reader, &run, 3, 16).unwrap(), 3);
    store.release_reader(pin);
    store.release_working_set(reader, store.current_epoch());
    store.retire_idle();
    assert_eq!(store.reclaim_cache(store.current_epoch(), |_| 0), 3);
    assert_eq!(store.available_pages(), 8);
}

fn a_suspend_short_of_host_room_drops_host_cache_first() {
    let mut store = KvStore::new_with_swap(8, 3, 4, h(32));
    kept_and_gone(&mut store, 13, 3, ProcessId::from_u128(22));
    let txn = store.prepare_demote(8, |_| 0).unwrap().unwrap();
    store.commit_suspend(txn).unwrap();
    assert_eq!(store.host_swap_available(), 0);
    let running = store.create_working_set();
    commit_fresh(&mut store, running, 2, 1);
    let Some(txn) = store.prepare_suspend(&HashSet::from([running])).unwrap() else {
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
        floor: None,
    };
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), keys.clone()),
        1,
        "A's tail"
    );
    assert_eq!(on_device(&mut store, &a_tokens), 3);
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), keys.clone()),
        4,
        "B whole"
    );
    assert_eq!(on_device(&mut store, &b_tokens), 0);
    assert_eq!(on_device(&mut store, &a_tokens), 3, "A's head stays");
    assert_eq!(store.reclaim_cache(store.current_epoch(), keys.clone()), 3);
    assert_eq!(store.available_pages(), 16);
}

fn a_demote_leaves_what_a_higher_key_covers() {
    let mut store = KvStore::new_with_swap(8, 8, 0, h(35));
    store.set_page_size(16);
    let a = ProcessId::from_u128(43);
    let tokens = prompt(17, 4);
    sealed_and_gone(&mut store, &tokens, a);
    // KVFlow: the fixed prompt (two pages) kept, the suffix below the floor.
    let keys = Steps {
        entries: vec![(a, None, -1), (a, Some(32), 10)],
        floor: Some(0),
    };
    let txn = store
        .prepare_demote(8, keys.clone())
        .unwrap()
        .expect("the head goes to the host");
    assert_eq!(
        txn.page_count(),
        2,
        "only the head: the tail is below the floor"
    );
    store.commit_suspend(txn).unwrap();
    store.retire_idle();
    assert_eq!(
        store.reclaim_cache(store.current_epoch(), keys.clone()),
        2,
        "the tail"
    );
    assert_eq!(store.available_pages(), 8);
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
    assert_eq!(store.reclaim_cache(store.current_epoch(), |_| 0), 1);
    assert_eq!(on_device(&mut store, &other), 2, "the older prompt stays");
}
