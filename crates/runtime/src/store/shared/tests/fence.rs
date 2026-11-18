//! A slot no one holds goes back only once the last device work naming it
//! has landed: every fire and every copy takes an epoch of the kv store and
//! touches the slots it names at it, a copy's epoch closing when the
//! engine's `CopyCompletion` resolves (landed), not when it is enqueued.
//! A copy still in flight cannot write into a slot handed to a new owner.
//! Fires and copies retire on watermarks of their own: a copy in flight
//! holds back only the slots it names, and one a fire and a copy both
//! named waits on both.

use ::engine::chunk::SlotCopy;
use ::engine::completion::copy_in_flight;
use ::engine::transfer::MemoryDomain;

use super::*;
use crate::scheduler::lane::CopyEpoch;
use crate::scheduler::worker::PreLaunchCopy;
use crate::scheduler::{TierKind, TierMove};
use crate::store::kv::page_table::{KvPageBacking, PhysicalKvPageId};
use crate::store::registry::{Stores, shared_stores};
use crate::store::slots::Touch;

const DEVICE: MemoryDomain = MemoryDomain::CudaDevice(0);

/// A shared pool with a host tier, and the stores over it as the runtime
/// builds them (the kv store fencing both pools on its epochs).
fn fenced() -> (Shared, Shared, Stores) {
    let mut facts = fixed_facts();
    facts.host_chunks = 8;
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let tiers = Tiers::of(&facts).unwrap();
    let host = tiers.host.clone().expect("a host tier");
    let stores = shared_stores(&shared, &tiers, 16, 256);
    (shared, host, stores)
}

fn list_pool(shared: &Shared) -> ShapeId {
    shared.lock().unwrap().list().unwrap().pool
}

/// Whether kv list slot `slot_ref` is out of the free list (held, or let
/// go and waiting on an epoch).
fn out(shared: &Shared, slot_ref: u32) -> bool {
    let pool = shared.lock().unwrap();
    let shape = pool.list().unwrap().pool;
    pool.pool().is_held(shape, pool.slot(shape, slot_ref))
}

fn retire(stores: &Stores) {
    crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.retire_idle());
}

/// A device slot a copy reads, let go while the copy is in flight, is not
/// handed out again until the engine says the copy landed; then it is.
#[test]
fn a_slot_let_go_under_a_copy_is_not_handed_out_until_the_copy_lands() {
    let (shared, host, stores) = fenced();
    let list = list_pool(&shared);
    let device_refs = shared.lock().unwrap().take_refs(list, 2).unwrap();
    let host_refs = host.lock().unwrap().take_refs(list, 2).unwrap();
    let copy = SlotCopy {
        shape: LIST,
        src: DEVICE,
        dst: MemoryDomain::HostPinned,
        slots: device_refs
            .iter()
            .copied()
            .zip(host_refs.iter().copied())
            .collect(),
        runs: Vec::new(),
    };
    let epoch = CopyEpoch::open_on(
        &stores.kv,
        stores.shared.as_ref(),
        stores.host.as_ref(),
        None,
        &PreLaunchCopy::Slots(copy),
    );
    let (landing, landed) = copy_in_flight();
    epoch.retire_on(&landed);

    // Its owner lets the source go (a suspend committed on the enqueue).
    shared.lock().unwrap().give_refs(list, &device_refs);
    retire(&stores);
    for &slot_ref in &device_refs {
        assert!(
            out(&shared, slot_ref),
            "ref {slot_ref} handed out under the copy"
        );
    }
    let taken = shared.lock().unwrap().take_refs(list, 1).unwrap();
    assert!(
        !device_refs.contains(&taken[0]),
        "a new owner took a slot the copy still reads"
    );

    landing.landed();
    for &slot_ref in &device_refs {
        assert!(
            !out(&shared, slot_ref),
            "ref {slot_ref} back once the copy landed"
        );
    }
    shared.lock().unwrap().give_refs(list, &taken);
    host.lock().unwrap().give_refs(list, &host_refs);
}

/// A slot waits for the last epoch that touched it, not the newest: a fire
/// opened after it, still running, does not hold it.
#[test]
fn a_slot_waits_for_its_own_last_epoch_not_the_newest() {
    let (shared, _host, stores) = fenced();
    let list = list_pool(&shared);
    let refs = shared.lock().unwrap().take_refs(list, 1).unwrap();
    let (own, later) = crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
        (kv.begin_fire(), kv.begin_fire())
    });
    shared
        .lock()
        .unwrap()
        .touch(LIST, refs.iter().copied(), Touch::Fire(own));
    shared.lock().unwrap().give_refs(list, &refs);
    retire(&stores);
    assert!(out(&shared, refs[0]), "its fire still runs");
    crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.settle(own));
    assert!(
        !out(&shared, refs[0]),
        "back with its own fire, the later one still running"
    );
    crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| kv.settle(later));
}

/// The copies a promote's move states, host to device.
fn promote_copies(txn: &crate::store::kv::KvResidencyTxn) -> Vec<SlotCopy> {
    let kind = TierKind::kv(txn.shape());
    txn.copy_plans()
        .into_iter()
        .map(|(tier, device, slots)| {
            TierMove {
                kind,
                tier,
                out: false,
                device,
                slots,
                source: None,
            }
            .slot_copy(DEVICE)
            .expect("a slot move")
        })
        .collect()
}

/// Restore, abort, rebind: a promote's host-to-device copy is in flight
/// when the move aborts (its waiter gone). The fresh device pages it writes
/// go back to the pool, and a new reservation must not take them while the
/// copy can still land on them: a late write would overwrite the new
/// owner's pages. They come back once the copy lands.
#[test]
fn a_restore_aborted_under_its_copy_recycles_at_the_copy_s_epoch() {
    use crate::store::kv::hash::{Hash256, chain_slots, page_hash};
    let (shared, _host, stores) = fenced();
    let seed: Hash256 = [3; 32];
    let tokens: Vec<Vec<Option<Hash256>>> =
        chain_slots(&seed, None, 0, &(0..48).collect::<Vec<_>>())
            .chunks(16)
            .map(|page| page.iter().map(|h| Some(*h)).collect())
            .collect();
    let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
    let kv = &stores.kv;
    let with = |f: &mut dyn FnMut(&mut crate::store::kv::KvStore)| {
        crate::store::registry::with_kv_lock(kv, "test", |kv| f(kv));
    };
    // Three cached pages, demoted to the host.
    with(&mut |kv| {
        kv.set_keeps(true);
        let pages = kv.reserve_device_pages(3).unwrap();
        assert!(kv.import_cached_at(
            &[],
            tokens.clone(),
            keys.clone(),
            pages.into_iter().map(KvPageBacking::Resident).collect(),
        ));
        let txn = kv.prepare_demote(64, |_| 1).unwrap().expect("a demote");
        assert_eq!(kv.commit_suspend(txn).unwrap(), 3);
        kv.retire_idle();
    });

    // A promote reserves fresh device pages and copies the host's onto
    // them.
    let mut aborted: Vec<u32> = Vec::new();
    let mut copies = Vec::new();
    let mut txn = None;
    with(&mut |kv| {
        let run = kv.cached_run_any_tier(&keys);
        let mut granted: Vec<PhysicalKvPageId> = kv.reserve_device_pages(3).unwrap();
        aborted = granted.iter().map(|page| page.0).collect();
        let promote = kv.prepare_promote(&run, &mut granted).unwrap().unwrap();
        copies = promote_copies(&promote);
        txn = Some(promote);
    });
    let mut landings = Vec::new();
    for copy in copies {
        let epoch = CopyEpoch::open_on(
            &stores.kv,
            stores.shared.as_ref(),
            stores.host.as_ref(),
            None,
            &PreLaunchCopy::Slots(copy),
        );
        let (landing, landed) = copy_in_flight();
        epoch.retire_on(&landed);
        landings.push(landing);
    }

    // The move aborts before the copy lands.
    let txn = txn.take().unwrap();
    crate::store::registry::with_kv_lock(kv, "test", |kv| kv.abort_restore(txn));
    for &page in &aborted {
        assert!(out(&shared, page), "page {page} handed out under its copy");
    }
    // The rebind: another owner takes every page there is room for now.
    let mut rebound = Vec::new();
    with(&mut |kv| {
        let free = kv.available_pages();
        rebound = kv.reserve_device_pages(free).unwrap_or_default();
    });
    assert!(
        rebound.iter().all(|page| !aborted.contains(&page.0)),
        "a new owner took a page the restore's copy still writes"
    );

    for landing in landings {
        landing.landed();
    }
    for &page in &aborted {
        assert!(!out(&shared, page), "page {page} back once the copy landed");
    }
    with(&mut |kv| kv.release_device_reservation(std::mem::take(&mut rebound)));
}

/// Opens the epoch of a copy of `refs` of the kv list device to host, as
/// the lane does.
fn copy_list(stores: &Stores, host_refs: &[u32], refs: &[u32]) -> CopyEpoch {
    CopyEpoch::open_on(
        &stores.kv,
        stores.shared.as_ref(),
        stores.host.as_ref(),
        None,
        &PreLaunchCopy::Slots(SlotCopy {
            shape: LIST,
            src: DEVICE,
            dst: MemoryDomain::HostPinned,
            slots: refs
                .iter()
                .copied()
                .zip(host_refs.iter().copied())
                .collect(),
            runs: Vec::new(),
        }),
    )
}

/// A copy the engine refuses enqueued nothing: its epoch closes as the
/// lane drops it, and a slot let go meanwhile goes back.
#[test]
fn a_copy_the_engine_refuses_holds_nothing() {
    let (shared, host, stores) = fenced();
    let list = list_pool(&shared);
    let refs = shared.lock().unwrap().take_refs(list, 1).unwrap();
    let host_refs = host.lock().unwrap().take_refs(list, 1).unwrap();
    let epoch = copy_list(&stores, &host_refs, &refs);
    shared.lock().unwrap().give_refs(list, &refs);
    retire(&stores);
    assert!(
        out(&shared, refs[0]),
        "fenced while the copy is being handed over"
    );
    // The engine's `copy_slots` answered Err: the lane drops the epoch.
    drop(epoch);
    assert!(!out(&shared, refs[0]), "a refused copy holds nothing");
    let (current, retired) = crate::store::registry::with_kv_lock(&stores.kv, "test", |kv| {
        (kv.current_epoch(), kv.watermarks().through())
    });
    assert_eq!(current, retired, "no epoch is left open");
    host.lock().unwrap().give_refs(list, &host_refs);
}

/// An engine that drops a copy's landing without resolving it has the copy
/// fail: its epoch closes, and nothing behind it is fenced for good.
#[test]
fn a_copy_whose_landing_is_dropped_closes_its_epoch() {
    let (shared, host, stores) = fenced();
    let list = list_pool(&shared);
    let refs = shared.lock().unwrap().take_refs(list, 1).unwrap();
    let host_refs = host.lock().unwrap().take_refs(list, 1).unwrap();
    let epoch = copy_list(&stores, &host_refs, &refs);
    let (landing, landed) = copy_in_flight();
    epoch.retire_on(&landed);
    shared.lock().unwrap().give_refs(list, &refs);
    retire(&stores);
    assert!(out(&shared, refs[0]));
    drop(landing);
    assert!(matches!(landed.check(), Some(Err(_))), "it failed");
    assert!(!out(&shared, refs[0]), "and holds nothing");
    host.lock().unwrap().give_refs(list, &host_refs);
}

fn with_kv<T>(stores: &Stores, f: impl FnOnce(&mut crate::store::kv::KvStore) -> T) -> T {
    crate::store::registry::with_kv_lock(&stores.kv, "test", f)
}

/// A copy in flight (an eviction's, say) holds back no slot it does not
/// name: a slot only fires named goes back as its fire lands, the copy
/// still running, while the slot the copy reads waits for the copy.
#[test]
fn a_copy_in_flight_holds_back_no_slot_only_fires_named() {
    let (shared, host, stores) = fenced();
    let list = list_pool(&shared);
    let copied = shared.lock().unwrap().take_refs(list, 1).unwrap();
    let host_refs = host.lock().unwrap().take_refs(list, 1).unwrap();
    let fired = shared.lock().unwrap().take_refs(list, 1).unwrap();
    // The copy opens first: older than the fire, its epoch is the oldest
    // still open.
    let epoch = copy_list(&stores, &host_refs, &copied);
    let (landing, landed) = copy_in_flight();
    epoch.retire_on(&landed);
    let fire = with_kv(&stores, |kv| kv.begin_fire());
    shared
        .lock()
        .unwrap()
        .touch(LIST, fired.iter().copied(), Touch::Fire(fire));
    shared.lock().unwrap().give_refs(list, &fired);
    shared.lock().unwrap().give_refs(list, &copied);
    retire(&stores);
    assert!(out(&shared, fired[0]), "its fire still runs");
    with_kv(&stores, |kv| kv.settle(fire));
    assert!(
        !out(&shared, fired[0]),
        "back as its fire lands, though an older copy is still in flight"
    );
    assert!(out(&shared, copied[0]), "the copy still reads it");
    landing.landed();
    assert!(!out(&shared, copied[0]), "back once the copy landed");
    host.lock().unwrap().give_refs(list, &host_refs);
}

/// A kv page let go while an eviction's copy of other pages is in flight
/// goes back to the store as the fire that read it lands: it waits on a
/// copy only where one named it.
#[test]
fn a_kv_page_let_go_under_an_unrelated_copy_comes_back_with_its_fires() {
    let (shared, host, stores) = fenced();
    let list = list_pool(&shared);
    let copied = shared.lock().unwrap().take_refs(list, 1).unwrap();
    let host_refs = host.lock().unwrap().take_refs(list, 1).unwrap();
    let epoch = copy_list(&stores, &host_refs, &copied);
    let (landing, landed) = copy_in_flight();
    epoch.retire_on(&landed);
    let (pages, fire) = with_kv(&stores, |kv| {
        let pages = kv.reserve_device_pages(1).unwrap();
        (pages, kv.begin_fire())
    });
    shared
        .lock()
        .unwrap()
        .touch(LIST, pages.iter().map(|page| page.0), Touch::Fire(fire));
    let before = with_kv(&stores, |kv| {
        let before = kv.available_pages();
        kv.release_device_reservation(pages.clone());
        assert_eq!(kv.available_pages(), before, "fenced while its fire runs");
        before
    });
    with_kv(&stores, |kv| kv.settle(fire));
    assert!(
        !out(&shared, pages[0].0),
        "back with its fire, the copy in flight"
    );
    with_kv(&stores, |kv| {
        assert_eq!(kv.available_pages(), before + 1);
    });
    landing.landed();
    shared.lock().unwrap().give_refs(list, &copied);
    host.lock().unwrap().give_refs(list, &host_refs);
}

/// A slot a fire and a copy both named waits on both, whichever lands
/// first.
#[test]
fn a_slot_a_fire_and_a_copy_both_named_waits_on_both() {
    for copy_first in [true, false] {
        let (shared, host, stores) = fenced();
        let list = list_pool(&shared);
        let refs = shared.lock().unwrap().take_refs(list, 1).unwrap();
        let host_refs = host.lock().unwrap().take_refs(list, 1).unwrap();
        let fire = with_kv(&stores, |kv| kv.begin_fire());
        shared
            .lock()
            .unwrap()
            .touch(LIST, refs.iter().copied(), Touch::Fire(fire));
        let epoch = copy_list(&stores, &host_refs, &refs);
        let (landing, landed) = copy_in_flight();
        epoch.retire_on(&landed);
        shared.lock().unwrap().give_refs(list, &refs);
        retire(&stores);
        assert!(out(&shared, refs[0]));
        if copy_first {
            landing.landed();
            assert!(out(&shared, refs[0]), "its fire still runs");
            with_kv(&stores, |kv| kv.settle(fire));
        } else {
            with_kv(&stores, |kv| kv.settle(fire));
            assert!(out(&shared, refs[0]), "the copy still reads it");
            landing.landed();
        }
        assert!(!out(&shared, refs[0]), "back once both landed");
        host.lock().unwrap().give_refs(list, &host_refs);
    }
}
