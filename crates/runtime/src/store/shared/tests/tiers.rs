//! CONTRACT-BC 22: the host and disk tiers of a shared pool. Each shape's
//! slots go to the host tier (a pool of every shape of its own) and come
//! back to fresh device slots with their bytes and their holders; kv list
//! slots go to disk under the journal. An engine is played by a map of
//! each tier's slots to a value, moved by the `SlotCopy`s the stores' moves
//! become.

use std::collections::{HashMap, HashSet};

use ::engine::chunk::SlotCopy;
use ::engine::transfer::{DeviceReach, MemoryDomain};

use super::*;
use crate::scheduler::{TierKind, TierMove};
use crate::store::kv::KvStore;
use crate::store::kv::page_table::{KvPageBacking, PhysicalKvPageId};

const DEVICE: MemoryDomain = MemoryDomain::Device {
    ordinal: 0,
    reach: DeviceReach::Peer,
};
const HOST_CHUNKS: u32 = 8;

/// Each tier's slots and the value each holds: what an engine's copies move.
#[derive(Default)]
struct Bytes(HashMap<(MemoryDomain, EngineShape, u32), u64>);

impl Bytes {
    fn set(&mut self, shape: EngineShape, slot: u32, value: u64) {
        self.0.insert((DEVICE, shape, slot), value);
    }

    fn device(&self, shape: EngineShape, slot: u32) -> Option<u64> {
        self.0.get(&(DEVICE, shape, slot)).copied()
    }

    /// Moves what `copy` names: every source slot holds a value, no two
    /// copies land on one slot, and a slot moves whole.
    fn copy(&mut self, copy: &SlotCopy) {
        assert!(copy.runs.is_empty(), "between tiers a slot moves whole");
        assert_ne!(copy.src, copy.dst);
        let dsts: HashSet<u32> = copy.slots.iter().map(|&(_, dst)| dst).collect();
        assert_eq!(dsts.len(), copy.slots.len(), "one copy a destination");
        let moved: Vec<(u32, u64)> = copy
            .slots
            .iter()
            .map(|&(src, dst)| {
                let value = self.0.get(&(copy.src, copy.shape, src)).copied();
                (
                    dst,
                    value.unwrap_or_else(|| {
                        panic!(
                            "{:?} slot {src} of shape {} holds nothing",
                            copy.src, copy.shape
                        )
                    }),
                )
            })
            .collect();
        for (dst, value) in moved {
            self.0.insert((copy.dst, copy.shape, dst), value);
        }
    }

    /// A device whose slots were given back and taken by others: what is
    /// left on the device is gone.
    fn clobber_device(&mut self) {
        self.0.retain(|(domain, ..), _| *domain != DEVICE);
    }
}

/// The engine copies a kv move states, `out` of the device or back.
fn kv_copies(txn: &crate::store::kv::KvResidencyTxn, out: bool) -> Vec<(MemoryDomain, SlotCopy)> {
    let kind = TierKind::kv(txn.shape());
    assert!(
        matches!(kind, TierKind::KvSlots(LIST)),
        "a shared pool's kv list"
    );
    txn.copy_plans()
        .into_iter()
        .map(|(tier, device, slots)| {
            let copy = TierMove {
                kind,
                tier,
                out,
                device,
                slots,
                source: None,
            }
            .slot_copy(DEVICE)
            .expect("a slot move");
            (tier, copy)
        })
        .collect()
}

/// (g) Idle cache parked on the host goes on down to disk when the host is
/// short of room: host refs copied to disk refs by `copy_slots` (HostPinned
/// -> LocalDisk, kv list only), journaled, the host refs back. The next
/// run takes it back from the journal alone, and a hit brings it up to
/// fresh device refs with its bytes.
#[test]
fn host_cache_goes_on_down_to_disk_and_a_restart_brings_it_back_bitwise() {
    use crate::store::chunks::ShapePool;
    use crate::store::kv::hash::{Hash256, chain_slots, page_hash};
    let mut facts = fixed_facts();
    facts.host_chunks = HOST_CHUNKS;
    facts.disk_chunks = 1;
    let seed: Hash256 = [9; 32];
    let tokens: Vec<Vec<Option<Hash256>>> =
        chain_slots(&seed, None, 0, &(0..48).collect::<Vec<_>>())
            .chunks(16)
            .map(|page| page.iter().map(|h| Some(*h)).collect())
            .collect();
    let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
    let dir = tempfile::tempdir().unwrap();
    let journal = dir.path().join("g0.journal");
    let store_for = |shared: &Shared| {
        let tiers = Tiers::of(&facts).unwrap();
        let list = shared.lock().unwrap().list().unwrap();
        let host = tiers
            .host
            .as_ref()
            .map(|host| ShapePool::shared(host, list));
        let mut kv = KvStore::with_pool(ShapePool::shared(shared, list), 0, 0, seed);
        kv.set_tiers(list.id, host, tiers.disk_slots);
        kv.set_page_size(16);
        kv.set_keeps(true);
        kv
    };
    let mut bytes = Bytes::default();

    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut kv = store_for(&shared);
    assert_eq!(kv.keep_disk(&journal, true).unwrap(), 0);
    let host_free = kv.host_swap_available();
    let pages = kv.reserve_device_pages(3).unwrap();
    for (at, page) in pages.iter().enumerate() {
        bytes.set(LIST, page.0, 700 + at as u64);
    }
    assert!(kv.import_cached_at(
        &[],
        tokens.clone(),
        keys.clone(),
        pages.into_iter().map(KvPageBacking::Resident).collect(),
    ));
    // The device wants room: it goes to the host, not journaled.
    let txn = kv.prepare_demote(64, |_| 1).unwrap().expect("a demote");
    for (tier, copy) in kv_copies(&txn, true) {
        assert_eq!(tier, MemoryDomain::HostPinned);
        bytes.copy(&copy);
    }
    assert_eq!(kv.commit_suspend(txn).unwrap(), 3);
    kv.retire_idle();
    bytes.clobber_device();
    assert_eq!(kv.persisted_pages(), 0);
    assert_eq!(kv.below_device_cache_pages(), (3, 0));

    // The host is short: it goes on down to disk.
    let txn = kv.prepare_spill(1, |_| 1).expect("a spill");
    assert_eq!(txn.shape(), LIST);
    let (host, disk) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::KvSlots(txn.shape()),
        tier: MemoryDomain::LocalDisk,
        out: true,
        device: host,
        slots: disk,
        source: Some(MemoryDomain::HostPinned),
    }
    .slot_copy(DEVICE)
    .expect("a slot move");
    assert_eq!(
        (copy.src, copy.dst, copy.shape),
        (MemoryDomain::HostPinned, MemoryDomain::LocalDisk, LIST)
    );
    assert_eq!(copy.slots.len(), 3);
    assert!(copy.slots.iter().all(|&(_, disk)| disk < 16));
    bytes.copy(&copy);
    assert_eq!(kv.commit_spill(txn), 3);
    assert_eq!(
        kv.host_swap_available(),
        host_free,
        "the host refs are back"
    );
    assert_eq!(kv.persisted_pages(), 3, "journaled");
    assert_eq!(
        kv.below_device_cache_pages(),
        (0, 3),
        "idle cache on disk is room the pressure report counts"
    );
    drop(kv);
    drop(shared);
    // A restart: the device and the host are gone; the slot file stays.
    bytes
        .0
        .retain(|(domain, ..), _| *domain == MemoryDomain::LocalDisk);

    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut kv = store_for(&shared);
    assert_eq!(kv.keep_disk(&journal, true).unwrap(), 3, "taken back");
    let run = kv.cached_run_any_tier(&keys);
    assert_eq!((run.len(), kv.below_device(&run)), (3, 3));
    let mut granted: Vec<PhysicalKvPageId> = kv.reserve_device_pages(3).unwrap();
    let txn = kv.prepare_promote(&run, &mut granted).unwrap().unwrap();
    for (tier, copy) in kv_copies(&txn, false) {
        assert_eq!(tier, MemoryDomain::LocalDisk);
        bytes.copy(&copy);
    }
    let pin = kv.commit_promote(txn).unwrap();
    let run = kv.cached_run(&keys);
    assert_eq!(run.len(), 3, "resident again");
    for (at, location) in run.iter().enumerate() {
        let KvPageBacking::Resident(page) = kv.table().backing_at(location).unwrap() else {
            panic!("resident");
        };
        assert_eq!(
            bytes.device(LIST, page.0),
            Some(700 + at as u64),
            "page {at}"
        );
    }
    kv.release_reader(pin);
}

/// A slot file of `chunks` whole disk chunks, for a store's disk tier.
struct DiskFile(Mutex<Vec<u32>>);

impl crate::store::roles::DiskSlots for DiskFile {
    fn take_chunks(&self, n: usize) -> Option<Vec<u32>> {
        let mut free = self.0.lock().unwrap();
        let at = free.len().checked_sub(n)?;
        Some(free.split_off(at))
    }
    fn give_chunks(&self, chunks: &[u32]) {
        self.0.lock().unwrap().extend_from_slice(chunks);
    }
    fn free_chunks(&self) -> usize {
        self.0.lock().unwrap().len()
    }
}

/// No host tier (unified memory): an evicted process's records and rs list
/// slots would stay on the device, their chunks held, since a suspend
/// finds no host row. Where no host row is free its state parks in the
/// disk tier's slot file, a disk chunk of each shape laid out as the
/// device's (CONTRACT-BC 22): the move copies to the disk, the device slots
/// come free, and a restore brings it back and gives the disk chunks back.
#[test]
fn with_no_host_tier_an_evicted_set_s_state_parks_on_disk_and_frees_the_device() {
    use crate::store::rs::{RsGeometry, RsStore};
    let facts = fixed_facts();
    let tiers = Tiers::of(&facts).unwrap();
    assert!(tiers.host.is_none(), "unified memory: no host tier");
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut s = RsStore::shared(&shared, None);
    assert!(s.host_available().is_zero(), "no host rows");
    let ws = s.create_working_set(RsGeometry {
        state_size: 4096,
        buffer_page_tokens: RS_TOKENS,
    });
    let prepared = s.prepare_group(ws, true, false, None, None, None).unwrap();
    let published = s.publish_prepared(prepared).unwrap();
    s.settle(published);
    s.fold(ws, 4);
    let sets = HashSet::from([ws]);
    let held = s.suspendable(&sets);
    assert!(!held.is_zero(), "its state holds device slots");
    assert!(s.prepare_suspend(&sets).is_none(), "no disk tier: it stays on the device");

    let file = Arc::new(DiskFile(Mutex::new((0..8).collect())));
    s.set_disk(&shared, file.clone());
    assert!(!s.disk_units(8).is_zero(), "the disk is room below");
    let device_refs = s.private_slot_refs(&sets);
    let txn = s.prepare_suspend(&sets).expect("it parks on disk");
    let plan = txn.copy_plan();
    assert!(plan.shared.is_empty(), "nothing to a host tier");
    let to_disk: usize = plan.disk.iter().map(|(_, pairs)| pairs.len()).sum();
    let named: usize = device_refs.iter().map(|(_, refs)| refs.len()).sum();
    assert_eq!(to_disk, named, "every device slot of its state copied to disk");
    for (shape, pairs) in &plan.disk {
        let device = &device_refs.iter().find(|(at, _)| at == shape).unwrap().1;
        assert!(pairs.iter().all(|(from, _)| device.contains(from)), "by shape");
    }
    let taken = 8 - crate::store::roles::DiskSlots::free_chunks(&*file);
    assert!(taken > 0, "disk chunks of its shapes opened");
    let freed = s.commit_suspend(txn);
    assert_eq!(freed, held, "its device slots come free");
    assert!(s.private_slot_refs(&sets).is_empty(), "nothing of it on the device");
    assert!(s.disk_rows() > 0);
    assert_eq!(s.swapped_units(&sets), held, "parked, counted as swapped");

    let swapped = s.swapped_units(&sets);
    let mut grant = s.reserve(swapped).expect("device room to come back");
    let txn = s.prepare_restore(&sets, &mut grant).expect("it comes back");
    let back: usize = txn.copy_plan().disk.iter().map(|(_, pairs)| pairs.len()).sum();
    assert_eq!(back, named, "read back from the disk");
    s.commit_restore(txn);
    assert_eq!(s.disk_rows(), 0, "no disk row held");
    assert_eq!(crate::store::roles::DiskSlots::free_chunks(&*file), 8, "its disk chunks given back");
    assert_eq!(s.suspendable(&sets), held, "on the device again");
}

/// A list compaction: kv pages scattered across partly used list chunks
/// (free list slots enough, no chunk free of them) move into the free slots
/// of the fullest chunk, device to device, and the chunks they leave stand
/// free whole; every page reads its new slot with its bytes, and a page a
/// move holds stays where it lies.
#[test]
fn a_list_compaction_frees_whole_chunks_of_scattered_pages() {
    use crate::store::chunks::ShapePool;
    use crate::store::kv::hash::{Hash256, chain_slots, page_hash};
    let facts = fixed_facts();
    let seed: Hash256 = [7; 32];
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let list = shared.lock().unwrap().list().unwrap();
    let spc = list.slots_per_chunk as usize;
    assert!(spc >= 4, "a list chunk holds a few pages");
    let mut kv = KvStore::with_pool(ShapePool::shared(&shared, list), 0, 0, seed);
    kv.set_tiers(list.id, None, 0);
    kv.set_page_size(16);
    // Three chunks filled, then all but a few pages of each given back:
    // one page in the first, two in the second, three in the third.
    let mut pages = kv.reserve_device_pages(3 * spc).unwrap();
    let mut kept = Vec::new();
    for at in 0..3 {
        let rest: Vec<_> = pages.drain(..spc).collect();
        let (keep, give): (Vec<_>, Vec<_>) =
            rest.into_iter().enumerate().partition(|&(i, _)| i <= at);
        kv.release_device_reservation(give.into_iter().map(|(_, page)| page).collect());
        kept.extend(keep.into_iter().map(|(_, page)| page));
    }
    let spread: HashSet<usize> = kept.iter().map(|page| page.0 as usize / spc).collect();
    assert_eq!(spread.len(), 3, "pages in each of three chunks");
    let free_before = shared.lock().unwrap().pool().free_chunks();
    let tokens: Vec<Vec<Option<Hash256>>> = chain_slots(&seed, None, 0, &(0..96).collect::<Vec<_>>())
        .chunks(16)
        .map(|page| page.iter().map(|h| Some(*h)).collect())
        .collect();
    let keys: Vec<Hash256> = tokens.iter().map(|page| page_hash(page)).collect();
    let mut bytes = Bytes::default();
    for (at, page) in kept.iter().enumerate() {
        bytes.set(LIST, page.0, 900 + at as u64);
    }
    assert!(kv.import_cached_at(
        &[],
        tokens,
        keys.clone(),
        kept.iter().copied().map(KvPageBacking::Resident).collect(),
    ));

    // Two chunks wanted: the emptiest two empty into the third.
    let plan = kv.plan_list_compaction(2, &|_, _| true).expect("a plan");
    assert_eq!((plan.chunks.len(), plan.pages), (2, 3), "{plan:?}");
    assert!(plan.readers.is_empty() && plan.parked.is_empty(), "idle cache: no reader");
    // Not more than the free slots elsewhere take.
    assert!(kv.plan_list_compaction(3, &|_, _| true).unwrap().chunks.len() == 2);
    let txn = kv.prepare_relocate(&plan, &HashSet::new()).expect("a move");
    // Pinned as a move's: a second plan finds nothing more to move.
    assert!(kv.plan_list_compaction(1, &|_, _| true).is_none());
    let (from, to) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::kv(txn.shape()),
        tier: DEVICE,
        out: true,
        device: from.clone(),
        slots: to.clone(),
        source: None,
    }
    .slot_copy(DEVICE)
    .expect("a slot move");
    assert_eq!((copy.src, copy.dst, copy.shape), (DEVICE, DEVICE, LIST));
    for &(src, dst) in &copy.slots {
        let value = bytes.device(LIST, src).expect("a page's bytes");
        bytes.set(LIST, dst, value);
    }
    assert_eq!(kv.commit_relocate(txn).unwrap(), 3);
    assert_eq!(
        shared.lock().unwrap().pool().free_chunks(),
        free_before + 2,
        "the chunks emptied stand free whole"
    );
    let run = kv.cached_run(&keys);
    assert_eq!(run.len(), 6);
    let mut chunks_now = HashSet::new();
    for (at, location) in run.iter().enumerate() {
        let KvPageBacking::Resident(page) = kv.table().backing_at(location).unwrap() else {
            panic!("resident");
        };
        chunks_now.insert(page.0 as usize / spc);
        assert_eq!(bytes.device(LIST, page.0), Some(900 + at as u64), "page {at}");
    }
    assert_eq!(chunks_now.len(), 1, "every page in one chunk");
}
