//! CONTRACT-BC 22: the host and disk tiers of a shared pool. Each shape's
//! slots go to the host tier (a pool of every shape of its own) and come
//! back to fresh device slots with their bytes and their holders; kv list
//! slots go to disk under the journal. An engine is played by a map of
//! each tier's slots to a value, moved by the `SlotCopy`s the stores' moves
//! become.

use std::collections::{HashMap, HashSet};

use ::engine::chunk::SlotCopy;
use ::engine::transfer::MemoryDomain;

use super::*;
use crate::scheduler::{TierKind, TierMove};
use crate::store::kv::KvStore;
use crate::store::kv::page_table::{KvPageBacking, PhysicalKvPageId};

const DEVICE: MemoryDomain = MemoryDomain::CudaDevice(0);
const HOST_CHUNKS: u32 = 8;

/// A host or disk tier with more slots than a `u32` slot ref names is
/// refused by name at load, as the device pool is: its refs are never
/// clamped or saturated onto another slot's.
#[test]
fn a_tier_past_a_u32_of_slot_refs_is_refused_by_name() {
    let facts = fixed_facts();
    let list = facts
        .shapes
        .iter()
        .find(|shape| {
            shape.role == SlotRole::State
                && matches!(shape.kind, SlotKind::List { window: None, .. })
        })
        .unwrap();
    assert!(list.slots_per_chunk >= 2, "the kv list packs a chunk");
    // The widest count the device pool itself would pass is not the
    // tier's: the tier's own chunks are checked.
    let chunks = u32::MAX;
    let wide = facts
        .shapes
        .iter()
        .find(|shape| shape.slots_per_chunk >= 2)
        .unwrap();
    assert_eq!(
        SharedPool::tier(&facts, chunks).unwrap_err(),
        SharedError::Tier {
            tier: "host",
            shape: wide.id.0,
            chunks,
            slots_per_chunk: wide.slots_per_chunk,
        }
    );
    let mut host = facts.clone();
    host.host_chunks = chunks;
    assert!(matches!(
        Tiers::of(&host),
        Err(SharedError::Tier { tier: "host", .. })
    ));
    let mut disk = facts.clone();
    disk.disk_chunks = chunks;
    let refused = Tiers::of(&disk).unwrap_err();
    assert_eq!(
        refused,
        SharedError::Tier {
            tier: "disk",
            shape: list.id.0,
            chunks,
            slots_per_chunk: list.slots_per_chunk,
        }
    );
    assert!(
        refused
            .to_string()
            .contains("more slots than a u32 slot ref names"),
        "{refused}"
    );
    // The most a `u32` names is taken whole on disk.
    let mut most = facts.clone();
    most.disk_chunks = u32::MAX / list.slots_per_chunk;
    assert_eq!(
        Tiers::of(&most).unwrap().disk_slots,
        most.disk_chunks * list.slots_per_chunk
    );
}

/// A tier of facts that do not hang together is refused as the device pool
/// of them is.
#[test]
fn a_tier_of_invalid_facts_is_refused() {
    let mut facts = fixed_facts();
    facts.shapes[0].slots_per_chunk = 0;
    assert!(matches!(
        SharedPool::tier(&facts, HOST_CHUNKS),
        Err(SharedError::Facts(_))
    ));
}

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
    assert_eq!(kv.host_cache_pages(), 3);

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
