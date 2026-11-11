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
use crate::store::kv::page_table::{KvPageBacking, PhysicalKvPageId, WorkingSetId};
use crate::store::kv::write::PageCommit;
use crate::store::rs::{RsResidencyTxn, RsStore, RsWorkingSetId};

const DEVICE: MemoryDomain = MemoryDomain::CudaDevice(0);
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

/// `facts` with a host tier of [`HOST_CHUNKS`] chunks and `disk` chunks of kv
/// list slots, and the stores of their shared pool.
fn tiered(mut facts: ChunkFacts, disk: u32) -> (Shared, Tiers, crate::store::registry::Stores) {
    facts.host_chunks = HOST_CHUNKS;
    facts.disk_chunks = disk;
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let tiers = Tiers::of(&facts).unwrap();
    let stores = crate::store::registry::shared_stores(&shared, &tiers, 16, 256);
    (shared, tiers, stores)
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

/// The engine copies an rs move states, shape by shape.
fn rs_copies(txn: &RsResidencyTxn, out: bool) -> Vec<SlotCopy> {
    txn.copy_plan()
        .shared
        .iter()
        .map(|(shape, pairs)| {
            let (device, slots) = pairs.iter().copied().unzip();
            TierMove {
                kind: TierKind::RsSlots(*shape),
                tier: MemoryDomain::HostPinned,
                out,
                device,
                slots,
                source: None,
            }
            .slot_copy(DEVICE)
            .expect("a slot move")
        })
        .collect()
}

fn commit_pages(kv: &mut KvStore, ws: WorkingSetId, n: u64) {
    let start = kv.page_len(ws).unwrap();
    kv.reserve(ws, n).unwrap();
    let indexes: Vec<u64> = (start..start + n).collect();
    let need = kv.write_demand(ws, &indexes).unwrap();
    let mut granted = kv.reserve_device_pages(need).unwrap();
    let prepared = kv
        .prepare_write_reserved(ws, &indexes, &mut granted)
        .unwrap();
    let commits: Vec<PageCommit> = (0..n)
        .map(|_| PageCommit {
            token_hashes: Vec::new(),
            page_hash: None,
        })
        .collect();
    let seq = kv.publish_prepared(prepared, &commits).unwrap();
    kv.settle(seq);
}

fn table(kv: &mut KvStore, ws: WorkingSetId) -> Vec<u32> {
    kv.flat_table(ws)
        .unwrap()
        .iter()
        .map(|page| page.0)
        .collect()
}

/// The host tier's live slots of pool shape `shape`.
fn host_live(tiers: &Tiers, shape: ShapeId) -> u64 {
    tiers
        .host
        .as_ref()
        .unwrap()
        .lock()
        .unwrap()
        .pool()
        .shape_demand(shape)
        .live
}

/// (a) kv list pages: suspended to host refs of the host tier's list shape
/// and restored to fresh device refs, every page's bytes where its table
/// entry now points; the host slots and the device ones all come back.
#[test]
fn kv_list_pages_go_to_the_host_tier_and_come_back_with_their_bytes() {
    let (_, tiers, stores) = tiered(fixed_facts(), 0);
    let mut kv = stores.kv.lock();
    let free = kv.available_pages();
    let host_free = kv.host_swap_available();
    assert_eq!(
        host_free,
        (HOST_CHUNKS * 16) as usize,
        "16 list slots a chunk"
    );
    let ws = kv.create_working_set();
    commit_pages(&mut kv, ws, 5);
    let before = table(&mut kv, ws);
    let mut bytes = Bytes::default();
    for (at, &page) in before.iter().enumerate() {
        bytes.set(LIST, page, 100 + at as u64);
    }

    let set = HashSet::from([ws]);
    let txn = kv.prepare_suspend(&set).unwrap().expect("five pages move");
    for (tier, copy) in kv_copies(&txn, true) {
        assert_eq!(tier, MemoryDomain::HostPinned, "host first");
        assert_eq!((copy.src, copy.dst), (DEVICE, MemoryDomain::HostPinned));
        bytes.copy(&copy);
    }
    assert_eq!(kv.commit_suspend(txn).unwrap(), 5);
    kv.retire_idle();
    assert_eq!(kv.available_pages(), free, "the device pages are free");
    assert_eq!(host_live(&tiers, LIST_POOL), 5, "held in the host tier");
    bytes.clobber_device();

    let mut granted = kv.reserve_device_pages(5).unwrap();
    let txn = kv.prepare_restore(&set, &mut granted).unwrap();
    for (_, copy) in kv_copies(&txn, false) {
        assert_eq!((copy.src, copy.dst), (MemoryDomain::HostPinned, DEVICE));
        bytes.copy(&copy);
    }
    assert_eq!(kv.commit_restore(txn).unwrap(), 5);
    let after = table(&mut kv, ws);
    assert_eq!(after.len(), 5);
    for (at, &page) in after.iter().enumerate() {
        assert_eq!(bytes.device(LIST, page), Some(100 + at as u64), "page {at}");
    }
    assert_eq!(
        kv.host_swap_available(),
        host_free,
        "the host slots go back"
    );
    assert_eq!(host_live(&tiers, LIST_POOL), 0);
    let epoch = kv.current_epoch();
    kv.release_working_set(ws, epoch);
    kv.retire_idle();
    assert_eq!(kv.available_pages(), free);
}

/// (b) A fork shares its pages: a suspend of the parent alone moves
/// nothing it shares; with the fork, a shared page moves once, both read
/// the host slot, and a restore of either brings it back for both.
#[test]
fn a_forks_shared_kv_pages_move_once_and_come_back_for_both() {
    let (_, tiers, stores) = tiered(fixed_facts(), 0);
    let mut kv = stores.kv.lock();
    let ws = kv.create_working_set();
    commit_pages(&mut kv, ws, 3);
    let fork = kv.fork(ws).unwrap();
    commit_pages(&mut kv, fork, 1);
    let parent = table(&mut kv, ws);
    let child = table(&mut kv, fork);
    assert_eq!(
        &child[..3],
        &parent[..],
        "the fork reads its parent's pages"
    );
    let mut bytes = Bytes::default();
    for (at, &page) in child.iter().enumerate() {
        bytes.set(LIST, page, 200 + at as u64);
    }
    let suspend = |kv: &mut KvStore, bytes: &mut Bytes, sets: HashSet<WorkingSetId>| {
        let txn = kv.prepare_suspend(&sets).unwrap().expect("pages to move");
        for (_, copy) in kv_copies(&txn, true) {
            bytes.copy(&copy);
        }
        kv.commit_suspend(txn).unwrap()
    };
    assert_eq!(
        suspend(&mut kv, &mut bytes, HashSet::from([fork])),
        1,
        "its own"
    );
    assert_eq!(
        suspend(&mut kv, &mut bytes, HashSet::from([ws])),
        3,
        "the shared pages, once"
    );
    assert_eq!(host_live(&tiers, LIST_POOL), 4);
    let swapped = |kv: &KvStore, ws| kv.swapped_page_count(&HashSet::from([ws])).unwrap();
    assert_eq!((swapped(&kv, ws), swapped(&kv, fork)), (3, 4));
    kv.retire_idle();
    bytes.clobber_device();

    let mut granted = kv.reserve_device_pages(4).unwrap();
    let txn = kv
        .prepare_restore(&HashSet::from([fork]), &mut granted)
        .unwrap();
    for (_, copy) in kv_copies(&txn, false) {
        bytes.copy(&copy);
    }
    assert_eq!(kv.commit_restore(txn).unwrap(), 4);
    assert_eq!(
        swapped(&kv, ws),
        0,
        "the shared pages are back for the parent"
    );
    let parent = table(&mut kv, ws);
    let child = table(&mut kv, fork);
    assert_eq!(&child[..3], &parent[..], "and still shared");
    for (at, &page) in child.iter().enumerate() {
        assert_eq!(bytes.device(LIST, page), Some(200 + at as u64));
    }
    assert_eq!(host_live(&tiers, LIST_POOL), 0);
}

/// The values a record's pieces hold, by shape, in piece order.
fn record_values(shared: &Shared, bytes: &Bytes, record: u32) -> Vec<(EngineShape, Vec<u64>)> {
    shared
        .lock()
        .unwrap()
        .record_refs(record)
        .into_iter()
        .map(|(shape, refs)| {
            let values = refs
                .iter()
                .map(|&r| bytes.device(shape, r).expect("a piece holds its bytes"))
                .collect();
            (shape, values)
        })
        .collect()
}

/// The rs list slot refs `ws`'s buffer holds, in order.
fn buffer_refs(rs: &RsStore, ws: RsWorkingSetId) -> Vec<u32> {
    rs.export(ws)
        .unwrap()
        .buffer
        .iter()
        .flatten()
        .map(|&id| rs.engine_id(id))
        .collect()
}

fn rs_suspend(rs: &mut RsStore, bytes: &mut Bytes, sets: &HashSet<RsWorkingSetId>) -> u32 {
    let txn = rs.prepare_suspend(sets).expect("rows to park");
    for copy in rs_copies(&txn, true) {
        assert_eq!((copy.src, copy.dst), (DEVICE, MemoryDomain::HostPinned));
        bytes.copy(&copy);
    }
    let freed = rs.commit_suspend(txn);
    rs.retire_idle();
    freed.slots + freed.pages
}

fn rs_restore(rs: &mut RsStore, bytes: &mut Bytes, sets: &HashSet<RsWorkingSetId>) -> usize {
    let need = rs.swapped_units(sets);
    let mut grant = rs.reserve(need).expect("room on the device");
    let txn = rs.prepare_restore(sets, &mut grant).unwrap();
    assert!(grant.is_empty(), "the grant is what was parked");
    for copy in rs_copies(&txn, false) {
        assert_eq!((copy.src, copy.dst), (MemoryDomain::HostPinned, DEVICE));
        bytes.copy(&copy);
    }
    rs.commit_restore(txn)
}

/// (c) A record of pieces (three of GDN, one of conv) and the rs list slots
/// of its unfolded rows park in the host tier, a slot of each shape a
/// piece, and come back as a fresh record and fresh rs list slots holding
/// the same bytes in the same order.
#[test]
fn a_record_of_pieces_and_its_rs_list_slots_park_and_come_back() {
    use crate::pipeline::fire::rs::{GroupFold, GroupPlan, RsPlan, prepare, prepare_many, settle};
    use crate::store::rs::RsGeometry;
    let (shared, tiers, stores) = tiered(fixed_facts(), 0);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 16,
    });
    let first = prepare(&mut rs, ws, 5).unwrap();
    settle(&mut rs, first.txn);
    let plan = RsPlan::Groups(vec![GroupPlan {
        lanes: vec![0],
        writes: Some(5..14),
        fold: GroupFold::None,
        snapshots: Vec::new(),
    }]);
    let out = prepare_many(&mut rs, &[ws], &plan, None).unwrap();
    settle(&mut rs, out.txn);
    rs.retire_idle();
    let record = rs.folded_slot(ws).unwrap().unwrap().0;
    let pages = buffer_refs(&rs, ws);
    assert_eq!(pages.len(), 3, "rows 4..14 in slots of 4");
    let mut bytes = Bytes::default();
    let mut value = 0u64;
    for (shape, refs) in shared.lock().unwrap().record_refs(record) {
        for r in refs {
            value += 1;
            bytes.set(shape, r, value);
        }
    }
    for &page in &pages {
        value += 1;
        bytes.set(RS, page, value);
    }
    let state = record_values(&shared, &bytes, record);
    let rows: Vec<u64> = pages
        .iter()
        .map(|&p| bytes.device(RS, p).unwrap())
        .collect();

    let set = HashSet::from([ws]);
    assert_eq!(
        rs_suspend(&mut rs, &mut bytes, &set),
        4,
        "a record and three slots"
    );
    assert!(
        rs.export(ws)
            .unwrap()
            .folded
            .is_some_and(|id| rs.on_host(id))
    );
    let (gdn, conv, rs_pool) = (1, 2, RS_POOL);
    assert_eq!(
        (
            host_live(&tiers, gdn),
            host_live(&tiers, conv),
            host_live(&tiers, rs_pool)
        ),
        (3, 1, 3),
        "a host slot of each shape a piece"
    );
    {
        let pool = shared.lock().unwrap();
        assert_eq!(
            pool.pool().shape_demand(gdn).live,
            0,
            "the device record is free"
        );
        assert_eq!(pool.pool().shape_demand(RS_POOL).live, 0);
    }
    bytes.clobber_device();

    assert_eq!(rs_restore(&mut rs, &mut bytes, &set), 4);
    let back = rs.folded_slot(ws).unwrap().unwrap();
    assert!(!rs.on_host(back));
    assert_eq!(
        record_values(&shared, &bytes, back.0),
        state,
        "every piece, in order"
    );
    let pages = buffer_refs(&rs, ws);
    let back_rows: Vec<u64> = pages
        .iter()
        .map(|&p| bytes.device(RS, p).unwrap())
        .collect();
    assert_eq!(back_rows, rows, "the unfolded rows, in order");
    assert_eq!(
        (host_live(&tiers, gdn), host_live(&tiers, rs_pool)),
        (0, 0),
        "the host slots go back"
    );
}

/// A set is in a move from a suspend's or a restore's prepare to its
/// commit or abort.
#[test]
fn a_set_is_in_a_move_from_prepare_to_commit() {
    use crate::pipeline::fire::rs::{prepare, settle};
    use crate::store::rs::RsGeometry;
    let (_shared, _tiers, stores) = tiered(fixed_facts(), 0);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 16,
    });
    let first = prepare(&mut rs, ws, 4).unwrap();
    settle(&mut rs, first.txn);
    let set = HashSet::from([ws]);
    assert!(!rs.in_move(ws));
    let txn = rs.prepare_suspend(&set).expect("rows to park");
    assert!(rs.in_move(ws));
    rs.commit_suspend(txn);
    assert!(!rs.in_move(ws));
    let need = rs.swapped_units(&set);
    let mut grant = rs.reserve(need).expect("room on the device");
    let txn = rs.prepare_restore(&set, &mut grant).unwrap();
    assert!(rs.in_move(ws));
    rs.abort_restore(txn);
    assert!(!rs.in_move(ws));
}

/// (d) A fork shares its parent's record: parked with it, the record moves
/// once and both hold the host copy (its holders kept); restored, each
/// reads a device record of the same bytes.
#[test]
fn a_forks_shared_record_parks_once_and_keeps_its_holders() {
    use crate::pipeline::fire::rs::{prepare, settle};
    use crate::store::rs::RsGeometry;
    let (shared, tiers, stores) = tiered(fixed_facts(), 0);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(RsGeometry {
        state_size: 1024,
        buffer_page_tokens: 16,
    });
    let first = prepare(&mut rs, ws, 4).unwrap();
    settle(&mut rs, first.txn);
    let fork = rs.fork(ws).unwrap();
    let record = rs.folded_slot(ws).unwrap().unwrap();
    assert_eq!(rs.folded_slot(fork).unwrap(), Some(record));
    let mut bytes = Bytes::default();
    for (shape, refs) in shared.lock().unwrap().record_refs(record.0) {
        for (at, r) in refs.into_iter().enumerate() {
            bytes.set(shape, r, 300 + at as u64 + u64::from(shape.0) * 10);
        }
    }
    let state = record_values(&shared, &bytes, record.0);

    assert!(
        rs.prepare_suspend(&HashSet::from([ws])).is_none(),
        "a record the fork shares stays for the fork"
    );
    let both = HashSet::from([ws, fork]);
    assert_eq!(
        rs_suspend(&mut rs, &mut bytes, &both),
        1,
        "one record, once"
    );
    let parked = rs.folded_slot(ws).unwrap().unwrap();
    assert!(rs.on_host(parked));
    assert_eq!(
        rs.folded_slot(fork).unwrap(),
        Some(parked),
        "both hold the host copy"
    );
    assert_eq!(host_live(&tiers, 1), 3);
    bytes.clobber_device();

    assert_eq!(rs_restore(&mut rs, &mut bytes, &both), 1);
    let back = rs.folded_slot(ws).unwrap().unwrap();
    assert_eq!(
        rs.folded_slot(fork).unwrap(),
        Some(back),
        "still one record for both"
    );
    assert_eq!(record_values(&shared, &bytes, back.0), state);
    assert_eq!(host_live(&tiers, 1), 0);
    // Both let go: the record goes once.
    rs.release_working_set(ws);
    rs.release_working_set(fork);
    rs.retire_idle();
    assert_eq!(shared.lock().unwrap().pool().shape_demand(1).live, 0);
}

/// (e) Two window lists, each of its own slot: a windowed set parks every
/// list's slots in host slots of that list's shape and comes back with its
/// tables (null where trimmed) pointing at the same bytes.
#[test]
fn every_window_list_parks_in_its_own_shape_and_comes_back() {
    use crate::pipeline::fire::rs::settle;
    let mut facts = window_facts();
    facts.host_chunks = HOST_CHUNKS;
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let tiers = Tiers::of(&facts).unwrap();
    let stores = crate::store::registry::shared_stores(&shared, &tiers, 16, 256);
    let mut rs = stores.rs.lock().unwrap();
    let ws = rs.create_working_set(window_geometry());
    let out = window_fire(&mut rs, ws, 0..90, None);
    settle(&mut rs, out.txn);
    rs.retire_idle();
    // List 0 is the narrow one (its window first), list 1 the wide.
    let shapes = [NARROW, RING];
    let before = rs.window_tables(ws, 90).unwrap();
    assert_eq!(before.len(), 2);
    assert!(
        before.iter().all(|table| table.contains(&0)),
        "both trimmed"
    );
    let mut bytes = Bytes::default();
    for (list, table) in before.iter().enumerate() {
        for (at, &r) in table.iter().enumerate().filter(|(_, r)| **r != 0) {
            bytes.set(shapes[list], r, (list as u64 + 1) * 1000 + at as u64);
        }
    }
    let held: Vec<usize> = before
        .iter()
        .map(|table| table.iter().filter(|&&r| r != 0).count())
        .collect();

    let set = HashSet::from([ws]);
    let parked = rs_suspend(&mut rs, &mut bytes, &set);
    assert_eq!(parked as usize, held.iter().sum::<usize>());
    assert_eq!(
        (host_live(&tiers, NARROW_POOL), host_live(&tiers, WIDE_POOL)),
        (held[0] as u64, held[1] as u64),
        "each list in host slots of its own shape"
    );
    assert_eq!(window_live(&shared), (0, 0), "the device slots are free");
    bytes.clobber_device();

    rs_restore(&mut rs, &mut bytes, &set);
    let after = rs.window_tables(ws, 90).unwrap();
    for (list, (old, new)) in before.iter().zip(&after).enumerate() {
        assert_eq!(old.len(), new.len());
        for (at, (&o, &n)) in old.iter().zip(new).enumerate() {
            assert_eq!(o == 0, n == 0, "list {list} entry {at} trimmed alike");
            if n != 0 {
                assert_eq!(
                    bytes.device(shapes[list], n),
                    Some((list as u64 + 1) * 1000 + at as u64)
                );
            }
        }
    }
    assert_eq!(
        (host_live(&tiers, NARROW_POOL), host_live(&tiers, WIDE_POOL)),
        (0, 0)
    );
}

/// (f) A kv list run persisted to disk under the journal: copied by slot
/// ref, taken back by the next run's store from the journal alone, and
/// brought up to fresh device refs with its bytes.
#[test]
fn a_kv_list_run_persists_to_disk_and_a_restart_brings_it_back() {
    use crate::store::chunks::ShapePool;
    use crate::store::kv::hash::{Hash256, chain_slots, page_hash};
    let mut facts = fixed_facts();
    facts.disk_chunks = 1;
    let seed: Hash256 = [7; 32];
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
        assert_eq!(tiers.disk_slots, 16, "a chunk of list slots");
        let list = shared.lock().unwrap().list().unwrap();
        let mut kv = KvStore::with_pool(ShapePool::shared(shared, list), 0, 0, seed);
        kv.set_tiers(list.id, None, tiers.disk_slots);
        kv.set_page_size(16);
        kv.set_keeps(true);
        kv
    };
    let mut bytes = Bytes::default();

    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut kv = store_for(&shared);
    assert_eq!(kv.keep_disk(&journal, true).unwrap(), 0);
    let pages = kv.reserve_device_pages(3).unwrap();
    for (at, page) in pages.iter().enumerate() {
        bytes.set(LIST, page.0, 500 + at as u64);
    }
    assert!(kv.import_cached_at(
        &[],
        tokens.clone(),
        keys.clone(),
        pages.into_iter().map(KvPageBacking::Resident).collect(),
    ));
    let txn = kv.prepare_persist(&keys, 0).expect("three pages to copy");
    assert_eq!(txn.shape(), Some(LIST));
    let (device, slots) = txn.copy_plan();
    let copy = TierMove {
        kind: TierKind::kv(txn.shape()),
        tier: MemoryDomain::LocalDisk,
        out: true,
        device,
        slots,
        source: None,
    }
    .slot_copy(DEVICE)
    .unwrap();
    assert_eq!(
        (copy.src, copy.dst, copy.shape),
        (DEVICE, MemoryDomain::LocalDisk, LIST)
    );
    assert!(copy.slots.iter().all(|&(_, disk)| disk < 16));
    bytes.copy(&copy);
    assert_eq!(kv.commit_persist(txn), 3);
    drop(kv);
    drop(shared);
    bytes.clobber_device();

    // The next run: a fresh pool and store, the journal and the slot file.
    let shared: Shared = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let mut kv = store_for(&shared);
    assert_eq!(
        kv.keep_disk(&journal, true).unwrap(),
        3,
        "the run is taken back"
    );
    let run = kv.cached_run_any_tier(&keys);
    assert_eq!(
        (run.len(), kv.below_device(&run)),
        (3, 3),
        "all of it on disk"
    );
    let mut granted: Vec<PhysicalKvPageId> = kv.reserve_device_pages(3).unwrap();
    let txn = kv.prepare_promote(&run, &mut granted).unwrap().unwrap();
    for (tier, copy) in kv_copies(&txn, false) {
        assert_eq!(tier, MemoryDomain::LocalDisk);
        bytes.copy(&copy);
    }
    let pin = kv.commit_promote(txn).unwrap();
    let run = kv.cached_run(&keys);
    assert_eq!(run.len(), 3, "resident again");
    let resident: Vec<u32> = run
        .iter()
        .map(|location| match kv.table().backing_at(location).unwrap() {
            KvPageBacking::Resident(page) => page.0,
            other => panic!("{other:?}"),
        })
        .collect();
    for (at, page) in resident.into_iter().enumerate() {
        assert_eq!(bytes.device(LIST, page), Some(500 + at as u64));
    }
    kv.release_reader(pin);
    assert_eq!(kv.persisted_pages(), 3, "the copies stay");
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
    let txn = kv.prepare_demote(64, |_| 0).unwrap().expect("a demote");
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
    let txn = kv.prepare_spill(1, |_| 0).expect("a spill");
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
