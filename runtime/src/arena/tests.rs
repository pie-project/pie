//! Arena unit tests — the Phase-1 gate.
//!
//! Covers alloc/free, copy-on-write, CPU residency (offload/restore),
//! non-evictable snapshot blobs, and transaction abort-cleanup, plus refcount
//! sharing and RS-slab CoW (the primitive delta's RS fork reuses).

use super::*;

fn arena() -> Arena {
    Arena::new(ArenaConfig {
        device: 0,
        block_size: 16,
        kv_pages: 8,
        rs_blocks: 8,
        scratch_blocks: 4,
        cpu_blocks: 8,
    })
}

// ---------------------------------------------------------------------------
// alloc / free
// ---------------------------------------------------------------------------

#[test]
fn alloc_kv_page_resolves_to_one_block() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    assert_eq!(h.kind, ArenaKind::KvPage);
    assert_eq!(h.device, 0);
    assert_eq!(h.size_blocks, 1);
    assert_eq!(h.residency, Residency::Gpu);
    assert_eq!(a.blocks(h.object_id).unwrap().len(), 1);
    assert_eq!(a.used(ArenaKind::KvPage), 1);
    assert_eq!(a.available(ArenaKind::KvPage), 7);
    assert_eq!(a.capacity(ArenaKind::KvPage), 8);
}

#[test]
fn kv_page_must_be_one_block() {
    let mut a = arena();
    assert!(matches!(
        a.alloc(ArenaKind::KvPage, 2),
        Err(ArenaError::BadSize { .. })
    ));
}

#[test]
fn alloc_exhaustion_reports_out_of_blocks() {
    let mut a = arena();
    for _ in 0..8 {
        a.alloc(ArenaKind::KvPage, 1).unwrap();
    }
    assert!(matches!(
        a.alloc(ArenaKind::KvPage, 1),
        Err(ArenaError::OutOfBlocks {
            kind: ArenaKind::KvPage,
            ..
        })
    ));
}

#[test]
fn decref_reclaims_blocks_and_recycles_object() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    assert_eq!(a.live_objects(), 1);
    a.decref(h.object_id).unwrap();
    assert_eq!(a.live_objects(), 0);
    assert_eq!(a.available(ArenaKind::KvPage), 8);
    assert!(matches!(
        a.residency(h.object_id),
        Err(ArenaError::UnknownObject(_))
    ));
}

// ---------------------------------------------------------------------------
// refcount sharing
// ---------------------------------------------------------------------------

#[test]
fn incref_keeps_object_alive_until_last_decref() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.incref(h.object_id).unwrap();
    assert_eq!(a.refcount(h.object_id).unwrap(), 2);
    a.decref(h.object_id).unwrap();
    assert_eq!(a.refcount(h.object_id).unwrap(), 1);
    assert_eq!(a.live_objects(), 1);
    a.decref(h.object_id).unwrap();
    assert_eq!(a.live_objects(), 0);
}

// ---------------------------------------------------------------------------
// copy-on-write
// ---------------------------------------------------------------------------

#[test]
fn cow_is_in_place_when_uniquely_owned() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    match a.cow(h.object_id).unwrap() {
        CowPlan::InPlace { handle } => assert_eq!(handle.object_id, h.object_id),
        other => panic!("expected InPlace, got {other:?}"),
    }
    assert_eq!(a.live_objects(), 1);
}

#[test]
fn cow_copies_when_shared_and_leaves_other_sharers_mapped() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.incref(h.object_id).unwrap(); // simulate a second sharer
    let original_blocks = a.blocks(h.object_id).unwrap().to_vec();

    match a.cow(h.object_id).unwrap() {
        CowPlan::Copy { handle, from, to } => {
            assert_eq!(from, original_blocks);
            assert_eq!(to, a.blocks(handle.object_id).unwrap().to_vec());
            assert_ne!(handle.object_id, h.object_id);
            // The private copy is uniquely owned; the original keeps the
            // remaining sharer (refcount dropped from 2 -> 1).
            assert_eq!(a.refcount(handle.object_id).unwrap(), 1);
            assert_eq!(a.refcount(h.object_id).unwrap(), 1);
        }
        other => panic!("expected Copy, got {other:?}"),
    }
    assert_eq!(a.live_objects(), 2);
}

#[test]
fn cow_requires_device_residency() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.offload(h.object_id).unwrap();
    assert!(matches!(
        a.cow(h.object_id),
        Err(ArenaError::InvalidResidency { .. })
    ));
}

#[test]
fn rs_slab_cow_copies_all_blocks() {
    // The exact primitive delta's RS fork CoW reuses: a shared multi-block
    // slab copies every block.
    let mut a = arena();
    let h = a.alloc(ArenaKind::RsSlab, 3).unwrap();
    assert_eq!(a.blocks(h.object_id).unwrap().len(), 3);
    a.incref(h.object_id).unwrap();
    match a.cow(h.object_id).unwrap() {
        CowPlan::Copy { from, to, .. } => {
            assert_eq!(from.len(), 3);
            assert_eq!(to.len(), 3);
        }
        other => panic!("expected Copy, got {other:?}"),
    }
}

#[test]
fn copy_duplicates_unconditionally_without_touching_refcounts() {
    // charlie's `PageBackend::copy`: the CAS layer owns refcounts and asks the
    // arena for a fresh physical duplicate. delta wraps the MovePlan with
    // copy_slot_d2d for RS fork.
    let mut a = arena();
    let src = a.alloc(ArenaKind::KvPage, 1).unwrap();
    let from = a.blocks(src.object_id).unwrap().to_vec();
    let (dup, plan) = a.copy(src.object_id).unwrap();
    assert_ne!(dup.object_id, src.object_id);
    assert_eq!(plan.from, from);
    assert_eq!(plan.to, a.blocks(dup.object_id).unwrap().to_vec());
    // Unconditional: refcounts of both src and dup remain 1.
    assert_eq!(a.refcount(src.object_id).unwrap(), 1);
    assert_eq!(a.refcount(dup.object_id).unwrap(), 1);
}

// ---------------------------------------------------------------------------
// CPU residency (offload / restore)
// ---------------------------------------------------------------------------

#[test]
fn offload_then_restore_roundtrip() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    let gpu_blocks = a.blocks(h.object_id).unwrap().to_vec();

    let plan = a.offload(h.object_id).unwrap();
    assert_eq!(plan.from, gpu_blocks);
    assert_eq!(a.residency(h.object_id).unwrap(), Residency::Cpu);
    assert_eq!(a.available(ArenaKind::KvPage), 8); // gpu block returned
    assert_eq!(a.used(ArenaKind::CpuStash), 1); // cpu block taken
    assert_eq!(a.blocks(h.object_id).unwrap().to_vec(), plan.to);

    let restore = a.restore(h.object_id).unwrap();
    assert_eq!(restore.from, plan.to); // cpu -> gpu
    assert_eq!(a.residency(h.object_id).unwrap(), Residency::Gpu);
    assert_eq!(a.used(ArenaKind::CpuStash), 0);
    assert_eq!(a.available(ArenaKind::KvPage), 7);
}

#[test]
fn offload_refuses_pinned_object() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.pin(h.object_id).unwrap();
    assert!(matches!(
        a.offload(h.object_id),
        Err(ArenaError::Pinned(_))
    ));
    a.unpin(h.object_id).unwrap();
    assert!(a.offload(h.object_id).is_ok());
}

#[test]
fn replayable_release_then_reserve_gpu() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.mark_replayable(h.object_id).unwrap();
    assert_eq!(a.residency(h.object_id).unwrap(), Residency::Replayable);
    assert!(a.blocks(h.object_id).unwrap().is_empty());
    assert_eq!(a.available(ArenaKind::KvPage), 8);

    let blocks = a.reserve_gpu(h.object_id).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(a.residency(h.object_id).unwrap(), Residency::Gpu);
}

// ---------------------------------------------------------------------------
// non-evictable snapshot blobs (W14)
// ---------------------------------------------------------------------------

#[test]
fn snapshot_blob_is_cpu_resident_and_non_evictable() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::SnapshotBlob, 2).unwrap();
    assert_eq!(h.residency, Residency::Cpu);
    assert!(!a.is_evictable(h.object_id).unwrap());
    assert_eq!(a.used(ArenaKind::CpuStash), 2);
    // A CPU-resident blob cannot be dropped to replay.
    assert!(matches!(
        a.mark_replayable(h.object_id),
        Err(ArenaError::NotEvictable(_))
    ));
}

#[test]
fn snapshot_backed_gpu_copy_is_evictable() {
    // GPU cache copies of snapshot-backed data may be evicted under contention.
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.set_evictable(h.object_id, true).unwrap();
    assert!(a.mark_replayable(h.object_id).is_ok());
    assert_eq!(a.residency(h.object_id).unwrap(), Residency::Replayable);
}

// ---------------------------------------------------------------------------
// transactions
// ---------------------------------------------------------------------------

#[test]
fn txn_commit_retains_allocs_and_releases_pins() {
    let mut a = arena();
    let mut txn = a.txn_begin();
    let out = a.txn_alloc(&mut txn, ArenaKind::KvPage, 1).unwrap();
    a.txn_pin(&mut txn, out.object_id).unwrap();
    a.txn_mark_write(&mut txn, out.object_id).unwrap();

    let report = a.txn_commit(txn).unwrap();
    assert_eq!(report.write_targets, vec![out.object_id]);
    assert_eq!(a.live_objects(), 1);
    assert!(!a.is_pinned(out.object_id).unwrap());
}

#[test]
fn txn_abort_frees_staged_objects_and_pins() {
    let mut a = arena();
    let baseline = a.available(ArenaKind::KvPage);

    let mut txn = a.txn_begin();
    let o1 = a.txn_alloc(&mut txn, ArenaKind::KvPage, 1).unwrap();
    let o2 = a.txn_alloc(&mut txn, ArenaKind::KvPage, 1).unwrap();
    a.txn_pin(&mut txn, o1.object_id).unwrap();
    assert_eq!(a.live_objects(), 2);

    a.txn_abort(txn);
    assert_eq!(a.live_objects(), 0);
    assert_eq!(a.available(ArenaKind::KvPage), baseline);
    assert!(matches!(
        a.residency(o2.object_id),
        Err(ArenaError::UnknownObject(_))
    ));
}

#[test]
fn txn_cow_commit_publishes_new_mapping() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.incref(h.object_id).unwrap(); // a second sharer exists

    let mut txn = a.txn_begin();
    let copy = match a.txn_cow(&mut txn, h.object_id).unwrap() {
        CowPlan::Copy { handle, .. } => handle.object_id,
        other => panic!("expected Copy, got {other:?}"),
    };
    // Mid-txn (before commit) the original still carries both sharers.
    assert_eq!(a.refcount(h.object_id).unwrap(), 2);

    let report = a.txn_commit(txn).unwrap();
    assert_eq!(report.write_targets, vec![copy]);
    // Commit folds the writer out of the original's sharing group.
    assert_eq!(a.refcount(h.object_id).unwrap(), 1);
    assert_eq!(a.refcount(copy).unwrap(), 1);
}

#[test]
fn txn_cow_abort_discards_copy_and_keeps_original() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.incref(h.object_id).unwrap(); // rc 2
    let live_before = a.live_objects();

    let mut txn = a.txn_begin();
    let copy = match a.txn_cow(&mut txn, h.object_id).unwrap() {
        CowPlan::Copy { handle, .. } => handle.object_id,
        other => panic!("expected Copy, got {other:?}"),
    };
    assert_eq!(a.live_objects(), live_before + 1);

    a.txn_abort(txn);
    // Copy discarded; the original mapping survives untouched.
    assert!(matches!(
        a.residency(copy),
        Err(ArenaError::UnknownObject(_))
    ));
    assert_eq!(a.refcount(h.object_id).unwrap(), 2);
    assert_eq!(a.live_objects(), live_before);
}

#[test]
fn txn_pin_blocks_offload_until_finished() {
    let mut a = arena();
    let h = a.alloc(ArenaKind::KvPage, 1).unwrap();

    let mut txn = a.txn_begin();
    a.txn_pin(&mut txn, h.object_id).unwrap();
    assert!(matches!(
        a.offload(h.object_id),
        Err(ArenaError::Pinned(_))
    ));

    a.txn_commit(txn).unwrap();
    assert!(a.offload(h.object_id).is_ok());
}
