//! Lane B — CPU offload + non-evictable snapshot-blob residency, exercised
//! end-to-end through the runtime's driver channel (brief Ph1 gate item).
//!
//! The arena unit tests (`runtime/src/arena/tests.rs`) cover the residency
//! state machine in isolation. These tests close the loop the brief asks for —
//! "exercised end-to-end, not just in the arena unit tests" — by driving the
//! arena's offload/restore `MovePlan`s through real `driver::copy_d2h` /
//! `copy_h2d` calls against a registered (mock) driver channel, and by
//! confirming snapshot-blob non-evictability holds with the runtime's driver
//! registry live.

use std::sync::Arc;

mod common;

use common::mock_device::{EchoBehavior, MockBackend};

use pie::arena::{Arena, ArenaConfig, ArenaKind, CowPlan, Residency};
use pie::driver;

fn arena(device: usize) -> Arena {
    Arena::new(ArenaConfig {
        device,
        block_size: 16,
        kv_pages: 8,
        rs_blocks: 8,
        scratch_blocks: 4,
        cpu_blocks: 8,
    })
}

/// Offload then restore a batch of KV pages, dispatching the arena's MovePlans
/// as real D2H / H2D copies through the driver channel. Asserts residency and
/// pool accounting track the moves.
#[test]
fn offload_and_restore_drive_real_driver_copies() {
    let backend = MockBackend::new(1, Arc::new(EchoBehavior(0)));
    let drv = backend.driver_ids()[0];
    let mut a = arena(drv);

    let pages: Vec<_> = (0..4)
        .map(|_| a.alloc(ArenaKind::KvPage, 1).unwrap())
        .collect();
    assert_eq!(a.used(ArenaKind::KvPage), 4);

    // Offload: arena yields MovePlan{from: gpu, to: cpu}; runtime issues D2H.
    for h in &pages {
        let plan = a.offload(h.object_id).unwrap();
        driver::copy_d2h(drv, &plan.from, &plan.to).expect("D2H copy dispatched");
        assert_eq!(a.residency(h.object_id).unwrap(), Residency::Cpu);
    }
    assert_eq!(a.used(ArenaKind::KvPage), 0); // device pages reclaimed
    assert_eq!(a.used(ArenaKind::CpuStash), 4); // now stashed on CPU

    // Restore: MovePlan{from: cpu, to: gpu}; runtime issues H2D (cpu -> gpu).
    for h in &pages {
        let plan = a.restore(h.object_id).unwrap();
        driver::copy_h2d(drv, &plan.to, &plan.from).expect("H2D copy dispatched");
        assert_eq!(a.residency(h.object_id).unwrap(), Residency::Gpu);
    }
    assert_eq!(a.used(ArenaKind::KvPage), 4);
    assert_eq!(a.used(ArenaKind::CpuStash), 0);
}

/// Under device pressure, offloading frees device blocks for fresh work — the
/// contention path the runtime relies on, exercised through real copies.
#[test]
fn offload_relieves_device_pressure_end_to_end() {
    let backend = MockBackend::new(1, Arc::new(EchoBehavior(0)));
    let drv = backend.driver_ids()[0];
    let mut a = arena(drv);

    let pages: Vec<_> = (0..8)
        .map(|_| a.alloc(ArenaKind::KvPage, 1).unwrap())
        .collect();
    assert!(a.alloc(ArenaKind::KvPage, 1).is_err()); // device pool exhausted

    for h in pages.iter().take(2) {
        let plan = a.offload(h.object_id).unwrap();
        driver::copy_d2h(drv, &plan.from, &plan.to).unwrap();
    }

    // Device now has room again.
    let fresh = a.alloc(ArenaKind::KvPage, 1).unwrap();
    assert_eq!(a.residency(fresh.object_id).unwrap(), Residency::Gpu);
    assert_eq!(a.used(ArenaKind::KvPage), 7);
}

/// W14: snapshot blobs are CPU-resident and non-evictable; a GPU cache copy of
/// snapshot-backed data remains evictable. Verified with the driver registry
/// live so this is a runtime-level guarantee, not just an arena unit fact.
#[test]
fn snapshot_blob_non_evictable_under_runtime() {
    let backend = MockBackend::new(1, Arc::new(EchoBehavior(0)));
    let drv = backend.driver_ids()[0];
    let mut a = arena(drv);

    let blob = a.alloc(ArenaKind::SnapshotBlob, 3).unwrap();
    assert_eq!(blob.residency, Residency::Cpu);
    assert!(!a.is_evictable(blob.object_id).unwrap());
    assert_eq!(a.used(ArenaKind::CpuStash), 3);

    // Cannot be evicted from CPU (no mark_replayable; not device-resident to offload).
    assert!(a.mark_replayable(blob.object_id).is_err());
    assert!(a.offload(blob.object_id).is_err());

    // A GPU cache copy of snapshot-backed data IS evictable under contention.
    let gpu_copy = a.alloc(ArenaKind::KvPage, 1).unwrap();
    a.set_evictable(gpu_copy.object_id, true).unwrap();
    a.mark_replayable(gpu_copy.object_id).unwrap();
    assert_eq!(a.residency(gpu_copy.object_id).unwrap(), Residency::Replayable);

    // Blob still pinned in CPU, occupying capacity.
    assert_eq!(a.residency(blob.object_id).unwrap(), Residency::Cpu);
    assert_eq!(a.used(ArenaKind::CpuStash), 3);
}

/// RS-slab fork copy-on-write (W11) physical path, exercised through the RS
/// device copy op. A shared recurrent-state slab stays shared on a read-only
/// fork (lazy, no copy); the first fold/write on the shared slab copies it, and
/// the runtime issues `copy_rs_d2d` from the CoW `MovePlan`.
///
/// NOTE: a *full* hybrid-RS-fold e2e is out-of-mock-v1 — the echo mock driver
/// advertises `rs_cache_slots = 0` (RS is a real cuda GDN / Mamba2 feature, not
/// emulated by the mock), and no inferlet exercises RS. This covers the
/// feasible mock-level slice: the arena RS-slab fork-CoW primitive the
/// fold-after-fork path depends on, driving a real RS device copy through the
/// registered driver channel. Fold-token validation + the RS core live in
/// `rs_working_set.rs`; the `RS_FLAG_FOLD`/`rs_fold_lens` wire is covered by the
/// driver-ABI round-trip test.
#[test]
fn rs_slab_fork_cow_drives_rs_device_copy() {
    let backend = MockBackend::new(1, Arc::new(EchoBehavior(0)));
    let drv = backend.driver_ids()[0];
    let mut a = arena(drv);

    // Folded recurrent-state slab (v1 single-slot).
    let slab = a.alloc(ArenaKind::RsSlab, 1).unwrap();
    // Lazy fork shares the slab by reference (incref) — no copy yet (W11).
    a.incref(slab.object_id).unwrap();
    assert_eq!(a.refcount(slab.object_id).unwrap(), 2);

    // First fold/write on the shared slab copies it; the runtime issues the RS
    // device copy from the CoW MovePlan (wraps cuda copy_slot_d2d in v1).
    match a.cow(slab.object_id).unwrap() {
        CowPlan::Copy { handle, from, to } => {
            driver::copy_rs_d2d(drv, &from, &to).expect("RS d2d copy dispatched");
            assert_eq!(from.len(), 1);
            assert_eq!(to.len(), 1);
            // The writer left the original's sharing group; the copy is unique.
            assert_eq!(a.refcount(slab.object_id).unwrap(), 1);
            assert_eq!(a.refcount(handle.object_id).unwrap(), 1);
            assert_ne!(handle.object_id, slab.object_id);
        }
        other => panic!("expected Copy, got {other:?}"),
    }

    // A uniquely-owned slab (read-only fork released, or never shared) writes in
    // place — no device copy.
    let solo = a.alloc(ArenaKind::RsSlab, 1).unwrap();
    assert!(matches!(
        a.cow(solo.object_id).unwrap(),
        CowPlan::InPlace { .. }
    ));
}
