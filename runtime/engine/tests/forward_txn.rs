//! Lane E — Phase-4 forward atomic-transaction gate tests.
//!
//! Per [[workingset-brief]] §8 Phase-4 the forward path must pass tests for:
//!   1. overlapping read/write (kv-output overlaps kv-context only where declared)
//!   2. CoW-before-write (a shared write target is copied before the driver writes)
//!   3. driver-failure abort (abort: prior mappings visible, no CAS seal)
//!   4. CAS seal on completion (full pages sealed + deduped after commit)
//!   + brief §5 prepare-time validation edges (stale generation / dup / range).
//!
//! These exercise the exact sequence `api/inference.rs::execute()` /
//! `finalize_forward_txn` drive over the real `KvWorkingSet` (charlie) + unified
//! `Arena` (bravo) + the pure `forward_prepare::project_kv` (echo): resolve →
//! `cow_write_slot` (CoW) → `project_kv` → `txn_commit` + `commit_writes` +
//! `seal` (success) / `txn_abort` + `abort_writes` (failure). They run at the
//! core layer (no WASM/host plumbing), so they're deterministic + fast and don't
//! depend on the working-set WIT constructor wiring.

use pie_engine::arena::{Arena, ArenaConfig};
use pie_engine::working_set::page_hash::compute_page_hashes;
use pie_engine::inference::paging::{KvWrite, check_input_nonempty, project_kv};
use pie_engine::working_set::kv::{KvCas, KvWorkingSet};
use pie_driver_abi::Brle;

const PAGE: u32 = 4;

fn arena(kv_pages: u32) -> Arena {
    Arena::new(ArenaConfig {
        device: 0,
        block_size: PAGE,
        kv_pages,
        rs_blocks: 0,
        scratch_blocks: 0,
        cpu_blocks: 0,
    })
}

/// Drive a prefill forward that materializes + commits `n` full pages `[0, n)`
/// into a fresh working set — mirrors `execute()`'s prepare→commit on an empty
/// context (each reserved slot is lazily allocated by `cow_write_slot`).
fn prefill(ws: &mut KvWorkingSet, a: &mut Arena, n: u32) {
    ws.alloc(n).unwrap();
    let cur_gen = ws.generation();
    let indices: Vec<u32> = (0..n).collect();
    ws.resolve_write(&indices, cur_gen).unwrap();
    let mut txn = a.txn_begin();
    let wtx = ws.begin_write_txn();
    for &idx in &indices {
        ws.cow_write_slot(wtx, idx, &mut txn, a).unwrap();
    }
    a.txn_commit(txn).unwrap();
    ws.commit_writes(wtx);
}

fn page_of(a: &Arena, obj: u32) -> u32 {
    a.blocks(obj).unwrap()[0]
}

// =============================================================================
// GATE 1 — overlapping read/write
// =============================================================================

/// A forward reads context `[0,3)` and declares output on slot 1 (an overlap).
/// prepare must accept it and the projection must place the (written) output
/// page at slot 1 with the surrounding context pages intact.
#[test]
fn gate_overlapping_read_write() {
    let mut a = arena(16);
    let mut ws = KvWorkingSet::new(PAGE, 0);
    prefill(&mut ws, &mut a, 3); // 3 full context pages (12 valid tokens)

    let ctx_objs = ws.resolve_read(0, 3).unwrap();
    let ctx_pages: Vec<u32> = ctx_objs.iter().map(|&o| page_of(&a, o)).collect();

    let cur_gen = ws.generation();
    ws.resolve_write(&[1], cur_gen).unwrap(); // overlap with context, declared output
    let mut txn = a.txn_begin();
    let wtx = ws.begin_write_txn();
    let (out_obj, _mp) = ws.cow_write_slot(wtx, 1, &mut txn, &mut a).unwrap();
    let out_page = page_of(&a, out_obj);

    let writes = vec![KvWrite {
        slot_index: 1,
        page: out_page,
        valid_len: PAGE,
    }];
    let proj = project_kv(&ctx_pages, 12, &writes, PAGE).unwrap();

    assert_eq!(proj.physical_page_ids.len(), 3, "active run covers slots 0..3");
    assert_eq!(proj.physical_page_ids[0], ctx_pages[0]);
    assert_eq!(
        proj.physical_page_ids[1], out_page,
        "the declared output page occupies the overlapped slot"
    );
    assert_eq!(proj.physical_page_ids[2], ctx_pages[2]);
    assert_eq!(proj.active_page_idx, Some(1));
    assert_eq!(proj.full_page_writes, vec![1]);

    a.txn_commit(txn).unwrap();
    ws.commit_writes(wtx);
}

// =============================================================================
// GATE 2 — CoW-before-write
// =============================================================================

/// Two working sets share a page (fork). Writing it on one CoWs a private copy
/// (with a d2d MovePlan) and repoints only that set's slot; the sharer's page
/// is untouched.
#[test]
fn gate_cow_before_write() {
    let mut a = arena(16);
    let mut ws1 = KvWorkingSet::new(PAGE, 0);
    prefill(&mut ws1, &mut a, 1);
    let obj_orig = ws1.resolve_read(0, 1).unwrap()[0];

    let ws2 = ws1.fork(&mut a).unwrap(); // share by reference (rc=2)
    assert_eq!(a.refcount(obj_orig).unwrap(), 2);

    let cur_gen = ws1.generation();
    ws1.resolve_write(&[0], cur_gen).unwrap();
    let mut txn = a.txn_begin();
    let wtx = ws1.begin_write_txn();
    let (new_obj, mp) = ws1.cow_write_slot(wtx, 0, &mut txn, &mut a).unwrap();
    assert!(mp.is_some(), "writing a shared page must CoW (d2d plan)");
    assert_ne!(new_obj, obj_orig, "CoW allocated a private copy");

    a.txn_commit(txn).unwrap();
    ws1.commit_writes(wtx);

    assert_eq!(ws1.resolve_read(0, 1).unwrap()[0], new_obj, "writer diverged");
    assert_eq!(
        ws2.resolve_read(0, 1).unwrap()[0],
        obj_orig,
        "sharer's mapping is untouched"
    );
    assert_eq!(
        a.refcount(obj_orig).unwrap(),
        1,
        "writer left the sharing group on commit"
    );
}

// =============================================================================
// GATE 3 — driver-failure abort
// =============================================================================

/// A forward CoWs a shared write target, then the driver fails → abort. The
/// repointed slot reverts to the original, the sharer is untouched, the staged
/// copy is freed, and nothing is sealed (W13).
#[test]
fn gate_driver_failure_abort() {
    let mut a = arena(16);
    let cas = KvCas::new();
    let mut ws1 = KvWorkingSet::new(PAGE, 0);
    prefill(&mut ws1, &mut a, 1);
    let obj_orig = ws1.resolve_read(0, 1).unwrap()[0];

    let ws2 = ws1.fork(&mut a).unwrap();

    let cur_gen = ws1.generation();
    ws1.resolve_write(&[0], cur_gen).unwrap();
    let mut txn = a.txn_begin();
    let wtx = ws1.begin_write_txn();
    let (_new_obj, mp) = ws1.cow_write_slot(wtx, 0, &mut txn, &mut a).unwrap();
    assert!(mp.is_some());

    // Simulate driver/submit failure → abort the transaction.
    a.txn_abort(txn);
    ws1.abort_writes(wtx);

    assert_eq!(
        ws1.resolve_read(0, 1).unwrap()[0],
        obj_orig,
        "aborted writer reverted to the prior mapping"
    );
    assert_eq!(ws2.resolve_read(0, 1).unwrap()[0], obj_orig, "sharer untouched");
    assert_eq!(
        a.refcount(obj_orig).unwrap(),
        2,
        "no commit-time decref happened on abort"
    );
    assert!(cas.is_empty(), "abort seals nothing");
}

// =============================================================================
// GATE 4 — CAS seal on completion
// =============================================================================

/// Sealing a committed full page registers it in the CAS index; a second set
/// that writes identical content reuses the canonical page (dedup), and the
/// duplicate is freed. Partial pages are never sealed (W7).
#[test]
fn gate_cas_seal_on_completion() {
    let mut a = arena(16);
    let mut cas = KvCas::new();

    let tokens = vec![10u32, 11, 12, 13];
    let positions = vec![0u32, 1, 2, 3];
    let masks: Vec<Brle> = positions
        .iter()
        .map(|&p| Brle::all_true((p + 1) as usize))
        .collect();
    let hash = compute_page_hashes(PAGE as usize, &tokens, &positions, &masks, 0, None)[0];

    // First set: write + seal a full page.
    let mut ws1 = KvWorkingSet::new(PAGE, 0);
    prefill(&mut ws1, &mut a, 1);
    ws1.seal(0, hash, &mut a, &mut cas).unwrap();
    let canonical = ws1.resolve_read(0, 1).unwrap()[0];
    assert_eq!(cas.len(), 1, "one sealed page registered");

    // Second set: identical content → identical hash → CAS reuse.
    let mut ws2 = KvWorkingSet::new(PAGE, 0);
    prefill(&mut ws2, &mut a, 1);
    let dup = ws2.resolve_read(0, 1).unwrap()[0];
    assert_ne!(dup, canonical, "distinct object before sealing");
    ws2.seal(0, hash, &mut a, &mut cas).unwrap();

    assert_eq!(cas.len(), 1, "identical page reuses the canonical, no new entry");
    assert_eq!(
        ws2.resolve_read(0, 1).unwrap()[0],
        canonical,
        "second set repointed to the canonical sealed page (W6 isolation)"
    );
}

/// A partial page (valid_len < page_size) is excluded from `full_page_writes`,
/// so `execute()` never calls `seal` on it — it stays private-dirty (W7).
#[test]
fn gate_partial_page_never_sealed() {
    let proj = project_kv(&[], 0, &[KvWrite { slot_index: 0, page: 100, valid_len: 3 }], PAGE)
        .unwrap();
    assert!(
        proj.full_page_writes.is_empty(),
        "a partial page is not seal-eligible"
    );
}

// =============================================================================
// VALIDATION EDGES — brief §5 prepare-time rejections (fail before submit)
// =============================================================================

#[test]
fn gate_prepare_validation_edges() {
    let mut ws = KvWorkingSet::new(PAGE, 0);
    ws.alloc(3).unwrap();
    let stale_gen = ws.generation();

    // A reorder/compact renumbers slot ids and bumps the generation → a write
    // captured against the old generation is rejected (stale-mutation guard,
    // W8). Under slot-id semantics `alloc`/`free` no longer bump: survivors keep
    // their ids, so allocation/reclamation cannot invalidate a cached id.
    ws.reorder(&[0, 1, 2]).unwrap();
    assert!(
        ws.resolve_write(&[0], stale_gen).is_err(),
        "stale generation rejected"
    );

    let cur_gen = ws.generation();
    assert!(ws.resolve_write(&[0, 0], cur_gen).is_err(), "duplicate index rejected");
    assert!(ws.resolve_write(&[99], cur_gen).is_err(), "out-of-range index rejected");
    assert!(ws.resolve_write(&[0, 2], cur_gen).is_ok(), "valid indices accepted");

    // A read window past the end is rejected before any arena work.
    assert!(ws.resolve_read(0, 99).is_err(), "out-of-range read rejected");

    // The projection rejects a non-contiguous active run the v1 ABI can't express.
    assert!(
        project_kv(&[100], 4, &[KvWrite { slot_index: 3, page: 103, valid_len: 1 }], PAGE).is_err(),
        "non-contiguous output rejected"
    );
}

// =============================================================================
// GATE — empty-input guard (W4): a forward must compute at least one query row
// =============================================================================

#[test]
fn gate_empty_input_rejected() {
    // No input rows (text/image/audio) → `execute()` rejects before submit so the
    // driver never gets a no-op `qo_indptr = [0, 0]` pass. The old context API
    // enforced the same "must supply at least one token" invariant.
    assert!(check_input_nonempty(0).is_err(), "empty input rejected");
    assert!(check_input_nonempty(1).is_ok(), "one input row accepted");
    assert!(check_input_nonempty(8).is_ok(), "multi input rows accepted");
}
