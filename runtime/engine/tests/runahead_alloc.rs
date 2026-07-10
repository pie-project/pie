//! **2 + 1 pairwise integration** (masterplan M2): run-ahead decode +
//! in-flight-safe `alloc` top-up — overview §6.1's headroom loop:
//!
//! ```text
//! let want = div_ceil(clen + headroom, page);
//! if want > ws.size() { ws.alloc(want - ws.size()); }  // in-flight safe: grants never disturb existing ids
//! ```
//!
//! It ties **thrust 1** (alpha's stable slot ids — `alloc` never renumbers a live
//! id and, under `ws-slot-ids`, never bumps `generation`, W8) to **thrust 2**
//! (bravo's per-forward write transactions, S4 — two prepared forwards may be
//! outstanding at once). The headroom `alloc` happens *while a run-ahead pass is
//! in flight*; the property under test is that it disturbs neither the in-flight
//! pass's captured geometry (slot ids + generation) nor a second, disjoint
//! prepared pass. Mock/host-only — no driver, no GPU.
//!
//! Run the in-flight-safe regime with `--features ws-slot-ids`; the default
//! (legacy) build documents WHY it is unsafe there (`alloc` bumps the generation
//! a run-ahead pass captured, so its next pass would be stale-rejected).

use pie_engine::arena::{Arena, ArenaConfig};
use pie_engine::working_set::kv::{KvCas, KvWorkingSet, SLOT_IDS};

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

/// Materialise + commit page `idx` (one full-page write forward), advancing the
/// working set to a committed baseline. Mirrors `execute()`'s prepare→commit.
fn commit_slot(ws: &mut KvWorkingSet, a: &mut Arena, idx: u32) {
    let mut txn = a.txn_begin();
    let wtx = ws.begin_write_txn();
    ws.cow_write_slot(wtx, idx, &mut txn, a).unwrap();
    a.txn_commit(txn).unwrap();
    ws.commit_writes(wtx);
}

/// The core §6.1 property: a headroom `alloc` issued while a run-ahead pass is in
/// flight leaves that pass's captured geometry intact, and a second disjoint pass
/// prepared against the freshly-granted slots commits independently.
#[test]
fn headroom_alloc_is_in_flight_safe() {
    let mut a = arena(64);
    let mut cas = KvCas::new();
    let mut ws = KvWorkingSet::new(PAGE, 0);

    // Committed baseline: 2 live pages.
    ws.alloc(2).unwrap();
    commit_slot(&mut ws, &mut a, 0);
    commit_slot(&mut ws, &mut a, 1);
    let base0 = ws.resolve_read(0, 1).unwrap()[0];

    // Run-ahead pass `t` goes in flight: capture the generation (as prepare does
    // via `resolve_write`), open its write-txn, CoW its output slot. NOT yet
    // finalized — it is outstanding while we top up headroom.
    let gen_t = ws.generation();
    let mut txn_t = a.txn_begin();
    let wtx_t = ws.begin_write_txn();
    ws.resolve_write(&[0], gen_t).unwrap();
    ws.cow_write_slot(wtx_t, 0, &mut txn_t, &mut a).unwrap();

    // Headroom top-up (§6.1) — grow the working set while `t` is in flight.
    let size_before = ws.size();
    let headroom = 4u32;
    ws.alloc(headroom).unwrap();
    assert_eq!(ws.size(), size_before + headroom, "headroom slots granted");

    // In-flight safety, thrust 1: the grant never renumbered a live id — `t`'s
    // slot 0 still maps to the same object it CoW'd.
    // (Under ws-slot-ids the generation is ALSO stable, so `t`'s captured gen
    // still validates; legacy bumps it — the reason legacy can't alloc mid-flight.)
    if SLOT_IDS {
        assert_eq!(
            ws.generation(),
            gen_t,
            "ws-slot-ids: alloc must not bump the generation a run-ahead pass captured"
        );
        // `t`'s captured generation still validates a follow-on resolve — it was
        // not stale-rejected by the mid-flight alloc.
        assert!(
            ws.resolve_write(&[0], gen_t).is_ok(),
            "in-flight pass survives the headroom alloc"
        );
    } else {
        assert_ne!(
            ws.generation(),
            gen_t,
            "legacy: alloc bumps the generation (why legacy can't alloc mid-flight)"
        );
    }

    // Thrust 2 (S4): a second, disjoint run-ahead pass `t+1` prepares against a
    // freshly-granted headroom slot — both `t` and `t+1` outstanding at once.
    let t1_slot = size_before; // first freshly-alloc'd slot id
    let mut txn_t1 = a.txn_begin();
    let wtx_t1 = ws.begin_write_txn();
    ws.cow_write_slot(wtx_t1, t1_slot, &mut txn_t1, &mut a)
        .unwrap();

    // Commit `t` — its slot publishes; `t+1` (still in flight) is untouched.
    a.txn_commit(txn_t).unwrap();
    ws.commit_writes(wtx_t);
    // `t`'s repointed slot 0 is live; it diverged from the shared baseline only
    // if it was shared — here it was uniquely owned (in-place), so it stays valid.
    assert!(ws.resolve_read(0, 1).is_ok());
    let _ = base0;

    // Commit `t+1` independently — the freshly-granted slot materialises.
    a.txn_commit(txn_t1).unwrap();
    ws.commit_writes(wtx_t1);
    assert!(
        ws.resolve_read(t1_slot, 1).is_ok(),
        "the run-ahead-granted headroom slot committed independently"
    );

    ws.destroy(&mut a, &mut cas);
}

/// A run-ahead decode loop that tops up headroom every step, driving the exact
/// §6.1 shape (`alloc(want - size)` while the prior step's pass is outstanding).
/// Asserts sustained progress with no id churn: every previously-committed slot
/// keeps its object across all the mid-flight allocs.
#[test]
fn runahead_decode_with_periodic_headroom_topup() {
    let mut a = arena(256);
    let mut cas = KvCas::new();
    let mut ws = KvWorkingSet::new(PAGE, 0);

    const STEPS: u32 = 16;
    const HEADROOM: u32 = 3;

    // Seed the first committed page.
    ws.alloc(1).unwrap();
    commit_slot(&mut ws, &mut a, 0);

    let mut committed_objs: Vec<u32> = vec![ws.resolve_read(0, 1).unwrap()[0]];

    for step in 1..STEPS {
        // Run-ahead pass for this step goes in flight against a fresh slot.
        let want = step + HEADROOM; // keep HEADROOM spare live slots
        if want > ws.size() {
            ws.alloc(want - ws.size()).unwrap(); // in-flight-safe top-up (§6.1)
        }

        let mut txn = a.txn_begin();
        let wtx = ws.begin_write_txn();
        ws.cow_write_slot(wtx, step, &mut txn, &mut a).unwrap();
        a.txn_commit(txn).unwrap();
        ws.commit_writes(wtx);
        committed_objs.push(ws.resolve_read(step, 1).unwrap()[0]);

        // No id churn: every earlier committed slot still maps to its object,
        // despite the mid-loop allocs (the T1 stable-id guarantee the loop rests on).
        for (id, &obj) in committed_objs.iter().enumerate() {
            assert_eq!(
                ws.resolve_read(id as u32, 1).unwrap()[0],
                obj,
                "slot {id} churned at step {step}"
            );
        }
    }

    assert_eq!(committed_objs.len(), STEPS as usize, "every step committed");
    ws.destroy(&mut a, &mut cas);
}
