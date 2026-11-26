//! A load told its weights start held whose routed experts do not fit its
//! pool seats none of them itself: it backs its seats' segment and the
//! span's alone, and the runtime places the experts in segments it commits.
//! Its own fires (the naming walk, the bind walk) admit no chunk past the
//! segments it backed, whatever unit a kv page id counts, and a step
//! reading experts from slots the runtime placed in a segment committed
//! after the load answers what the step answered with them staged.

use std::collections::BTreeSet;

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{argmax, fixture, load_capped_with, near, word};

/// Four MoE layers of 32 experts of 384 KiB each.
const MODEL: &str = "qwen3-a3b-micro";
const LAYER_EXPERTS: u64 = 32;

/// Kv pages of 28 tokens (56 KiB): no chunk near the target holds a whole
/// number of them and of experts both, so the chunk is the experts' and a
/// page id counts `gcd(chunk, page)` bytes, a seventh of a page.
const PAGE: u32 = 28;
const CONTEXT: u32 = 280;
const TARGET: u64 = 1 << 20;

/// Two lanes prefilling then decoding a step together on seats 0 and 1,
/// fresh: every logit row.
fn serve(shell: &mut Shell, trace: &poem_ir::Trace) -> Vec<Vec<f32>> {
    let rows = 4u32;
    for slot in 0..2 {
        shell.open(slot).expect("the seat opens");
    }
    let prompts: Vec<Vec<u32>> = (0..2u32)
        .map(|lane| {
            (0..rows)
                .map(|at| (at * 37 + lane * 11 + 5) % 2048)
                .collect()
        })
        .collect();
    let seated: Vec<Seated<'_>> = (0..2u32)
        .map(|lane| Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: lane,
                word: word(trace, rows),
                tokens: &prompts[lane as usize],
            })
        })
        .collect();
    let mut got = shell.fire_seated(&seated).expect("the prefill lands");
    let fed: Vec<Vec<u32>> = got.iter().map(|row| vec![argmax(row)]).collect();
    let seated: Vec<Seated<'_>> = (0..2u32)
        .map(|lane| Seated {
            held: Some(rows),
            ..Seated::of(Lane {
                slot: lane,
                word: word(trace, 1),
                tokens: &fed[lane as usize],
            })
        })
        .collect();
    got.extend(shell.fire_seated(&seated).expect("the decode lands"));
    got
}

/// The atoms of `shape` the fires since the last ask read.
fn used(shell: &mut Shell, shape: engine::chunk::ShapeId) -> BTreeSet<u32> {
    shell
        .weight_uses()
        .expect("the uses read")
        .uses
        .iter()
        .filter(|(atom, _)| atom.shape == shape)
        .map(|(atom, _)| atom.atom)
        .collect()
}

#[test]
fn a_held_load_whose_experts_overflow_its_pool_serves_what_the_runtime_places() {
    let fixture = fixture(MODEL, None);
    let residency = |kv_budget: u64| engine::load::Residency {
        device_kv_budget: Some(kv_budget),
        weights_held: true,
        ..engine::load::Residency::default()
    };
    let Some(probe) = load_capped_with(
        &fixture,
        PAGE,
        CONTEXT,
        u64::MAX,
        TARGET,
        residency(1 << 30),
    ) else {
        return;
    };
    let chunk = probe.pool_layout().chunk_bytes;
    let expert = probe
        .pool_chunk_facts()
        .weights
        .iter()
        .find(|shape| shape.kind == engine::chunk::WeightKind::Expert)
        .expect("the model streams its routed experts")
        .slot_bytes;
    drop(probe);
    // Three segments, each one binding: a layer's experts staged fit one,
    // and the pool holds short of every expert below the span.
    let segment = (LAYER_EXPERTS * expert).div_ceil(chunk).max(10);
    let Some(mut shell) = load_capped_with(
        &fixture,
        PAGE,
        CONTEXT,
        segment * chunk,
        TARGET,
        residency(3 * segment * chunk),
    ) else {
        return;
    };
    let layout = shell.pool_layout();
    let facts = shell.pool_chunk_facts();
    let list = layout.list.expect("the model pages its kv");
    eprintln!(
        "chunk {} bytes, kv page {} bytes, {} chunks in segments of {}, span {:?}, seats {} \
         chunks, {} segments committed of {}",
        layout.chunk_bytes,
        list.slot_bytes,
        layout.total_chunks,
        layout.segment_chunks,
        facts.act_span_chunks,
        layout.seat_chunks(),
        facts.segments.committed,
        facts.segments.max
    );
    assert_ne!(
        layout.chunk_bytes % list.slot_bytes,
        0,
        "no whole number of kv pages in a chunk: a page id counts less than a page"
    );
    let experts = facts
        .weights
        .iter()
        .find(|shape| shape.kind == engine::chunk::WeightKind::Expert)
        .expect("the model streams its routed experts")
        .clone();
    let per = experts.slots_per_chunk.max(1);
    let span = facts.act_span_chunks.unwrap_or(0);
    let below_span = layout.total_chunks - span;
    assert!(
        1 + layout.seat_chunks() + experts.count.div_ceil(per) > below_span,
        "the {} experts ({per} a chunk) overflow the {below_span} chunks below the span",
        experts.count
    );
    assert!(
        experts.seated.is_none(),
        "a held load whose experts overflow its pool seats none of them"
    );
    // A pool that resizes is backed at its seats' segments (and the
    // span's) alone; one laid whole (wgpu) at every one. The load's naming
    // walk is backed as one that resizes is, on every backend.
    let seats = (1 + layout.seat_chunks()).div_ceil(layout.segment_chunks);
    assert!(facts.segments.committed >= seats);

    // Every expert staged: what the steps read with none in a slot, and
    // the experts they route to.
    let staged = serve(&mut shell, &fixture.trace);
    let routed = used(&mut shell, experts.id);
    assert!(!routed.is_empty(), "the steps route to experts");

    // The runtime commits the segments past the seats' below the span's,
    // and places there as many of the routed experts as they hold.
    let first = seats * layout.segment_chunks;
    let top = below_span.min(facts.segments.max * layout.segment_chunks) / layout.segment_chunks
        * layout.segment_chunks;
    assert!(
        top > first,
        "a segment lies between the seats' and the span's"
    );
    shell
        .commit_segments(top / layout.segment_chunks)
        .expect("the runtime's segments commit");
    let placed: Vec<u32> = routed
        .iter()
        .copied()
        .take(((top - first) * per) as usize)
        .collect();
    let resident = placed
        .iter()
        .enumerate()
        .map(|(at, &atom)| engine::chunk::ResidentWeight {
            atom: engine::chunk::WeightAtom {
                shape: experts.id,
                atom,
            },
            slot: first * per + at as u32,
        })
        .collect();
    shell
        .weight_residency(&engine::chunk::WeightResidency {
            staging: Vec::new(),
            resident,
            held: false,
        })
        .expect("the experts land where the runtime placed them");
    let from_slots = serve(&mut shell, &fixture.trace);
    let read = used(&mut shell, experts.id);
    assert!(
        placed.iter().all(|atom| read.contains(atom)),
        "the steps read every expert the runtime placed: {placed:?} of {read:?}"
    );
    near(
        "experts the runtime placed against the same experts staged",
        &from_slots,
        &staged,
    );
}
