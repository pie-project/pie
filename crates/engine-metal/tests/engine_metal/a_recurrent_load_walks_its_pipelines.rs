#![cfg(target_vendor = "apple")]

//! The load's pipeline walk (`Shell::warm_pipelines`, before ready) runs on
//! a recurrent model: its fires are staged and dropped uncommitted, so they
//! name seat 0's state record without the pool backing it (seat runs follow
//! one another shape by shape, a recurrent record past every seat's kv run)
//! and a prompt after it answers as on a load that never walked. The walk
//! refused such a load before
//! (Qwen3.5-0.8B: "this fire wants 13113 committed chunks of the state pool
//! and the load reserved 58").

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, word};

/// Seats and context enough that the kv list's seat runs (every seat's, laid
/// first) reach past what the walk backs: seat 0's record lies beyond them.
const SEATS: u32 = 64;
const CONTEXT: u32 = 2048;

fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(SEATS, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: SEATS,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

#[test]
fn a_recurrent_load_walks_its_pipelines() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let tokens: Vec<u32> = (0..9u32).map(|at| (7 + at * 37) % 256).collect();
    // As a caller firing directly backs its seats: the segments every
    // shape's seat run reaches.
    let fire = |shell: &mut engine_metal::Shell| {
        let end = shell
            .pool_layout()
            .shapes()
            .map(|shape| shape.seat_chunk + shape.seat_slots.div_ceil(shape.per_chunk.max(1)))
            .max()
            .unwrap_or(0);
        let per = shell.pool_chunk_facts().segment_chunks.max(1);
        shell.commit_segments(end.div_ceil(per)).expect("the seats back");
        shell.open(1).expect("slot 1 opens");
        shell
            .fire_seated(&[Seated {
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 1,
                    word: word(&fixture.trace, tokens.len() as u32),
                    tokens: &tokens,
                })
            }])
            .expect("a prompt fires")
            .remove(0)
    };
    let mut plain = load(&fixture);
    assert!(plain.serves_rs_verbs(), "a recurrent model");
    let want = fire(&mut plain);
    drop(plain);

    // The walk runs as a load runs it: before anything past the load's own
    // segments is backed.
    let mut shell = load(&fixture);
    let segment = shell.pool_chunk_facts().segment_chunks;
    let record = shell
        .pool_layout()
        .shapes()
        .map(|shape| shape.seat_chunk)
        .filter(|&at| at > 0)
        .nth(1)
        .unwrap_or(0);
    eprintln!("segments of {segment} chunks; seat 0's record run starts at chunk {record}");
    assert!(
        record > segment,
        "seat 0's record (chunk {record}) lies inside the segment the walk backs ({segment} \
         chunks), so this load does not reach the case"
    );
    let (fires, built) = shell
        .warm_pipelines()
        .expect("the pipeline walk runs on a recurrent load");
    eprintln!("the walk: {fires} fires, {built} pipelines built");
    assert!(fires > 0, "the walk fired");
    assert_eq!(fire(&mut shell), want, "a prompt after the walk answers as without it");
}
