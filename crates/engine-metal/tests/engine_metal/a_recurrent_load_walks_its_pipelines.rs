#![cfg(target_vendor = "apple")]

//! The load's pipeline walk (`Shell::warm_pipelines`, before ready) runs on
//! a recurrent model: its fires are staged and dropped uncommitted, so they
//! name seat 0's state record without the pool backing it (seat runs follow
//! one another shape by shape, a recurrent record past every seat's kv run)
//! and a prompt after it answers as on a load that never walked.

use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Load, Fixture, fixture_of, word};

/// Seats and context enough that the kv list's seat runs (every seat's, laid
/// first) reach past what the walk backs: seat 0's record lies beyond them.
const SEATS: u32 = 64;
const CONTEXT: u32 = 2048;

fn load(fixture: &Fixture) -> Shell {
    Load {
        budget: Budget::new(SEATS, 256),
        context: CONTEXT,
        slots: SEATS,
        ..Load::default()
    }
    .boot(fixture)
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
                replay_stated: None,
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
    assert!(
        record > segment,
        "seat 0's record (chunk {record}) lies inside the segment the walk backs ({segment} \
         chunks), so this load does not reach the case"
    );
    let (fires, _) = shell
        .warm_pipelines()
        .expect("the pipeline walk runs on a recurrent load");
    assert!(fires > 0, "the walk fired");
    assert_eq!(fire(&mut shell), want, "a prompt after the walk answers as without it");
}

/// Every form the plan serves is built by the walk: after it, with the
/// pipelines sealed (a key met after is refused, not compiled), a recurrent
/// load serves a plain prompt, a buffering one (`RsVerb::Buffer` folding
/// nothing: the verify conv), a replay folding what a lane held (the decode
/// drain), and a drafter's word where the plan has one.
#[test]
fn every_form_a_recurrent_load_serves_is_built_by_its_walk() {
    use engine::fire::{FoldLen, RsVerb};
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load(&fixture);
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the fixed pipelines build");
    // Every fixed pipeline (a fire names it by its lanes' shape, no rung
    // keys it) is built at load, `rs_copy_words4` (four row copies side by
    // side, a beam's replay) among them.
    let built = shell.pipeline_keys();
    for fire in kernels_metal::runtime::fixed_pipelines(true) {
        assert!(
            built
                .iter()
                .any(|(file, entry, _, _)| file == fire.file && entry == fire.entrypoint),
            "`{}` of {} is not built at load",
            fire.entrypoint,
            fire.file
        );
    }
    let end = shell
        .pool_layout()
        .shapes()
        .map(|shape| shape.seat_chunk + shape.seat_slots.div_ceil(shape.per_chunk.max(1)))
        .max()
        .unwrap_or(0);
    let per = shell.pool_chunk_facts().segment_chunks.max(1);
    shell.commit_segments(end.div_ceil(per)).expect("the seats back");
    let per_page = shell.pool_layout().rs_tokens.max(1);
    let pages: Vec<u32> = (0..16u32.div_ceil(per_page)).collect();
    let tokens: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let next = [5u32];
    let fold = RsVerb::Fold;
    let buffer = |fold: u32| RsVerb::Buffer {
        pages: pages.clone(),
        base: 0,
        folded: 0,
        fold: FoldLen::Host(fold),
        lineage: engine::fire::Lineage::Set,
    };
    let (hold, replay) = (buffer(0), buffer(1));
    let lane = |slot: u32, tokens: &'static [u32], word: u64, rs: &'static RsVerb, held: u32| Seated {
        replay_stated: None,
        held: Some(held),
        rs,
        ..Seated::of(Lane { slot, word, tokens })
    };
    let leak = |v: Vec<u32>| -> &'static [u32] { Box::leak(v.into_boxed_slice()) };
    let (tokens, next) = (leak(tokens), leak(next.to_vec()));
    let (fold, hold, replay): (&'static RsVerb, &'static RsVerb, &'static RsVerb) = (
        Box::leak(Box::new(fold)),
        Box::leak(Box::new(hold)),
        Box::leak(Box::new(replay)),
    );
    let prompt = word(&fixture.trace, tokens.len() as u32);
    let one = word(&fixture.trace, 1);
    for slot in 1..4 {
        shell.open(slot).expect("a slot opens");
    }
    shell
        .fire_seated(&[lane(1, tokens, prompt, fold, 0)])
        .expect("a folding prompt is served");
    shell
        .fire_seated(&[lane(2, tokens, prompt, hold, 0)])
        .expect("a buffering prompt is served");
    shell
        .fire_seated(&[lane(2, next, one, replay, tokens.len() as u32)])
        .expect("a decode replaying what the lane held is served");
    let drafter = fixture
        .trace
        .facts
        .requests()
        .into_iter()
        .any(|request| request.drafts() && request.query_len() == 1);
    if drafter {
        let drafts = fixture
            .trace
            .facts
            .word(&poem::Request::new(1, false).drafting(true));
        shell
            .fire_seated(&[lane(3, next, drafts, fold, 0)])
            .expect("a drafter's word is served");
    }
}
