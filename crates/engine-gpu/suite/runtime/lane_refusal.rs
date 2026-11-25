//! A lane the engine refuses for its own sake (here one routed to an
//! adapter on a load that corrects nothing) refuses that lane alone: at a
//! frame's first step the frame is refused naming it and nothing ran; at a
//! later step the frame stops there, its ticket naming it, and the steps
//! before it land whole. The plain lanes beside it fire on, and read out
//! their own rows. Each refusal is counted (`Tally::Refused`), a clean
//! frame not.

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::live::Tally;
use poem::Request;

use super::outputs::{Loaded, Rows, WIDTH, loaded};

/// A step of kv-less lanes, one a token, `routed` the ones naming an
/// adapter.
fn step_of(trace: &poem::Trace, tokens: &[(u32, bool)]) -> Step {
    Step {
        lanes: tokens
            .iter()
            .enumerate()
            .map(|(slot, &(token, routed))| Lane {
                slot: slot as u32,
                word: trace.facts.word(&Request::new(1, false)),
                tokens: vec![token],
                readout: Readout::Rows(vec![0]),
                kv_less: true,
                adapter: routed.then_some(0),
                ..Lane::default()
            })
            .collect(),
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
        shape: None,
    }
}

/// The lane's one row against the table's row of `token`.
fn reads(table: &[f32], token: u32, values: &[f32]) {
    let want = &table[(token * WIDTH) as usize..][..WIDTH as usize];
    assert_eq!(values.len(), WIDTH as usize);
    for (c, (&have, &want)) in values.iter().zip(want).enumerate() {
        assert!(
            (have - want).abs() <= 1e-2 * want.abs().max(1.0),
            "token {token} column {c}: device {have} against host {want}"
        );
    }
}

#[test]
fn a_refused_lane_is_refused_alone() {
    let Some(Loaded { mut engine, trace, planes, dir, .. }) = loaded("lane-refusal", &Rows) else {
        return;
    };
    let table = &planes["embed"].1;

    // At the first step: the frame is refused, naming lane 1 alone.
    let refused = engine
        .submit(&FrameSubmission::of(step_of(&trace, &[(3, false), (7, true), (11, false)])))
        .expect_err("a lane routed to an adapter this load lacks is refused");
    let engine::Error::Lanes(lanes) = &refused else {
        panic!("the refusal names its lane, not the frame: {refused}");
    };
    assert_eq!(lanes.iter().map(|fault| fault.lane).collect::<Vec<_>>(), vec![1], "lane 1 alone: {refused}");
    let live = engine.live();
    assert_eq!(live.tallied(Tally::Refused), 1, "the refused frame is counted");

    // At the second step: the first lands whole, and the ticket names the
    // second's lane 1.
    let frame = FrameSubmission {
        steps: vec![
            step_of(&trace, &[(3, false), (11, false)]),
            step_of(&trace, &[(5, false), (7, true)]),
        ],
    };
    let mut ticket = engine
        .submit(&frame)
        .expect("a lane refused past the first step stops the frame there");
    assert_eq!(ticket.steps.len(), 1, "the steps before the refusing one fired");
    assert_eq!(ticket.refused.iter().map(|fault| fault.lane).collect::<Vec<_>>(), vec![1]);
    assert_eq!(live.tallied(Tally::Refused), 2, "a frame stopped at a refusing step is counted");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    for (readout, token) in ticket.steps[0].readouts.iter().zip([3, 11]) {
        reads(table, token, &readout.values);
    }

    // The plain lanes fire on, the refused one gone.
    let mut ticket = engine
        .submit(&FrameSubmission::of(step_of(&trace, &[(5, false)])))
        .expect("the rest fires again");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    reads(table, 5, &ticket.steps[0].readouts[0].values);
    assert!(ticket.refused.is_empty());
    assert_eq!(live.tallied(Tally::Refused), 2, "a clean frame is not");

    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
