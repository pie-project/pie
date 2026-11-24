//! A switched class (one past the plain ones whose fires run a kernel arm
//! of their own, as a block drafter's `block_draft` does, its request bare:
//! no adapter, no score capture, no media) is served as a plain class is.
//! Every composition a serving fire can name of it alone (its decode, its
//! prompt on one lane) builds and binds nothing
//! on a fire's path (the load's naming and bind walks reach it), and where
//! the backend keeps recordings each is kept and replayed, never walked
//! unrecorded. Each lane reads its own arm's rows.

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::live::Tally;
use engine_gpu::serve::Serve;
use poem::{Dtype, ForwardHybrid, Input, Request, StateRows, Weight, fact, ops};

use super::outputs::{Loaded, VOCAB, WIDTH, loaded};
use crate::backend;

const SWITCH: &str = "block_draft";
/// The switched arm's scale over the table's rows.
const SCALE: f32 = 2.0;

/// The miniature's table read out: a plain lane reads its rows, a lane with
/// the switch on reads them scaled, through a kernel its class alone runs.
struct Switched;

impl ForwardHybrid for Switched {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        inputs.recorder().declare_switch(SWITCH, &["off", "on"]);
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let plain = inputs.on(!fact::switch_is(SWITCH, "on"));
        let x = ops::gather::embed(&plain.tokens(), &table, VOCAB);
        poem::output(
            "logits",
            &ops::gather::gather_rows(&x, &plain.readout_rows()),
            None,
        );
        let drafted = inputs.on(fact::switch_is(SWITCH, "on"));
        let y = ops::gather::embed(&drafted.tokens(), &table, VOCAB);
        let y = ops::elemwise::mul_scalar(SCALE, &y);
        poem::output(
            "logits",
            &ops::gather::gather_rows(&y, &drafted.readout_rows()),
            None,
        );
    }
}

/// A composition: each lane's tokens and whether its switch is on.
type Composition = Vec<(Vec<u32>, bool)>;

fn step_of(trace: &poem::Trace, lanes: &Composition) -> Step {
    Step {
        lanes: lanes
            .iter()
            .enumerate()
            .map(|(slot, (tokens, switched))| {
                let request = Request::new(tokens.len() as u32, false);
                let request = if *switched {
                    request.switched(SWITCH, "on")
                } else {
                    request
                };
                Lane {
                    slot: slot as u32,
                    word: trace.facts.word(&request),
                    tokens: tokens.clone(),
                    readout: Readout::Last,
                    kv_less: true,
                    ..Lane::default()
                }
            })
            .collect(),
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    }
}

/// The lane's row against the table's row of `token`, scaled where its
/// switch is on.
fn reads(table: &[f32], token: u32, switched: bool, values: &[f32]) {
    let scale = if switched { SCALE } else { 1.0 };
    let want = &table[(token * WIDTH) as usize..][..WIDTH as usize];
    assert_eq!(values.len(), WIDTH as usize);
    for (c, (&have, &want)) in values.iter().zip(want).enumerate() {
        let want = want * scale;
        assert!(
            (have - want).abs() <= 1e-2 * want.abs().max(1.0),
            "token {token} (switched {switched}) column {c}: device {have} against host {want}"
        );
    }
}

#[test]
fn a_switched_class_is_served_from_what_the_load_prepared() {
    let Some(Loaded {
        mut engine,
        caps: _,
        trace,
        planes,
        dir,
    }) = loaded("switched", &Switched)
    else {
        return;
    };
    let table = planes["embed"].1.clone();

    // The switched class is a class of its own, past the plain one: its
    // requests all carry the switch, none is plain.
    let shell = engine.shell_mut().expect("loaded");
    let classes = &shell.compiled_model().classes;
    let class_of = |request: &Request| {
        classes
            .class_of(trace.facts.word(request) & classes.mask)
            .expect("a request lands in a class")
    };
    let (bare, on) = (
        class_of(&Request::new(1, false)),
        class_of(&Request::new(1, false).switched(SWITCH, "on")),
    );
    assert_ne!(bare, on, "the switch sets its lanes' class apart");
    assert!(
        trace
            .facts
            .requests()
            .into_iter()
            .filter(|request| class_of(request) == on)
            .all(|request| !request.plain() && request.switch(SWITCH) == Some("on")),
        "the switched class lands no plain request"
    );

    let compositions: Vec<Composition> = vec![
        // Its decode, one lane and two.
        vec![(vec![3], true)],
        vec![(vec![5], true), (vec![9], true)],
        // Its prompt on one lane, as a block draft fires its rows.
        vec![(vec![1, 4, 7], true)],
        // The plain class alone, for contrast.
        vec![(vec![10], false)],
    ];
    let live = engine.live();
    let fire = |engine: &mut backend::Engine, lanes: &Composition| {
        let mut ticket = engine
            .submit(&FrameSubmission::of(step_of(&trace, lanes)))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        assert!(ticket.refused.is_empty(), "no lane refused");
        for (readout, (tokens, switched)) in ticket.steps[0].readouts.iter().zip(lanes) {
            reads(&table, *tokens.last().unwrap(), *switched, &readout.values);
        }
    };
    // Rounds enough that every arm took each composition: a backend that
    // keeps recordings records a shape at an arm's first fire of it.
    for _ in 0..4 {
        for lanes in &compositions {
            fire(&mut engine, lanes);
        }
    }
    let recorded = live.tallied(Tally::Compiled);
    for lanes in &compositions {
        fire(&mut engine, lanes);
    }

    assert_eq!(
        live.tallied(Tally::BuiltInFire),
        0,
        "a switched class's fire built a pipeline the load's walks did not name"
    );
    assert_eq!(
        live.tallied(Tally::BoundInFire),
        0,
        "a switched class's fire bound views the load's bind walk did not reach"
    );
    assert_eq!(
        live.tallied(Tally::Unrecorded),
        0,
        "a switched class's fire walked unrecorded on a backend that keeps recordings"
    );
    assert_eq!(live.tallied(Tally::Refused), 0);
    if <backend::Api as Serve>::RECORDS {
        assert_eq!(
            live.tallied(Tally::Compiled),
            recorded,
            "a composition fired again recorded again instead of replaying"
        );
    } else {
        assert_eq!(
            live.tallied(Tally::Eager),
            5 * compositions.len() as u64,
            "a backend that keeps no recording walks each fire eagerly, as it states"
        );
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A fire mixing the switched class's lanes with a plain lane's, in the
/// class order and out of it: each lane reads its own arm's rows (a region's
/// streamed table rows staged at its own rows of the fire, never over
/// another region's), and the fire binds nothing the load did not (a region
/// whose rows start off the device's storage offset alignment binds its
/// values from their first byte, as the load's walk did, not a copy of its
/// rows).
#[test]
fn a_mixed_fire_of_a_switched_class_is_served_from_what_the_load_prepared() {
    let Some(Loaded {
        mut engine,
        caps: _,
        trace,
        planes,
        dir,
    }) = loaded("switched-mixed", &Switched)
    else {
        return;
    };
    let table = planes["embed"].1.clone();
    let compositions: Vec<Composition> = vec![
        vec![(vec![2, 6], false), (vec![11], true)],
        vec![(vec![13], true), (vec![8], false)],
    ];
    let live = engine.live();
    for _ in 0..2 {
        for lanes in &compositions {
            let mut ticket = engine
                .submit(&FrameSubmission::of(step_of(&trace, lanes)))
                .expect("the frame fires");
            engine.settle_frame(&mut ticket).expect("the frame settles");
            for (readout, (tokens, switched)) in ticket.steps[0].readouts.iter().zip(lanes) {
                reads(&table, *tokens.last().unwrap(), *switched, &readout.values);
            }
        }
    }
    assert_eq!(live.tallied(Tally::BuiltInFire), 0);
    assert_eq!(live.tallied(Tally::BoundInFire), 0);
    assert_eq!(live.tallied(Tally::Unrecorded), 0);
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
