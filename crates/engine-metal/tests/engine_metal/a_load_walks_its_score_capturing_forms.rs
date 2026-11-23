#![cfg(target_vendor = "apple")]

//! The load's pipeline walk (`Shell::warm_pipelines`) builds the forms a
//! request capturing its attention scores fires (the score-capture kernel
//! beside the plain attention), at a prompt and at one row. The walk fired
//! plain and drafting words alone before, so trackb-h2o, trackb-snapkv and
//! snapkv-eviction on Qwen3-0.6B were refused at their first capturing fire
//! ("`attn_score_capture_bfloat16_d_128` would not enqueue: ... which the
//! load did not build before ready").

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of};
use super::a_score_reader_holds_its_rows_through_a_park::{at_lane, bind_reader, reader};

fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(4, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: 16,
        context: 1024,
        slots: 4,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

#[test]
fn a_load_walks_its_score_capturing_forms() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let captures = fixture
        .trace
        .facts
        .requests()
        .into_iter()
        .any(|request| request.captures_scores());
    assert!(captures, "the plan has a word capturing its scores");
    let mut shell = load(&fixture);
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the fixed pipelines build");
    let scores = |rows: u32| {
        fixture
            .trace
            .facts
            .word(&poem::Request::new(rows, false).capturing_scores(true))
    };
    let tokens: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    shell.open(0).expect("a slot opens");
    // A capturing lane's rows are its reader's: readers bound after ready,
    // at the prompt's rows and at one.
    let program = shell
        .register_program(&reader(1024))
        .expect("the reader registers");
    let (prompt_reader, ..) = bind_reader(&mut shell, program, tokens.len() as u32, 1);
    let (decode_reader, ..) = bind_reader(&mut shell, program, 1, 2);
    shell
        .fire_attached(
            &[Seated {
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 0,
                    word: scores(tokens.len() as u32),
                    tokens: &tokens,
                })
            }],
            &[at_lane(0, prompt_reader)],
        )
        .expect("a prompt capturing its scores is served after the sealed walk");
    shell
        .fire_attached(
            &[Seated {
                held: Some(tokens.len() as u32),
                ..Seated::of(Lane {
                    slot: 0,
                    word: scores(1),
                    tokens: &[5],
                })
            }],
            &[at_lane(0, decode_reader)],
        )
        .expect("a decode capturing its scores is served after the sealed walk");
}
