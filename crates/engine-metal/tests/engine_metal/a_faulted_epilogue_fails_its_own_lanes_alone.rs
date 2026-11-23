#![cfg(target_vendor = "apple")]

//! A guest epilogue that faults fails the lanes its instance is attached at,
//! and no other (`engine::StepOutcome::faulted`). Two lanes of one step, a
//! score reader attached at the second whose kernels write a fault status
//! and return. The sink hears that reader's lane faulted by its instance and
//! nothing else; the step's rows read back, and the first lane's are the
//! rows of the same step with nothing attached. (One shell holds one
//! reader: the stage cache keys a stage by its plan, not its source, so a
//! healthy reader registered beside it would lend the faulting one its
//! stages.)

use std::sync::{Arc, Mutex};

use engine::frame::Shell as FrameShell;
use engine::program::ProgramRegistration;
use engine_metal::serve::{Seated, StepView};
use engine_metal::{Done, Lane, Shell};
use poem::Request;

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};
use super::a_score_reader_holds_its_rows_through_a_park::{at_lane, bind_reader, reader};

const CONTEXT: u32 = 256;
const PROMPT: u32 = 12;

/// The reader whose kernels fault: each kernel, past its guard on the
/// status word (`if (status->state != 1) return;`, the lane's commit slot
/// in a grouped or streamed region, its own argument in a fused or
/// single-lane one), writes a fault and returns. The epilogue runs and its
/// verdict is a fault. A test-only rewrite of the registration's sources.
fn faulting(context: u32) -> ProgramRegistration {
    let mut registration = reader(context);
    registration.program_hash ^= 0xfa17_fa17_fa17_fa17;
    let guard = "  if (status->state != 1) return;\n";
    let fault = "  if (status->state != 1) return;\n  status->state = 3u; status->fault = 0x77u; return;\n";
    let mut rewritten = 0;
    for kernel in &mut registration.emitted_kernels {
        if kernel.source.contains(guard) {
            kernel.source = kernel.source.replace(guard, fault);
            rewritten += 1;
        }
    }
    assert!(rewritten > 0, "the reader's kernels guard on the status word");
    registration
}

/// Two lanes prefilled, then one scored decode of both with the readers
/// `with` attached (lane, program); what the sink heard and each lane's rows.
fn step(
    fixture: &Fixture,
    with: &[(u32, &ProgramRegistration)],
) -> (engine::StepOutcome, Vec<engine::InstanceId>, Vec<(String, Vec<f32>)>) {
    let mut shell: Shell = load_with(fixture, CONTEXT);
    shell.warm_pipelines().expect("the walk runs");
    shell.seal_pipelines().expect("the fixed pipelines build");
    let prompts: Vec<Vec<u32>> = (0..2u32)
        .map(|lane| (0..PROMPT).map(|t| (t * 7 + 11 + lane * 3) % 256).collect())
        .collect();
    for (slot, prompt) in prompts.iter().enumerate() {
        shell.open(slot as u32).expect("the slot opens");
        shell
            .fire_seated(&[Seated {
                replay_stated: None,
                held: Some(0),
                ..Seated::of(Lane {
                    slot: slot as u32,
                    word: word(&fixture.trace, PROMPT),
                    tokens: prompt,
                })
            }])
            .expect("the prompt prefills");
    }
    let mut ids = Vec::new();
    let mut attached = Vec::new();
    for &(lane, registration) in with {
        let program = shell
            .register_program(registration)
            .expect("the reader registers");
        let (id, ..) = bind_reader(&mut shell, program, 1, 1 + lane);
        ids.push(id);
        attached.push(at_lane(lane, id));
    }
    // A lane captures its scores where a reader is attached at it: only the
    // fire that runs a pass's program captures (Step::validate_captures).
    let scored = fixture
        .trace
        .facts
        .word(&Request::new(1, false).capturing_scores(true));
    let plain = word(&fixture.trace, 1);
    let lanes: Vec<Seated<'_>> = (0..2u32)
        .map(|slot| Seated {
            replay_stated: None,
            held: Some(PROMPT),
            ..Seated::of(Lane {
                slot,
                word: if with.iter().any(|&(lane, _)| lane == slot) { scored } else { plain },
                tokens: &[5],
            })
        })
        .collect();
    let heard: Arc<Mutex<Vec<engine::StepOutcome>>> = Arc::default();
    let sink: engine::CompletionSink = {
        let heard = heard.clone();
        Arc::new(move |_at, outcome| heard.lock().unwrap().push(outcome))
    };
    let at = engine::StepDone {
        frame: engine::FrameId(0),
        step: 0,
    };
    let prepared = FrameShell::prepare(
        &mut shell,
        StepView {
            lanes: &lanes,
            attachments: &attached,
            media: &[],
            clips: &[],
            done: Some(Done { at, sink }),
        },
        None,
    )
    .expect("the fire prepares");
    let enqueued = FrameShell::enqueue(&mut shell, prepared).expect("the fire enqueues");
    let landed = FrameShell::settle(&mut shell, enqueued).expect("the fire settles");
    let rows = shell
        .rows_of(&landed)
        .expect("the step's rows read back though one epilogue faulted");
    let heard = heard.lock().unwrap();
    assert_eq!(heard.len(), 1, "the sink heard the step once");
    (heard[0].clone(), ids, rows)
}

#[test]
fn a_faulted_epilogue_fails_its_own_lanes_alone() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let bad = faulting(CONTEXT);

    let (outcome, ids, rows) = step(&fixture, &[(1, &bad)]);
    assert_eq!(
        outcome.faulted.len(),
        1,
        "one lane faulted, not the step: {:?}",
        outcome.faulted
    );
    let fault = &outcome.faulted[0];
    assert_eq!(fault.lane, 1, "the faulting reader's lane");
    assert_eq!(fault.instance, Some(ids[0]), "named by its instance");
    assert!(
        fault.reason.contains("faulted"),
        "the reason says the epilogue faulted: {}",
        fault.reason
    );

    let (plain, _, control) = step(&fixture, &[]);
    assert!(plain.faulted.is_empty(), "the step alone commits: {:?}", plain.faulted);
    assert_eq!(
        rows[0], control[0],
        "the first lane's rows are the step's with nothing attached"
    );
}
