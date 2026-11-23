#![cfg(target_vendor = "apple")]

//! With one frame in flight every stage harvests the step before it, so a
//! frame's steps have all landed by the time its rows are asked. A frame of
//! more steps than the settled rows used to be kept for (two frames of the
//! most a frame takes) lost its first steps' rows and was refused. Every
//! step's rows are held until they are taken: the first of ten steps taken
//! after all ten landed, the first last, and a second ask is refused.

use engine::frame::Shell as FrameShell;
use engine_metal::serve::{Seated, StepView};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};

const STEPS: u32 = 10;

fn step(shell: &mut Shell, fixture: &Fixture, slot: u32, token: u32, held: u32) -> engine_metal::serve::Landed {
    let tokens = [token];
    let lanes = [Seated {
        replay_stated: None,
        held: Some(held),
        ..Seated::of(Lane {
            slot,
            word: word(&fixture.trace, 1),
            tokens: &tokens,
        })
    }];
    let prepared = FrameShell::prepare(
        shell,
        StepView {
            lanes: &lanes,
            attachments: &[],
            media: &[],
            clips: &[],
            done: None,
        },
        None,
    )
    .expect("the step prepares");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the step enqueues");
    FrameShell::settle(shell, enqueued).expect("the step settles")
}

#[test]
fn a_frames_steps_keep_their_rows_until_taken() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, 256);
    assert_eq!(shell.frames_in_flight(), 1, "one frame in flight");
    shell.open(0).expect("slot 0 opens");
    let tokens: Vec<u32> = (0..STEPS).map(|at| (11 + at * 37) % 256).collect();
    let landed: Vec<_> = tokens
        .iter()
        .enumerate()
        .map(|(at, &token)| step(&mut shell, &fixture, 0, token, at as u32))
        .collect();
    // Taken after all of them landed, the first last: none is refused.
    for (at, step) in landed.iter().enumerate().rev() {
        shell
            .rows_of(step)
            .unwrap_or_else(|why| panic!("step {at}'s rows, after {STEPS} steps landed: {why}"));
    }
    // Taken once: a second ask is refused by name.
    let again = shell.rows_of(&landed[0]).expect_err("a step's rows are taken once");
    assert!(again.to_string().contains("already been taken"), "{again}");
    eprintln!("{STEPS} steps landed; every step's rows were still there");
}
