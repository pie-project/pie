#![cfg(target_vendor = "apple")]

//! A frame whose command buffer fails commits nothing of the instances it
//! carried: their passes are let go, so each stands where it stood before
//! the fire. The step is told lost; the same instance's next fire stages,
//! runs and commits (its pass left airborne refused every later staging),
//! and it closes. At one frame in flight and at two, where the harvest that
//! meets the failure is the next fire's staging and must not refuse it.

use std::sync::mpsc;
use std::time::Duration;

use engine::frame::Shell as FrameShell;
use engine_metal::serve::{Attached, Seated, StepView};
use engine_metal::{Done, Lane, Shell};

use super::a_kv_only_lane_reports_its_device_fold::bind_reporting;
use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_ahead, word};

const ROWS: u32 = 5;

type Heard = (engine::StepDone, engine::StepOutcome);

/// Fires the instance's lane as step `frame`, the buffer failing where
/// `fail`; the sink's word arrives on `rx`.
fn fire(
    shell: &mut Shell,
    id: engine::InstanceId,
    tokens: &[u32],
    frame: u64,
    fail: bool,
    tx: &mpsc::Sender<Heard>,
) {
    let tx = std::sync::Mutex::new(tx.clone());
    let sink: engine::CompletionSink = std::sync::Arc::new(move |at, outcome| {
        let _ = tx.lock().unwrap().send((at, outcome));
    });
    let lanes = [Seated {
        replay_stated: None,
        held: Some(0),
        reports: 7,
        ..Seated::of(Lane {
            slot: 0,
            word: word(shell.trace(), ROWS),
            tokens,
        })
    }];
    let attached = [Attached {
        lane: 0,
        lanes: 1,
        instance: id,
        adapter_only: false,
    }];
    let at = engine::StepDone {
        frame: engine::FrameId(frame),
        step: 0,
    };
    let prepared = FrameShell::prepare(
        shell,
        StepView {
            lanes: &lanes,
            attachments: &attached,
            media: &[],
            clips: &[],
            done: Some(Done { at, sink }),
        },
        None,
    )
    .unwrap_or_else(|why| panic!("step {frame} prepares: {why}"));
    if fail {
        engine_metal::device::fail_next_frame();
    }
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the fire enqueues");
    let landed = FrameShell::settle(shell, enqueued).expect("the fire settles");
    if shell.frames_in_flight() == 1 {
        // One frame in flight: the harvest is the readout, and it answers
        // the failure to the frame that failed.
        assert_eq!(shell.rows_of(&landed).is_err(), fail, "step {frame}'s rows");
    }
}

fn heard(rx: &mpsc::Receiver<Heard>, frame: u64) -> engine::StepOutcome {
    let (at, outcome) = rx
        .recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("step {frame} is told"));
    assert_eq!(at.frame, engine::FrameId(frame));
    outcome
}

#[test]
fn an_instance_outlives_its_failed_frame() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    for frames in [1u8, 2] {
        let mut shell = load_ahead(&fixture, 1024, None, engine::runahead::Runahead::of(frames));
        assert_eq!(shell.frames_in_flight(), usize::from(frames));
        let tokens: Vec<u32> = (0..ROWS).map(|at| (7 + at * 37) % 256).collect();
        shell.open(3).expect("slot 3 opens");
        let vocab = shell
            .fire_seated(&[Seated {
                replay_stated: None,
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 3,
                    word: word(&fixture.trace, ROWS),
                    tokens: &tokens,
                })
            }])
            .expect("a plain fire lands")[0]
            .len() as u32;
        let id = bind_reporting(&mut shell, vocab, &tokens, true, 0);
        shell.open(0).expect("the slot opens");
        let (tx, rx) = mpsc::channel();

        fire(&mut shell, id, &tokens, 1, true, &tx);
        let lost = heard(&rx, 1);
        assert!(lost.wrote.is_empty(), "frames {frames}: a failed step wrote nothing");
        assert_eq!(lost.faulted.len(), 1, "frames {frames}: its lane is lost: {lost:?}");

        fire(&mut shell, id, &tokens, 2, false, &tx);
        let next = heard(&rx, 2);
        assert!(next.faulted.is_empty(), "frames {frames}: the next fire commits: {next:?}");
        assert_eq!(next.wrote.len(), 1, "frames {frames}: and reports its lane");

        shell.reap().expect("the landed steps publish");
        shell.close_program_instance(id).expect("the instance closes");
        assert!(rx.try_recv().is_err(), "frames {frames}: each step told once");
    }
}
