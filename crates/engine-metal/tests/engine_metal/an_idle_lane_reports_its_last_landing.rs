#![cfg(target_vendor = "apple")]

//! With more than one frame in flight, a frame carrying an attached
//! program's epilogue is reported landed from its command buffer's
//! completion, not when the lane's next activity harvests it: a lane that
//! goes idle with one such frame in flight still tells the runtime, so a
//! set waiting out its last fire (an eviction, a teardown) moves on. Here
//! the frame is submitted and the lane does nothing more: the sink must
//! hear the step, from Metal's completion thread, with its epilogue's
//! verdict; the commit itself follows when the runtime publishes the landed
//! step (`publish_landed`, a reap).

use std::sync::mpsc;
use std::time::Duration;

use engine::frame::Shell as FrameShell;
use engine_metal::serve::{Attached, Seated, StepView};
use engine_metal::{Done, Lane};

use super::a_kv_only_lane_reports_its_device_fold::bind_reporting;
use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_ahead, word};

const ROWS: u32 = 5;

#[test]
fn an_idle_lane_reports_its_last_landing() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_ahead(&fixture, 1024, None, engine::runahead::Runahead::of(2));
    assert_eq!(shell.frames_in_flight(), 2, "two frames in flight");
    let tokens: Vec<u32> = (0..ROWS).map(|at| (7 + at * 37) % 256).collect();
    shell.open(3).expect("slot 3 opens");
    let vocab = shell
        .fire_seated(&[Seated {
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
    let tx = std::sync::Mutex::new(tx);
    let sink: engine::CompletionSink = std::sync::Arc::new(move |at, outcome| {
        let _ = tx.lock().unwrap().send((at, outcome, std::thread::current().id()));
    });
    let lanes = [Seated {
        held: Some(0),
        reports: 7,
        ..Seated::of(Lane {
            slot: 0,
            word: word(shell.trace(), ROWS),
            tokens: &tokens,
        })
    }];
    let attached = [Attached {
        lane: 0,
        lanes: 1,
        instance: id,
        adapter_only: false,
    }];
    let at = engine::StepDone {
        frame: engine::FrameId(4),
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
    let _landed = FrameShell::settle(&mut shell, enqueued).expect("the fire settles");
    assert_eq!(shell.airborne_steps(), 1, "the frame is in flight, unharvested");
    // The lane goes idle: no submit, no harvest, no read.
    let (told_at, outcome, thread) = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the idle lane's last frame is reported landed without another submit");
    assert_eq!(told_at, at);
    assert_ne!(
        thread,
        std::thread::current().id(),
        "told from the command buffer's completion, not the lane thread"
    );
    assert!(outcome.faulted.is_empty(), "the epilogue committed: {outcome:?}");
    assert_eq!(outcome.wrote.len(), 1, "one reporting lane");
    assert_eq!(outcome.wrote[0].tag, 7);
    // What the runtime does on hearing it: publish the landed step, which
    // commits the epilogue on the lane thread.
    shell.reap().expect("the landed step publishes");
    assert_eq!(shell.airborne_steps(), 0, "harvested by the publish");
    shell.close_program_instance(id).expect("the instance closes");
    assert!(rx.try_recv().is_err(), "the step was told once");
}
