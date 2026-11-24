#![cfg(target_vendor = "apple")]

//! With one frame in flight the engine settles inside submit, and the
//! runtime lifts a set's fold mark at that settle, so a fire's fold report
//! must be in the completion sink's hands before the settle returns, never
//! after it from Metal's completion thread. A buffering lane that reports
//! fires with a sink: by the time its rows are taken (the harvest the
//! in-line settle runs) the sink has its `Wrote`, with the fold it froze,
//! and told it on the settling thread.

use std::sync::{Arc, Mutex};

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine::frame::Shell as FrameShell;
use engine_metal::serve::{Seated, StepView};
use engine_metal::{Done, Lane};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};

type Heard = Vec<(engine::StepOutcome, std::thread::ThreadId)>;

#[test]
fn a_settled_fires_fold_is_told_before_it_returns() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert_eq!(shell.frames_in_flight(), 1, "one frame in flight");
    assert!(shell.serves_rs_verbs(), "a hybrid model buffers recurrent rows");
    shell.open(0).expect("slot 0 opens");
    let heard: Arc<Mutex<Heard>> = Arc::default();
    let sink: engine::CompletionSink = {
        let heard = heard.clone();
        Arc::new(move |_at, outcome| {
            heard.lock().unwrap().push((outcome, std::thread::current().id()));
        })
    };
    let layout = shell.pool_layout();
    let pages = 8u32.div_ceil(layout.rs_tokens.max(1)) + 1;
    let tokens: Vec<u32> = (0..5).map(|at| (7 + at * 37) % 256).collect();
    let verb = RsVerb::Buffer {
        pages: (0..pages).collect(),
        base: 0,
        folded: 0,
        fold: FoldLen::Host(3),
        lineage: engine::fire::Lineage::Set,
    };
    let lanes = [Seated {
        replay_stated: None,
        held: Some(0),
        rs: &verb,
        rs_reset: RsReset::Fresh,
        reports: 7,
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, tokens.len() as u32),
            tokens: &tokens,
        })
    }];
    let at = engine::StepDone {
        frame: engine::FrameId(0),
        step: 0,
    };
    let prepared = FrameShell::prepare(
        &mut shell,
        StepView {
            lanes: &lanes,
            attachments: &[],
            media: &[],
            clips: &[],
            done: Some(Done { at, sink }),
        },
        None,
    )
    .expect("the fire prepares");
    let enqueued = FrameShell::enqueue(&mut shell, prepared).expect("the fire enqueues");
    let landed = FrameShell::settle(&mut shell, enqueued).expect("the fire settles");
    shell.rows_of(&landed).expect("the fire's rows");
    let heard = heard.lock().unwrap();
    assert_eq!(heard.len(), 1, "the sink heard the fire once, before the settle returned");
    let (outcome, thread) = &heard[0];
    assert_eq!(
        *thread,
        std::thread::current().id(),
        "told on the settling thread, not Metal's completion thread"
    );
    assert!(outcome.faulted.is_empty(), "no lane faulted: {outcome:?}");
    let writes = &outcome.wrote;
    assert_eq!(writes.len(), 1, "one reporting lane");
    assert_eq!(writes[0].tag, 7);
    assert_eq!(writes[0].fold, Some(3), "the fold the fire froze");
    eprintln!("the fold report was in hand, on the settling thread, before the settle returned");
}
