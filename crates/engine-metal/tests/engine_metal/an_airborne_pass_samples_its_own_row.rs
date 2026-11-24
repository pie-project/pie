#![cfg(target_vendor = "apple")]

//! A pass in flight samples the row its own fire gave its lane, whatever the
//! next pass staged on its other set gives that lane. Frame N fires a plain
//! lane first and the decode envelope's lane after it (its row 1); frame
//! N+1, staged while N flies, fires the envelope's lane alone (its row 0):
//! staging N+1 must not rebind the output plane in the set N's epilogue is
//! still to read. The envelope's reader ring is shared with a second, idle
//! envelope, as served instances' rings are: a member on a shared ring fires
//! alone, from its own address table. Each pass's argmax
//! (the envelope's reader channel) equals what the same two fires give when
//! N lands before N+1 is staged. Frame N is held on the device
//! (`device::hold_next_frame`) until N+1 is staged, so its epilogue is truly
//! still to run then.

use std::sync::mpsc;
use std::time::Duration;

use engine::frame::Shell as FrameShell;
use engine_metal::serve::{Attached, Seated, StepView};
use engine_metal::{Done, Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_ahead, word};
use super::an_envelope_keeps_two_passes_in_flight::{Heard, bind_envelope};

/// The plain lane's slot and its token: a different context from the
/// envelope's, so its row's argmax is another lane's.
const PLAIN_SLOT: u32 = 3;
const PLAIN_TOKEN: u32 = 99;

/// Submits step `frame`: the envelope's lane (slot 0), after a plain lane
/// (slot [`PLAIN_SLOT`]) where `beside` is set, harvesting nothing.
fn submit(
    shell: &mut Shell,
    fixture: &Fixture,
    id: engine::InstanceId,
    frame: u64,
    beside: bool,
    tx: &mpsc::Sender<Heard>,
) {
    let tx = std::sync::Mutex::new(tx.clone());
    let sink: engine::CompletionSink = std::sync::Arc::new(move |at, outcome| {
        let _ = tx.lock().unwrap().send((at, outcome));
    });
    let envelope_tokens = [7u32];
    let plain_tokens = [PLAIN_TOKEN];
    let envelope = Seated {
        held: Some(0),
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, 1),
            tokens: &envelope_tokens,
        })
    };
    let plain = Seated {
        held: Some(0),
        ..Seated::of(Lane {
            slot: PLAIN_SLOT,
            word: word(&fixture.trace, 1),
            tokens: &plain_tokens,
        })
    };
    let lanes: Vec<Seated<'_>> = if beside {
        vec![plain, envelope]
    } else {
        vec![envelope]
    };
    let attached = [Attached {
        lane: u32::from(beside),
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
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the fire enqueues");
    FrameShell::settle(shell, enqueued).expect("the fire settles");
}

fn heard(rx: &mpsc::Receiver<Heard>, frame: u64) {
    let (at, outcome) = rx
        .recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("step {frame} is told"));
    assert_eq!(at.frame, engine::FrameId(frame), "told in order");
    assert!(outcome.faulted.is_empty(), "step {frame} commits: {outcome:?}");
}

/// The envelope's argmax cells, in order, after frames 1 (after the plain
/// lane) and 2 (alone); `airborne` stages frame 2 while frame 1 flies.
fn argmaxes(fixture: &Fixture, airborne: bool) -> Vec<Vec<u8>> {
    let mut shell = load_ahead(fixture, 1024, None, engine::runahead::Runahead::of(2));
    shell.open(PLAIN_SLOT).expect("the plain slot opens");
    let probe = [PLAIN_TOKEN];
    let logits = shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: PLAIN_SLOT,
                word: word(&fixture.trace, 1),
                tokens: &probe,
            })
        }])
        .expect("a plain fire lands");
    let vocab = logits[0].len() as u32;
    // The envelope's reader ring is a shared channel a second, idle
    // envelope also binds, as served instances share rings: a member on a
    // ring with another attachment fires alone, from its own address table
    // (a batch copies its members' seats when it encodes them).
    let shape = engine_metal::program::ChannelShape {
        capacity: 4,
        numel: 1,
        dtype: eta_ir::Dtype::I32,
    };
    let footprint = shell.ring_facts(shape).expect("a ring shape").footprint.slots;
    let layout = shell.pool_layout();
    let per_chunk = layout.channel.expect("a channel shape").per_chunk;
    let first = (layout.free_chunk() + 6) * per_chunk;
    let run: Vec<u32> = (first..first + footprint).collect();
    shell
        .register_shared_channel(engine::ChannelId(90), shape, &run)
        .expect("the shared reader ring registers");
    let id = bind_envelope(&mut shell, vocab, 0);
    let idle = bind_envelope(&mut shell, vocab, 1);
    shell.open(0).expect("the envelope's slot opens");
    let (tx, rx) = mpsc::channel();
    // Frame 1 held on the device until frame 2 is staged: its epilogue is
    // still to run when frame 2's staging binds its lane's row.
    if airborne {
        engine_metal::device::hold_next_frame();
    }
    submit(&mut shell, fixture, id, 1, true, &tx);
    if !airborne {
        heard(&rx, 1);
        shell.reap().expect("frame 1 publishes");
    }
    submit(&mut shell, fixture, id, 2, false, &tx);
    if airborne {
        assert_eq!(shell.airborne_steps(), 2, "frame 2 staged while frame 1 flies");
        engine_metal::device::release_held_frame();
        heard(&rx, 1);
    }
    heard(&rx, 2);
    shell.reap().expect("the landed steps publish");
    let mut cells = Vec::new();
    for _ in 0..2 {
        let cell = shell
            .program_instance_taking(id, 0)
            .expect("the reader channel reads")
            .expect("the instance is bound")
            .take(0)
            .expect("a take")
            .expect("each pass put its argmax");
        cells.push(cell);
    }
    shell.close_program_instance(id).expect("the instance closes");
    shell.close_program_instance(idle).expect("the idle instance closes");
    cells
}

#[test]
fn an_airborne_pass_samples_its_own_row() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let serial = argmaxes(&fixture, false);
    let airborne = argmaxes(&fixture, true);
    assert_eq!(
        airborne, serial,
        "a pass in flight samples its own lane's row: frame 2's staging, its lane moved to \
         row 0, leaves frame 1's binding as it was"
    );
}
