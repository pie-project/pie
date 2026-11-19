#![cfg(target_vendor = "apple")]

//! A decode envelope keeps two passes in flight: its next pass is staged in
//! its other set of tables while the last one runs, on the cursors that
//! one's commit will leave. Both commit and report, in order. A pass whose
//! frame fails takes the one staged after it along (staged on a commit
//! that never came), and the instance's next fire commits.

use std::sync::mpsc;
use std::time::Duration;

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::frame::Shell as FrameShell;
use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated, StepView};
use engine_metal::{Done, Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_ahead, word};

const ROWS: u32 = 1;
const FIRST_TOKEN: i32 = 42;

/// A decode envelope's least form: its token rides channel 1 (`tok`,
/// seeded), which `embed_tokens` reads; the epilogue puts the logits'
/// argmax on channel 0 for the host and as the next token on `tok`.
fn program(vocab: u32) -> ProgramRegistration {
    let logits = OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: vocab,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    };
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(EtaDtype::I32),
                capacity: 4,
                host_role: HostRole::Reader,
                seeded: false,
            },
            ChannelDecl {
                shape: Shape::vector(1),
                dtype: ChanDType::Concrete(EtaDtype::I32),
                capacity: 2,
                host_role: HostRole::None,
                seeded: true,
            },
        ],
        ports: vec![PortBinding {
            port: Port::EmbedTokens,
            source: PortSource::Channel(1),
        }],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, vocab),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
                Op::ChanTake(1),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let bound = eta_ir::validate::bind(
        container,
        vec![vec![logits]],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the program binds");
    let stages = compile_bound(&bound);
    let backend = Backend::parse("metal").expect("the metal codegen backend");
    ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    }
}

/// Binds the envelope as a decode envelope at `slot`'s places, `tok`
/// seeded with [`FIRST_TOKEN`].
pub(super) fn bind_envelope(shell: &mut Shell, vocab: u32, slot: u32) -> engine::InstanceId {
    let program = shell.register_program(&program(vocab)).expect("the program registers");
    let facts = shell.program_facts(program).expect("the program's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    let free = layout.free_chunk();
    let base = (free + slot) * instance.per_chunk;
    let slots: Vec<u32> = (base..base + facts.instance.slots).collect();
    let mut next = (free + 2 + slot) * channel.per_chunk;
    let rings: Vec<engine::RingSlots> = facts
        .rings
        .iter()
        .map(|ring| {
            let slots = (next..next + ring.footprint.slots).collect();
            next += ring.footprint.slots;
            engine::RingSlots {
                channel: ring.channel,
                slots,
            }
        })
        .collect();
    let extents = eta_exec::Extents {
        row_count: ROWS,
        token_count: ROWS,
        sampled_rows: 1,
        query_len: ROWS,
        ..eta_exec::Extents::default()
    };
    let (scratch, pair) = crate::guest_places(shell, program, &extents, slot);
    shell
        .bind_program(
            program,
            &[(1, FIRST_TOKEN.to_le_bytes().to_vec())],
            extents,
            GeometryClass::DecodeEnvelope,
            PortMask::DECODE_ENVELOPE,
            &[engine::ChannelId(90), engine::ChannelId(91)],
            &slots,
            &rings,
            &scratch,
            &pair,
            0,
        )
        .expect("the decode envelope binds")
}

pub(super) type Heard = (engine::StepDone, engine::StepOutcome);

/// Submits the instance's lane as step `frame` (its buffer failing where
/// `fail`), harvesting nothing.
pub(super) fn submit(
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
        held: Some(0),
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
    FrameShell::settle(shell, enqueued).expect("the fire settles");
}

pub(super) fn heard(rx: &mpsc::Receiver<Heard>, frame: u64) -> engine::StepOutcome {
    let (at, outcome) = rx
        .recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("step {frame} is told"));
    assert_eq!(at.frame, engine::FrameId(frame), "told in order");
    outcome
}

#[test]
fn an_envelope_keeps_two_passes_in_flight() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_ahead(&fixture, 1024, None, engine::runahead::Runahead::of(2));
    let tokens: Vec<u32> = vec![7];
    shell.open(3).expect("slot 3 opens");
    let vocab = shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: 3,
                word: word(&fixture.trace, 1),
                tokens: &tokens,
            })
        }])
        .expect("a plain fire lands")[0]
        .len() as u32;
    let id = bind_envelope(&mut shell, vocab, 0);
    shell.open(0).expect("the slot opens");
    let (tx, rx) = mpsc::channel();

    // Two passes in flight at once: both commit and report, in order.
    submit(&mut shell, id, &tokens, 1, false, &tx);
    submit(&mut shell, id, &tokens, 2, false, &tx);
    assert_eq!(shell.airborne_steps(), 2, "the second staged while the first flies");
    for frame in [1, 2] {
        let outcome = heard(&rx, frame);
        assert!(outcome.faulted.is_empty(), "step {frame} commits: {outcome:?}");
    }
    shell.reap().expect("the landed steps publish");

    // The first's frame fails: it is lost, and the one staged on its commit
    // goes with it.
    submit(&mut shell, id, &tokens, 3, true, &tx);
    submit(&mut shell, id, &tokens, 4, false, &tx);
    let lost = heard(&rx, 3);
    assert_eq!(lost.faulted.len(), 1, "the failed step's lane is lost: {lost:?}");
    let after = heard(&rx, 4);
    assert_eq!(after.faulted.len(), 1, "the one staged on it is not committed: {after:?}");
    assert!(
        after.faulted[0].reason.contains("staged before it"),
        "{}",
        after.faulted[0].reason
    );
    shell.reap().expect("the landed steps publish");

    // The instance stands where it stood: its next fire commits.
    submit(&mut shell, id, &tokens, 5, false, &tx);
    let next = heard(&rx, 5);
    assert!(next.faulted.is_empty(), "the next fire commits: {next:?}");
    shell.reap().expect("the landed step publishes");
    shell.close_program_instance(id).expect("the instance closes");
    assert!(rx.try_recv().is_err(), "each step told once");
}
