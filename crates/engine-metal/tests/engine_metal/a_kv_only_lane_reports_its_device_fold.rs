#![cfg(target_vendor = "apple")]

//! A lane of a model with no recurrent state (kv only) whose attached
//! program states its fold (a `fold` port) and which asks a report tells, at
//! settle, the fold its program stated for it, as a buffering lane does: the
//! runtime adds it to the set's frozen length, and a set whose fold went
//! unreported never comes to rest (a fork or a save of it is refused).
//! Negative control: the same lane under a program with no `fold` port
//! reports no fold, so the count told is the port's, not one made up.

use std::sync::{Arc, Mutex};

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::frame::Shell as FrameShell;
use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated, StepView};
use engine_metal::{Done, Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};
use crate::common::{const_port};

const ROWS: u32 = 5;
const FOLD: u32 = 3;

/// One lane of `ROWS` tokens; with `fold`, a `fold` port stating `FOLD` for
/// it. The epilogue puts the logits' argmax in a ring.
pub(super) fn program(vocab: u32, tokens: &[u32], fold: bool) -> ProgramRegistration {
    let logits = OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: vocab,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    };
    let mut ports = vec![
        const_port(Port::EmbedTokens, tokens),
        const_port(Port::EmbedIndptr, &[0, ROWS]),
    ];
    if fold {
        ports.push(const_port(Port::Fold, &[FOLD]));
    }
    ports.sort_by_key(|binding| binding.port);
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(EtaDtype::I32),
            capacity: 4,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports,
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

/// Registers the program (with or without its `fold` port) and binds a
/// device-geometry instance of it at `slot`'s places.
pub(super) fn bind_reporting(
    shell: &mut Shell,
    vocab: u32,
    tokens: &[u32],
    fold: bool,
    slot: u32,
) -> engine::InstanceId {
    let program = shell
        .register_program(&program(vocab, tokens, fold))
        .expect("the program registers");
    let facts = shell.program_facts(program).expect("the program's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    // From the first chunk no slot holds: past the engine's tables run,
    // which no slot names, and the seats' runs.
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
            &[],
            extents,
            GeometryClass::DeviceGeometry,
            PortMask::NONE,
            &[engine::ChannelId(70 + u64::from(slot))],
            &slots,
            &rings,
            &scratch,
            &pair,
            0,
        )
        .expect("the device-geometry instance binds")
}

/// Fires one reporting kv-only lane under a program (with or without its
/// `fold` port) and returns what the completion sink heard.
fn report(shell: &mut Shell, vocab: u32, tokens: &[u32], fold: bool, slot: u32) -> engine::StepOutcome {
    let id = bind_reporting(shell, vocab, tokens, fold, slot);
    shell.open(slot).expect("the slot opens");
    let heard: Arc<Mutex<Vec<engine::StepOutcome>>> = Arc::default();
    let sink: engine::CompletionSink = {
        let heard = heard.clone();
        Arc::new(move |_at, outcome| heard.lock().unwrap().push(outcome))
    };
    let lanes = [Seated {
        replay_stated: None,
        held: Some(0),
        reports: 7,
        ..Seated::of(Lane {
            slot,
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
        frame: engine::FrameId(0),
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
    .expect("the fire prepares");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the fire enqueues");
    let landed = FrameShell::settle(shell, enqueued).expect("the fire settles");
    shell.rows_of(&landed).expect("the fire's rows");
    shell.close_program_instance(id).expect("the instance closes");
    let heard = heard.lock().unwrap();
    assert_eq!(heard.len(), 1, "the sink heard the fire once");
    heard[0].clone()
}

#[test]
fn a_kv_only_lane_reports_its_device_fold() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(!shell.serves_rs_verbs(), "a kv-only model");
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
    match report(&mut shell, vocab, &tokens, true, 0) {
        engine::StepOutcome {
            wrote: writes,
            faulted,
            ..
        } if faulted.is_empty() && !writes.is_empty() => {
            assert_eq!(writes.len(), 1, "one reporting lane");
            assert_eq!(writes[0].tag, 7);
            assert_eq!(writes[0].fold, Some(FOLD), "the fold its program stated");
        }
        other => panic!("the lane's report, not {other:?}"),
    }
    match report(&mut shell, vocab, &tokens, false, 1) {
        engine::StepOutcome {
            wrote: writes,
            faulted,
            ..
        } if faulted.is_empty() && !writes.is_empty() => {
            assert_eq!(writes.len(), 1, "one reporting lane");
            assert_eq!(writes[0].fold, None, "no `fold` port, no fold told");
        }
        other => panic!("the lane's report, not {other:?}"),
    }
}
