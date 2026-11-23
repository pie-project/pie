#![cfg(target_vendor = "apple")]

//! A guest reads only the rows its own lanes hold. A logits reader whose
//! epilogue declares more readout rows than its lane holds would read the
//! next lanes' rows, another request's, from the lane's seat on: its
//! binding is refused by name, its lane alone (`Fault::OutputRows`), and it
//! never launches. One declaring the row its lane holds fires. Two lanes of
//! one decode, the reader attached at the first, the second a plain
//! request beside it.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::frame::Shell as FrameShell;
use engine::program::ProgramRegistration;
use engine_metal::serve::{Seated, StepView};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, word};
use super::a_score_reader_holds_its_rows_through_a_park::at_lane;

const CONTEXT: u32 = 256;
const PROMPT: u32 = 12;

/// An epilogue that sums `rows` readout rows of `logits` from its lane on
/// into a reader channel.
fn logits_reader(vocab: u32, rows: u32) -> ProgramRegistration {
    let logits = OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: vocab,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    };
    let container = TraceContainer {
        names: vec!["logits".into()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(rows),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 8,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(rows, vocab),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceSum(0),
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
    .expect("the reader binds");
    let stages = compile_bound(&bound);
    let backend = Backend::parse("metal").expect("the metal codegen backend");
    ProgramRegistration {
        program_hash: bound.hash ^ u64::from(rows),
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    }
}

/// Binds `program` for a one-row decode lane that samples `rows` rows, its
/// tables in `Instance` chunk `chunk` and its ring in `Channel` chunk
/// `chunk + 2`.
fn bind(shell: &mut Shell, program: engine::ProgramId, rows: u32, chunk: u32) -> engine::InstanceId {
    let facts = shell.program_facts(program).expect("the reader's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    let chunk = layout.free_chunk() + chunk - 1;
    let first = chunk * instance.per_chunk;
    let slots: Vec<u32> = (first..first + facts.instance.slots).collect();
    let mut next = (chunk + 2) * channel.per_chunk;
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
        row_count: 1,
        token_count: 1,
        sampled_rows: rows,
        query_len: 1,
        ..eta_exec::Extents::default()
    };
    let (scratch, pair) = crate::guest_places(shell, program, &extents, chunk);
    shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[engine::ChannelId(0x5e0f_0000 + u64::from(chunk))],
            &slots,
            &rings,
            &scratch,
            &pair,
            0,
        )
        .expect("the reader binds")
}

/// Two lanes prefilled, then one decode of both with a reader of `rows`
/// rows attached at lane 0: the enqueue's verdict, and the reader.
fn step(fixture: &Fixture, rows: u32) -> (Result<(), String>, engine::InstanceId) {
    let mut shell: Shell = load_with(fixture, CONTEXT);
    let mut vocab = 0;
    for slot in 0..2u32 {
        let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11 + slot * 3) % 256).collect();
        shell.open(slot).expect("the slot opens");
        let logits = shell
            .fire_seated(&[Seated {
                replay_stated: None,
                held: Some(0),
                ..Seated::of(Lane {
                    slot,
                    word: word(&fixture.trace, PROMPT),
                    tokens: &prompt,
                })
            }])
            .expect("the prompt prefills");
        vocab = u32::try_from(logits[0].len()).expect("a vocab a u32 counts");
    }
    let program = shell
        .register_program(&logits_reader(vocab, rows))
        .expect("the reader registers");
    let id = bind(&mut shell, program, rows, 1);
    let attached = [at_lane(0, id)];
    let plain = word(&fixture.trace, 1);
    let lanes: Vec<Seated<'_>> = (0..2u32)
        .map(|slot| Seated {
            replay_stated: None,
            held: Some(PROMPT),
            ..Seated::of(Lane {
                slot,
                word: plain,
                tokens: &[5],
            })
        })
        .collect();
    let verdict = FrameShell::prepare(
        &mut shell,
        StepView {
            lanes: &lanes,
            attachments: &attached,
            media: &[],
            clips: &[],
            done: None,
        },
        None,
    )
    .and_then(|prepared| FrameShell::enqueue(&mut shell, prepared))
    .and_then(|enqueued| FrameShell::settle(&mut shell, enqueued))
    .map(|_| ())
    .map_err(|error| error.to_string());
    (verdict, id)
}

#[test]
fn a_guest_reads_only_its_own_lanes_rows() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);

    // Control: the row its lane holds.
    let (own, _) = step(&fixture, 1);
    assert!(own.is_ok(), "a reader of its own row fires: {own:?}");

    // Four rows from a lane that holds one: rows 1..3 are the next lane's
    // (another request's) and past it.
    let (wide, id) = step(&fixture, 4);
    let why = wide.expect_err("a reader of rows its lane does not hold is refused");
    assert!(
        why.contains(&format!("instance {} attached at lane 0 is refused", id.0))
            && why.contains("another lane's rows"),
        "the refusal names the instance, its lane and why: {why}"
    );
}
