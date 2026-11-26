#![cfg(target_vendor = "apple")]

//! A guest attached over two lanes reads each lane's own readout row, where
//! the fire's gathered readouts do not keep the lanes' order. The fire's
//! lanes are [the guest's one-row lane, its two-row lane, another request's
//! two-row lane]: the fire composes its lanes by class, so its readouts are
//! [the guest's second lane, the other request's, the guest's first], the
//! guest's two apart and out of its lanes' order. The guest puts its
//! readout rows' argmaxes, in its lanes' order, on its channel: each is the
//! argmax of the logits the host reads for that lane, never its neighbour's.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{argmax, fixture_of, load_with, word};

const CONTEXT: u32 = 256;
const PROMPT: u32 = 12;

/// An epilogue putting its two readout rows' argmaxes, lane 0's then lane
/// 1's, on its reader channel.
fn per_lane_argmax(vocab: u32) -> ProgramRegistration {
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
            shape: Shape::vector(2),
            dtype: ChanDType::Concrete(EtaDtype::I32),
            capacity: 4,
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
                    shape: Shape::matrix(2, vocab),
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
        vec![vec![logits.clone()], vec![logits]],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the reader binds");
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

/// Binds `program` over two lanes of `rows` rows in all, one readout row
/// each.
fn bind(shell: &mut Shell, program: engine::ProgramId, rows: u32) -> engine::InstanceId {
    let facts = shell.program_facts(program).expect("the reader's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    let chunk = layout.free_chunk();
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
        row_count: rows,
        token_count: rows,
        sampled_rows: 2,
        query_len: rows,
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
            &[engine::ChannelId(0x2a0f_0000)],
            &slots,
            &rings,
            &scratch,
            &pair,
            0,
        )
        .expect("the reader binds over two lanes")
}

#[test]
fn a_guest_over_two_lanes_reads_each_lanes_own_row() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    let mut vocab = 0;
    for slot in 0..3u32 {
        let prompt: Vec<u32> = (0..PROMPT)
            .map(|t| (t * 7 + 11 + slot * 29) % 256)
            .collect();
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
        .register_program(&per_lane_argmax(vocab))
        .expect("the reader registers");
    // The guest's lanes: a one-row lane and a two-row one (another class).
    let id = bind(&mut shell, program, 3);
    let tokens: [&[u32]; 3] = [&[5], &[61, 9], &[200, 17]];
    let lanes: Vec<Seated<'_>> = (0..3u32)
        .map(|slot| Seated {
            replay_stated: None,
            held: Some(PROMPT),
            ..Seated::of(Lane {
                slot,
                word: word(&fixture.trace, tokens[slot as usize].len() as u32),
                tokens: tokens[slot as usize],
            })
        })
        .collect();
    let rows = shell
        .fire_attached(
            &lanes,
            &[Attached {
                lane: 0,
                lanes: 2,
                instance: id,
                adapter_only: false,
            }],
        )
        .expect("the mixed fire runs");
    let want: Vec<u32> = rows.iter().map(|row| argmax(row)).collect();
    assert!(
        want[0] != want[1] && want[0] != want[2],
        "the lanes answer apart, so a row read from another lane shows: {want:?}"
    );
    shell.reap().expect("the landed step publishes");
    let cell = shell
        .program_instance_taking(id, 0)
        .expect("the reader channel reads")
        .expect("the instance is bound")
        .take(0)
        .expect("a take")
        .expect("the epilogue put its argmaxes");
    let got: Vec<u32> = cell
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().expect("an i32 cell")))
        .collect();
    assert_eq!(
        got,
        want[..2],
        "each of the guest's lanes reads its own row (the host's argmaxes {want:?})"
    );
    shell
        .close_program_instance(id)
        .expect("the instance closes");
}
