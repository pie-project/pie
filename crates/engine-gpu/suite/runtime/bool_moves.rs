//! Every op that moves elements writes a Bool output: a gather of a mask's
//! rows by parent (beam's `inherited = gather(mask, parent)`), a gather of a
//! 1-D mask, a row gather and a transpose, each into a ring the host reads,
//! over two fires with different masks. An op that skipped a Bool store
//! answers what its scratch held (the first fire's bits, or another
//! value's), not the second fire's.

use engine::ChannelId;
use engine::Engine;
use engine::channel::{ChannelRegistration, ChannelSeed};
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::{InstanceBinding, ProgramRegistration};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;
use poem_ir::InputAxis;

use super::outputs::{Loaded, Outputs, loaded};
use super::slots::Slots;

const ROWS: u32 = 4;
const COLS: u32 = 6;
const LONG: u32 = 8;
/// The parents, each a row of the mask and a column of it.
const AT: [u32; 4] = [2, 0, 3, 2];

/// The channels: the mask, the line, the parents, the four readers, the
/// miniature's timestep. Shape, dtype, role, seeded.
fn decls() -> Vec<(Vec<u32>, EtaDtype, HostRole, bool)> {
    vec![
        (vec![ROWS, COLS], EtaDtype::Bool, HostRole::Writer, false),
        (vec![LONG], EtaDtype::Bool, HostRole::Writer, false),
        (vec![ROWS], EtaDtype::U32, HostRole::None, true),
        (vec![ROWS, COLS], EtaDtype::Bool, HostRole::Reader, false),
        (vec![ROWS], EtaDtype::Bool, HostRole::Reader, false),
        (vec![ROWS], EtaDtype::Bool, HostRole::Reader, false),
        (vec![COLS, ROWS], EtaDtype::Bool, HostRole::Reader, false),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer, false),
    ]
}

fn program() -> TraceContainer {
    let shape = |dims: &[u32]| match dims {
        [n] => Shape::vector(*n),
        [r, c] => Shape::matrix(*r, *c),
        _ => unreachable!("vectors and matrices"),
    };
    TraceContainer {
        names: Vec::new(),
        channels: decls()
            .into_iter()
            .map(|(dims, dtype, host_role, seeded)| ChannelDecl {
                shape: shape(&dims),
                dtype: ChanDType::Concrete(dtype),
                capacity: if host_role == HostRole::Reader { 1 } else { 2 },
                host_role,
                seeded,
            })
            .collect(),
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::ChanRead(2),
                Op::Gather { src: 0, idx: 2 },
                Op::ChanPut { chan: 3, value: 3 },
                Op::Gather { src: 1, idx: 2 },
                Op::ChanPut { chan: 4, value: 4 },
                Op::GatherRow { src: 0, idx: 2 },
                Op::ChanPut { chan: 5, value: 5 },
                Op::Transpose(0),
                Op::ChanPut { chan: 6, value: 6 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn bits(v: &[bool]) -> Vec<u8> {
    let mut out = vec![0u8; v.len().div_ceil(8)];
    for (k, &bit) in v.iter().enumerate() {
        if bit {
            out[k / 8] |= 1 << (k % 8);
        }
    }
    out
}

fn mask_of(salt: u32) -> Vec<bool> {
    (0..ROWS * COLS).map(|k| (k * 7 + salt) % 3 == 0).collect()
}

fn line_of(salt: u32) -> Vec<bool> {
    (0..LONG).map(|k| (k + salt) % 2 == 0).collect()
}

/// What each reader holds after a fire over `mask` and `line`, as the wire
/// carries it.
fn want(mask: &[bool], line: &[bool]) -> [Vec<u8>; 4] {
    let (r, c) = (ROWS as usize, COLS as usize);
    let gathered: Vec<bool> = AT
        .iter()
        .flat_map(|&p| mask[p as usize * c..][..c].to_vec())
        .collect();
    let picked: Vec<bool> = AT.iter().map(|&p| line[p as usize]).collect();
    let by_row: Vec<bool> = (0..r).map(|row| mask[row * c + AT[row] as usize]).collect();
    let flipped: Vec<bool> = (0..c * r).map(|k| mask[(k % r) * c + k / r]).collect();
    [
        bits(&gathered),
        bits(&picked),
        bits(&by_row),
        bits(&flipped),
    ]
}

#[test]
fn bool_outputs_are_written_by_every_op_that_moves_them() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("bool-moves", &Outputs)
    else {
        return;
    };
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    let mut outputs: Vec<OutputDecl> = Vec::new();
    for decl in loaded.caps.classes.classes.iter().flat_map(|c| &c.outputs) {
        if outputs.iter().any(|seen| seen.name == decl.name) {
            continue;
        }
        outputs.push(OutputDecl {
            name: decl.name.clone(),
            axis: match decl.axis {
                InputAxis::Lane => Axis::Lane,
                InputAxis::Row => Axis::Row,
                InputAxis::Voxel => Axis::Voxel,
            },
            width: decl.width,
            dtype: match decl.dtype {
                poem_ir::Dtype::I32 => EtaDtype::I32,
                _ => EtaDtype::F32,
            },
            layers: decl.layers,
            readout: decl.readout,
        });
    }
    let bound = eta_ir::validate::bind(program(), vec![outputs], EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0xb00e,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = decls()
        .into_iter()
        .enumerate()
        .map(|(at, (shape, dtype, host_role, seeded))| {
            let id = ChannelId(at as u64 + 1);
            slots.register(
                &mut engine,
                ChannelRegistration {
                    id,
                    shape,
                    dtype: ChanDType::Concrete(dtype),
                    host_role,
                    seeded,
                    extern_dir: None,
                    capacity: if host_role == HostRole::Reader { 1 } else { 2 },
                    extern_name: Vec::new(),
                    slots: Vec::new(),
                },
            );
            id
        })
        .collect();
    let rows = 3u32;
    let binding = slots.binding(
        &engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: vec![ChannelSeed {
                channel: 2,
                bytes: AT.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }],
            geometry: GeometryClass::Host,
            extents: engine::program::BindExtents {
                row_count: 1,
                token_count: rows,
                sampled_rows: 1,
                query_len: rows,
                ..engine::program::BindExtents::default()
            },
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            pair_generation: 0,
        },
        &channels,
    );
    let instance = engine
        .bind_instance(&binding)
        .expect("the instance binds")
        .id;
    let readers = [
        ("gather of rows", 3u32),
        ("gather of a line", 4),
        ("gather_row", 5),
        ("transpose", 6),
    ];
    let mut wrong = Vec::new();
    for salt in [0u32, 1] {
        for (channel, cell) in [
            (7u32, 0.375f32.to_le_bytes().to_vec()),
            (0, bits(&mask_of(salt))),
            (1, bits(&line_of(salt))),
        ] {
            assert!(
                engine
                    .publish_channel(instance, channel, &cell)
                    .expect("the cell publishes"),
                "the ring had room"
            );
        }
        let lane = Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(rows, false)),
            tokens: vec![3, 11, 7],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed {
                id: t_input,
                channel: channels[7],
            }],
            ..Lane::default()
        };
        let mut ticket = engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![lane],
                attachments: vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: ::engine::ScratchPlace::None,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");
        let expected = want(&mask_of(salt), &line_of(salt));
        for ((name, channel), want) in readers.iter().zip(&expected) {
            let got = engine
                .take_channel(instance, *channel)
                .expect("the channel reads")
                .expect("the channel holds a cell");
            if &got != want {
                wrong.push(format!("fire {salt}, {name}: {got:02x?} != {want:02x?}"));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    eprintln!("every Bool output that moves elements: exact over two fires");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
