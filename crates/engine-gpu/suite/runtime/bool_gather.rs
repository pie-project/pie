//! Bool values a stage moves rather than computes: a row gathered by a
//! parent index, then or-ed with an `eq` (beam search's
//! `or(gather(mask, parent), eq(col, slot))`, the ancestry a beam's mask
//! carries), the mask transposed, and a cell of each row gathered. Each
//! holds the cells it names, every one: a stage whose op writes no bool
//! cells leaves its value what the scratch held, and beam's mask loses its
//! ancestry.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
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

const CELLS: u32 = 8;

/// The parent's row: row 1 of the mask channel's two.
const PARENT_ROW: [bool; 8] = [true, false, true, true, false, false, false, false];
/// The other row, which a gather that read the wrong row would answer.
const OTHER_ROW: [bool; 8] = [true, true, false, false, false, false, false, false];
/// The slot the step writes: `0 + 1 + 2 + 3`.
const SLOT: usize = 6;

/// The cell of each row `gather_row` reads: row 0's 2, row 1's 3.
const ROW_CELLS: [usize; 2] = [2, 3];

/// The epilogue: channel 0 takes the `[2, 8]` mask, channel 1 gets
/// `cast(or(gather(mask, [1]), eq(col, 6)), i32)`, channel 2 is the
/// miniature's timestep, channel 3 the mask transposed and channel 4
/// `gather_row(mask, [2, 3])`, as i32.
fn program() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(Shape::matrix(2, CELLS), EtaDtype::Bool, HostRole::Writer),
            decl(Shape::matrix(1, CELLS), EtaDtype::I32, HostRole::Reader),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
            decl(Shape::matrix(CELLS, 2), EtaDtype::I32, HostRole::Reader),
            decl(Shape::vector(2), EtaDtype::I32, HostRole::Reader),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // 0: the mask.
                Op::ChanTake(0),
                // 1..=4: the parent's row, `gather(mask, [1])`.
                Op::Iota { len: 2 },
                Op::ReduceSum(1),
                Op::Reshape {
                    value: 2,
                    shape: Shape::vector(1),
                },
                Op::Gather { src: 0, idx: 3 },
                // 5..=10: the slot's cell, `eq(col, 6)`.
                Op::Iota { len: CELLS },
                Op::Reshape {
                    value: 5,
                    shape: Shape::matrix(1, CELLS),
                },
                Op::Iota { len: 4 },
                Op::ReduceSum(7),
                Op::Reshape {
                    value: 8,
                    shape: Shape::matrix(1, 1),
                },
                Op::Broadcast {
                    value: 9,
                    shape: Shape::matrix(1, CELLS),
                },
                Op::Eq(6, 10),
                // 12..=13: the new mask, as words the host reads.
                Op::Or(4, 11),
                Op::Cast {
                    value: 12,
                    dtype: EtaDtype::I32,
                },
                Op::ChanPut { chan: 1, value: 13 },
                // 14..=15: the mask transposed.
                Op::Transpose(0),
                Op::Cast {
                    value: 14,
                    dtype: EtaDtype::I32,
                },
                Op::ChanPut { chan: 3, value: 15 },
                // 16..=21: `gather_row(mask, iota(2) + 2)`.
                Op::Iota { len: 2 },
                Op::Const(eta_ir::types::Literal::U32(2)),
                Op::Broadcast {
                    value: 17,
                    shape: Shape::vector(2),
                },
                Op::Add(16, 18),
                Op::GatherRow { src: 0, idx: 19 },
                Op::Cast {
                    value: 20,
                    dtype: EtaDtype::I32,
                },
                Op::ChanPut { chan: 4, value: 21 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// A bool cell's wire bytes: a bit a cell, the first the lowest.
fn bits(cells: &[bool]) -> Vec<u8> {
    let mut bytes = vec![0u8; cells.len().div_ceil(8)];
    for (at, &cell) in cells.iter().enumerate() {
        if cell {
            bytes[at / 8] |= 1 << (at % 8);
        }
    }
    bytes
}

#[test]
fn a_gathered_bool_row_keeps_its_parents_cells() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("bool-gather", &Outputs)
    else {
        return;
    };
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    let mut decls: Vec<OutputDecl> = Vec::new();
    for decl in loaded.caps.classes.classes.iter().flat_map(|c| &c.outputs) {
        if decls.iter().any(|seen| seen.name == decl.name) {
            continue;
        }
        decls.push(OutputDecl {
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
    let bound = eta_ir::validate::bind(program(), vec![decls], EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0xb001,
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = [
        (vec![2, CELLS], EtaDtype::Bool, HostRole::Writer),
        (vec![1, CELLS], EtaDtype::I32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
        (vec![CELLS, 2], EtaDtype::I32, HostRole::Reader),
        (vec![2], EtaDtype::I32, HostRole::Reader),
    ]
    .into_iter()
    .enumerate()
    .map(|(at, (shape, dtype, host_role))| {
        let id = ChannelId(at as u64 + 1);
        slots.register(
            &mut engine,
            ChannelRegistration {
                id,
                shape,
                dtype: ChanDType::Concrete(dtype),
                host_role,
                seeded: false,
                extern_dir: None,
                capacity: 2,
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
            seeds: Vec::new(),
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
    assert!(
        engine
            .publish_channel(instance, 2, &0.375f32.to_le_bytes())
            .expect("the timestep publishes"),
        "the ring had room"
    );
    let mask_rows: Vec<bool> = OTHER_ROW.iter().chain(&PARENT_ROW).copied().collect();
    assert!(
        engine
            .publish_channel(instance, 0, &bits(&mask_rows))
            .expect("the mask publishes"),
        "the ring had room"
    );
    let lane = Lane {
        slot: 0,
        word: trace.facts.word(&Request::new(rows, false)),
        tokens: vec![3, 11, 7],
        readout: Readout::Last,
        kv_less: true,
        inputs: vec![InputFeed {
            id: t_input,
            channel: channels[2],
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
            shape: None,
        }))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let cell = engine
        .take_channel(instance, 1)
        .expect("the channel reads")
        .expect("the channel holds a cell");
    let got: Vec<bool> = cell
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word) != 0)
        .collect();
    let mut want = PARENT_ROW;
    want[SLOT] = true;
    assert_eq!(got, want, "the parent's cells and the slot's");
    eprintln!("or(gather(mask, parent), eq(col, slot)): {got:?}");

    let mut words = |dense: u32| -> Vec<bool> {
        engine
            .take_channel(instance, dense)
            .expect("the channel reads")
            .expect("the channel holds a cell")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word) != 0)
            .collect()
    };
    let transposed: Vec<bool> = (0..CELLS as usize)
        .flat_map(|col| [OTHER_ROW[col], PARENT_ROW[col]])
        .collect();
    assert_eq!(words(3), transposed, "the mask transposed");
    let picked = vec![OTHER_ROW[ROW_CELLS[0]], PARENT_ROW[ROW_CELLS[1]]];
    assert_eq!(words(4), picked, "a cell of each row");
    eprintln!("transpose and gather_row of the mask: exact");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
