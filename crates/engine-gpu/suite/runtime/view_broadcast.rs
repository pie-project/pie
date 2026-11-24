//! A reshape the stage elides as a view, broadcast: the broadcast reads its
//! shape from the view as written, its bytes from the value it views.
//! `broadcast(reshape(iota(8), [1, 8]), [1, 8])` is `[0..8]`, so the cell
//! it equals 3 at is cell 3 (beam search's `eq(col, wpos)`). Broadcast by
//! the viewed `[8]` instead, its axis lines up with the output's first and
//! every cell reads element 0: no cell matches and the argmax answers 0.

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

/// The epilogue: channel 0 takes the argmax of
/// `cast(eq(broadcast(reshape(iota(8), [1, 8]), [1, 8]), 3))`; channel 1
/// is the miniature's timestep.
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
            decl(Shape::vector(1), EtaDtype::I32, HostRole::Reader),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // 0..=2: col, `[0..8]` as a row.
                Op::Iota { len: CELLS },
                Op::Reshape {
                    value: 0,
                    shape: Shape::matrix(1, CELLS),
                },
                Op::Broadcast {
                    value: 1,
                    shape: Shape::matrix(1, CELLS),
                },
                // 3..=6: the slot, 0 + 1 + 2, as a row.
                Op::Iota { len: 3 },
                Op::ReduceSum(3),
                Op::Reshape {
                    value: 4,
                    shape: Shape::matrix(1, 1),
                },
                Op::Broadcast {
                    value: 5,
                    shape: Shape::matrix(1, CELLS),
                },
                // 7..=11: the cell they are equal at.
                Op::Eq(2, 6),
                Op::Cast {
                    value: 7,
                    dtype: EtaDtype::F32,
                },
                Op::Reshape {
                    value: 8,
                    shape: Shape::vector(CELLS),
                },
                Op::ReduceArgmax(9),
                Op::Reshape {
                    value: 10,
                    shape: Shape::vector(1),
                },
                Op::ChanPut { chan: 0, value: 11 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn a_view_reshape_broadcasts_by_its_own_shape() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("view-broadcast", &Outputs)
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
    assert!(
        launch.plans.iter().any(|plan| {
            !eta_compiler::codegen::wgsl_analysis::analyze_stage(plan)
                .elided
                .is_empty()
        }),
        "a reshape is elided as a view"
    );
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0xb0ad,
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = [
        (vec![1], EtaDtype::I32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
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
            .publish_channel(instance, 1, &0.375f32.to_le_bytes())
            .expect("the timestep publishes"),
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
            channel: channels[1],
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
        .take_channel(instance, 0)
        .expect("the channel reads")
        .expect("the channel holds a cell");
    let got = i32::from_le_bytes(cell[..4].try_into().expect("an i32"));
    assert_eq!(got, 3, "the cell `eq(col, 3)` holds at is cell 3");
    eprintln!("argmax of eq(col, 3): {got}");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
