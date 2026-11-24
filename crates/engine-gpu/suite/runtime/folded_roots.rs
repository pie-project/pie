//! A stage whose compile folds two of its roots into one value (two equal
//! constants) and two of its puts into one value lays each value once in
//! its pass's berth of the guest ring: the berth the pass takes and the
//! spans its copies ask for are one layout, so a fire never asks for more
//! of its berth than it took (beam's width-1 replay asked 100 bytes of a
//! 96-byte berth, a constant staged once per name it had).

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::{InstanceBinding, ProgramRegistration};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Literal, Shape};
use poem::Request;
use poem_ir::InputAxis;

use super::outputs::{Loaded, Outputs, loaded};
use super::slots::Slots;

const CELLS: u32 = 4;
const FILL: f32 = 0.25;

/// The epilogue: `broadcast(0.25)` put to channels 0 and 1, the constant
/// and its broadcast written twice (globals 0 and 2, 1 and 3), which the
/// compile folds into one each; channel 2 is the miniature's timestep.
fn program() -> TraceContainer {
    let decl = |shape: Shape, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    let row = Shape::matrix(1, CELLS);
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(row, HostRole::Reader),
            decl(row, HostRole::Reader),
            decl(Shape::matrix(1, 1), HostRole::Writer),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Const(Literal::F32(FILL)),
                Op::Broadcast {
                    value: 0,
                    shape: row,
                },
                Op::Const(Literal::F32(FILL)),
                Op::Broadcast {
                    value: 2,
                    shape: row,
                },
                Op::ChanPut { chan: 0, value: 1 },
                Op::ChanPut { chan: 1, value: 3 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn folded_roots_and_puts_lay_once_in_their_pass() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("folded-roots", &Outputs)
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
    // The shape under test: the two constants are one value of the stage,
    // and the two puts name one value.
    assert_eq!(
        launch.plan_local(0),
        launch.plan_local(2),
        "the compile folds the equal constants into one value"
    );
    assert_eq!(
        launch.plan_local(1),
        launch.plan_local(3),
        "the compile folds the two broadcasts into one value"
    );
    let program = engine::Programs::register(&mut engine, &ProgramRegistration {
            program_hash: 0xf01d,
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = [
        (vec![1, CELLS], HostRole::Reader),
        (vec![1, CELLS], HostRole::Reader),
        (vec![1, 1], HostRole::Writer),
    ]
    .into_iter()
    .enumerate()
    .map(|(at, (shape, host_role))| {
        let id = ChannelId(at as u64 + 1);
        slots.register(
            &mut engine,
            ChannelRegistration {
                id,
                shape,
                dtype: ChanDType::Concrete(EtaDtype::F32),
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
            ..InstanceBinding::default()
        },
        &channels,
    );
    let instance = engine::Programs::bind(&mut engine, &binding)
        .expect("the instance binds")
        .id;
    // Twice: the second fire replays the first's recording where the
    // backend records.
    for round in 0..2 {
        assert!(
            engine::Channels::publish(&mut engine, instance, 2, &0.375f32.to_le_bytes())
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
        engine
            .settle_frame(&mut ticket)
            .expect("the frame settles: its pass asked for no more than its berth");
        for channel in [0, 1] {
            let cell = engine::Channels::take(&mut engine, instance, channel)
                .expect("the channel reads")
                .expect("the channel holds a cell");
            let got: Vec<f32> = cell
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| f32::from_le_bytes(*word))
                .collect();
            assert_eq!(
                got,
                vec![FILL; CELLS as usize],
                "round {round}: channel {channel} holds the folded broadcast"
            );
        }
    }
    eprintln!("two folded constants and two puts of one value: each laid once, exact");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
