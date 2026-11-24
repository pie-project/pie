//! A guest frame that records two readout widens: an epilogue reading the
//! bf16 `logits` output twice, each root through its own widen into its
//! own channel. A frame records as many widens as its program reads roots
//! (a margin epilogue reads logits' sample and its top-2, a drafting one
//! the drafts beside them), each carrying its own offsets, so the second
//! runs with its own and both channels hold the row read out.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, Lane, Readout, Step};
use engine::program::{InstanceBinding, ProgramRegistration};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;
use poem_ir::InputAxis;

use super::outputs::{Loaded, Rows, WIDTH, loaded};

/// Two roots of `logits`, each put to its own channel.
fn epilogue() -> TraceContainer {
    let decl = || ChannelDecl {
        shape: Shape::matrix(1, WIDTH),
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    };
    let logits = || Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, WIDTH),
        dtype: EtaDtype::F32,
    };
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![decl(), decl()],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                logits(),
                logits(),
                Op::ChanPut { chan: 0, value: 0 },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn a_frame_of_two_readout_widens_serves() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        planes,
        dir,
    }) = loaded("two-widens", &Rows)
    else {
        return;
    };
    let decls: Vec<OutputDecl> = loaded
        .caps
        .classes
        .classes
        .iter()
        .flat_map(|class| &class.outputs)
        .filter(|decl| decl.name == "logits")
        .take(1)
        .map(|decl| OutputDecl {
            name: decl.name.clone(),
            axis: match decl.axis {
                InputAxis::Lane => Axis::Lane,
                InputAxis::Row => Axis::Row,
                InputAxis::Voxel => Axis::Voxel,
            },
            width: decl.width,
            dtype: EtaDtype::F32,
            layers: decl.layers,
            readout: decl.readout,
        })
        .collect();
    let bound = eta_ir::validate::bind(epilogue(), vec![decls], EngineCaps::default())
        .expect("the epilogue binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x2e1d,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("a program of two widens a frame registers");
    let chunks = &loaded.caps.chunks;
    let mut slots =
        super::slots::Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = (0..2u64)
        .map(|at| {
            let id = ChannelId(at + 1);
            slots.register(
                &mut engine,
                ChannelRegistration {
                    id,
                    shape: vec![1, WIDTH],
                    dtype: ChanDType::Concrete(EtaDtype::F32),
                    host_role: HostRole::Reader,
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
    let tokens = vec![3u32, 11, 7];
    let rows = tokens.len() as u32;
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
    let mut ticket = engine
        .submit(&FrameSubmission::of(Step {
            lanes: vec![Lane {
                slot: 0,
                word: trace.facts.word(&Request::new(rows, false)),
                tokens: tokens.clone(),
                readout: Readout::Last,
                kv_less: true,
                ..Lane::default()
            }],
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
    let row = &planes["embed"].1[(tokens[2] * WIDTH) as usize..][..WIDTH as usize];
    for dense in 0..2u32 {
        let got: Vec<f32> = engine
            .take_channel(instance, dense)
            .expect("the channel reads")
            .expect("the channel holds a cell")
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(got.len(), WIDTH as usize, "root {dense}: one row");
        for (c, (&have, &want)) in got.iter().zip(row).enumerate() {
            assert!(
                (have - want).abs() <= 1e-2 * want.abs().max(1.0),
                "root {dense} column {c}: widened {have}, the row read out holds {want}"
            );
        }
    }
    assert_eq!(
        engine.live().tallied(engine::live::Tally::BuiltInFire),
        0,
        "a frame of two widens built a pipeline the load did not"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
