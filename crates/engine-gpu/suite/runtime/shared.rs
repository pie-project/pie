//! A guest whose stage launches a chain of whole-row ops, three of its
//! values read by the host after the last: every cell is exact against the
//! host's arithmetic over three passes, the second's id unlike the others',
//! whichever values of the chain lie in one another's scratch bytes.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;
use poem_ir::InputAxis;

use super::outputs::{NAME, Outputs, VOCAB, WIDTH, contract_for, planes, write};
use super::slots::Slots;
use crate::backend;

/// A row of the chain: wider than a scratch alignment, so two values lie
/// apart or in the same bytes and never by chance side by side in a line.
const WIDE: u32 = 4096;

/// What the host reads of a pass whose readout id is `id`: `doubled` and
/// `flat` and `last`, each `WIDE` copies of one word.
fn wanted(id: i32) -> [i32; 3] {
    let doubled = id + id;
    let tripled = doubled + id;
    let six = tripled + tripled;
    let eight = six + doubled;
    let two = eight - six;
    let ten = two + eight;
    let product = ten * two;
    [doubled, ten, product + ten]
}

/// The chain: each value is read by the launch after it and by one two or
/// three later, so one whose readers are done lies free while later ones
/// are written; `doubled`, the reshape of `ten` and `last` are put.
fn program() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: vec!["ids".to_string()],
        channels: vec![
            decl(Shape::matrix(1, WIDE), EtaDtype::I32, HostRole::Reader),
            decl(Shape::vector(WIDE), EtaDtype::I32, HostRole::Reader),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
            decl(Shape::matrix(1, WIDE), EtaDtype::I32, HostRole::Reader),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // 0: the readout id.
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, 1),
                    dtype: EtaDtype::I32,
                },
                // 1: a row of it.
                Op::Broadcast {
                    value: 0,
                    shape: Shape::matrix(1, WIDE),
                },
                // 2: doubled, put.
                Op::Add(1, 1),
                Op::ChanPut { chan: 0, value: 2 },
                // 3: tripled; 4: six; 5: eight; 6: two; 7: ten.
                Op::Add(2, 1),
                Op::Add(3, 3),
                Op::Add(4, 2),
                Op::Sub(5, 4),
                Op::Add(6, 5),
                // 8: ten as one row, put.
                Op::Reshape {
                    value: 7,
                    shape: Shape::vector(WIDE),
                },
                Op::ChanPut { chan: 1, value: 8 },
                // 9: ten times two; 10: that and ten, put.
                Op::Mul(7, 6),
                Op::Add(9, 7),
                Op::ChanPut { chan: 3, value: 10 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

#[test]
fn a_chain_of_rows_answers_the_host_cell_for_cell() {
    let boot = crate::gpu::device_boot();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-shared-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    let planes = planes();
    write(&path, &planes);
    let argmax: Vec<i32> = (0..VOCAB as usize)
        .map(|token| {
            let row = &planes["embed"].1[token * WIDTH as usize..][..WIDTH as usize];
            (0..row.len())
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .expect("a row") as i32
        })
        .collect();
    // Two tokens whose ids are unlike and neither zero: a pass that read
    // the other's bytes, or none, answers another word.
    let first = (0..VOCAB)
        .find(|&token| argmax[token as usize] != 0)
        .expect("a row whose argmax is not its first column");
    let second = (0..VOCAB)
        .find(|&token| {
            let id = argmax[token as usize];
            id != 0 && wanted(id) != wanted(argmax[first as usize])
        })
        .expect("a second row of another id");

    let trace = poem::trace_hybrid(NAME, &Outputs, backend::PLATFORM);
    let facts = trace.facts.clone();
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    let mut engine = backend::Engine::new(crate::gpu::fires_stated(boot), contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 16,
                buckets: vec![16],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
            steps: Default::default(),
        })
        .expect("the miniature loads");

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
        .expect("the chain binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x5a4e,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channels: Vec<ChannelId> = [
        (vec![1, WIDE], EtaDtype::I32, HostRole::Reader),
        (vec![WIDE], EtaDtype::I32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
        (vec![1, WIDE], EtaDtype::I32, HostRole::Reader),
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

    // The timestep's one cell feeds every pass: a fire reads it where it
    // lies and leaves it there.
    assert!(
        engine
            .publish_channel(instance, 2, &0.375f32.to_le_bytes())
            .expect("the timestep publishes"),
        "the ring had room"
    );
    for token in [first, second, first] {
        let lane = Lane {
            slot: 0,
            word: facts.word(&Request::new(rows, false)),
            tokens: vec![3, 11, token],
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
            }))
            .expect("the frame fires");
        engine.settle_frame(&mut ticket).expect("the frame settles");

        let id = argmax[token as usize];
        for (dense, want) in [0u32, 1, 3].into_iter().zip(wanted(id)) {
            let cell = engine
                .take_channel(instance, dense)
                .expect("the channel reads")
                .expect("the channel holds a cell");
            assert_eq!(cell.len(), WIDE as usize * 4, "channel {dense}'s row");
            let unlike = cell
                .as_chunks::<4>()
                .0
                .iter()
                .position(|word| i32::from_le_bytes(*word) != want);
            assert_eq!(
                unlike, None,
                "token {token} (id {id}), channel {dense}: a cell is not {want}"
            );
        }
        eprintln!("token {token} (id {id}): {:?} in every cell", wanted(id));
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
