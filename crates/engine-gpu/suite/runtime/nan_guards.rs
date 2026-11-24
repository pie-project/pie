//! The guest runtime's NaN guards on the device, each on a row that holds
//! real NaNs (the first element among them, and the one past the largest):
//! a max and a min that skip them, an argmax that never names one, a top-k
//! and a sort that rank them last, and rank cuts (a whole row past a
//! workgroup, which the one-row select serves, and seven short rows) that
//! keep none. Each lands what the host interpreter does, bit for bit. A
//! guard written as a self-compare (`x != x`) folds to false under WGSL's
//! compilers and never fires: the NaN at the head of a row then wins the
//! max, leads the sort and takes a place in the cut.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::InstanceBinding;
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Literal, Predicate, Shape};
use poem::Request;

use super::outputs::{Loaded, Outputs, loaded, output_decls, registration_of};
use super::slots::Slots;

/// 7 * 11 * 13: odd, and past a workgroup's 256 invocations.
const N: u32 = 1001;
const ROWS: u32 = 7;
const COLS: u32 = 143;
const TOP: u32 = 5;

/// The readers, in channel order past the writers (0: the values, 1: the
/// miniature's timestep).
const READERS: [(&str, u32, EtaDtype); 8] = [
    ("max", 1, EtaDtype::F32),
    ("min", 1, EtaDtype::F32),
    ("argmax", 1, EtaDtype::I32),
    ("top-5 values", TOP, EtaDtype::F32),
    ("top-5 ids", TOP, EtaDtype::U32),
    ("sorted descending", N, EtaDtype::F32),
    ("rank-5 cut of the row", N, EtaDtype::I32),
    ("rank-5 cut of each [7, 143] row", N, EtaDtype::I32),
];

fn program() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    };
    let mut channels = vec![
        decl(Shape::vector(N), EtaDtype::F32, HostRole::Writer),
        decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
    ];
    channels.extend(
        READERS
            .iter()
            .map(|&(_, len, dtype)| decl(Shape::vector(len), dtype, HostRole::Reader)),
    );
    let one = |value| Op::Reshape {
        value,
        shape: Shape::vector(1),
    };
    let words = |value| Op::Cast {
        value,
        dtype: EtaDtype::I32,
    };
    let ops = vec![
        Op::ChanTake(0),                                         // 0 f32 [N]
        Op::ReduceMax(0),                                        // 1
        one(1),                                                  // 2
        Op::ChanPut { chan: 2, value: 2 },
        Op::ReduceMin(0),                                        // 3
        one(3),                                                  // 4
        Op::ChanPut { chan: 3, value: 4 },
        Op::ReduceArgmax(0),                                     // 5 i32
        one(5),                                                  // 6
        Op::ChanPut { chan: 4, value: 6 },
        Op::TopK { input: 0, k: TOP },                           // 7 values, 8 ids
        Op::ChanPut { chan: 5, value: 7 },
        Op::ChanPut { chan: 6, value: 8 },
        Op::SortDesc(0),                                         // 9 values, 10 order
        Op::ChanPut { chan: 7, value: 9 },
        Op::Const(Literal::I32(TOP as i32)),                     // 11
        Op::PivotThreshold { input: 0, predicate: Predicate::RankLe(11) }, // 12 bool [N]
        words(12),                                               // 13
        Op::ChanPut { chan: 8, value: 13 },
        Op::Reshape { value: 0, shape: Shape::matrix(ROWS, COLS) }, // 14
        Op::PivotThreshold { input: 14, predicate: Predicate::RankLe(11) }, // 15
        Op::Reshape { value: 15, shape: Shape::vector(N) },      // 16
        words(16),                                               // 17
        Op::ChanPut { chan: 9, value: 17 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// Distinct signed values, a NaN at the head of the row and of every
/// [7, 143] row, every 89th, and one just past the largest value.
fn values() -> Vec<f32> {
    let mut x = 0x2545_f491u32;
    let mut v: Vec<f32> = (0..N)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if i % 89 == 0 || i % COLS == 0 {
                f32::NAN
            } else {
                (x % 40_000) as f32 / 4093.0 - 4.5
            }
        })
        .collect();
    let top = (0..N as usize)
        .filter(|&i| !v[i].is_nan())
        .max_by(|&a, &b| v[a].total_cmp(&v[b]))
        .expect("a value");
    if top + 1 < v.len() {
        v[top + 1] = f32::NAN;
    }
    v
}

/// What the interpreter lands in each reader, as f32.
fn interpreted(decls: &[eta_ir::registry::OutputDecl], x: &[f32]) -> Vec<Vec<f32>> {
    let bound = eta_ir::validate::bind(program(), vec![decls.to_vec()], EngineCaps::default())
        .expect("the program binds");
    let mut inst = Instance::new(&bound, &[]).expect("an instance");
    inst.host_put(&bound, 0, Value::F32(x.to_vec())).unwrap();
    inst.host_put(&bound, 1, Value::F32(vec![0.375])).unwrap();
    let stepped = inst
        .step(&bound, &PassInputs::default(), &mut NoKernels)
        .expect("the interpreter steps");
    assert!(stepped.committed);
    (0..READERS.len() as u32)
        .map(|at| match inst.host_take(&bound, 2 + at).expect("a cell") {
            Value::F32(v) => v,
            Value::U32(v) => v.into_iter().map(|u| u as f32).collect(),
            Value::I32(v) => v.into_iter().map(|u| u as f32).collect(),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn cells(raw: &[u8], dtype: EtaDtype) -> Vec<f32> {
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| match dtype {
            EtaDtype::I32 => i32::from_le_bytes(*c) as f32,
            EtaDtype::U32 => u32::from_le_bytes(*c) as f32,
            _ => f32::from_le_bytes(*c),
        })
        .collect()
}

#[test]
fn nan_guards_skip_real_nans_on_the_device() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("nan-guards", &Outputs)
    else {
        return;
    };
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    let decls = output_decls(&loaded);
    let program = engine
        .register_program(&registration_of(&decls, program(), 0x7fc0))
        .expect("the NaN program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let shapes: Vec<(Vec<u32>, EtaDtype, HostRole)> = [
        (vec![N], EtaDtype::F32, HostRole::Writer),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
    ]
    .into_iter()
    .chain(
        READERS
            .iter()
            .map(|&(_, len, dtype)| (vec![len], dtype, HostRole::Reader)),
    )
    .collect();
    let channels: Vec<ChannelId> = shapes
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
    let x = values();
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert!(engine.publish_channel(instance, 0, &bytes).expect("the values publish"));
    assert!(
        engine
            .publish_channel(instance, 1, &0.375f32.to_le_bytes())
            .expect("the timestep publishes")
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
    let want = interpreted(&decls, &x);
    // The guards are what is under test: the interpreter skips every NaN.
    assert!(!want[0][0].is_nan() && !want[1][0].is_nan(), "the reference skips NaNs");
    for (at, &(what, len, dtype)) in READERS.iter().enumerate() {
        let raw = engine
            .take_channel(instance, 2 + at as u32)
            .expect("the channel reads")
            .expect("the channel holds a cell");
        let got = cells(&raw, dtype);
        assert_eq!(got.len(), len as usize, "{what}: its cell");
        for (i, (&g, &w)) in got.iter().zip(&want[at]).enumerate() {
            assert!(
                g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                "{what}[{i}]: the device landed {g}, the interpreter {w}"
            );
        }
        eprintln!("{what}: {len} values as the interpreter's");
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
