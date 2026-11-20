//! bf16 guest values on the device: a cast in, elementwise ops mixing an
//! f32 constant, a scan over rows of odd length, a gather, a scatter-add, a
//! transpose, a top-k, a sum and two Gumbel-max draws (one over the bf16
//! row, masked and divided, one over the f32 row), each stored two elements a word by one
//! invocation and read back against the host interpreter, which computes
//! in f32 and rounds every store to nearest even. The values are an odd
//! number of elements, past one workgroup, so the last word holds padding
//! and every invocation writes several words.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::InstanceBinding;
use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Literal, Shape};
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
const READERS: [(&str, u32, EtaDtype); 11] = [
    ("x * 0.5 + 1 (bf16 bits)", N, EtaDtype::Bf16),
    ("exp(x * 0.5 + 1)", N, EtaDtype::F32),
    ("cumsum over [7, 143] rows", N, EtaDtype::F32),
    ("x reversed (gather)", N, EtaDtype::F32),
    ("scatter_add of 1 at 0, 1, 2", N, EtaDtype::F32),
    ("x [7, 143] transposed", N, EtaDtype::F32),
    ("top-5 values", TOP, EtaDtype::F32),
    ("top-5 ids", TOP, EtaDtype::U32),
    ("sum", 1, EtaDtype::F32),
    ("gumbel_argmax of bf16 x, x > 0 kept, * (1 / 0.7)", 1, EtaDtype::I32),
    ("gumbel_argmax of f32 x * (1 / 0.7)", 1, EtaDtype::I32),
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
    let bf = EtaDtype::Bf16;
    let wide = |value| Op::Cast {
        value,
        dtype: EtaDtype::F32,
    };
    let ops = vec![
        Op::ChanTake(0),                                          // 0 f32 [N]
        Op::Cast { value: 0, dtype: bf },                         // 1 x
        Op::Const(Literal::F32(0.5)),                             // 2
        Op::Mul(1, 2),                                            // 3 bf16
        Op::Const(Literal::F32(1.0)),                             // 4
        Op::Add(3, 4),                                            // 5 bf16
        Op::ChanPut { chan: 2, value: 5 },
        Op::Exp(5),                                               // 6 bf16
        wide(6),                                                  // 7
        Op::ChanPut { chan: 3, value: 7 },
        Op::Reshape { value: 1, shape: Shape::matrix(ROWS, COLS) }, // 8
        Op::CumSum(8),                                            // 9 bf16 [7, 143]
        Op::Reshape { value: 9, shape: Shape::vector(N) },        // 10
        wide(10),                                                 // 11
        Op::ChanPut { chan: 4, value: 11 },
        Op::Iota { len: N },                                      // 12
        Op::Const(Literal::U32(N - 1)),                           // 13
        Op::Broadcast { value: 13, shape: Shape::vector(N) },     // 14
        Op::Sub(14, 12),                                          // 15 reversed ids
        Op::Gather { src: 1, idx: 15 },                           // 16 bf16
        wide(16),                                                 // 17
        Op::ChanPut { chan: 5, value: 17 },
        Op::Iota { len: 3 },                                      // 18
        Op::ScatterAdd { base: 1, idx: 18, vals: 4 },             // 19 bf16
        wide(19),                                                 // 20
        Op::ChanPut { chan: 6, value: 20 },
        Op::Transpose(8),                                         // 21 bf16 [143, 7]
        Op::Reshape { value: 21, shape: Shape::vector(N) },       // 22
        wide(22),                                                 // 23
        Op::ChanPut { chan: 7, value: 23 },
        Op::TopK { input: 1, k: TOP },                            // 24 bf16, 25 u32
        wide(24),                                                 // 26
        Op::ChanPut { chan: 8, value: 26 },
        Op::ChanPut { chan: 9, value: 25 },
        Op::ReduceSum(1),                                         // 27 f32 []
        Op::Reshape { value: 27, shape: Shape::vector(1) },       // 28
        Op::ChanPut { chan: 10, value: 28 },
        Op::Const(Literal::F32(0.0)),                             // 29
        Op::Gt(1, 29),                                            // 30 keep
        Op::Const(Literal::F32(1.0 / 0.7)),                       // 31
        Op::Iota { len: 2 },                                      // 32 rng state
        Op::GumbelArgmax { logits: 1, keep: 30, scale: 31, state: 32 }, // 33
        Op::Reshape { value: 33, shape: Shape::vector(1) },       // 34
        Op::ChanPut { chan: 11, value: 34 },
        Op::Const(Literal::Bool(true)),                           // 35
        Op::GumbelArgmax { logits: 0, keep: 35, scale: 31, state: 32 }, // 36
        Op::Reshape { value: 36, shape: Shape::vector(1) },       // 37
        Op::ChanPut { chan: 12, value: 37 },
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

/// The values: distinct, signed, most not bf16-exact, a few ties.
fn values() -> Vec<f32> {
    let mut x = 0x2545_f491u32;
    (0..N)
        .map(|i| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if i % 97 == 0 {
                3.0
            } else {
                (x % 40_000) as f32 / 4093.0 - 4.5
            }
        })
        .collect()
}

/// What the interpreter lands in each reader, as f32 (a bf16 reader's
/// values widened).
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
            Value::Bf16(v) => v.into_iter().map(eta_ir::types::bf16_to_f32).collect(),
            other => panic!("{other:?}"),
        })
        .collect()
}

fn cells(raw: &[u8], dtype: EtaDtype) -> Vec<f32> {
    match dtype {
        EtaDtype::Bf16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| eta_ir::types::bf16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        EtaDtype::I32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c) as f32)
            .collect(),
        EtaDtype::U32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c) as f32)
            .collect(),
        _ => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
    }
}

#[test]
fn bf16_values_are_stored_a_word_at_a_time() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("bf16-values", &Outputs)
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
        .register_program(&registration_of(&decls, program(), 0xbf16))
        .expect("the bf16 program registers");
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
        }))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let want = interpreted(&decls, &x);
    for (at, &(what, len, dtype)) in READERS.iter().enumerate() {
        let raw = engine
            .take_channel(instance, 2 + at as u32)
            .expect("the channel reads")
            .expect("the channel holds a cell");
        let got = cells(&raw, dtype);
        assert_eq!(got.len(), len as usize, "{what}: its cell");
        // exp is the device's own: within one bf16 ulp of the host's. Every
        // other op is exact.
        let ulps = if at == 1 { 1.0 } else { 0.0 };
        let mut worst = 0f32;
        for (i, (&g, &w)) in got.iter().zip(&want[at]).enumerate() {
            let ulp = (w.abs() * 2f32.powi(-7)).max(f32::MIN_POSITIVE);
            let apart = (g - w).abs() / ulp;
            assert!(
                apart <= ulps || (at == 8 && (g - w).abs() <= 1e-3 * w.abs()),
                "{what}[{i}]: the device stored {g}, the interpreter {w}"
            );
            worst = worst.max(apart);
        }
        eprintln!("{what}: {len} values, worst {worst:.2} bf16 ulps from the interpreter");
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
