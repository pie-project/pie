//! A reduction over rows far wider than a workgroup (Qwen3's vocabulary),
//! fired with more rows than CUDA's split-row launch splits
//! (`LANE_SAMPLER_BLOCKS`, 16) and fewer: every row's argmax and sum is
//! the host's, whatever the row count. On CUDA a split row's region past 16
//! rows ran its first phase only and left every output unwritten; here a
//! reduction runs a ladder of levels whose count follows the row's length
//! alone, and this holds that every level runs at each row count.
//!
//! The values are made on the device from an iota in exact u32 arithmetic,
//! `((i mod P) * M) mod P` over the [rows, vocab] grid with P prime, so
//! each row's values are distinct integers under 2^24 (exact in f32) and its
//! argmax has one answer.

use engine::ChannelId;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::program::InstanceBinding;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Literal, Shape};
use poem::Request;

use super::outputs::{Loaded, Outputs, loaded, output_decls, registration_of};
use super::slots::Slots;

/// Qwen3's vocabulary: a row folded over four levels.
const VOCAB: u32 = 151_936;
/// A prime past a row, so `i mod P` is distinct along one.
const P: u32 = 1_000_003;
/// Coprime with `P`; `(P - 1) * M` stays under 2^32.
const M: u32 = 4093;
/// One row, CUDA's split bound and one past it, and two beyond.
const ROWS: [u32; 5] = [1, 16, 17, 33, 64];
/// The channels a row count's program reads and writes.
const CHANNELS: usize = 3;

fn program(rows: u32) -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    };
    let channels = vec![
        decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer),
        decl(Shape::vector(rows), EtaDtype::I32, HostRole::Reader),
        decl(Shape::vector(rows), EtaDtype::F32, HostRole::Reader),
    ];
    let n = rows * VOCAB;
    let ops = vec![
        Op::Iota { len: n },                                       // 0 u32 [n]
        Op::Const(Literal::U32(P)),                                // 1
        Op::Broadcast { value: 1, shape: Shape::vector(n) },       // 2
        Op::Rem(0, 2),                                             // 3
        Op::Const(Literal::U32(M)),                                // 4
        Op::Broadcast { value: 4, shape: Shape::vector(n) },       // 5
        Op::Mul(3, 5),                                             // 6
        Op::Rem(6, 2),                                             // 7
        Op::Cast { value: 7, dtype: EtaDtype::F32 },               // 8
        Op::Reshape { value: 8, shape: Shape::matrix(rows, VOCAB) }, // 9
        Op::ReduceArgmax(9),                                       // 10 [rows]
        Op::Cast { value: 10, dtype: EtaDtype::I32 },              // 11
        Op::Reshape { value: 11, shape: Shape::vector(rows) },     // 12
        Op::ChanPut { chan: 1, value: 12 },
        Op::ReduceSum(9),                                          // 13 [rows]
        Op::Reshape { value: 13, shape: Shape::vector(rows) },     // 14
        Op::ChanPut { chan: 2, value: 14 },
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

/// Element `i` of the grid, as the device makes it.
fn value(i: u32) -> u32 {
    ((i % P) * M) % P
}

/// Each row's argmax and its sum's magnitude (exact, in u64).
fn host(rows: u32) -> Vec<(u32, u64)> {
    (0..rows)
        .map(|r| {
            let (mut best, mut at, mut sum) = (0u32, 0u32, 0u64);
            for j in 0..VOCAB {
                let v = value(r * VOCAB + j);
                if v > best {
                    (best, at) = (v, j);
                }
                sum += u64::from(v);
            }
            (at, sum)
        })
        .collect()
}

#[test]
fn every_row_of_a_wide_reduction_is_folded_whatever_the_rows() {
    let Some(Loaded {
        mut engine,
        caps: loaded,
        trace,
        dir,
        ..
    }) = loaded("split-rows", &Outputs)
    else {
        return;
    };
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    let decls = output_decls(&loaded);
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let mut wrong = Vec::new();
    for (k, &rows) in ROWS.iter().enumerate() {
        let program = engine::Programs::register(&mut engine, &registration_of(&decls, program(rows), 0x5917_0000 + u64::from(rows)))
            .expect("the program registers");
        let shapes = [
            (vec![1, 1], EtaDtype::F32, HostRole::Writer),
            (vec![rows], EtaDtype::I32, HostRole::Reader),
            (vec![rows], EtaDtype::F32, HostRole::Reader),
        ];
        let channels: Vec<ChannelId> = shapes
            .into_iter()
            .enumerate()
            .map(|(at, (shape, dtype, host_role))| {
                let id = ChannelId((k * CHANNELS + at) as u64 + 1);
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
        let binding = slots.binding(
            &engine,
            InstanceBinding {
                program,
                channels: channels.clone(),
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                extents: engine::program::BindExtents {
                    row_count: 1,
                    token_count: 1,
                    sampled_rows: 1,
                    query_len: 1,
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
        let instance = engine::Programs::bind(&mut engine, &binding).expect("the instance binds").id;
        assert!(
            engine::Channels::publish(&mut engine, instance, 0, &0.375f32.to_le_bytes())
                .expect("the timestep publishes")
        );
        let lane = Lane {
            slot: 0,
            word: trace.facts.word(&Request::new(1, false)),
            tokens: vec![3],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed {
                id: t_input,
                channel: channels[0],
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
        let mut take = |at: u32| {
            engine::Channels::take(&mut engine, instance, at)
                .expect("the channel reads")
                .expect("the channel holds a cell")
        };
        let ids: Vec<i32> = take(1)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect();
        let sums: Vec<f32> = take(2)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        assert_eq!(ids.len(), rows as usize, "{rows} rows: an argmax a row");
        assert_eq!(sums.len(), rows as usize, "{rows} rows: a sum a row");
        // Every element is exact in f32 and nonnegative; each add of the fold
        // rounds once, and no element passes through more than `depth` adds
        // (31 a level, the levels of a row, the finish's copy none), so the
        // sum is within `depth * 2^-24` of the exact one, relative.
        let levels = {
            let (mut len, mut levels) = (VOCAB, 0u32);
            while len > 1 {
                len = len.div_ceil(32);
                levels += 1;
            }
            levels
        };
        let bound = f64::from(31 * levels + 1) * 2f64.powi(-24);
        let mut worst = 0f64;
        for (r, ((&id, &sum), &(at, exact))) in ids.iter().zip(&sums).zip(&host(rows)).enumerate() {
            if id != at as i32 {
                wrong.push(format!("{rows} rows, row {r}: argmax {id}, the host's {at}"));
            }
            let apart = (f64::from(sum) - exact as f64).abs() / exact as f64;
            worst = worst.max(apart);
            if !(apart <= bound) {
                wrong.push(format!(
                    "{rows} rows, row {r}: sum {sum}, the host's {exact} ({apart:.3e} apart, bound {bound:.3e})"
                ));
            }
        }
        eprintln!("{rows} rows of {VOCAB}: worst sum {worst:.3e} relative (bound {bound:.3e})");
    }
    assert!(wrong.is_empty(), "{} rows disagree:\n{}", wrong.len(), wrong.join("\n"));
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
