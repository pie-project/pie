//! A beam-style program bound in `GeometryClass::DeviceGeometry`: two beams
//! forked off one prompt publish their token ids, positions, page tables
//! (relative to the working set, translated by the lane's table) and KV
//! extents on channels, and the engine resolves the fire's geometry from
//! those cells (engine-cuda `program/ports.rs` + `serve/prepare.rs`). Each
//! beam's logits, read back through the guest's `logits` intrinsic, answer
//! what the host path answers for the same beams stated on the lanes, for
//! two decode steps.

mod common_dit;

use common_dit::{Rig, Weights, assert_close, attach};
use engine::Engine;
use engine::fire::{Lane, Readout};
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, Port, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Trace, Value, Weight,
    ops, seam, trace_hybrid,
};

const VOCAB: u32 = 64;
const WIDTH: u32 = 32;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 1;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "trunk.kv";
const PAGE: u32 = 16;
const BEAMS: u32 = 2;
/// Pages per beam: the shared prompt's page, then the beam's own.
const PAGES: u32 = 2;

struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

struct Tiny;

impl ForwardHybrid for Tiny {
    type Facts = NoFacts;

    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        let space = spec.kv_space(Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(space, KV_ROW, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };
        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let x = ops::layout::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", HEAD_DIM, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", HEAD_DIM, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", HEAD_DIM, WIDTH));
        let (q, k) =
            ops::elemwise::rope_full(&q, &k, &inputs.positions(), HEAD_DIM, 10_000.0, false);
        let pages = inputs.kv(KV_ROW);
        ops::attn::kv_append(
            &k,
            &v,
            pages,
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let plan = ops::attn::plan_prefill(&inputs, HEADS, HEADS, HEAD_DIM, None);
        let o = ops::attn::prefill(&q, &plan, pages, None, HEAD_DIM, HEADS, SM_SCALE);
        let h = ops::linear::matmul(&o, &w("o", WIDTH, HEAD_DIM));
        let logits = ops::linear::matmul(&h, &w("head", VOCAB, WIDTH));
        seam::at(seam::OUT, &[&logits]);
        logits
    }
}

fn plan() -> Trace {
    trace_hybrid("tiny-paged", &Tiny, Platform::Xla)
}

fn channel(shape: Shape, dtype: EtaDtype, host_role: HostRole) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role,
        seeded: false,
    }
}

fn const_port(port: Port, words: &[u32]) -> PortBinding {
    PortBinding {
        port,
        source: PortSource::Const {
            dtype: EtaDtype::U32,
            shape: Shape::vector(words.len() as u32),
            data: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        },
    }
}

/// The beam program: token ids, positions, page table and extents on
/// channels 0..4, the token and page CSRs constant, each beam's logits
/// put on channel 4.
fn beam_program() -> TraceContainer {
    let bound = |port: Port, chan: u32| PortBinding {
        port,
        source: PortSource::Channel(chan),
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            channel(Shape::vector(BEAMS), EtaDtype::I32, HostRole::Writer),
            channel(Shape::vector(BEAMS), EtaDtype::U32, HostRole::Writer),
            channel(Shape::matrix(BEAMS, PAGES), EtaDtype::U32, HostRole::Writer),
            channel(Shape::vector(BEAMS), EtaDtype::U32, HostRole::Writer),
            channel(Shape::matrix(BEAMS, VOCAB), EtaDtype::F32, HostRole::Reader),
            channel(Shape::vector(BEAMS), EtaDtype::U32, HostRole::Writer),
            channel(Shape::vector(BEAMS), EtaDtype::U32, HostRole::Writer),
        ],
        ports: vec![
            bound(Port::EmbedTokens, 0),
            const_port(Port::EmbedIndptr, &(0..=BEAMS).collect::<Vec<_>>()),
            bound(Port::Positions, 1),
            bound(Port::Pages, 2),
            const_port(
                Port::PageIndptr,
                &(0..=BEAMS).map(|b| b * PAGES).collect::<Vec<_>>(),
            ),
            bound(Port::KvLen, 3),
            bound(Port::WSlot, 5),
            bound(Port::WOff, 6),
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ChanTake(1),
                Op::ChanTake(2),
                Op::ChanTake(3),
                Op::ChanTake(5),
                Op::ChanTake(6),
                Op::IntrinsicVal {
                    intr: IntrinsicId::Logits,
                    shape: Shape::matrix(BEAMS, VOCAB),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 4, value: 6 },
            ],
        }],
        externs: Vec::new(),
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn host_lane(slot: u32, tokens: Vec<u32>, pages: Vec<u32>, held: u32) -> Lane {
    let mut lane = Lane {
        slot,
        word: 0,
        tokens,
        readout: Readout::Last,
        ..Lane::default()
    };
    lane.kv.pages = pages;
    lane.kv.held = held;
    lane
}

#[test]
fn a_device_resolved_beam_answers_what_the_host_path_answers() {
    let Some(_device) = common_dit::device() else {
        return;
    };
    let plan = plan();
    let weights = Weights::random(&plan, 17);
    let mut rig = Rig::load(plan, &weights, 64, vec![1, 2, 16, 32, 64], None);
    assert_eq!(rig.loaded.caps.geometry, GeometryClass::DeviceGeometry);
    assert!(
        rig.loaded
            .caps
            .ports
            .covers(eta_ir::registry::PortMask::DEVICE_GEOMETRY)
    );

    // The prompt: one page, shared by every beam.
    let prompt: Vec<u32> = (0..PAGE).map(|i| (i * 7 + 3) % VOCAB).collect();
    let shared = 1u32;
    rig.fire(
        vec![host_lane(0, prompt, vec![shared], 0)],
        Vec::new(),
        Vec::new(),
    );

    // Host beams write their own pages 2, 3; device beams 4, 5, which the
    // device path names relatively through the lane's translation table.
    let host_own = [2u32, 3];
    let device_own = [4u32, 5];
    let translation = vec![shared, device_own[0], device_own[1]];

    let program = rig.register(beam_program());
    let channels = vec![
        rig.channel_of(vec![BEAMS], EtaDtype::I32, HostRole::Writer),
        rig.channel_of(vec![BEAMS], EtaDtype::U32, HostRole::Writer),
        rig.channel_of(vec![BEAMS, PAGES], EtaDtype::U32, HostRole::Writer),
        rig.channel_of(vec![BEAMS], EtaDtype::U32, HostRole::Writer),
        rig.channel_of(vec![BEAMS, VOCAB], EtaDtype::F32, HostRole::Reader),
        rig.channel_of(vec![BEAMS], EtaDtype::U32, HostRole::Writer),
        rig.channel_of(vec![BEAMS], EtaDtype::U32, HostRole::Writer),
    ];
    let instance = rig
        .engine
        .bind_instance(&engine::program::InstanceBinding {
            program,
            channels,
            seeds: Vec::new(),
            geometry: GeometryClass::DeviceGeometry,
            extents: engine::program::BindExtents {
                row_count: BEAMS,
                token_count: BEAMS,
                sampled_rows: BEAMS,
                query_len: 1,
                ..engine::program::BindExtents::default()
            },
        })
        .expect("a device-geometry instance binds")
        .id;

    let steps: [[u32; 2]; 2] = [[11, 42], [5, 29]];
    for (step, beam_tokens) in steps.iter().enumerate() {
        let held = PAGE + step as u32;
        // The host path: each beam states its page table and extent.
        let host: Vec<Lane> = (0..BEAMS as usize)
            .map(|b| {
                host_lane(
                    1 + b as u32,
                    vec![beam_tokens[b]],
                    vec![shared, host_own[b]],
                    held,
                )
            })
            .collect();
        let want = rig.fire(host, Vec::new(), Vec::new());

        // The device path: the same beams, stated on the program's ports.
        for (chan, cell) in [
            (0u32, words(beam_tokens)),
            (1, words(&[held, held])),
            (2, words(&[0, 1, 0, 2])),
            (3, words(&[held + 1, held + 1])),
            // Each beam writes its row into its own page (relative 1, 2).
            (5, words(&[1, 2])),
            (6, words(&[held - PAGE, held - PAGE])),
        ] {
            assert!(
                rig.engine
                    .publish_channel(instance, chan, &cell)
                    .expect("the cell publishes"),
                "the ring had room"
            );
        }
        let lanes: Vec<Lane> = (0..BEAMS)
            .map(|b| {
                let mut lane = Lane {
                    slot: 3 + b,
                    word: 0,
                    tokens: vec![0],
                    readout: Readout::Last,
                    ..Lane::default()
                };
                lane.kv.translation = translation.clone();
                lane
            })
            .collect();
        rig.fire(lanes, vec![attach(0, instance)], Vec::new());
        let got = rig.take(instance, 4);
        assert_eq!(
            got.len(),
            (BEAMS * VOCAB) as usize,
            "one logits row per beam"
        );
        for b in 0..BEAMS as usize {
            let row = &got[b * VOCAB as usize..(b + 1) * VOCAB as usize];
            assert_eq!(want[b].values.len(), VOCAB as usize);
            assert_close(
                row,
                &want[b].values,
                &format!("step {step} beam {b} logits"),
            );
        }
    }
}
