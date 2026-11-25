//! A buffering lane whose fold length is the device's replays what one
//! folding the same count on the host replays, bit for bit, when the
//! runtime's count lags: the device-folding fires of one set are pipelined
//! (the next submitted before the previous settles), each stating the
//! `folded` the runtime held at submit, before the device fold it carries
//! landed. qwen35-tiny (recurrent layers); the fold count rides a device-geometry program's `fold` port.
//! Skipped without a PJRT plugin.

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::{ShapeId, SlotKind, SlotRole};
use engine::fire::{Attachment, FoldLen, FrameSubmission, Lane, RsReset, RsVerb, Step};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{EngineCaps, GeometryClass, Port, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use crate::common_llm::{Model, Refs, Seq, tokens};

/// The slots each buffering fire's fold freezes.
const N: u32 = 4;
/// The prompt's rows (folded) and each buffering fire's.
const PROMPT: u32 = 32;
const ROWS: u32 = 8;

/// A program that tells each fire's tokens and fold length, from host
/// channels: channel 0 the lane's `ROWS` tokens, channel 1 its fold.
fn fold_program() -> TraceContainer {
    let channel = |shape: Shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::U32),
        capacity: 2,
        host_role: HostRole::Writer,
        seeded: false,
    };
    let indptr: Vec<u8> = [0u32, ROWS].iter().flat_map(|w| w.to_le_bytes()).collect();
    TraceContainer {
        names: Vec::new(),
        channels: vec![channel(Shape::vector(ROWS)), channel(Shape::vector(1))],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            },
            PortBinding {
                port: Port::EmbedIndptr,
                source: PortSource::Const {
                    dtype: EtaDtype::U32,
                    shape: Shape::vector(2),
                    data: indptr,
                },
            },
            PortBinding {
                port: Port::Fold,
                source: PortSource::Channel(1),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::ChanTake(0), Op::ChanTake(1)],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The fold program bound on `model`, device geometry.
fn bind_fold(model: &mut Model) -> InstanceId {
    let caps = &model.loaded.caps;
    let engine_caps = EngineCaps {
        lora: caps.has_lora,
        attn_page_mask: caps.has_attn_page_mask,
    };
    let container = fold_program();
    let bound = eta_ir::validate::bind(container, vec![Vec::new()], engine_caps)
        .expect("the fold program binds");
    let launch = eta_compiler::codegen::launch::build(&bound, &compile_bound(&bound));
    let program = engine::Programs::register(&mut model.engine, &ProgramRegistration {
            launch,
            ..Default::default()
        })
        .expect("the fold program registers");
    let channels: Vec<ChannelId> = [ROWS, 1]
        .iter()
        .enumerate()
        .map(|(at, &width)| {
            let id = ChannelId((1 << 20) + at as u64);
            engine::Channels::open(&mut model.engine, &ChannelRegistration {
                    id,
                    shape: vec![width],
                    dtype: ChanDType::Concrete(EtaDtype::U32),
                    host_role: HostRole::Writer,
                    seeded: false,
                    extern_dir: None,
                    capacity: 2,
                    extern_name: Vec::new(),
                    slots: Vec::new(),
                })
                .expect("the channel registers");
            id
        })
        .collect();
    engine::Programs::bind(&mut model.engine, &InstanceBinding {
            program,
            channels,
            seeds: Vec::new(),
            geometry: GeometryClass::DeviceGeometry,
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            extents: engine::program::BindExtents {
                row_count: ROWS,
                token_count: ROWS,
                sampled_rows: ROWS,
                query_len: ROWS,
                ..engine::program::BindExtents::default()
            },
            pair_generation: 0,
        })
        .expect("the fold program binds an instance")
        .id
}

/// The rs list's shape and the slots a slot ref of it holds.
fn rs_list(refs: &Refs) -> (ShapeId, u32) {
    let shape = refs
        .shapes(|s| s.role == SlotRole::RsBuffer)
        .into_iter()
        .next()
        .expect("a recurrent plan states an rs list");
    let SlotKind::List { tokens, .. } = shape.kind else {
        panic!("the rs list is a list shape");
    };
    (shape.id, tokens)
}

/// One sequence: its kv and record (`Seq`), the rs list slots its buffer
/// holds from the prompt's end on.
struct Set {
    seq: Seq,
    rs: Vec<u32>,
}

impl Set {
    fn new(model: &mut Model, refs: &mut Refs, slot: u32, rs_tokens: u32, rs_shape: ShapeId) -> Set {
        let mut seq = Seq::new(slot);
        let prompt = seq.lane(model, refs, tokens(7, PROMPT as usize, model.vocab));
        model.fire(vec![prompt]);
        seq.held = PROMPT;
        let pages = (3 * ROWS).div_ceil(rs_tokens) as usize + 1;
        let rs = (0..pages).map(|_| refs.take(rs_shape)).collect();
        Set {
            seq,
            rs,
        }
    }

    /// A buffering lane of `ROWS` rows whose state holds the slots below
    /// `folded`, folding `fold`.
    fn lane(&mut self, model: &Model, refs: &mut Refs, seed: u64, folded: u32, fold: FoldLen) -> Lane {
        let mut lane = self.seq.lane(model, refs, tokens(seed, ROWS as usize, model.vocab));
        lane.rs = RsVerb::Buffer {
            pages: self.rs.clone(),
            base: PROMPT,
            folded,
            fold,
            lineage: engine::fire::Lineage::Set,
        };
        lane.rs_reset = RsReset::Held;
        self.seq.held += ROWS;
        lane
    }
}

fn submit(model: &mut Model, lane: Lane, attach: Option<InstanceId>) -> engine::fire::FrameTicket {
    model
        .engine
        .submit(&FrameSubmission::of(Step {
            lanes: vec![lane],
            attachments: attach
                .map(|instance| Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: ::engine::ScratchPlace::None,
                })
                .into_iter()
                .collect(),
            media: Vec::new(),
            voxels: Vec::new(),
            shape: None,
        }))
        .expect("the frame fires")
}

fn settled(model: &mut Model, mut ticket: engine::fire::FrameTicket) -> Vec<u32> {
    model
        .engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    ticket.steps[0].readouts[0]
        .values
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Publishes `lane`'s tokens and the fold length on the program's channels.
fn tell(model: &mut Model, instance: InstanceId, lane: &Lane) {
    let tokens: Vec<u8> = lane.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    for (channel, bytes) in [(0, tokens), (1, N.to_le_bytes().to_vec())] {
        assert!(
            engine::Channels::publish(&mut model.engine, instance, channel, &bytes)
                .expect("the cell publishes"),
            "the ring had room"
        );
    }
}

#[test]
fn a_device_fold_replays_what_a_host_fold_replays() {
    if crate::common_dit::device().is_none() {
        return;
    }
    let mut model = Model::load_with("qwen35-tiny", None, 256, 7, None);
    let mut refs = Refs::of(&model);
    let (rs_shape, rs_tokens) = rs_list(&refs);
    let instance = bind_fold(&mut model);
    let bits = |row: Vec<f32>| -> Vec<u32> { row.iter().map(|v| v.to_bits()).collect() };
    // What a set's state and buffer answer once every fold has settled: a
    // host lane that replays from the exact count and folds nothing (a lane
    // whose pass runs on the device answers through its program, not here).
    let probe = |model: &mut Model, refs: &mut Refs, set: &mut Set, folded: u32| {
        let lane = set.lane(model, refs, 99, folded, FoldLen::Host(0));
        bits(model.fire(vec![lane]).remove(0))
    };

    // Two buffering fires folding N each. The host's counts are known at
    // submit, so the runtime's `folded` is exact.
    let mut host = Set::new(&mut model, &mut refs, 0, rs_tokens, rs_shape);
    let a = host.lane(&model, &mut refs, 11, PROMPT, FoldLen::Host(N));
    model.fire(vec![a]);
    let b = host.lane(&model, &mut refs, 12, PROMPT + N, FoldLen::Host(N));
    model.fire(vec![b]);
    let want = probe(&mut model, &mut refs, &mut host, PROMPT + 2 * N);
    // The device's ride the program's `fold` port: the second is submitted
    // before the first settles, stating the `folded` the runtime held then.
    let mut device = Set::new(&mut model, &mut refs, 1, rs_tokens, rs_shape);
    let a = device.lane(&model, &mut refs, 11, PROMPT, FoldLen::Device);
    tell(&mut model, instance, &a);
    let first = submit(&mut model, a, Some(instance));
    let b = device.lane(&model, &mut refs, 12, PROMPT, FoldLen::Device);
    tell(&mut model, instance, &b);
    let second = submit(&mut model, b, Some(instance));
    drop(first);
    let _ = settled(&mut model, second);
    assert_eq!(
        probe(&mut model, &mut refs, &mut device, PROMPT + 2 * N),
        want,
        "a pipelined device fold leaves what the host fold leaves"
    );
}
