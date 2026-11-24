//! A buffering lane whose fold length is the device's replays what one
//! folding the same count on the host replays, bit for bit, when the
//! runtime's count lags: the device-folding fires of one set are pipelined
//! (the next submitted before the previous settles), each stating the
//! `folded` the runtime held at submit, before the device fold it carries
//! landed. And a fork of that set, fired while its parent's device fold is
//! unsettled, replays what the host fork replays. qwen35-tiny (recurrent
//! layers); the fold count rides a device-geometry program's `fold` port.
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
    let program = model
        .engine
        .register_program(&ProgramRegistration {
            launch,
            ..Default::default()
        })
        .expect("the fold program registers");
    let channels: Vec<ChannelId> = [ROWS, 1]
        .iter()
        .enumerate()
        .map(|(at, &width)| {
            let id = ChannelId((1 << 20) + at as u64);
            model
                .engine
                .register_channel(&ChannelRegistration {
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
    model
        .engine
        .bind_instance(&InstanceBinding {
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
/// holds from the prompt's end on, and the slot copies its next lane makes
/// (a fork's).
struct Set {
    seq: Seq,
    rs: Vec<u32>,
    copies: Vec<(ShapeId, Vec<[u32; 2]>)>,
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
            copies: Vec::new(),
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
        lane.slot_copies = std::mem::take(&mut self.copies);
        self.seq.held += ROWS;
        // A fork's record is its own from its first fire on.
        for (_, pieces) in &mut self.seq.record {
            for piece in pieces.iter_mut() {
                piece[0] = piece[1];
            }
        }
        lane
    }

    /// A fork of the set as it stands: the parent's record copied onto
    /// one of its own at its first fire, and the kv page and rs slots its
    /// rows land in copied to slots of its own.
    fn fork(&self, refs: &mut Refs, rs_shape: ShapeId, rs_tokens: u32, slot: u32) -> Set {
        let kv_shape = refs
            .shapes(|s| {
                matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State
            })
            .into_iter()
            .next()
            .expect("a kv list")
            .id;
        let held = self.seq.held;
        let mut kv = self.seq.kv.clone();
        let page = (held / crate::common_llm::PAGE) as usize;
        let mut kv_copies = Vec::new();
        if page < kv.len() {
            let fresh = refs.take(kv_shape);
            kv_copies.push([kv[page], fresh]);
            kv[page] = fresh;
        }
        let record = self
            .seq
            .record
            .iter()
            .map(|(id, pieces)| {
                let id = *id;
                (id, pieces.iter().map(|&[_, cow]| [cow, refs.take(id)]).collect())
            })
            .collect();
        let seq = Seq {
            slot,
            held,
            kv,
            window: self.seq.window.clone(),
            record,
            ..Seq::default()
        };
        let landed = ((held - PROMPT) / rs_tokens) as usize;
        let mut rs = self.rs.clone();
        let rs_copies: Vec<[u32; 2]> = (landed..rs.len())
            .map(|at| {
                let fresh = refs.take(rs_shape);
                let old = std::mem::replace(&mut rs[at], fresh);
                [old, fresh]
            })
            .collect();
        Set {
            seq,
            rs,
            copies: vec![(kv_shape, kv_copies), (rs_shape, rs_copies)],
        }
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
            model
                .engine
                .publish_channel(instance, channel, &bytes)
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

    // A fork of a set whose device fold is unsettled, fired at once.
    let mut host = Set::new(&mut model, &mut refs, 2, rs_tokens, rs_shape);
    let a = host.lane(&model, &mut refs, 21, PROMPT, FoldLen::Host(N));
    model.fire(vec![a]);
    let mut child = host.fork(&mut refs, rs_shape, rs_tokens, 3);
    let c = child.lane(&model, &mut refs, 22, PROMPT + N, FoldLen::Host(N));
    model.fire(vec![c]);
    let want = probe(&mut model, &mut refs, &mut child, PROMPT + 2 * N);
    let mut device = Set::new(&mut model, &mut refs, 4, rs_tokens, rs_shape);
    let a = device.lane(&model, &mut refs, 21, PROMPT, FoldLen::Device);
    tell(&mut model, instance, &a);
    let first = submit(&mut model, a, Some(instance));
    let mut child = device.fork(&mut refs, rs_shape, rs_tokens, 5);
    let c = child.lane(&model, &mut refs, 22, PROMPT, FoldLen::Device);
    tell(&mut model, instance, &c);
    let second = submit(&mut model, c, Some(instance));
    drop(first);
    let _ = settled(&mut model, second);
    assert_eq!(
        probe(&mut model, &mut refs, &mut child, PROMPT + 2 * N),
        want,
        "a fork of a set whose device fold is unsettled leaves what the host fork leaves"
    );

    // A step refused for a lane whose fold nothing resolved (no program is
    // attached to it) leaves no position behind for its first lane's fold,
    // resolved but never run: that set replays next from what the runtime
    // states.
    let mut host = Set::new(&mut model, &mut refs, 6, rs_tokens, rs_shape);
    let a = host.lane(&model, &mut refs, 31, PROMPT, FoldLen::Host(N));
    model.fire(vec![a]);
    let want = probe(&mut model, &mut refs, &mut host, PROMPT + N);
    let mut device = Set::new(&mut model, &mut refs, 7, rs_tokens, rs_shape);
    let mut other = Set::new(&mut model, &mut refs, 8, rs_tokens, rs_shape);
    let a = device.lane(&model, &mut refs, 31, PROMPT, FoldLen::Device);
    let b = other.lane(&model, &mut refs, 32, PROMPT, FoldLen::Device);
    tell(&mut model, instance, &a);
    let refused = model
        .engine
        .submit(&FrameSubmission::of(Step {
            lanes: vec![a, b],
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
        .expect_err("a device-fold lane no program is attached to resolves no fold");
    assert!(
        refused.to_string().contains("device-resident fold length"),
        "refused for the unresolved fold: {refused}"
    );
    device.seq.held -= ROWS;
    let a = device.lane(&model, &mut refs, 31, PROMPT, FoldLen::Host(N));
    model.fire(vec![a]);
    assert_eq!(
        probe(&mut model, &mut refs, &mut device, PROMPT + N),
        want,
        "a refused step's fold, which never ran, moved no state's position"
    );
}

/// A KV-only set (no recurrent state) is folded by its program's `fold`
/// count just the same: each reporting lane's fire reports that count as
/// its `fold`, two frames in flight included (the second submitted before
/// the first settles), so the runtime's cursor comes to rest at the tokens
/// folded. A reporting lane no program is attached to reports none (the
/// negative control).
#[test]
fn a_kv_only_device_fold_reports_its_count() {
    if crate::common_dit::device().is_none() {
        return;
    }
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let mut refs = Refs::of(&model);
    let instance = bind_fold(&mut model);
    let heard: std::sync::Arc<std::sync::Mutex<Vec<engine::KvWrites>>> = Default::default();
    let into = std::sync::Arc::clone(&heard);
    model.engine.on_complete(std::sync::Arc::new(move |_, outcome| {
        into.lock().expect("heard").extend(outcome.wrote);
    }));
    let mut seq = Seq::new(0);
    let mut lane = |model: &Model, refs: &mut Refs, seed: u64, tag: u64| {
        let mut lane = seq.lane(model, refs, tokens(seed, ROWS as usize, model.vocab));
        seq.held += ROWS;
        lane.reports = tag;
        lane
    };
    let a = lane(&model, &mut refs, 41, 101);
    tell(&mut model, instance, &a);
    let first = submit(&mut model, a, Some(instance));
    let b = lane(&model, &mut refs, 42, 102);
    tell(&mut model, instance, &b);
    let second = submit(&mut model, b, Some(instance));
    drop(first);
    let _ = settled(&mut model, second);
    let mut other = Seq::new(1);
    let mut plain = other.lane(&model, &mut refs, tokens(43, ROWS as usize, model.vocab));
    plain.reports = 103;
    let ticket = submit(&mut model, plain, None);
    let _ = settled(&mut model, ticket);
    let heard = heard.lock().expect("heard");
    let fold = |tag: u64| heard.iter().find(|w| w.tag == tag).map(|w| w.fold);
    assert_eq!(fold(101), Some(Some(N)), "the first fire reports its count");
    assert_eq!(fold(102), Some(Some(N)), "the second, in flight with it, its own");
    assert_eq!(fold(103), Some(None), "a lane no program folds reports no count");
}
