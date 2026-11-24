//! An adapter an instance carries lands at its bind in the pool's `Adapter`
//! slots, one plane a bank, and a fire reads its own lanes' adapters alone
//! (k, the power of two at or past their count, planes a bank). An adapter
//! of zero planes serves the unadapted logits bit for bit, other planes
//! move them to what the banked path answered, bit for bit; its slots hold
//! the resolver's planes byte for byte; a second holder of the same slots
//! joins the landing (nothing written again); a fire of k adapters answers
//! each lane what it answers alone at k = 1, 2, 4, 8; after its slots move
//! and the instance is rebound there its lanes read the same logits; and an
//! adapter registered by id is refused. Skipped without a PJRT plugin.

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::SlotCopy;
use engine::fire::{Attachment, FrameSubmission, Lane, Step};
use engine::program::{InstanceBinding, InstanceId, ProgramId, ProgramRegistration, Rebind};
use engine::transfer::MemoryDomain;
use engine_xla::store::layout::ADAPTER;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype as EtaDtype, Shape};

use crate::common_llm::{Model, Refs, Seq, bits, tokens};

const PROMPT: usize = 24;

/// A prologue whose `lora` sink reads its A and B planes out of two seeded
/// channels: what `Pass::adapter` lowers to.
fn lora_program(layers: u32, rank: u32, hidden: u32) -> TraceContainer {
    let chan = |shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 1,
        host_role: HostRole::None,
        seeded: true,
    };
    let a = Shape::new(&[layers, rank, hidden]).expect("A is [layers, rank, hidden]");
    let b = Shape::new(&[layers, hidden, rank]).expect("B is [layers, hidden, rank]");
    TraceContainer {
        names: vec!["lora".to_string()],
        channels: vec![chan(a), chan(b), chan(Shape::vector(1))],
        ports: vec![],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::ChanRead(0),
                Op::ChanRead(1),
                Op::ChanRead(2),
                Op::SinkCall {
                    name: 0,
                    args: vec![0, 1, 2],
                },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn register(model: &mut Model, layers: u32, rank: u32, hidden: u32) -> ProgramId {
    let bound = eta_ir::validate::bind(
        lora_program(layers, rank, hidden),
        vec![vec![eta_ir::registry::OutputDecl {
            name: "logits".to_string(),
            axis: eta_ir::registry::Axis::Row,
            width: model.vocab,
            dtype: EtaDtype::F32,
            layers: 0,
            readout: true,
        }]],
        eta_ir::registry::EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    model
        .engine
        .register_program(&ProgramRegistration {
            launch,
            ..Default::default()
        })
        .expect("the lora program registers")
}

/// F32 values of a small spread, a seed apart: one plane's wire.
fn wire(elements: usize, seed: u64) -> Vec<u8> {
    (0..elements as u64)
        .flat_map(|at| {
            let x = (at.wrapping_mul(2_654_435_761).wrapping_add(seed * 97)) % 2001;
            ((x as f32 - 1000.0) / 20_000.0).to_le_bytes()
        })
        .collect()
}

struct Rig {
    model: Model,
    refs: Refs,
    program: ProgramId,
    /// Elements of the A and B wires.
    a: usize,
    b: usize,
    layers: u32,
    rank: u32,
    hidden: u32,
    next_channel: u32,
    /// `Adapter` slots one adapter takes.
    planes: usize,
    next_slot: u32,
}

impl Rig {
    fn bind(&mut self, seed: Option<u64>, slots: &[u32]) -> InstanceId {
        let (a, b) = match seed {
            Some(seed) => (wire(self.a, seed), wire(self.b, seed + 1)),
            None => (vec![0u8; self.a * 4], vec![0u8; self.b * 4]),
        };
        let seeds = vec![
            engine::channel::ChannelSeed {
                channel: 0,
                bytes: a,
            },
            engine::channel::ChannelSeed {
                channel: 1,
                bytes: b,
            },
            engine::channel::ChannelSeed {
                channel: 2,
                bytes: vec![0u8; 4],
            },
        ];
        let shapes = [
            vec![self.layers, self.rank, self.hidden],
            vec![self.layers, self.hidden, self.rank],
            vec![1],
        ];
        let channels: Vec<ChannelId> = shapes
            .into_iter()
            .map(|shape| {
                let id = ChannelId(u64::from(self.next_channel));
                self.next_channel += 1;
                self.model
                    .engine
                    .register_channel(&ChannelRegistration {
                        id,
                        shape,
                        dtype: ChanDType::Concrete(EtaDtype::F32),
                        host_role: HostRole::None,
                        seeded: true,
                        extern_dir: None,
                        capacity: 1,
                        extern_name: Vec::new(),
                        slots: Vec::new(),
                    })
                    .expect("the channel registers");
                id
            })
            .collect();
        self.model
            .engine
            .bind_instance(&InstanceBinding {
                program: self.program,
                channels,
                seeds,
                adapter_slots: slots.to_vec(),
                ..Default::default()
            })
            .expect("the adapter-carrying instance binds")
            .id
    }

    fn slots(&mut self) -> Vec<u32> {
        (0..self.planes).map(|_| self.refs.take(ADAPTER)).collect()
    }

    /// One fire of a fresh prompt a lane, each lane attached to its
    /// instance (none: plain): each lane's logits, as bits.
    fn fire(&mut self, lanes: &[(u64, Option<InstanceId>)]) -> Vec<Vec<u32>> {
        let mut fired: Vec<Lane> = Vec::new();
        let mut attachments = Vec::new();
        for (at, &(seed, instance)) in lanes.iter().enumerate() {
            let mut seq = Seq::new(self.next_slot % crate::common_llm::slots());
            self.next_slot += 1;
            let prompt = tokens(seed, PROMPT, self.model.vocab);
            fired.push(seq.lane(&self.model, &mut self.refs, prompt));
            if let Some(instance) = instance {
                attachments.push(Attachment {
                    lane: at as u32,
                    lanes: 1,
                    instance,
                    adapter_only: true,
                    scratch: ::engine::ScratchPlace::None,
                });
            }
        }
        let mut ticket = self
            .model
            .engine
            .submit(&FrameSubmission::of(Step {
                lanes: fired,
                attachments,
                media: Vec::new(),
                voxels: Vec::new(),
                shape: None,
            }))
            .expect("the frame fires");
        self.model
            .engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        ticket.steps[0]
            .readouts
            .iter()
            .map(|r| bits(&r.values))
            .collect()
    }

    fn one(&mut self, seed: u64, instance: Option<InstanceId>) -> Vec<u32> {
        self.fire(&[(seed, instance)]).remove(0)
    }
}

#[test]
fn an_adapter_lies_in_its_slots() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    crate::common_llm::set_knob("SLOTS", 8);
    crate::common_llm::set_knob("TOKENS", 256);
    let mut model = Model::load_with("qwen3-micro-text", None, 256, 7, None);
    let seats = model.engine.shell().expect("loaded").bank_seats();
    let lora_a: Vec<_> = seats
        .iter()
        .filter(|s| s.name.ends_with("lora_a"))
        .collect();
    let layers = lora_a.len() as u32;
    let (rank, hidden) = (lora_a[0].rows as u32, lora_a[0].cols as u32);
    assert!(layers > 0, "the model text corrects through LoRA banks");
    let program = register(&mut model, layers, rank, hidden);
    let facts = model
        .engine
        .program_facts(program)
        .expect("the program states its facts");
    assert_eq!(
        facts.adapter.slots,
        2 * layers,
        "one `Adapter` slot a plane: an A and a B a layer"
    );
    let shape = model
        .loaded
        .caps
        .chunks
        .shape(ADAPTER)
        .expect("a model with banks states `Adapter` slots")
        .clone();
    assert_eq!(
        shape.slot_bytes,
        u64::from(rank) * u64::from(hidden) * 2,
        "a slot is one plane"
    );
    let refs = Refs::of(&model);
    let elements = (layers * rank * hidden) as usize;
    let mut rig = Rig {
        model,
        refs,
        program,
        a: elements,
        b: elements,
        layers,
        rank,
        hidden,
        next_channel: 0,
        planes: facts.adapter.slots as usize,
        next_slot: 0,
    };

    let plain = rig.one(1, None);

    // Zero planes correct nothing.
    let zero_slots = rig.slots();
    let zero = rig.bind(None, &zero_slots);
    assert_eq!(
        rig.one(1, Some(zero)),
        plain,
        "an adapter of zero planes corrects nothing"
    );

    // An adapter's planes move its lane's logits, to what the banked path
    // answered for the same planes (memory-redesign f83dd82de, seat 0 of 8).
    let one_slots = rig.slots();
    let one = rig.bind(Some(7), &one_slots);
    let landed = rig.one(1, Some(one));
    assert_ne!(landed, plain, "an adapter's planes move its lane's logits");
    assert_eq!(
        digest(&landed),
        BANKED,
        "the banked path's logits, bit for bit"
    );

    // Its slots hold the planes the resolver builds, byte for byte.
    let seats = rig.model.engine.shell().expect("loaded").bank_seats();
    let mut want =
        engine_xla::adapter::planes_of(engine_xla::adapter::Role::A, None, &wire(rig.a, 7), &seats)
            .expect("A resolves");
    want.extend(
        engine_xla::adapter::planes_of(engine_xla::adapter::Role::B, None, &wire(rig.b, 8), &seats)
            .expect("B resolves"),
    );
    let device = rig.model.loaded.caps.device.domain;
    let host: Vec<u32> = (0..want.len() as u32).collect();
    rig.model
        .engine
        .copy_slots(&SlotCopy {
            shape: ADAPTER,
            src: device,
            dst: MemoryDomain::HostPinned,
            slots: one_slots
                .iter()
                .copied()
                .zip(host.iter().copied())
                .collect(),
            runs: Vec::new(),
        })
        .expect("the slots park");
    let back = rig
        .model
        .engine
        .read_tier_slots(&[(ADAPTER, host)])
        .expect("the host slots read");
    let slot = back.len() / want.len();
    for (at, (bank, bytes)) in want.iter().enumerate() {
        assert_eq!(
            &back[at * slot..at * slot + bytes.len()],
            &bytes[..],
            "slot {at} holds bank `{bank}`'s plane"
        );
    }

    // A second holder of the same slots joins: other seeds land nothing.
    let joined = rig.bind(Some(99), &one_slots);
    assert_eq!(rig.one(1, Some(joined)), landed, "joining wrote nothing");

    // A fire of k adapters answers each lane what it answers alone, at
    // every k (the planes a bank gathered: 1, 2, 4, 8).
    let mut others = Vec::new();
    for seed in 0..7u64 {
        let slots = rig.slots();
        others.push(rig.bind(Some(200 + seed), &slots));
    }
    let alone: Vec<Vec<u32>> = others.iter().map(|&o| rig.one(5, Some(o))).collect();
    for k in [2usize, 4, 8] {
        let mut lanes = vec![(1u64, Some(one))];
        lanes.extend(others[..k - 1].iter().map(|&o| (5u64, Some(o))));
        let fired = rig.fire(&lanes);
        assert_eq!(
            fired[0], landed,
            "k = {k}: the first adapter's lane answers alone"
        );
        for (at, got) in fired[1..].iter().enumerate() {
            assert_eq!(
                got, &alone[at],
                "k = {k}: adapter {at}'s lane answers alone"
            );
        }
    }
    let shared = rig.fire(&[(1, Some(one)), (1, Some(others[0]))]);
    let both = rig.fire(&[(1, Some(one)), (5, Some(others[0]))]);
    assert_eq!(shared[0], landed, "two adapters in one fire");
    assert_eq!(both[1], alone[0], "and each its own");

    // The adapter's slots move and both holders are rebound there.
    let moved = rig.slots();
    let device = rig.model.loaded.caps.device.domain;
    rig.model
        .engine
        .copy_slots(&SlotCopy {
            shape: ADAPTER,
            src: device,
            dst: device,
            slots: one_slots
                .iter()
                .copied()
                .zip(moved.iter().copied())
                .collect(),
            runs: Vec::new(),
        })
        .expect("the adapter's slots move");
    rig.model
        .engine
        .copy_slots(&SlotCopy {
            shape: ADAPTER,
            src: device,
            dst: device,
            slots: zero_slots
                .iter()
                .copied()
                .zip(one_slots.iter().copied())
                .collect(),
            runs: Vec::new(),
        })
        .expect("the old slots are overwritten");
    for holder in [one, joined] {
        rig.model
            .engine
            .rebind_slots(Rebind::Instance {
                id: holder,
                instance: Vec::new(),
                rings: Vec::new(),
                adapter: moved.clone(),
                scratch: engine::ScratchPlace::None,
                pair: engine::ScratchPlace::None,
                pair_generation: 0,
            })
            .expect("the holder is rebound");
    }
    assert_eq!(
        rig.one(1, Some(one)),
        landed,
        "after the move its lanes read the planes where they lie now"
    );

    let refused = rig
        .model
        .engine
        .register_adapter(&engine::adapter::AdapterRegistration {
            id: 0,
            planes: Vec::new(),
        })
        .expect_err("an adapter registered by id is refused");
    assert!(
        refused.to_string().contains("registered by id"),
        "refused by name: {refused}"
    );
}

/// The FNV-1a digest of a lane's logits bits.
fn digest(bits: &[u32]) -> u64 {
    bits.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &w| {
        (h ^ u64::from(w)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The adapted logits the banked path answered for seed 7's planes on this
/// fixture (memory-redesign f83dd82de, the adapter registered at seat 0 of
/// the model text's 8, the lane's adapter id 0).
const BANKED: u64 = 0x4a51_e0aa_001b_bb30;
