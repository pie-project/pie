#![cfg(target_vendor = "apple")]

//! A lane that buffers recurrent rows with a device-resident fold length
//! (`FoldLen::Device`) folds what its attached program's `fold` port states
//! for that lane, and the fire reports it at settle: a one-entry list states
//! it for every lane, a longer one each lane's own entry. The runtime takes
//! a successful device-fold fire's report as that lane's own count, so a
//! lane the port states nothing for is refused at prepare, by name: a lane
//! past a per-lane list (never another lane's length in its place, a silent
//! wrong fold), and a lane with no attached program at all.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::fire::{FoldLen, RsVerb};
use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};
use crate::common::{const_port};

const ROWS: u32 = 5;

/// A program describing `lanes` lanes of `ROWS` tokens each, whose `fold`
/// port carries `fold`; its epilogue puts the logits' argmax in a ring.
fn lanes_with_fold(vocab: u32, tokens: &[u32], lanes: u32, fold: &[u32]) -> ProgramRegistration {
    let logits = OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: vocab,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: true,
    };
    let mut ports = vec![
        const_port(Port::EmbedTokens, tokens),
        // Its tokens split evenly over the lanes it describes: `ROWS` a
        // lane where it carries each lane's own.
        const_port(
            Port::EmbedIndptr,
            &(0..=lanes)
                .map(|lane| lane * tokens.len() as u32 / lanes)
                .collect::<Vec<_>>(),
        ),
        const_port(Port::Fold, fold),
    ];
    // Describing more lanes than it carries tokens for, it writes their kv
    // slots too: what a guest needs to reach another lane's slots.
    if tokens.len() as u32 != lanes * ROWS {
        let slots: Vec<u32> = (0..tokens.len() as u32).collect();
        ports.push(const_port(Port::Slots, &slots));
    }
    ports.sort_by_key(|binding| binding.port);
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(lanes),
            dtype: ChanDType::Concrete(EtaDtype::I32),
            capacity: 4,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports,
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(lanes, vocab),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let bound = eta_ir::validate::bind(
        container,
        vec![vec![logits]],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the program binds");
    let stages = compile_bound(&bound);
    let backend = Backend::parse("metal").expect("the metal codegen backend");
    ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    }
}

fn buffering(shell: &Shell) -> RsVerb {
    buffering_with(shell, FoldLen::Device)
}

fn buffering_with(shell: &Shell, fold: FoldLen) -> RsVerb {
    let layout = shell.pool_layout();
    let pages = ROWS.div_ceil(layout.rs_tokens.max(1)) + 1;
    RsVerb::Buffer {
        pages: (0..pages).collect(),
        base: 0,
        folded: 0,
        fold,
        lineage: engine::fire::Lineage::Set,
    }
}

fn refusal(why: impl std::fmt::Display) -> String {
    let why = why.to_string();
    assert!(
        why.contains("device-resident fold length") && why.contains("`fold` port"),
        "the refusal names the lane's missing fold: {why}"
    );
    why
}

/// Fires `lanes` device-fold lanes under one attached program whose
/// `fold` port carries `fold`.
fn fire_with_fold(lanes: u32, fold: &[u32]) -> Option<String> {
    fire_attached_at(lanes, lanes, fold)
}

/// Fires `lanes` device-fold lanes, the program describing them all
/// through its ports and attached at the first `owned` of them.
fn fire_attached_at(lanes: u32, owned: u32, fold: &[u32]) -> Option<String> {
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(shell.serves_rs_verbs(), "a hybrid model buffers recurrent rows");
    let tokens: Vec<u32> = (0..lanes * ROWS).map(|at| (7 + at * 37) % 256).collect();
    // The logits' width, from one plain fire of a spare slot.
    let spare = lanes;
    shell.open(spare).expect("the spare slot opens");
    let vocab = shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(0),
            ..Seated::of(Lane {
                slot: spare,
                word: word(&fixture.trace, ROWS),
                tokens: &tokens[..ROWS as usize],
            })
        }])
        .expect("a plain fire lands")[0]
        .len() as u32;
    // It carries the tokens of the lanes it is attached at, split over the
    // lanes it describes.
    let carried = &tokens[..(owned * ROWS) as usize];
    let program = shell
        .register_program(&lanes_with_fold(vocab, carried, lanes, fold))
        .expect("the program registers");
    let facts = shell.program_facts(program).expect("the program's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    // From the first chunk no slot holds: past the engine's tables run,
    // which no slot names, and the seats' runs.
    let free = layout.free_chunk();
    let first = free * instance.per_chunk;
    let slots: Vec<u32> = (first..first + facts.instance.slots).collect();
    let mut next = (free + 1) * channel.per_chunk;
    let rings: Vec<engine::RingSlots> = facts
        .rings
        .iter()
        .map(|ring| {
            let slots = (next..next + ring.footprint.slots).collect();
            next += ring.footprint.slots;
            engine::RingSlots {
                channel: ring.channel,
                slots,
            }
        })
        .collect();
    let extents = eta_exec::Extents {
        row_count: lanes * ROWS,
        token_count: lanes * ROWS,
        sampled_rows: lanes,
        query_len: ROWS,
        ..eta_exec::Extents::default()
    };
    let (scratch, pair) = crate::guest_places(&mut shell, program, &extents, 0);
    let id = shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::DeviceGeometry,
            PortMask::NONE,
            &[engine::ChannelId(7)],
            &slots,
            &rings,
            &scratch,
            &pair,
            0,
        )
        .expect("the device-geometry instance binds");
    // A lane past the attachment is a plain request's: its fold is its own.
    let verbs: Vec<RsVerb> = (0..lanes)
        .map(|lane| {
            if lane < owned {
                buffering(&shell)
            } else {
                buffering_with(&shell, FoldLen::Host(0))
            }
        })
        .collect();
    for slot in 0..lanes {
        shell.open(slot).expect("the lane's slot opens");
    }
    let seated: Vec<Seated> = verbs
        .iter()
        .zip(0u32..)
        .map(|(verb, slot)| Seated {
            replay_stated: None,
            held: Some(0),
            rs: verb,
            reports: 1 + u64::from(slot),
            ..Seated::of(Lane {
                slot,
                word: word(&fixture.trace, ROWS),
                tokens: &tokens[(slot * ROWS) as usize..((slot + 1) * ROWS) as usize],
            })
        })
        .collect();
    shell
        .fire_attached(
            &seated,
            &[Attached {
                lane: 0,
                lanes: owned,
                instance: id,
                adapter_only: false,
            }],
        )
        .err()
        .map(|refused| refused.to_string())
}

/// A program attached at one lane whose `embed_indptr` port describes two
/// would set the second lane's positions and kv slots, another request's:
/// refused by name, its own lane alone.
#[test]
fn geometry_ports_describing_lanes_past_the_attachment_are_refused() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let refused = fire_attached_at(2, 1, &[3])
        .expect("ports describing a lane the attachment does not own are refused");
    assert!(
        refused.contains("attached at lane 0 is refused")
            && refused.contains("describes 2 lane(s) and its attachment owns 1"),
        "the refusal names the instance's lane and the lanes it described: {refused}"
    );
}

#[test]
fn a_lane_past_the_fold_ports_list_is_refused() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let refused = fire_with_fold(3, &[3, 2])
        .expect("a lane past a per-lane fold list is refused");
    let why = refusal(&refused);
    assert!(why.contains("lane 2"), "the refusal names the third lane: {why}");
}

#[test]
fn a_one_entry_fold_port_serves_every_lane() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    if let Some(why) = fire_with_fold(2, &[3]) {
        panic!("a one-entry fold states every lane's fold: {why}");
    }
}

#[test]
fn a_device_fold_lane_without_a_program_is_refused() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(shell.serves_rs_verbs(), "a hybrid model buffers recurrent rows");
    shell.open(0).expect("slot 0 opens");
    let tokens: Vec<u32> = (0..ROWS).map(|at| (7 + at * 37) % 256).collect();
    let verb = buffering(&shell);
    let lane = Seated {
        replay_stated: None,
        held: Some(0),
        rs: &verb,
        reports: 1,
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, ROWS),
            tokens: &tokens,
        })
    };
    let refused = shell
        .fire_seated(&[lane])
        .expect_err("a device-fold lane with no program to state its fold is refused");
    let why = refusal(&refused);
    assert!(why.contains("lane 0"), "the refusal names the lane: {why}");
}
