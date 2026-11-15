#![cfg(target_vendor = "apple")]

//! A lane that buffers recurrent rows with a device-resident fold length
//! (`FoldLen::Device`) folds what its attached program's `fold` port states
//! for that lane, and the fire reports it at settle. The runtime takes a
//! successful device-fold fire's report as that lane's own count, so a lane
//! the port states nothing for is refused at prepare, by name: a lane past
//! the port's list (never the first lane's length in its place, a silent
//! wrong fold), and a lane with no attached program at all.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::fire::{FoldLen, RsVerb};
use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with, word};

const ROWS: u32 = 5;

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

/// A program describing two lanes of `ROWS` tokens each, whose `fold` port
/// states one lane's fold length only; its epilogue puts the logits'
/// argmax in a ring.
fn two_lanes_one_fold(vocab: u32, tokens: &[u32]) -> ProgramRegistration {
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
        const_port(Port::EmbedIndptr, &[0, ROWS, 2 * ROWS]),
        const_port(Port::Fold, &[3]),
    ];
    ports.sort_by_key(|binding| binding.port);
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(2),
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
                    shape: Shape::matrix(2, vocab),
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
    let layout = shell.pool_layout();
    let pages = (2 * ROWS).div_ceil(layout.rs_tokens.max(1)) + 1;
    RsVerb::Buffer {
        pages: (0..pages).collect(),
        base: 0,
        folded: 0,
        fold: FoldLen::Device,
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

#[test]
fn a_lane_past_the_fold_ports_list_is_refused() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(shell.serves_rs_verbs(), "a hybrid model buffers recurrent rows");
    let tokens: Vec<u32> = (0..2 * ROWS).map(|at| (7 + at * 37) % 256).collect();
    // The logits' width, from one plain fire of a spare slot.
    shell.open(3).expect("slot 3 opens");
    let vocab = shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: 3,
                word: word(&fixture.trace, ROWS),
                tokens: &tokens[..ROWS as usize],
            })
        }])
        .expect("a plain fire lands")[0]
        .len() as u32;
    let program = shell
        .register_program(&two_lanes_one_fold(vocab, &tokens))
        .expect("the program registers");
    let facts = shell.program_facts(program).expect("the program's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    let slots: Vec<u32> = (instance.per_chunk..instance.per_chunk + facts.instance.slots).collect();
    let mut next = 2 * channel.per_chunk;
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
    let id = shell
        .bind_program(
            program,
            &[],
            eta_exec::Extents {
                row_count: 2 * ROWS,
                token_count: 2 * ROWS,
                sampled_rows: 2,
                query_len: ROWS,
                ..eta_exec::Extents::default()
            },
            GeometryClass::DeviceGeometry,
            PortMask::NONE,
            &[engine::ChannelId(7)],
            &slots,
            &rings,
        )
        .expect("the device-geometry instance binds");
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let first = buffering(&shell);
    let second = buffering(&shell);
    let lanes: Vec<Seated> = [(0u32, &first), (1, &second)]
        .into_iter()
        .map(|(slot, verb)| Seated {
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
    let refused = shell
        .fire_attached(
            &lanes,
            &[Attached {
                lane: 0,
                lanes: 2,
                instance: id,
                adapter_only: false,
            }],
        )
        .expect_err("a lane the fold port states nothing for is refused");
    let why = refusal(&refused);
    assert!(why.contains("lane 1"), "the refusal names the second lane: {why}");
    eprintln!("refused by name: {why}");
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
    eprintln!("refused by name: {why}");
}
