#![cfg(target_vendor = "apple")]

//! A fire whose verb states a frozen length behind where the engine's last
//! fold left the record (the runtime learns a device-counted fold only when
//! its fire settles, and states the length before it to a fire staged
//! sooner) replays from where the record stands, not from the stated
//! length: the walk with the stale statement lands the logits and the state
//! of the walk stated exactly, bit for bit. Without it the rows the record
//! already froze would be replayed onto it a second time, silently.
//!
//! The fold is device-counted (`FoldLen::Device`, the attached program's
//! `fold` port), the one kind the runtime can state stale: a host count is
//! committed when its fire is staged, so the next fire is told it.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{Fixture, fixture_of, load_with, same, word};

fn buffer(pages: u32, folded: u32, fold: FoldLen) -> RsVerb {
    RsVerb::Buffer {
        pages: (0..pages).collect(),
        base: 0,
        folded,
        fold,
        lineage: engine::fire::Lineage::Set,
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

/// An instance describing one lane of `tokens` whose `fold` port states
/// `fold`: the second fire's device-counted fold. Its epilogue puts the
/// logits' argmax in a ring, as an instance must publish something.
fn folding(
    shell: &mut Shell,
    vocab: u32,
    tokens: &[u32],
    fold: u32,
) -> engine::program::InstanceId {
    let rows = tokens.len() as u32;
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
        const_port(Port::EmbedIndptr, &[0, rows]),
        const_port(Port::Fold, &[fold]),
    ];
    ports.sort_by_key(|binding| binding.port);
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
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
                    shape: Shape::matrix(1, vocab),
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
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: bound.hash,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the program registers");
    let facts = shell.program_facts(program).expect("the program's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    // The instance and its rings in the pool's top chunks, clear of the
    // lane's records and its rs list pages, which sit in the lowest.
    let top = layout.total_chunks;
    let slots: Vec<u32> = ((top - 2) * instance.per_chunk
        ..(top - 2) * instance.per_chunk + facts.instance.slots)
        .collect();
    let mut next = (top - 1) * channel.per_chunk;
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
    shell
        .bind_program(
            program,
            &[],
            eta_exec::Extents {
                row_count: rows,
                token_count: rows,
                sampled_rows: 1,
                query_len: rows,
                ..eta_exec::Extents::default()
            },
            GeometryClass::DeviceGeometry,
            PortMask::NONE,
            &[engine::ChannelId(7)],
            &slots,
            &rings,
        )
        .expect("the device-geometry instance binds")
}

fn fire(
    shell: &mut Shell,
    fixture: &Fixture,
    slot: u32,
    tokens: &[u32],
    rs: &RsVerb,
    reset: RsReset,
    held: u32,
) -> Vec<Vec<f32>> {
    shell
        .fire_seated(&[Seated {
            held: Some(held),
            rs,
            rs_reset: reset,
            ..Seated::of(Lane {
                slot,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens,
            })
        }])
        .expect("the fire lands")
}

const SECOND: [u32; 2] = [42, 43];

/// Buffers five rows, then two more folding three on the device, then two
/// more: the last fire stated `third_folded` as its frozen length. Every
/// logit row and the slot's state after.
fn walk(
    shell: &mut Shell,
    fixture: &Fixture,
    instance: engine::program::InstanceId,
    slot: u32,
    third_folded: u32,
) -> (Vec<Vec<f32>>, Vec<u8>) {
    let layout = shell.pool_layout();
    let pages = 12u32.div_ceil(layout.rs_tokens.max(1)) + 1;
    let prompt: Vec<u32> = (0..5).map(|at| (7 + at * 37) % 256).collect();
    let third = [44u32, 45];
    let host = |folded| buffer(pages, folded, FoldLen::Host(0));
    shell.open(slot).expect("the slot opens");
    let mut out = fire(shell, fixture, slot, &prompt, &host(0), RsReset::Fresh, 0);
    let device = buffer(pages, 0, FoldLen::Device);
    out.extend(
        shell
            .fire_attached(
                &[Seated {
                    held: Some(5),
                    rs: &device,
                    rs_reset: RsReset::Held,
                    reports: 1,
                    ..Seated::of(Lane {
                        slot,
                        word: word(&fixture.trace, SECOND.len() as u32),
                        tokens: &SECOND,
                    })
                }],
                &[Attached {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                }],
            )
            .expect("the device-fold fire lands"),
    );
    out.extend(fire(
        shell,
        fixture,
        slot,
        &third,
        &host(third_folded),
        RsReset::Held,
        7,
    ));
    (out, shell.state_bytes(slot).expect("the slot reads back"))
}

#[test]
fn a_stale_frozen_length_replays_from_where_the_record_stands() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen35-tiny", None);
    let mut shell = load_with(&fixture, 1024);
    assert!(
        shell.serves_rs_verbs(),
        "a hybrid model buffers recurrent rows"
    );
    // The second fire froze the record through slot 3: stated exactly, and
    // stale (the length before that fold, as a fire staged before it
    // settled is told).
    // The logits' width, from one plain fire. Both walks run on slot 0,
    // which each reopens: the instance's own slots sit past it.
    shell.open(0).expect("the slot opens");
    let vocab = fire(
        &mut shell,
        &fixture,
        0,
        &SECOND,
        &RsVerb::Fold,
        RsReset::Fresh,
        0,
    )[0]
    .len() as u32;
    let instance = folding(&mut shell, vocab, &SECOND, 3);
    let (exact, exact_state) = walk(&mut shell, &fixture, instance, 0, 3);
    let (stale, stale_state) = walk(&mut shell, &fixture, instance, 0, 0);
    // The last fire's logits: the rows a stale statement could replay
    // wrong. (The device-fold fire's own readout is compared by the walks
    // that state it; on this fixture an attached fire's row is not
    // repeatable bit for bit from one instance fire to the next.)
    same(
        "a stale frozen length against the exact one",
        &stale[2..],
        &exact[2..],
    );
    assert_eq!(
        stale_state, exact_state,
        "the record after the stale statement is the record after the exact one"
    );
    eprintln!(
        "a fire told a stale frozen length replays from where its record stands, bit for bit"
    );
}
