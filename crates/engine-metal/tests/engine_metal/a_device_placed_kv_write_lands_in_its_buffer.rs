#![cfg(target_vendor = "apple")]

//! A device-geometry lane whose program states its rows' `slots` writes its
//! keys and values where those slots' pages lie, in whichever buffer of the
//! pool that is: its prompt prefilled through the guest's write descriptor
//! into pages of a chunk past the first buffer, then decoded on the host
//! path over those pages, serves the logits of the same run over one buffer,
//! which are the host path's own.

use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Port, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};

use engine::program::ProgramRegistration;
use engine_metal::serve::{Attached, Seated};
use engine_metal::{Lane, Shell};

use super::a_scattered_pool_serves_the_same_logits::{
    Fixture, argmax, fixture_of, load_with, same, word,
};

const CONTEXT: u32 = 256;
const PROMPT: u32 = 40;
const STEPS: u32 = 4;

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

/// A program whose ports state one lane's prompt: its tokens, and slots
/// `0..PROMPT` of its working set (positions the slots); its epilogue puts
/// the logits in a ring the host reads.
fn prompt_program(vocab: u32, tokens: &[u32]) -> ProgramRegistration {
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
        const_port(Port::Slots, &(0..rows).collect::<Vec<_>>()),
        const_port(Port::KvLen, &[rows]),
    ];
    ports.sort_by_key(|binding| binding.port);
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(1, vocab),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
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
                Op::ChanPut { chan: 0, value: 0 },
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

fn prompt() -> Vec<u32> {
    (0..PROMPT).map(|at| (at * 37 + 11) % 256).collect()
}

/// The prompt prefilled into `pages` (slot refs of the kv list), through
/// the guest's ports where `device` states the vocabulary, else on the host path; then `STEPS`
/// greedy decodes over the same pages. Every logit row.
fn serve(
    shell: &mut Shell,
    fixture: &Fixture,
    pages: &[u32],
    device: Option<u32>,
) -> Vec<Vec<f32>> {
    let tokens = prompt();
    let mut rows = Vec::new();
    let first = Seated {
        pages,
        held: Some(0),
        refs: true,
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, PROMPT),
            tokens: &tokens,
        })
    };
    if let Some(vocab) = device {
        let program = shell
            .register_program(&prompt_program(vocab, &tokens))
            .expect("the program registers");
        let facts = shell.program_facts(program).expect("the program's facts");
        let layout = shell.pool_layout();
        let instance = layout.instance.expect("an instance shape");
        let channel = layout.channel.expect("a channel shape");
        // The instance's tables in the chunk above the pages', its ring in
        // the chunk above that.
        let above = pages[0] / layout.list.expect("a kv list").per_chunk + 1;
        let at = above * instance.per_chunk;
        let slots: Vec<u32> = (at..at + facts.instance.slots).collect();
        let mut next = (above + 1) * channel.per_chunk;
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
                    row_count: PROMPT,
                    token_count: PROMPT,
                    sampled_rows: 1,
                    query_len: PROMPT,
                    ..eta_exec::Extents::default()
                },
                GeometryClass::DeviceGeometry,
                PortMask::NONE,
                &[engine::ChannelId(7)],
                &slots,
                &rings,
            )
            .expect("the device-geometry instance binds");
        let lane = Seated {
            pages: &[],
            translation: pages,
            ..first
        };
        rows.extend(
            shell
                .fire_attached(
                    &[lane],
                    &[Attached {
                        lane: 0,
                        lanes: 1,
                        instance: id,
                        adapter_only: false,
                    }],
                )
                .expect("the device-placed prefill lands"),
        );
        shell
            .close_program_instance(id)
            .expect("the instance closes");
    } else {
        rows.extend(shell.fire_seated(&[first]).expect("the prefill lands"));
    }
    let mut fed = vec![argmax(rows.last().expect("a prefill row"))];
    for step in 0..STEPS {
        let got = shell
            .fire_seated(&[Seated {
                pages,
                held: Some(PROMPT + step),
                refs: true,
                ..Seated::of(Lane {
                    slot: 0,
                    word: word(&fixture.trace, 1),
                    tokens: &fed,
                })
            }])
            .expect("the decode lands");
        fed = vec![argmax(&got[0])];
        rows.extend(got);
    }
    rows
}

/// The pages of a chunk starting a segment past the seats (under the
/// buffer cap, a buffer past the first), its segment and the next
/// committed.
fn pages_past_the_seats(shell: &mut Shell) -> Vec<u32> {
    let layout = shell.pool_layout();
    let list = layout.list.expect("a kv list");
    let segment = shell.pool_chunk_facts().segment_chunks;
    let first = layout.free_chunk().div_ceil(segment) * segment;
    shell
        .commit_segments((first + 2).div_ceil(segment) + 1)
        .expect("the segments commit");
    let pages = (PROMPT + STEPS).div_ceil(shell.paging().page_size);
    // Scattered over the chunk's slots, last first.
    (0..pages)
        .map(|page| first * list.per_chunk + (list.per_chunk - 1 - 2 * page))
        .collect()
}

#[test]
fn a_device_placed_kv_write_lands_in_its_buffer() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);

    let mut whole = load_with(&fixture, CONTEXT);
    let pages = pages_past_the_seats(&mut whole);
    let host = serve(&mut whole, &fixture, &pages, None);
    let vocab = host[0].len() as u32;
    let one = serve(&mut whole, &fixture, &pages, Some(vocab));
    same(
        "the device-placed prefill against the host path's",
        &one,
        &host,
    );
    drop(whole);

    engine_metal::store::pool::cap_parts(1);
    let spread = {
        let mut shell = load_with(&fixture, CONTEXT);
        let pages = pages_past_the_seats(&mut shell);
        let facts = shell.pool_chunk_facts();
        let chunk = pages[0] / shell.pool_layout().list.expect("a kv list").per_chunk;
        assert!(
            chunk >= facts.segment_chunks,
            "the pages lie past the first buffer (chunk {chunk}, a segment {})",
            facts.segment_chunks
        );
        serve(&mut shell, &fixture, &pages, Some(vocab))
    };
    engine_metal::store::pool::cap_parts(0);
    same(
        "the device-placed prefill past the first buffer against one buffer's",
        &spread,
        &one,
    );
    eprintln!(
        "a device-placed kv write lands in its buffer ({} rows)",
        spread.len()
    );
}
