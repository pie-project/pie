#![cfg(target_vendor = "apple")]

//! The score capture's contract on Metal. A fire lays one score row a
//! capturing lane and none for a lane that captures nothing. Its rows are
//! the CPU oracle's (`engine_cpu::Forward`): the attention mass a key of the
//! load's context, summed over the layers and averaged over the query heads,
//! and that summed a kv page. A fire of 256 lanes, one capturing, lays one
//! lane's rows. A capturing lane no program attached in its fire reads lays
//! none: its rows are a term of their reader's.

use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem::Request;
use poem_compiler::Budget;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, seeded_bf16, word};

/// A program reading a lane's `attn.scores` row (`keys` wide) and putting
/// its argmax in a channel, bound for slot `slot`: the reader a capturing
/// lane's rows are a term of.
pub(super) fn bind_score_reader(shell: &mut Shell, keys: u32, slot: u32) -> engine::InstanceId {
    use eta_compiler::codegen::program::{Backend, emit_program};
    use eta_compiler::plan::compile_bound;
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, PortMask, Stage};
    use eta_ir::types::{Dtype as EtaDtype, Shape};
    let scores = OutputDecl {
        name: "attn.scores".into(),
        axis: Axis::Lane,
        width: keys,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: false,
    };
    let container = TraceContainer {
        names: vec!["attn.scores".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(EtaDtype::I32),
            capacity: 4,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, keys),
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
        vec![vec![scores]],
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
    .expect("the reader binds");
    let stages = compile_bound(&bound);
    let backend = Backend::parse("metal").expect("the metal codegen backend");
    let registration = engine::program::ProgramRegistration {
        program_hash: bound.hash,
        emitted_kernels: emit_program(backend, &stages, &bound),
        emitter_version: backend.emitter_version(),
        launch: eta_compiler::codegen::launch::build(&bound, &stages),
        ..Default::default()
    };
    let program = shell.register_program(&registration).expect("the reader registers");
    let facts = shell.program_facts(program).expect("the reader's facts");
    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    let base = (1 + slot) * instance.per_chunk;
    let slots: Vec<u32> = (base..base + facts.instance.slots).collect();
    let mut next = (3 + slot) * channel.per_chunk;
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
                row_count: 1,
                token_count: 1,
                sampled_rows: 1,
                query_len: 1,
                ..eta_exec::Extents::default()
            },
            GeometryClass::Host,
            PortMask::NONE,
            &[engine::ChannelId(900 + u64::from(slot))],
            &slots,
            &rings,
        )
        .expect("the reader instance binds")
}

/// `lanes` fired as one step, lane 0 attached to `reader`.
pub(super) fn fire_read(shell: &mut Shell, lanes: &[Seated<'_>], reader: engine::InstanceId) {
    use engine::frame::Shell as FrameShell;
    let attached = [engine_metal::serve::Attached {
        lane: 0,
        lanes: 1,
        instance: reader,
        adapter_only: false,
    }];
    let prepared = FrameShell::prepare(
        shell,
        engine_metal::serve::StepView {
            lanes,
            attachments: &attached,
            media: &[],
            clips: &[],
            done: None,
        },
        None,
    )
    .expect("the fire prepares");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the fire enqueues");
    let landed = FrameShell::settle(shell, enqueued).expect("the fire settles");
    let _ = shell.rows_of(&landed);
}

const MODEL: &str = "qwen3-micro-text";
const LANES: u32 = 256;
const CONTEXT: u32 = 64;
const PAGE: u32 = 16;

#[test]
fn the_score_rows_are_the_oracles() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of(MODEL, None);
    let mut shell = Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(LANES, 512),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: LANES,
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"));
    let prompt: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let one: Vec<Vec<u32>> = (1..LANES).map(|lane| vec![(lane * 7) % 256]).collect();
    let capturing = fixture
        .trace
        .facts
        .word(&Request::new(prompt.len() as u32, false).capturing_scores(true));

    // A fire of no capturing lane lays no row.
    for slot in 0..LANES {
        shell.open(slot).expect("a slot opens");
    }
    let plain: Vec<Seated<'_>> = (0..LANES)
        .map(|slot| {
            let tokens: &[u32] = if slot == 0 { &prompt } else { &one[slot as usize - 1] };
            Seated {
                held: Some(0),
                ..Seated::of(Lane {
                    slot,
                    word: word(&fixture.trace, tokens.len() as u32),
                    tokens,
                })
            }
        })
        .collect();
    shell.fire_seated(&plain).expect("the plain fire lands");
    assert!(
        shell.score_rows().expect("the rows read back").is_empty(),
        "a fire of no capturing lane laid a score row"
    );

    // One capturing lane among 256 lays one lane's rows.
    for slot in 0..LANES {
        shell.open(slot).expect("a slot reopens");
    }
    let mixed: Vec<Seated<'_>> = (0..LANES)
        .map(|slot| {
            let tokens: &[u32] = if slot == 0 { &prompt } else { &one[slot as usize - 1] };
            let word = if slot == 0 {
                capturing
            } else {
                word(&fixture.trace, tokens.len() as u32)
            };
            Seated {
                held: Some(0),
                ..Seated::of(Lane { slot, word, tokens })
            }
        })
        .collect();
    // With no program reading its rows, a capturing lane lays none: its
    // rows are a term of their reader's, and nothing can read them.
    shell.fire_seated(&mixed).expect("a capturing lane no program reads still serves");
    assert!(
        shell.score_rows().expect("the rows read back").is_empty(),
        "a capturing lane no program reads laid a score row"
    );
    for slot in 0..LANES {
        shell.open(slot).expect("a slot reopens");
    }
    let reader = bind_score_reader(&mut shell, CONTEXT, 0);
    fire_read(&mut shell, &mixed, reader);
    let rows = shell.score_rows().expect("the rows read back");
    assert_eq!(rows.len(), 1, "one capturing lane of {LANES} laid {} rows", rows.len());
    let (source, keys, pages) = &rows[0];
    assert_eq!(*source, 0, "the row is the capturing lane's");
    assert_eq!(keys.len(), CONTEXT as usize, "a mass a key of the context");
    // The load's kv page, as its chunk recipe lays it.
    let page = shell.paging().page_size;
    assert_eq!(pages.len(), CONTEXT.div_ceil(page) as usize, "a mass a page of it");

    // The oracle: the same trace and planes on the host, one lane.
    let host_trace = fixture.trace.clone();
    let host_word = host_trace
        .facts
        .word(&Request::new(prompt.len() as u32, false).capturing_scores(true));
    let mut oracle = engine_cpu::Forward::new(host_trace, None, &seeded_bf16)
        .expect("the oracle loads")
        .at_context(CONTEXT, page);
    let outs = oracle.step(&prompt, 0, host_word).expect("the oracle steps");
    let floats = |name: &str| -> Vec<f32> {
        match outs.iter().find(|out| out.name == name) {
            Some(engine_cpu::forward::Output {
                values: engine_cpu::forward::Values::F32(v),
                ..
            }) => v.clone(),
            _ => panic!("the oracle reads out no `{name}`"),
        }
    };
    let (want_keys, want_pages) = (floats("attn.scores"), floats("attn.scores.pages"));
    let apart = |got: &[f32], want: &[f32]| -> f32 {
        got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0.0, f32::max)
    };
    let mass: f32 = want_keys.iter().sum();
    let (keys_apart, pages_apart) = (apart(keys, &want_keys), apart(pages, &want_pages));
    eprintln!(
        "the row against the oracle: keys {keys_apart:e}, pages {pages_apart:e} apart at most, a \
         row of mass {mass} (the layers' count)"
    );
    // bf16 activations against the oracle's f32: a few hundredths of one
    // layer's mass a key at most; a row of another head, layer or lane is
    // off by tenths.
    assert!(keys_apart < 2e-2, "the key row is not the oracle's: {keys_apart} apart");
    assert!(pages_apart < 4e-2, "the page row is not the oracle's: {pages_apart} apart");
}
