#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine::fire::Readout;
use engine::runahead::Runahead;
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};
use poem_compiler::Budget;

pub(crate) const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
pub(crate) const PAGE: u32 = 16;
pub(crate) const PROMPT: u32 = 20;
const LANES: u32 = 4;
/// The keys a score plane holds a run of per query head (`attend`'s).
const KV_MAX: u64 = 2048;

/// One attention layer that captures its scores.
pub(crate) struct Scored;

impl ForwardHybrid for Scored {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, 0.125, None, None, None, false, None,
            None, false, true,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

/// A shell of `trace`, recording bodies or not as `graphs` says.
pub(crate) fn shell(trace: poem::Trace, path: &Path, graphs: Graphs) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 256),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: LANES,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(graphs, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    for slot in 0..LANES {
        shell.open(slot).expect("the slot opens");
    }
    shell
}

/// An epilogue that sums lane 0's `attn.scores` row into a reader channel.
pub(crate) fn reader() -> TraceContainer {
    let width = u32::try_from(u64::from(HEADS) * KV_MAX).expect("a row's width");
    TraceContainer {
        names: vec!["attn.scores".into()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 1,
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
                    shape: Shape::matrix(1, width),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceSum(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// No load backs a score plane, and nothing backs the span after the load:
/// the span the arena states is what it backed, a program reading
/// `attn.scores` states the planes a fire carrying it lays in its term
/// (`GuestState::per_pair`: the widest carve's, `HEADS × KV_MAX` f32 a lane of
/// it), and neither its register, its bind nor a fire that captures the
/// scores it reads maps anything: its programs and its planes lie in the
/// span, under the decode's carve.
#[test]
fn a_score_reader_states_its_planes_and_nothing_backs_them() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("scored-micro", &Scored, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4409).write(dir.path());
    let mut shell = shell(trace.clone(), &path, Graphs::Off);
    let (span, backed, holds) = shell.activation_span();
    assert_eq!(backed, span, "the span is backed whole at load");
    let facts = shell.chunk_facts().expect("the chunk facts");
    let stated = facts.act_chunks.last().expect("the top rung").1;
    // The top rung's chunks are the activation's whole span (the arena and
    // the runs past it): what its runs were asked, to a chunk, all backed.
    let ledger = shell.activation_ledger();
    assert_eq!(
        stated, ledger.chunks,
        "the top rung states the span's chunks"
    );
    assert_eq!(
        u64::from(stated),
        ledger.asked.div_ceil(facts.chunk_bytes),
        "the span states the chunks its runs reach and no more: {ledger:?}"
    );
    assert_eq!(
        ledger.held_end, ledger.asked,
        "every run asked is backed: {ledger:?}"
    );
    assert!(ledger.held >= span, "the arena is backed in it: {ledger:?}");
    eprintln!(
        "span {span} bytes, backed {backed}, {holds} holds at load, rungs {:?}",
        facts.act_chunks
    );

    let pages: Vec<u32> = (0..64 / PAGE).collect();
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    shell
        .fire_rows(
            &[Seated {
                pages: &pages,
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 0,
                    word: trace.facts.word(&Request::new(PROMPT, false)),
                    tokens: &prompt,
                })
            }],
            &[Readout::Last],
        )
        .expect("the prompt prefills");

    // What the runtime would have committed before a bind: the first
    // segment. From here on nothing maps: not the register, the bind, nor
    // the scored fire.
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let maps = engine_cuda::device::elastic::maps();
    let decls = crate::bound::loaded_decls_of(&shell.classes());
    let bound = eta_ir::validate::bind(reader(), vec![decls], crate::bound::caps())
        .expect("the reader binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0x5c0e,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the reader registers");
    let extents = Extents {
        row_count: 1,
        token_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let state = shell
        .guest_state(program, extents)
        .expect("the reader's guest state");
    // The widest carve's planes: a row of `HEADS × KV_MAX` f32 for each of
    // the fire's lanes, whoever's they are, counted once a fire carrying it.
    let planes = u64::from(LANES) * u64::from(HEADS) * KV_MAX * 4;
    let term = state.per_pair;
    assert!(
        term >= planes,
        "a reader's term {term} holds the widest fire's planes, {planes}"
    );
    // The decode below lays its planes in the reader's pair's state
    // (`Plane::planes_at`, `Arena::fire_slots`), never in the span.

    let slots = crate::guest::Guest::of(&facts, segments * facts.segment_chunks).shell_slots(
        program,
        &shell.program_facts(program).expect("the program's facts"),
        extents,
        state,
    );
    let ids = [engine::ChannelId(1)];
    let instance = shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &[None],
            &ids,
            slots,
        )
        .expect("the reader binds");
    assert_eq!(
        shell.activation_span().2,
        holds,
        "a reader's bind backs nothing of the span"
    );
    assert_eq!(
        engine_cuda::device::elastic::maps(),
        maps,
        "a reader's register and bind map nothing"
    );

    let token = [prompt[0]];
    let depth = [PROMPT];
    let fired = shell.fire_media(
        &[Seated {
            pages: &pages,
            held: Some(PROMPT),
            positions: &depth,
            ..Seated::of(Lane {
                slot: 0,
                word: trace
                    .facts
                    .word(&Request::new(1, true).capturing_scores(true)),
                tokens: &token,
            })
        }],
        &[Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }],
        &[],
    );
    match &fired {
        Ok(_) => eprintln!("the scored decode fired inside the span"),
        Err(refused) => {
            let said = refused.to_string();
            assert!(said.contains("attn.scores"), "{said}");
            eprintln!("the scored decode was refused by name: {said}");
        }
    }
    assert_eq!(
        shell.activation_span().2,
        holds,
        "a scored fire backs nothing of the span"
    );
    assert_eq!(
        engine_cuda::device::elastic::maps(),
        maps,
        "a scored fire maps nothing"
    );
    fired.expect("a one-token fire's planes fit under its carve");
}
