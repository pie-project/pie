#![cfg(feature = "device")]
//! A lane whose prologue does not commit folds nothing into its recurrent
//! state, on every body: a decode step on a set that folds all it writes
//! (`record::RsBody::Folds`) served from a body the load armed holds back
//! the row its gate refused, as the eager walk does. The step after reads
//! the state as if the refused step never ran.

use std::path::Path;

use engine::ProgramRegistration;
use engine::frame::Shell as FrameShell;
use engine_cuda::serve::{Attached, StepView};
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

use crate::common_encoder::{Weights, contract_for};

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const TAPS: u32 = 4;
const KV_ROW: &str = "kv";
const CONV_ROW: &str = "conv";
const PAGE: u32 = 16;
const PROMPT: u32 = 24;
/// The refused step, the folding twin and the step that reads them.
const SLOTS: u32 = 3;

/// A causal conv over the embeddings (its last taps the recurrent state),
/// then one attention layer.
struct Micro;

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(KV_ROW, Dtype::Bf16, [plane, plane], HEAD_DIM);
        spec.state(CONV_ROW, [u64::from(TAPS), WIDTH], Dtype::Bf16);
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let c = ops::ssm::causal_conv1d(
            &x,
            &w("conv", WIDTH, u64::from(TAPS)),
            None,
            inputs.state(CONV_ROW),
            TAPS,
        );
        let q = ops::linear::matmul(&c, &w("q", plane, WIDTH));
        let k = ops::linear::matmul(&c, &w("k", plane, WIDTH));
        let v = ops::linear::matmul(&c, &w("v", plane, WIDTH));
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, false, None,
            None, false, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: poem::Trace, path: &Path, graphs: Graphs) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(1, 64),
        patches: None,
        page_size: PAGE,
        context: 256,
        slots: SLOTS,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((graphs, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    for slot in 0..SLOTS {
        shell.open(slot).expect("a slot opens");
    }
    shell
}

/// A prologue taking from a host-written channel nothing was published on:
/// it never commits.
fn refusing_prologue() -> TraceContainer {
    let decl = |host_role| ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(EtaDtype::I32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![decl(HostRole::Writer), decl(HostRole::Reader)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn bind(shell: &mut Shell) -> engine::InstanceId {
    let bound = crate::bound::logits(refusing_prologue(), VOCAB).expect("the prologue binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0x9a7e,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (1..=count as u64).map(engine::ChannelId).collect();
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };
    let (scratch, pair) = guest.places(
        (program, extents),
        &shell
            .guest_state(program, extents)
            .expect("the program's guest state"),
    );
    shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::NONE,
            &vec![None; count],
            &ids,
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
                scratch,
                pair,
                pair_generation: 0,
            },
        )
        .expect("the instance binds")
}

/// One lane of `tokens` in `slot`, `held` rows before them; its last row of
/// logits.
fn fire(
    shell: &mut Shell,
    trace: &poem::Trace,
    slot: u32,
    tokens: &[u32],
    held: u32,
    attachments: &[Attached],
) -> Vec<f32> {
    let per = 256 / PAGE;
    let pages: Vec<u32> = (slot * per..(slot + 1) * per).collect();
    let word = trace.facts.word(&Request::new(tokens.len() as u32, false));
    let seated = [Seated {
        pages: &pages,
        held: Some(held),
        reports: 10 + u64::from(slot),
        ..Seated::of(Lane {
            slot,
            word,
            tokens,
        })
    }];
    let prepared = FrameShell::prepare(
        shell,
        StepView {
            lanes: &seated,
            attachments,
            media: &[],
            voxels: &[],
        },
        None,
    )
    .expect("the step prepares");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the step enqueues");
    let mut settled = shell
        .settle_step(enqueued, None)
        .expect("a guest fault is not a refused step");
    shell.read_out(&mut settled).expect("the step reads out");
    std::mem::take(&mut settled.logits).swap_remove(0)
}

fn worst(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "one row of logits each");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

/// Slot 0 is refused a decode step, slot 1 folds it, slot 2 never fires
/// it; each then decodes the same token at the same place. What slot 0 and
/// slot 2 answer, and what slot 1 answers apart.
fn serve(trace: &poem::Trace, path: &Path, graphs: Graphs) -> (f32, f32) {
    let mut shell = shell(trace.clone(), path, graphs);
    let instance = bind(&mut shell);
    let prompt: Vec<u32> = (0..PROMPT).map(|t| (t * 7 + 11) % VOCAB).collect();
    for slot in 0..SLOTS {
        fire(&mut shell, trace, slot, &prompt, 0, &[]);
    }
    let step = [prompt[PROMPT as usize - 1] ^ 5];
    let refused = [Attached {
        lane: 0,
        lanes: 1,
        instance,
        adapter_only: false,
    }];
    let hits = shell.body_stats().tally.hits;
    fire(&mut shell, trace, 0, &step, PROMPT, &refused);
    fire(&mut shell, trace, 1, &step, PROMPT, &[]);
    if matches!(graphs, Graphs::On) {
        assert!(
            shell.body_stats().tally.hits > hits,
            "the refused step and its folding twin are served from armed bodies"
        );
        let key = shell
            .replayed_body()
            .expect("the decode step replays a body")
            .to_string();
        assert!(
            !key.contains("~rs"),
            "the decode step folds all it writes: {key} buffers"
        );
    }
    // Each answers the token again where it stands: the refused step's kv
    // row is written over with the same row, its state is what it folded.
    let after_refused = fire(&mut shell, trace, 0, &step, PROMPT, &[]);
    let after_folded = fire(&mut shell, trace, 1, &step, PROMPT, &[]);
    let alone = fire(&mut shell, trace, 2, &step, PROMPT, &[]);
    assert!(
        after_refused.iter().all(|x| x.is_finite()),
        "{graphs:?}: the step after a refused one answers a non-finite logit"
    );
    (worst(&after_refused, &alone), worst(&after_folded, &alone))
}

#[test]
fn a_prologue_that_does_not_commit_folds_no_state() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("refused-fold-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 3307).write(dir.path());
    for graphs in [Graphs::Off, Graphs::On] {
        let (held_back, folded) = serve(&trace, &path, graphs);
        eprintln!("{graphs:?}: refused {held_back}, folded {folded}");
        assert!(
            folded > 1e-2,
            "{graphs:?}: a folded step moves the next one's logits ({folded}), so a fold \
             the gate let through would be seen"
        );
        assert!(
            held_back <= 1e-3,
            "{graphs:?}: the step after a refused one answers {held_back} away from a lane \
             that never fired it: the refused row was folded"
        );
    }
}
