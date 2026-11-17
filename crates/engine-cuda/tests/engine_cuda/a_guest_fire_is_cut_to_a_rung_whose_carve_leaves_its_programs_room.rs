#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
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

const VOCAB: u32 = 64;
/// Wide enough that the arena spans several chunks, so a rung under the
/// top leaves whole chunks under its carve.
const WIDTH: u64 = 16384;
/// The widest fire the load accepts.
const TOKENS: u32 = 4096;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
const PAGE: u32 = 16;
const LANES: u32 = 4;
/// The keys a score plane holds a run of per query head (`attend`'s).
const KV_MAX: u64 = 2048;

/// One attention layer that captures its scores.
struct Scored;

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

fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, TOKENS),
        patches: None,
        page_size: PAGE,
        context: TOKENS,
        slots: LANES,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, None)),
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
fn reader() -> TraceContainer {
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

/// A fire of a bound program lays the program's term under its carve, so
/// the engine states that term as its standing guest bytes, and the
/// runtime, setting them aside below every rung's carve, cuts the fire to
/// the widest rung that still holds them. A fire past the top named rung
/// carves the whole arena and leaves nothing under it: stated as none (the
/// engine's old answer), the runtime let such a fire through and it was
/// refused at launch ("guests' programs lay ... reserved 0").
#[test]
fn a_guest_fire_is_cut_to_a_rung_whose_carve_leaves_its_programs_room() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("scored-micro", &Scored, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4409).write(dir.path());
    let mut shell = shell(trace.clone(), &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let decls = crate::bound::loaded_decls_of(&shell.classes());
    let bound = eta_ir::validate::bind(reader(), vec![decls], crate::bound::caps())
        .expect("the reader binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0x5c0f,
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
    assert_eq!(shell.guest_standing(), 0, "nothing bound stands");
    let after = shell
        .guest_standing_after(program, extents)
        .expect("the standing a bind would leave");
    let slots = crate::guest::Guest::of(&facts, segments * facts.segment_chunks).slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
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
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
            },
        )
        .expect("the reader binds");
    let standing = shell.guest_standing();
    assert!(standing > 0, "a bound program's fires lay its term");
    assert_eq!(standing, after, "the bind leaves what was asked before it");

    // The runtime's cut (`SlotPool::act_chunks`, `widest_fit`): a rung's
    // chunks with the standing bytes beside them, no more than the top's.
    // Every rung it admits fires with the program aboard; the top rung,
    // whose carve is the whole arena, is never admitted.
    let chunk = facts.chunk_bytes;
    let &(widest, top) = facts.act_chunks.last().expect("the top rung");
    let beside = u32::try_from(standing.div_ceil(chunk)).expect("a chunk count");
    let admitted: Vec<u32> = facts
        .act_chunks
        .iter()
        .filter(|&&(_, chunks)| chunks + beside <= top)
        .map(|&(tokens, _)| tokens)
        .collect();
    eprintln!(
        "standing {standing} bytes ({beside} chunks of {chunk}), rungs {:?}, admitted {admitted:?}",
        facts.act_chunks
    );
    let pages: Vec<u32> = (0..TOKENS / PAGE).collect();
    for rows in admitted.iter().copied().chain(std::iter::once(widest)) {
        let prompt: Vec<u32> = (0..rows).map(|t| (t * 7 + 11) % VOCAB).collect();
        let fired = shell.fire_media(
            &[Seated {
                pages: &pages,
                held: Some(0),
                ..Seated::of(Lane {
                    slot: 0,
                    word: trace.facts.word(&Request::new(rows, true)),
                    tokens: &prompt,
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
        if admitted.contains(&rows) {
            fired.expect("a fire the runtime admits lays its programs under its carve");
        } else {
            let refused = fired.expect_err("the top rung's carve leaves its programs no room");
            eprintln!("a fire of {rows} rows, never admitted, is refused: {refused}");
        }
    }
}
