#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, GeometryClass, OutputDecl, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{
    Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Trace, Weight, ops, trace_hybrid,
};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const LANES: u32 = 4;
const ROWS: u32 = 7;
const KV_ROW: &str = "kv";
/// The keys a capture holds a head (`poem::ops::attn::attend`).
const KEYS: u32 = 2048;

/// One attention layer, its scores captured or not.
struct Micro {
    scores: bool,
}

impl ForwardHybrid for Micro {
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
        let pages = inputs.kv(KV_ROW);
        ops::attn::kv_append(
            &k,
            Some(&v),
            pages,
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q,
            &inputs,
            KV_ROW,
            HEADS,
            HEADS,
            HEAD_DIM,
            0.125,
            None,
            None,
            None,
            false,
            None,
            None,
            false,
            self.scores,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        let logits = ops::linear::lm_head(&h, &table);
        poem::output("logits", &logits, None);
    }
}

fn shell(trace: Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(LANES, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: 2,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    shell.open(0).expect("the slot opens");
    shell.open(1).expect("the slot opens");
    shell
}

/// An epilogue reading its lane's scores into a channel of its own.
fn reader() -> TraceContainer {
    let width = HEADS * KEYS;
    TraceContainer {
        names: vec!["attn.scores".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(1, width),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
            host_role: HostRole::None,
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
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn bind(shell: &mut Shell) -> engine::InstanceId {
    let decl = OutputDecl {
        name: "attn.scores".into(),
        axis: Axis::Lane,
        width: HEADS * KEYS,
        dtype: eta_ir::Dtype::F32,
        layers: 1,
        readout: false,
    };
    let bound = eta_ir::validate::bind(reader(), vec![vec![decl]], crate::bound::caps())
        .expect("the reader binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0x5c07e,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let slots = crate::guest::Guest::of(&facts, segments * facts.segment_chunks).slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let backed = shell.footprint().1;
    let instance = shell
        .bind_program(
            program,
            &[],
            Extents {
                row_count: ROWS,
                token_count: ROWS,
                sampled_rows: 1,
                query_len: ROWS,
                ..Extents::default()
            },
            GeometryClass::Host,
            PortMask::default(),
            &[None],
            &[engine::ChannelId(1)],
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
            },
        )
        .expect("the reader binds");
    assert_eq!(
        shell.footprint().1,
        backed,
        "the bind of a score reader backed activation"
    );
    instance
}

// A score reader's rows are its instance's: a load backs nothing for the
// capture planes, a bind backs nothing, and a fire carrying a plain lane
// beside a capturing one answers the plain lane as alone and writes each
// head's mass where the capturing lane's reader reads it.
#[test]
fn a_score_reader_keeps_its_rows_and_a_bind_backs_nothing() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plain = trace_hybrid("score-micro", &Micro { scores: false }, Platform::Cuda);
    let scored = trace_hybrid("score-micro", &Micro { scores: true }, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&plain, 1103).write(dir.path());

    let plain = shell(plain, &path).footprint().1;
    let word = scored
        .facts
        .word(&Request::new(ROWS, false).capturing_scores(true));
    let plain_word = scored.facts.word(&Request::new(ROWS, false));
    let mut shell = shell(scored, &path);
    assert_eq!(
        shell.footprint().1,
        plain,
        "the capturing load backs more activation than the plain one"
    );
    let instance = bind(&mut shell);

    let tokens: Vec<u32> = (0..ROWS).map(|at| (at * 13 + 5) % VOCAB).collect();
    let other: Vec<u32> = (0..ROWS).map(|at| (at * 7 + 3) % VOCAB).collect();
    let plain_lane = || {
        Seated::of(Lane {
            slot: 1,
            word: plain_word,
            tokens: &other,
        })
    };
    let alone = shell
        .fire_media(&[plain_lane()], &[], &[])
        .expect("the plain fire runs")
        .remove(0);
    shell.open(1).expect("the slot reopens");
    let capturing = Seated::of(Lane {
        slot: 0,
        word,
        tokens: &tokens,
    });
    // The capturing lane second: its row is the fire's lane 1.
    let attached = [Attached {
        lane: 1,
        lanes: 1,
        instance,
        adapter_only: false,
    }];
    let beside = shell
        .fire_media(&[plain_lane(), capturing], &attached, &[])
        .expect("the mixed fire runs")
        .remove(0);
    let off = alone
        .iter()
        .zip(&beside)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        alone.len() == beside.len() && off <= 1e-2,
        "the plain lane beside a capturing one answers {off} away from it alone"
    );

    let at = shell
        .score_rows(instance, 0)
        .expect("the reader keeps its lane's rows");
    let mut bytes = vec![0u8; (HEADS * KEYS * 4) as usize];
    engine_cuda::device::Buffer::over(at, bytes.len())
        .read(0, &mut bytes)
        .expect("the rows read back");
    let rows: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect();
    for (head, run) in rows.chunks_exact(KEYS as usize).enumerate() {
        let mass: f32 = run[..ROWS as usize].iter().sum();
        assert!(
            (mass - 1.0).abs() <= 1e-3,
            "head {head}: the mass over the lane's {ROWS} keys is {mass}"
        );
        assert!(
            run[ROWS as usize..].iter().all(|&x| x == 0.0),
            "head {head}: mass past the lane's keys"
        );
    }
}
