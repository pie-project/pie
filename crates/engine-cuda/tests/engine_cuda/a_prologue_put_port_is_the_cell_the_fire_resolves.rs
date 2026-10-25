#![cfg(feature = "device")]

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Graphs, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{
    ChanDType, ChannelDecl, HostRole, PortBinding, PortSource, StageProgram, TraceContainer,
};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Port, Stage};
use eta_ir::types::{Dtype as EtaDtype, Literal, Shape};
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops};
use poem_compiler::Budget;

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const SM_SCALE: f32 = 0.125;
const KV_ROW: &str = "kv";
const ROWS: u32 = 7;
const FIRST_TOKEN: u32 = 5;

/// One causal attention layer over a paged cache.
struct Micro;

impl ForwardHybrid for Micro {
    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        let space = spec.kv_space(Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(space, KV_ROW, [plane, plane], HEAD_DIM);
        spec
    }

    fn forward(&self, inputs: Input) -> Value {
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
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, SM_SCALE, None, None, None, false, false,
            None, None, None, false,
        );
        let h = ops::linear::matmul(&a, &w("o", WIDTH, plane));
        ops::linear::lm_head(&h, &table)
    }
}

fn shell(trace: poem::Trace, path: &Path) -> Shell {
    let contract = contract_for(&trace, path).expect("the random planes fit the trace");
    let mut shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: path,
        budget: Budget::new(1, 64),
        patches: None,
        page_size: 16,
        context: 256,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            chip: Some(engine_cuda::chip::pinned(Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    shell.open(0).expect("the slot opens");
    shell
}

/// A program whose prologue puts the lane's tokens and positions on the
/// channels its `embed_tokens` and `positions` ports read; nothing else
/// writes them.
fn stating_prologue() -> TraceContainer {
    let decl = || ChannelDecl {
        shape: Shape::vector(ROWS),
        dtype: ChanDType::Concrete(EtaDtype::U32),
        capacity: 2,
        host_role: HostRole::None,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![decl(), decl()],
        ports: vec![
            PortBinding {
                port: Port::EmbedTokens,
                source: PortSource::Channel(0),
            },
            PortBinding {
                port: Port::Positions,
                source: PortSource::Channel(1),
            },
        ],
        stages: vec![StageProgram {
            stage: Stage::Prologue,
            ops: vec![
                Op::Iota { len: ROWS },
                Op::Const(Literal::U32(FIRST_TOKEN)),
                Op::Broadcast {
                    value: 1,
                    shape: Shape::vector(ROWS),
                },
                Op::Add(0, 2),
                Op::ChanPut { chan: 0, value: 3 },
                Op::ChanPut { chan: 1, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn bind(shell: &mut Shell) -> engine::InstanceId {
    let bound = crate::bound::logits(stating_prologue(), VOCAB).expect("the prologue binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0x9071,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (1..=count as u64).map(engine::ChannelId).collect();
    shell
        .bind_program(
            program,
            &[],
            Extents {
                row_count: ROWS,
                token_count: ROWS,
                sampled_rows: ROWS,
                query_len: ROWS,
                ..Extents::default()
            },
            GeometryClass::DecodeEnvelope,
            &vec![None; count],
            &ids,
        )
        .expect("the instance binds")
}

fn fire(shell: &mut Shell, word: u64, tokens: &[u32], attachments: &[Attached]) -> Vec<f32> {
    let every: Vec<u32> = (0..ROWS).collect();
    let seated = Seated {
        readout: Some(&every),
        ..Seated::of(Lane {
            slot: 0,
            word,
            tokens,
        })
    };
    let mut scores = Vec::new();
    let mut out = shell
        .fire_media(&[seated], attachments, &[], &mut scores)
        .expect("the fire runs");
    out.remove(0)
}

// A prologue that puts a descriptor port's channel commits before the fire
// resolves its ports: the forward embeds and places the cells it put.
#[test]
fn the_forward_reads_the_ports_its_prologue_put() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("port-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 1213).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<u32> = (0..ROWS).map(|at| at + FIRST_TOKEN).collect();

    let mut plain = shell(trace.clone(), &path);
    let want = fire(&mut plain, word, &tokens, &[]);
    drop(plain);

    let mut shell = shell(trace, &path);
    let instance = bind(&mut shell);
    let attached = [Attached {
        lane: 0,
        lanes: 1,
        instance,
        adapter_only: false,
    }];
    let placeholder = vec![0u32; ROWS as usize];
    let stale = fire(&mut shell, word, &placeholder, &[]);
    assert!(
        stale.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-2),
        "the placeholder tokens answer what the stated ones do"
    );
    // Twice: each fire's prologue puts the cells its epilogue takes back.
    for at in 0..2 {
        shell.open(0).expect("the slot reopens");
        let got = fire(&mut shell, word, &placeholder, &attached);
        assert_eq!(got.len(), want.len(), "fire {at}: logits");
        let off = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            off <= 1e-2,
            "fire {at}: the forward over the prologue's ports answers {off} away from the \
             host-stated lane"
        );
    }
}
