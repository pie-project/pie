#![cfg(feature = "device")]
//! A guest program's scratch is state, not the fire's: each instance holds
//! its lane's value scratch and the pair its tables, both in `Instance`
//! slots, and a fire of 64 bound lanes lays nothing of the guest in the
//! activation's span, which stays what the rungs state.

use std::path::Path;

use crate::common_encoder::{Weights, contract_for};
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

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 2;
const KV_ROW: &str = "kv";
/// A decode row a lane: 64 lanes are a fire of 64 rows.
const ROWS: u32 = 1;
const LANES: u32 = 64;

struct Micro;

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
        ops::attn::kv_append(
            &k,
            Some(&v),
            inputs.kv(KV_ROW),
            &inputs.write_page(KV_ROW),
            &inputs.write_offset(KV_ROW),
        );
        let a = ops::attn::attend(
            &q, &inputs, KV_ROW, HEADS, HEADS, HEAD_DIM, 0.125, None, None, None, false, None,
            None, false, false,
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
        budget: Budget::new(LANES, LANES * ROWS),
        patches: None,
        page_size: 16,
        context: 64,
        slots: LANES,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
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

/// An epilogue putting the lane's argmax.
fn epilogue() -> TraceContainer {
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
            dtype: ChanDType::Concrete(EtaDtype::I32),
            capacity: 2,
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
                    shape: Shape::matrix(1, VOCAB),
                    dtype: EtaDtype::F32,
                },
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 0, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn register(shell: &mut Shell) -> engine::ProgramId {
    let bound = crate::bound::logits(epilogue(), VOCAB).expect("the epilogue binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    shell
        .register_program(&ProgramRegistration {
            program_hash: 0x9e57,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers")
}

fn fire(
    shell: &mut Shell,
    word: u64,
    tokens: &[Vec<u32>],
    attachments: &[Attached],
) -> Vec<Vec<f32>> {
    let seated: Vec<Seated<'_>> = tokens
        .iter()
        .enumerate()
        .map(|(lane, tokens)| Seated {
            reports: 10 + lane as u64,
            ..Seated::of(Lane {
                slot: lane as u32,
                word,
                tokens,
            })
        })
        .collect();
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
    let mut settled = shell.settle_step(enqueued, None).expect("the step settles");
    shell.read_out(&mut settled).expect("the step reads out");
    std::mem::take(&mut settled.logits)
}

#[test]
fn a_bound_guest_holds_its_scratch_as_state() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("guest-state-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4243).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = (0..LANES)
        .map(|lane| (0..ROWS).map(|row| (lane * 7 + row * 13) % VOCAB).collect())
        .collect();

    let mut plain = shell(trace.clone(), &path);
    let want = fire(&mut plain, word, &tokens, &[]);
    drop(plain);

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let program = register(&mut shell);
    let extents = Extents {
        row_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..Extents::default()
    };

    // Each instance holds its lane's scratch, the pair its tables.
    let state = shell
        .guest_state(program, extents)
        .expect("the program's guest state");
    assert!(state.per_instance > 0, "{state:?}");
    assert!(state.per_pair > 0, "{state:?}");

    let rungs = shell.chunk_facts().expect("the chunk facts").act_chunks;
    let activation = shell.footprint().1;
    let instances: Vec<engine::InstanceId> = (0..LANES)
        .map(|_| {
            let (scratch, pair) = guest.places((program, extents), &state);
            assert!(
                matches!(
                    (&scratch, &pair),
                    (engine::ScratchPlace::Run { .. }, engine::ScratchPlace::Run { .. })
                ),
                "a small guest's state lies in runs of Instance slots: {scratch:?} {pair:?}"
            );
            let slots = guest.slots(
                &shell.program_facts(program).expect("the program's facts"),
                |_| false,
            );
            shell
                .bind_program(
                    program,
                    &[],
                    extents,
                    GeometryClass::Host,
                    PortMask::NONE,
                    &[None],
                    &[engine::ChannelId(1)],
                    engine_cuda::program::Slots {
                        instance: slots.0,
                        rings: slots.1,
                        scratch,
                        pair,
                    },
                )
                .expect("the instance binds")
        })
        .collect();
    // 64 bound lanes ask the same state, and nothing of the span.
    assert_eq!(
        shell
            .guest_state(program, extents)
            .expect("the guest state"),
        state
    );
    assert_eq!(
        shell.chunk_facts().expect("the chunk facts").act_chunks,
        rungs,
        "the rungs' activation is what it was before any guest bound"
    );
    assert_eq!(
        shell.footprint().1,
        activation,
        "nothing backed past the span"
    );

    let attachments: Vec<Attached> = instances
        .iter()
        .enumerate()
        .map(|(lane, &instance)| Attached {
            lane: lane as u32,
            lanes: 1,
            instance,
            adapter_only: false,
        })
        .collect();
    let got = fire(&mut shell, word, &tokens, &attachments);
    shell.drain().expect("the device drains");
    assert_eq!(
        shell.footprint().1,
        activation,
        "the fire laid nothing past the span"
    );

    for (lane, instance) in instances.iter().enumerate() {
        let off = got[lane]
            .iter()
            .zip(&want[lane])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            off <= 1e-2,
            "lane {lane} answers {off} away from the plain fire"
        );
        let cell = shell
            .program_instance(*instance)
            .expect("the instance settles")
            .expect("the instance")
            .take(0)
            .expect("its channel reads")
            .expect("its epilogue put a cell");
        let argmax = i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"));
        // Its own lane's greatest logit (bf16 logits tie: either index).
        let most = got[lane].iter().copied().fold(f32::MIN, f32::max);
        assert_eq!(
            got[lane][argmax as usize],
            most,
            "lane {lane}'s epilogue put {argmax}, not an argmax of its own logits"
        );
    }

    // The pair's state is given back at its last close.
    for instance in instances {
        shell
            .close_program_instance(instance)
            .expect("the instance closes");
    }
}
