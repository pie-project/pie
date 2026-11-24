#![cfg(feature = "device")]
//! One instance's epilogue faulting fails its own lanes: the step settles
//! with those lanes in `faulted`, and every other lane commits its tokens.

use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::common_encoder::{Weights, contract_for};
use engine::ProgramRegistration;
use engine::frame::Shell as FrameShell;
use engine_cuda::serve::{Attached, Done, StepView};
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
const ROWS: u32 = 3;

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
        knobs: engine_cuda::Knobs {
            pin: Some((Graphs::Off, None)),
            ..engine_cuda::Knobs::default()
        },
        ..crate::common::boot(trace, &contract, path, Budget::new(2, 64), 256, 2)
    })
    .expect("the shell loads");
    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    shell
}

/// An epilogue putting the lane's argmax; `faulting`, it also takes from a
/// host-written channel nothing was published on, so it never commits.
fn epilogue(faulting: bool) -> TraceContainer {
    let decl = |host_role| ChannelDecl {
        shape: Shape::vector(1),
        dtype: ChanDType::Concrete(EtaDtype::I32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    let mut ops = vec![
        Op::Output {
            index: 0,
            lane: 0,
            shape: Shape::matrix(1, VOCAB),
            dtype: EtaDtype::F32,
        },
        Op::ReduceArgmax(0),
    ];
    let mut channels = vec![decl(HostRole::Reader)];
    if faulting {
        channels.push(decl(HostRole::Writer));
        ops.extend([
            Op::ChanTake(1),
            Op::Add(1, 2),
            Op::ChanPut { chan: 0, value: 3 },
        ]);
    } else {
        ops.push(Op::ChanPut { chan: 0, value: 1 });
    }
    TraceContainer {
        names: vec!["logits".to_string()],
        channels,
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

fn bind(
    shell: &mut Shell,
    guest: &mut crate::guest::Guest,
    faulting: bool,
    first_channel: u64,
) -> engine::InstanceId {
    let bound = crate::bound::logits(epilogue(faulting), VOCAB).expect("the epilogue binds");
    let stages = compile_bound(&bound);
    let launch = eta_compiler::codegen::launch::build(&bound, &stages);
    let backend = Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
    let program = shell
        .register_program(&ProgramRegistration {
            program_hash: 0xfa17 + u64::from(faulting),
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        })
        .expect("the program registers");
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (0..count as u64)
        .map(|at| engine::ChannelId(first_channel + at))
        .collect();
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

type Told = Arc<Mutex<Vec<engine::StepOutcome>>>;

/// Fires `tokens` a lane each, told to a sink where `told`; each lane's last
/// row of logits.
fn fire(
    shell: &mut Shell,
    word: u64,
    tokens: &[Vec<u32>],
    attachments: &[Attached],
    told: Option<&Told>,
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
    let done = told.map(|told| {
        let told = Arc::clone(told);
        Done {
            at: engine::StepDone {
                frame: engine::FrameId(0),
                step: 0,
            },
            sink: Arc::new(move |_, outcome| told.lock().expect("the sink's lock").push(outcome)),
        }
    });
    let prepared = FrameShell::prepare(
        shell,
        StepView {
            lanes: &seated,
            attachments,
            media: &[],
            voxels: &[],
            shape: None,
        },
        None,
    )
    .expect("the step prepares");
    let enqueued = FrameShell::enqueue(shell, prepared).expect("the step enqueues");
    let mut settled = shell
        .settle_step(enqueued, done)
        .expect("a guest fault is not a refused step");
    shell.read_out(&mut settled).expect("the step reads out");
    std::mem::take(&mut settled.logits)
}

#[test]
fn a_faulted_epilogue_faults_its_own_lanes() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem::trace_hybrid("lane-fault-micro", &Micro, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let path = Weights::random(&trace, 4242).write(dir.path());
    let word = trace.facts.word(&Request::new(ROWS, false));
    let tokens: Vec<Vec<u32>> = vec![vec![3, 9, 27], vec![5, 25, 61]];

    let mut plain = shell(trace.clone(), &path);
    let want = fire(&mut plain, word, &tokens, &[], None);
    drop(plain);

    let mut shell = shell(trace, &path);
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let sound = bind(&mut shell, &mut guest, false, 1);
    let faulting = bind(&mut shell, &mut guest, true, 11);
    let attachments = [
        Attached {
            lane: 0,
            lanes: 1,
            instance: sound,
            adapter_only: false,
        },
        Attached {
            lane: 1,
            lanes: 1,
            instance: faulting,
            adapter_only: false,
        },
    ];
    let told: Told = Arc::default();
    let got = fire(&mut shell, word, &tokens, &attachments, Some(&told));
    shell.drain().expect("the device drains");

    let outcomes = told.lock().expect("the sink's lock").clone();
    let [outcome] = outcomes.as_slice() else {
        panic!("the step is told once, told {outcomes:?}");
    };
    let [fault] = outcome.faulted.as_slice() else {
        panic!("one lane faults: {:?}", outcome.faulted);
    };
    assert_eq!(
        fault.lane, 1,
        "the faulting instance's lane faults: {fault:?}"
    );
    assert_eq!(fault.instance, Some(faulting), "{fault:?}");
    assert!(
        fault.reason.contains(&faulting.to_string()),
        "the reason names the instance: {}",
        fault.reason
    );

    // The sound lane commits: its writes are its tokens, its logits the
    // unattached fire's.
    let wrote = outcome
        .wrote
        .iter()
        .find(|writes| writes.tag == 10)
        .expect("the sound lane reports its writes");
    assert_eq!(wrote.tokens, tokens[0], "the sound lane's tokens land");
    let off = got[0]
        .iter()
        .zip(&want[0])
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        off <= 1e-2,
        "the sound lane answers {off} away from the plain fire"
    );
    let cell = shell
        .program_instance(sound)
        .expect("the instance settles")
        .expect("the sound instance")
        .take(0)
        .expect("its channel reads")
        .expect("its epilogue put a cell");
    let argmax = i32::from_le_bytes(cell[..4].try_into().expect("an i32 cell"));
    let host = want[0]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(at, _)| at as i32)
        .expect("a logit");
    assert_eq!(argmax, host, "the sound epilogue committed its argmax");

    // The next step reaps the faulted step's batch without failing.
    shell.open(0).expect("slot 0 reopens");
    shell.open(1).expect("slot 1 reopens");
    let again = fire(
        &mut shell,
        word,
        &tokens[..1],
        &attachments[..1],
        Some(&told),
    );
    assert_eq!(again.len(), 1);
    shell.drain().expect("the device drains");
    let outcomes = told.lock().expect("the sink's lock");
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes[1].faulted.is_empty(), "{:?}", outcomes[1].faulted);
}
