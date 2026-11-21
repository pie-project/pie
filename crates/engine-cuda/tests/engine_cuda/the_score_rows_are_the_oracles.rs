#![cfg(feature = "device")]

//! The score capture's contract on CUDA. A capturing lane's rows are its
//! reader instance's state: a load backs nothing for them (its activation is
//! a plain load's) and a reader's bind backs nothing. A fire lays one lane's
//! rows a capturing lane and none for a lane that captures nothing: a fire
//! of 256 lanes, one capturing, answers its plain lanes as a fire of none
//! does. The rows are the CPU oracle's (`engine_cpu::Forward`): the
//! attention mass a key of the load's context, summed over the layers and
//! averaged over the query heads, and that summed a kv page.

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
const Q_HEADS: u32 = 4;
const KV_HEADS: u32 = 2;
const LAYERS: usize = 2;
const LANES: u32 = 256;
const ROWS: u32 = 512;
const CONTEXT: u32 = 64;
const PAGE: u32 = 16;
/// The capturing lane: not the fire's first.
const CAPTURING: u32 = 5;

/// Two attention layers, their scores captured or not.
struct Micro {
    scores: bool,
}

fn kv(layer: usize) -> String {
    format!("kv{layer}")
}

impl ForwardHybrid for Micro {
    fn states(&self) -> StateRows {
        let mut spec = StateRows::new();
        let plane = u64::from(KV_HEADS) * u64::from(HEAD_DIM);
        for layer in 0..LAYERS {
            spec.kv(&kv(layer), Dtype::Bf16, [plane, plane], HEAD_DIM);
        }
        spec
    }

    fn forward(&self, inputs: Input) {
        let w = |name: &str, out: u64, inner: u64| Weight::sym(name, [out, inner], Dtype::Bf16);
        let q_plane = u64::from(Q_HEADS) * u64::from(HEAD_DIM);
        let kv_plane = u64::from(KV_HEADS) * u64::from(HEAD_DIM);
        let table = w("embed", u64::from(VOCAB), WIDTH);
        let mut x = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        for layer in 0..LAYERS {
            let at = |name: &str| format!("l{layer}.{name}");
            let q = ops::linear::matmul(&x, &w(&at("q"), q_plane, WIDTH));
            let k = ops::linear::matmul(&x, &w(&at("k"), kv_plane, WIDTH));
            let v = ops::linear::matmul(&x, &w(&at("v"), kv_plane, WIDTH));
            let row = kv(layer);
            let pages = inputs.kv(&row);
            ops::attn::kv_append(
                &k,
                Some(&v),
                pages,
                &inputs.write_page(&row),
                &inputs.write_offset(&row),
            );
            let a = ops::attn::attend(
                &q,
                &inputs,
                &row,
                Q_HEADS,
                KV_HEADS,
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
            x = ops::linear::matmul(&a, &w(&at("o"), WIDTH, q_plane));
        }
        let logits = ops::linear::lm_head(&x, &table);
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
        budget: Budget::new(LANES, ROWS),
        patches: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: LANES,
        kv_budget: Some(engine_cuda::store::KvSize::Bytes(crate::POOL)),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs {
            // Every lane fires in one fire: the load lays its rungs to the
            // rows its budget states, not to the top the estimate would
            // choose for serving (which fires a wider wave as several).
            every_rung: Some(engine_cuda::EveryRung::To(ROWS)),
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
    open_all(&mut shell);
    shell
}

fn open_all(shell: &mut Shell) {
    for slot in 0..LANES {
        shell.open(slot).expect("a slot opens");
    }
}

/// An epilogue reading its lane's key row and summing it into a channel.
fn reader() -> TraceContainer {
    TraceContainer {
        names: vec!["attn.scores".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::vector(1),
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
                    shape: Shape::matrix(1, CONTEXT),
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

fn bind(shell: &mut Shell, rows: u32) -> engine::InstanceId {
    let decl = OutputDecl {
        name: "attn.scores".into(),
        axis: Axis::Lane,
        width: CONTEXT,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
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
    let mut guest = crate::guest::Guest::of(&facts, segments * facts.segment_chunks);
    let slots = guest.slots(
        &shell.program_facts(program).expect("the program's facts"),
        |_| false,
    );
    let extents = Extents {
        row_count: rows,
        token_count: rows,
        sampled_rows: 1,
        query_len: rows,
        ..Extents::default()
    };
    let (scratch, pair) = guest.places(
        (program, extents),
        &shell
            .guest_state(program, extents)
            .expect("the program's guest state"),
    );
    let backed = shell.footprint().1;
    let instance = shell
        .bind_program(
            program,
            &[],
            extents,
            GeometryClass::Host,
            PortMask::default(),
            &[None],
            &[engine::ChannelId(1)],
            engine_cuda::program::Slots {
                resident: engine::ScratchPlace::None,
                instance: slots.0,
                rings: slots.1,
                scratch,
                pair,
                pair_generation: 0,
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

fn floats(at: u64, count: u32) -> Vec<f32> {
    let mut bytes = vec![0u8; count as usize * 4];
    engine_cuda::device::Buffer::over(at, bytes.len())
        .read(0, &mut bytes)
        .expect("the rows read back");
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

fn apart(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "rows of one width");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

#[test]
fn the_score_rows_are_the_oracles() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plain = trace_hybrid("score-micro", &Micro { scores: false }, Platform::Cuda);
    let scored = trace_hybrid("score-micro", &Micro { scores: true }, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let weights = Weights::random(&plain, 1103);
    let path = weights.write(dir.path());

    // A capturing load backs a plain load's activation: its rows are no
    // carve's.
    let plain_activation = shell(plain, &path).footprint().1;
    let mut shell = shell(scored.clone(), &path);
    assert_eq!(
        shell.footprint().1,
        plain_activation,
        "the capturing load backs more activation than the plain one"
    );

    let prompt: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % VOCAB).collect();
    let one: Vec<Vec<u32>> = (0..LANES).map(|lane| vec![(lane * 7) % VOCAB]).collect();
    let word = |tokens: &[u32], captures: bool| {
        scored
            .facts
            .word(&Request::new(tokens.len() as u32, false).capturing_scores(captures))
    };
    let tokens = |slot: u32| -> &[u32] {
        if slot == CAPTURING {
            &prompt
        } else {
            &one[slot as usize]
        }
    };

    // A fire of no capturing lane.
    let seated: Vec<Seated<'_>> = (0..LANES)
        .map(|slot| {
            Seated::of(Lane {
                slot,
                word: word(tokens(slot), false),
                tokens: tokens(slot),
            })
        })
        .collect();
    let alone = shell
        .fire_media(&seated, &[], &[])
        .expect("the plain fire lands");

    // One capturing lane among 256, its reader attached.
    open_all(&mut shell);
    let instance = bind(&mut shell, prompt.len() as u32);
    let seated: Vec<Seated<'_>> = (0..LANES)
        .map(|slot| {
            Seated::of(Lane {
                slot,
                word: word(tokens(slot), slot == CAPTURING),
                tokens: tokens(slot),
            })
        })
        .collect();
    let attached = [Attached {
        lane: CAPTURING,
        lanes: 1,
        instance,
        adapter_only: false,
    }];
    let mixed = shell
        .fire_media(&seated, &attached, &[])
        .expect("the mixed fire lands");
    for slot in (0..LANES).filter(|&slot| slot != CAPTURING) {
        let off = apart(&mixed[slot as usize], &alone[slot as usize]);
        assert!(
            off <= 1e-2,
            "plain lane {slot} beside a capturing one answers {off} away from it in a fire of \
             none"
        );
    }

    let at = shell
        .score_rows(instance)
        .expect("the reader keeps a lane's rows");
    let pages = CONTEXT.div_ceil(PAGE);
    let keys = floats(at, CONTEXT);
    let page_rows = floats(at + u64::from(CONTEXT) * 4, pages);

    // The oracle: the same trace and planes on the host, one lane.
    let mut oracle = engine_cpu::Forward::new(scored.clone(), None, &|name, count| {
        let plane = weights.get(name);
        assert_eq!(plane.len(), count, "plane {name}'s elements");
        plane.to_vec()
    })
    .expect("the oracle loads")
    .at_context(CONTEXT, PAGE);
    let outs = oracle
        .step(&prompt, 0, word(&prompt, true))
        .expect("the oracle steps");
    let oracle_row = |name: &str| -> Vec<f32> {
        match outs.iter().find(|out| out.name == name) {
            Some(engine_cpu::forward::Output {
                values: engine_cpu::forward::Values::F32(v),
                ..
            }) => v.clone(),
            _ => panic!("the oracle reads out no `{name}`"),
        }
    };
    let (want_keys, want_pages) = (oracle_row("attn.scores"), oracle_row("attn.scores.pages"));
    let mass: f32 = keys.iter().sum();
    let (keys_apart, pages_apart) = (apart(&keys, &want_keys), apart(&page_rows, &want_pages));
    eprintln!(
        "the row against the oracle: keys {keys_apart:e}, pages {pages_apart:e} apart at most, a \
         row of mass {mass} (the layers' count, {LAYERS})"
    );
    assert!(
        (mass - LAYERS as f32).abs() < 1e-2,
        "the key row's mass is {mass}, not the layers' count"
    );
    assert!(
        keys_apart < 2e-2,
        "the key row is not the oracle's: {keys_apart} apart"
    );
    assert!(
        pages_apart < 4e-2,
        "the page row is not the oracle's: {pages_apart} apart"
    );
}
