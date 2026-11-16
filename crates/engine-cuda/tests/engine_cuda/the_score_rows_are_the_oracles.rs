#![cfg(feature = "device")]

//! The score capture's contract on CUDA. A fire lays one score row a
//! capturing lane and none for a lane that captures nothing. Its rows are
//! the CPU oracle's (`engine_cpu::Forward`): the attention mass a key of the
//! load's context, summed over the layers and averaged over the query heads,
//! and that summed a kv page of the load's own pages. A fire of 256 lanes,
//! one capturing, lays one lane's rows, in the span of the program reading
//! them (a capturing lane's rows are its reader's term).

use std::path::{Path, PathBuf};

use engine::ProgramRegistration;
use engine_cuda::serve::Attached;
use engine_cuda::{Boot, Lane, Seated, Shell};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_exec::Extents;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, GeometryClass, OutputDecl, PortMask, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const MODEL: &str = "qwen3-micro-text";
const LANES: u32 = 256;
const CONTEXT: u32 = 64;
const PAGE: u32 = 16;

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A plane's values by its name, bf16-rounded as the checkpoint holds
/// them: norms near one, the rest small.
fn seeded_bf16(name: &str, count: usize) -> Vec<f32> {
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            f32::from_bits(value.to_bits() & 0xffff_0000)
        })
        .collect()
}

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        assert_eq!(param.dtype, Dtype::Bf16, "`{}` is bf16", param.name);
        let count = param.shape.iter().product::<u64>() as usize;
        let bytes: Vec<u8> = seeded_bf16(&param.name, count)
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                ztensor::Leaf::BF16,
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the checkpoint closes");
}

/// An instance of an epilogue reading lane 0's key row (`keys` wide) and
/// page row (`pages` wide) onto channels of its own.
fn bind_reader(shell: &mut Shell, keys: u32, pages: u32) -> engine::InstanceId {
    let reader = |width: u32| ChannelDecl {
        shape: Shape::matrix(1, width),
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    };
    let output = |index: u16, width: u32| Op::Output {
        index,
        lane: 0,
        shape: Shape::matrix(1, width),
        dtype: EtaDtype::F32,
    };
    let decl = |name: &str, width: u32| OutputDecl {
        name: name.to_string(),
        axis: Axis::Lane,
        width,
        dtype: eta_ir::Dtype::F32,
        layers: 0,
        readout: false,
    };
    let container = TraceContainer {
        names: vec!["attn.scores".to_string(), "attn.scores.pages".to_string()],
        channels: vec![reader(keys), reader(pages)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                output(0, keys),
                output(1, pages),
                Op::ChanPut { chan: 0, value: 0 },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let bound = crate::bound::reading(
        container,
        vec![decl("attn.scores", keys), decl("attn.scores.pages", pages)],
    )
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
        .expect("the reader registers");
    let count = bound.container.channels.len();
    let ids: Vec<engine::ChannelId> = (1..=count as u64).map(engine::ChannelId).collect();
    let facts = shell.chunk_facts().expect("the chunk facts");
    let segments = facts.segments.committed.max(1);
    shell
        .commit_segments(segments)
        .expect("the first segment commits");
    let slots = crate::guest::Guest::of(&facts, segments * facts.segment_chunks).slots(
        &shell.program_facts(program).expect("the reader's facts"),
        |_| false,
    );
    shell
        .bind_program(
            program,
            &[],
            Extents {
                row_count: 6,
                token_count: 6,
                sampled_rows: 1,
                query_len: 6,
                ..Extents::default()
            },
            GeometryClass::Host,
            PortMask::default(),
            &vec![None; count],
            &ids,
            engine_cuda::program::Slots {
                instance: slots.0,
                rings: slots.1,
            },
        )
        .expect("the reader binds")
}

#[test]
fn the_score_rows_are_the_oracles() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let trace = poem_compiler::catalog::trace_of(MODEL, Dtype::Bf16, Dtype::Bf16, Platform::Cuda);
    let dir = tempfile::tempdir().expect("a scratch directory");
    let container: PathBuf = dir.path().join("model.zt");
    write_checkpoint(&container, &trace);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .expect("a container of the plan's own planes");
    drop(source);
    let mut shell = Shell::load(Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: &container,
        budget: Budget::new(LANES, 512),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: LANES,
        kv_budget: Some(crate::POOL),
        allowance: engine::Allowance::WHOLE,
        context_declared: false,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("the shell loads: {why}"));
    let prompt: Vec<u32> = (0..6u32).map(|at| (11 + at * 37) % 256).collect();
    let one: Vec<Vec<u32>> = (1..LANES).map(|lane| vec![(lane * 7) % 256]).collect();
    let word = |rows: usize, captures: bool| {
        trace
            .facts
            .word(&Request::new(rows as u32, false).capturing_scores(captures))
    };
    let page = shell.paging().page_size;
    let instance = bind_reader(&mut shell, CONTEXT, CONTEXT.div_ceil(page));
    let fire = |shell: &mut Shell, captures: bool| {
        for slot in 0..LANES {
            shell.open(slot).expect("a slot opens");
        }
        let lanes: Vec<Seated<'_>> = (0..LANES)
            .map(|slot| {
                let tokens: &[u32] = if slot == 0 {
                    &prompt
                } else {
                    &one[slot as usize - 1]
                };
                Seated::of(Lane {
                    slot,
                    word: word(tokens.len(), captures && slot == 0),
                    tokens,
                })
            })
            .collect();
        let attached = [Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }];
        let attached: &[Attached] = if captures { &attached } else { &[] };
        shell
            .fire_media(&lanes, attached, &[])
            .expect("the fire lands");
        shell.score_rows().expect("the rows read back")
    };

    // A fire of no capturing lane lays no row.
    assert!(
        fire(&mut shell, false).is_empty(),
        "a fire of no capturing lane laid a row"
    );

    // One capturing lane among 256 lays one lane's rows.
    let rows = fire(&mut shell, true);
    assert_eq!(
        rows.len(),
        1,
        "one capturing lane of {LANES} laid {} rows",
        rows.len()
    );
    let (lane, keys, pages) = &rows[0];
    assert_eq!(*lane, 0, "the row is the capturing lane's");
    assert_eq!(keys.len(), CONTEXT as usize, "a mass a key of the context");
    // The load's kv page, as its pools lay it.
    assert_eq!(
        pages.len(),
        CONTEXT.div_ceil(page) as usize,
        "a mass a page of it"
    );

    // The oracle: the same trace and planes on the host, one lane.
    let mut oracle = engine_cpu::Forward::new(trace.clone(), None, &seeded_bf16)
        .expect("the oracle loads")
        .at_context(CONTEXT, page);
    let outs = oracle
        .step(&prompt, 0, word(prompt.len(), true))
        .expect("the oracle steps");
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
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs())
            .fold(0.0, f32::max)
    };
    let mass: f32 = want_keys.iter().sum();
    let (keys_apart, pages_apart) = (apart(keys, &want_keys), apart(pages, &want_pages));
    eprintln!(
        "the row against the oracle: keys {keys_apart:e}, pages {pages_apart:e} apart at most, a \
         row of mass {mass} (the layers' count), {} pages of {page}",
        pages.len()
    );
    // bf16 activations against the oracle's f32: a few hundredths of one
    // layer's mass a key at most; a row of another head, layer or lane is
    // off by tenths.
    assert!(
        keys_apart < 2e-2,
        "the key row is not the oracle's: {keys_apart} apart"
    );
    assert!(
        pages_apart < 4e-2,
        "the page row is not the oracle's: {pages_apart} apart"
    );
}
