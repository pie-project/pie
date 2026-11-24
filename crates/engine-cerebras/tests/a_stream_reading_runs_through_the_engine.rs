//! flux2-mini's denoise through the cerebras engine as Pie drives any engine: the model loaded
//! from a diffusers checkpoint, the program compiled with no capture, and the denoise loop a
//! guest program's. The engine's static arrays match the capture compile's word for word; on the
//! host fabric the guest's loop lands bit for bit on the host's own and within the bf16 tolerance
//! of diffusers'. `flux_2_minis_denoise_on_the_fabric` (ignored) runs it on a fabric
//! (`READING_PROGRAM`, `READING_LINK`, `READING_OUT`, `READING_GOLDEN`).

mod common;

use std::path::{Path, PathBuf};

use common::guest::Ran;
use common::{MINI, grid, rows, trace};
use engine::Engine;
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine_cerebras::reading::{Denoise, Reading};
use engine_cpu::Forward;
use poem_ir::{Request, Trace};

/// diffusers' latent after the 2-step loop at 64 rows a lane.
const LOOP_GOLDEN: &[u8] = include_bytes!("flux_2_mini_loop2_latent.f32");

fn f32s(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn rel_cos(a: &[f32], b: &[f32]) -> (f64, f64) {
    let dot = |x: &[f32], y: &[f32]| {
        x.iter()
            .zip(y)
            .map(|(p, q)| f64::from(*p) * f64::from(*q))
            .sum::<f64>()
    };
    let d: Vec<f32> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    (
        (dot(&d, &d) / dot(b, b)).sqrt(),
        dot(a, b) / (dot(a, a) * dot(b, b)).sqrt(),
    )
}

/// The program compiled with no capture into `dir`.
fn compile(trace: &Trace, dir: &Path, gold: Option<&Path>) {
    let o = kernels_cerebras::compile::stream::Options {
        reading: "denoise".into(),
        rows: rows() as u32,
        round16: true,
        upto: None,
        taps: Vec::new(),
        anneal: Default::default(),
    };
    kernels_cerebras::compile::stream::compile(trace, &o, dir, gold).expect("the program compiles");
}

/// The step engine-cpu's test runs, its lanes the reading's classes.
fn step_of<'a>(
    trace: &Trace,
    inputs: &'a [(String, Vec<f32>)],
) -> engine_cpu::forward::StreamStep<'a> {
    let word = |stream: &str| {
        let r = Request::new(rows() as u32, false)
            .switched("reading", "denoise")
            .switched("stream", stream);
        trace.facts.word(&r)
    };
    engine_cpu::forward::StreamStep {
        lanes: vec![
            engine_cpu::forward::StreamLane {
                word: word("context"),
                rows: rows(),
                clip: None,
            },
            engine_cpu::forward::StreamLane {
                word: word("latents"),
                rows: (grid()[1] * grid()[2] / 4) as usize,
                clip: Some(grid()),
            },
        ],
        inputs: inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect(),
        tokens: Vec::new(),
    }
}

/// The static arrays the engine cuts from the checkpoint are the capture compile's, word for
/// word; the no-capture compile states the same arrays, with no words of a value on file.
#[test]
fn flux_2_minis_statics_are_the_checkpoints() {
    let trace = trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (gold, with, bare, ckpt) = (
        tmp.path().join("gold"),
        tmp.path().join("with"),
        tmp.path().join("bare"),
        tmp.path().join("ckpt"),
    );
    // the capture compile: every array's words from engine-cpu's step
    let mut forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let inputs = common::inputs();
    let step = step_of(&trace, &inputs);
    capture(&mut forward, &trace, &step, &gold);
    compile(&trace, &with, Some(&gold));
    compile(&trace, &bare, None);
    let (with, bare) = (
        Reading::open(&with).expect("the capture compile's host.json"),
        Reading::open(&bare).expect("the bare compile's host.json"),
    );
    assert_eq!(with.h2d.len(), bare.h2d.len());

    // the checkpoint, through the import contract, as the engine lands it
    common::checkpoint(&trace, &ckpt);
    let contract = common::contract_for(&trace, &ckpt).expect("the contract reads the checkpoint");
    let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt).expect("it lands");
    let mut weights = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        let p =
            <engine_cerebras::prepare::Landed as kernels_cerebras::compile::tiles::Params>::get(
                &landed, n,
            )
            .unwrap_or_else(|| panic!("{n} landed"));
        p.bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(c)
            .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
            .collect()
    })
    .expect("flux2-mini loads from its checkpoint");
    let statics = bare
        .statics(&mut weights, &step)
        .expect("the statics compute");
    let (mut spans, mut files) = (0, 0);
    for ((a, b), got) in with.h2d.iter().zip(&bare.h2d).zip(&statics) {
        assert_eq!((&a.sym, a.rect.x, a.rect.y), (&b.sym, b.rect.x, b.rect.y));
        let Some(got) = got else {
            assert!(b.entry.is_some(), "`{}`: no words and no entry", b.sym);
            continue;
        };
        let want =
            kernels_cerebras::compile::npy::read_f32_or_i32(a.file.as_deref().expect("on file"))
                .expect("an h2d file")
                .1;
        let want: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(got, &want, "`{}` at ({}, {})", b.sym, b.rect.x, b.rect.y);
        if b.statics.is_some() {
            assert!(b.file.is_none(), "`{}`: a value's words on file", b.sym);
            spans += 1;
        } else {
            files += 1;
        }
    }
    eprintln!("{spans} arrays from the checkpoint, {files} the program's own");
    assert!(spans > 0 && files > 0);
}

/// Every node's values of the step, as the stream compile reads a capture.
fn capture(
    forward: &mut Forward,
    trace: &Trace,
    step: &engine_cpu::forward::StreamStep<'_>,
    dir: &Path,
) {
    use engine_cpu::forward::Values;
    use poem_ir::Operands;
    let all: Vec<usize> = (0..trace.nodes.len()).collect();
    let (_, caps) = forward.stream_capture(step, &all).expect("the step runs");
    std::fs::create_dir_all(dir).expect("a capture directory");
    let mut index = Vec::new();
    for &node in &all {
        let (mut ins, mut outs) = (0, 0);
        let mut files = Vec::new();
        for c in caps.iter().filter(|c| c.node == node) {
            let k = if c.input { &mut ins } else { &mut outs };
            let name = format!("n{node}_{}{k}.npy", if c.input { "in" } else { "out" });
            *k += 1;
            let v: Vec<f32> = match &c.values {
                Values::F32(v) => v.clone(),
                Values::I32(v) => v.iter().map(|&x| x as f32).collect(),
            };
            kernels_cerebras::compile::npy::write_f32(&dir.join(&name), &[c.rows, c.width], &v)
                .expect("a capture file");
            files.push(format!("\"{name}\": {}", c.value.0));
        }
        if !files.is_empty() {
            index.push(format!(
                "\"{node}\": {{\"op\": \"{}\", \"values\": {{{}}}}}",
                trace.nodes[node].op.name(),
                files.join(", ")
            ));
        }
    }
    std::fs::write(
        dir.join("nodes.json"),
        format!("{{{}}}\n", index.join(",\n")),
    )
    .expect("the capture's index");
}

/// The engine loaded from `ckpt` with `program` (and its link), the guest's loop of `steps`
/// steps submitted a step a frame, as the runtime submits a pass.
fn run_loop(
    program: &Path,
    link: Option<PathBuf>,
    ckpt: &Path,
    fabric: Option<&str>,
    steps: u32,
) -> Ran {
    let trace = trace();
    let cache = tempfile::tempdir().expect("a cache directory");
    let mut cfg = engine_cerebras::Config {
        cache_dir: Some(cache.path().to_path_buf()),
        num_threads: Some(8),
        ..engine_cerebras::Config::default()
    };
    cfg.debug = Some(engine_cerebras::config::Debug {
        program: Some(program.to_path_buf()),
        link,
        stream_fabric: fabric.map(str::to_string),
        ..engine_cerebras::config::Debug::default()
    });
    let mut engine = engine_cerebras::CerebrasEngine::new(cfg, common::contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace: trace.clone(),
            checkpoint: Checkpoint::Path(ckpt.to_path_buf()),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 2 * rows() as u32,
                buckets: Vec::new(),
                max_adapters: 0,
                page_size: 16,
                max_context: 4 * rows() as u32,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: Some(grid()[1] * grid()[2]),
                max_clips: Some(1),
            },
            residency: Residency::default(),
            ordinal: 0,
            frames_in_flight: 1,
            steps: Default::default(),
        })
        .expect("flux2-mini loads on the cerebras engine");
    common::guest::loop_on(&mut engine, &loaded.caps, steps, &common::context())
}

/// The host's own loop over the same schedule, each velocity engine-cpu's whole step.
fn host_loop(steps: u32) -> Vec<Vec<f32>> {
    let trace = trace();
    let mut forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let d = Denoise::of(MINI, steps, Some(grid()[1] * grid()[2] / 4)).expect("a flow schedule");
    let mut inputs = common::inputs();
    let mut out = Vec::new();
    let (_, vs) = d
        .run(&mut inputs, 2, |_, ins| {
            let step = step_of(&trace, ins);
            let outs = forward.stream(&step)?;
            Ok(match outs.into_iter().find(|o| o.name == "velocity") {
                Some(engine_cpu::forward::Output {
                    values: engine_cpu::forward::Values::F32(v),
                    ..
                }) => v,
                _ => return Err("no velocity".into()),
            })
        })
        .expect("the host's loop runs");
    let mut x = common::latents();
    for (i, v) in vs.iter().enumerate() {
        let dt = d.sigmas[i + 1] - d.sigmas[i];
        for (x, v) in x.iter_mut().zip(v) {
            *x += dt * v;
        }
        out.push(x.clone());
    }
    out
}

/// The guest's loop through the engine on the host fabric: the host's loop bit for bit, and
/// diffusers' within the bf16 tolerance.
#[test]
fn flux_2_minis_denoise_runs_through_the_engine() {
    let trace = trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (prog, ckpt) = (tmp.path().join("prog"), tmp.path().join("ckpt"));
    compile(&trace, &prog, None);
    common::checkpoint(&trace, &ckpt);
    let ran = run_loop(&prog, None, &ckpt, Some("host"), 2);
    let want = host_loop(2);
    for (i, (got, want)) in ran.latents.iter().zip(&want).enumerate() {
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            "the latent after step {i}"
        );
    }
    if rows() == 64 {
        let (rel, cos) = rel_cos(ran.latents.last().expect("a latent"), &f32s(LOOP_GOLDEN));
        eprintln!(
            "flux2-mini through the engine, 2 steps, vs diffusers: rel L2 {rel:.3e}, cosine {cos:.8}"
        );
        assert!(rel <= 1e-2 && cos >= 0.999, "rel L2 {rel}, cosine {cos}");
    }
}

/// A batch's entries: the engine's (`Reading::entries_batch`) are word for word the ones the
/// compile writes from two requests' captures laid out as a batch.
#[test]
fn a_batch_of_twos_entries_are_its_requests_in_turn() {
    let trace = trace();
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let mut forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(&trace, n, c)
    })
    .expect("flux2-mini loads");
    let a = common::inputs();
    let b: Vec<(String, Vec<f32>)> = a
        .iter()
        .map(|(n, v)| match n.as_str() {
            "timestep" => (n.clone(), vec![700.0; v.len()]),
            "latents" => (n.clone(), v.iter().rev().copied().collect()),
            _ => (n.clone(), v.clone()),
        })
        .collect();
    let (ca, cb, gold) = (
        tmp.path().join("a"),
        tmp.path().join("b"),
        tmp.path().join("gold"),
    );
    capture(&mut forward, &trace, &step_of(&trace, &a), &ca);
    capture(&mut forward, &trace, &step_of(&trace, &b), &cb);
    std::fs::create_dir_all(&gold).expect("a capture directory");
    for e in std::fs::read_dir(&ca).expect("a's capture") {
        let name = e.expect("a file").file_name();
        if !name.to_string_lossy().ends_with(".npy") {
            std::fs::copy(ca.join(&name), gold.join(&name)).expect("the index");
            continue;
        }
        let (sa, mut va) =
            kernels_cerebras::compile::npy::read_f32_or_i32(&ca.join(&name)).expect("a");
        let (_, vb) = kernels_cerebras::compile::npy::read_f32_or_i32(&cb.join(&name)).expect("b");
        va.extend(vb);
        let mut shape = sa.clone();
        shape[0] *= 2;
        kernels_cerebras::compile::npy::write_f32(&gold.join(&name), &shape, &va)
            .expect("a batch value");
    }
    let o = kernels_cerebras::compile::stream::Options {
        reading: "denoise".into(),
        rows: rows() as u32,
        round16: true,
        upto: None,
        taps: Vec::new(),
        anneal: Default::default(),
    };
    let p = kernels_cerebras::compile::stream::gemm::program(&trace, &o, &[], None, (762, 1172), 2)
        .expect("a batch of two places");
    let dir = tmp.path().join("prog");
    kernels_cerebras::compile::stream::gemm::compile_program(&p, &dir, Some(&gold))
        .expect("it compiles");
    let reading = Reading::open(&dir).expect("its host.json");
    assert_eq!(reading.batch, 2);
    let mut host = Forward::host(trace.clone());
    let steps = [step_of(&trace, &a), step_of(&trace, &b)];
    let entries = reading
        .entries_batch(&mut host, &steps)
        .expect("the entries");
    let mut seen = 0;
    for (arr, got) in reading.h2d.iter().zip(&entries) {
        let Some(got) = got else { continue };
        let want =
            kernels_cerebras::compile::npy::read_f32_or_i32(arr.file.as_deref().expect("on file"))
                .expect("the compile's entry")
                .1;
        assert_eq!(got.len(), want.len(), "`{}`", arr.sym);
        for (x, y) in got.iter().zip(&want) {
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "`{}`",
                arr.sym
            );
        }
        seen += 1;
    }
    assert!(seen > 0);
}
