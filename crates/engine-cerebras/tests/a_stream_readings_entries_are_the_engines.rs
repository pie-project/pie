//! flux2-mini's denoise as the wafer program `wafer-compile stream` makes of it, with no fabric:
//! the entries the engine's host forward computes ([`Reading::entries`]) and the exit's ops over
//! the rows read back match engine-cpu's capture bit for bit, and the host's denoise loop around
//! such a step lands where diffusers' does.

use std::path::Path;

use engine_cerebras::reading::{Denoise, Reading};
use engine_cpu::Forward;
use engine_cpu::forward::{Captured, Output, StreamStep, Values};
use poem_ir::{Request, Trace};

mod common;

use common::{MINI, grid, rows, trace};

/// diffusers' latent after the 2-step loop, f32 [256, 32].
const LOOP_GOLDEN: &[u8] = include_bytes!("flux_2_mini_loop2_latent.f32");

/// flux2-mini with the golden's weights, and its step's guest inputs by name.
fn setup(trace: &Trace) -> (Forward, Vec<(String, Vec<f32>)>) {
    let forward = Forward::new(trace.clone(), None, &|n: &str, c: usize| {
        common::plane(trace, n, c)
    })
    .expect("flux2-mini loads");
    (forward, common::inputs())
}

/// The step the engine-cpu test runs: its lanes are the reading's classes.
fn engine_cpu_step<'a>(trace: &Trace, inputs: &'a [(String, Vec<f32>)]) -> StreamStep<'a> {
    let word = |stream: &str| {
        let r = Request::new(rows() as u32, false)
            .switched("reading", "denoise")
            .switched("stream", stream);
        trace.facts.word(&r)
    };
    StreamStep {
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

fn velocity(outs: &[Output]) -> Vec<f32> {
    match outs.iter().find(|o| o.name == "velocity") {
        Some(Output {
            values: Values::F32(v),
            ..
        }) => v.clone(),
        _ => panic!("the step reads out no velocity"),
    }
}

/// The program's directory, compiled with the capture's values.
fn compile(trace: &Trace, gold: &Path, out: &Path) {
    let o = kernels_cerebras::compile::stream::Options {
        reading: "denoise".into(),
        rows: rows() as u32,
        round16: true,
        upto: None,
        taps: Vec::new(),
        anneal: Default::default(),
    };
    kernels_cerebras::compile::stream::compile(trace, &o, out, Some(gold))
        .expect("the program compiles");
}

fn words(path: &Path) -> Vec<u32> {
    let (_, v) = kernels_cerebras::compile::npy::read_f32_or_i32(path).expect("an h2d file");
    v.iter().map(|x| x.to_bits()).collect()
}

/// A device's step on the host forward: the values the program returns (the reading's `outs`),
/// as the whole step makes them.
fn device(forward: &mut Forward, reading: &Reading, step: &StreamStep<'_>) -> Vec<Captured> {
    forward
        .stream_values(step, &[], &reading.out_values())
        .expect("the step runs")
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

#[test]
fn flux_2_minis_entries_are_the_captures() {
    let trace = trace();
    let (mut forward, inputs) = setup(&trace);
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (gold, prog) = (tmp.path().join("gold"), tmp.path().join("prog"));
    let step = engine_cpu_step(&trace, &inputs);
    common::capture(&mut forward, &trace, &step, &gold);
    compile(&trace, &gold, &prog);

    let reading = Reading::open(&prog).expect("host.json states the reading");
    assert_eq!(reading.model, MINI);
    let mine = reading.step(&trace, Some(grid()), &inputs);
    assert_eq!(
        (mine.lanes.iter())
            .map(|l| (l.word, l.rows, l.clip))
            .collect::<Vec<_>>(),
        (step.lanes.iter())
            .map(|l| (l.word, l.rows, l.clip))
            .collect::<Vec<_>>(),
        "a lane a class, the latent's lane on the grid"
    );
    let mut host = Forward::host(trace.clone());
    let entries = reading
        .entries(&mut host, &mine)
        .expect("the entries compute");
    let mut n = 0;
    for (a, e) in reading.h2d.iter().zip(&entries) {
        let Some(e) = e else { continue };
        let want = words(
            a.file
                .as_deref()
                .expect("a capture's compile writes every array"),
        );
        let got: Vec<u32> = e.iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            got, want,
            "entry `{}` at ({}, {})",
            a.sym, a.rect.x, a.rect.y
        );
        n += 1;
    }
    // the lane prologue's input (the sinusoids), the entries' host rows, every unit's scalars
    assert!(n >= 3, "{n} entries");

    // the exit: the projection's rows, as a program returns them, unpatchified on the host
    let back = device(&mut forward, &reading, &mine);
    let v = reading
        .finish(&mut host, &trace, &mine, &back, &["velocity"])
        .expect("the exit runs")
        .remove(0);
    let want = velocity(&forward.stream(&step).expect("the step runs"));
    assert_eq!(
        v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

/// The host's loop over two steps of the poem's schedule, each step's velocity the engine's
/// split, against diffusers' loop over the same sigmas.
#[test]
fn flux_2_minis_denoise_loop_answers_diffusers() {
    let trace = trace();
    let (mut forward, mut inputs) = setup(&trace);
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (gold, prog) = (tmp.path().join("gold"), tmp.path().join("prog"));
    {
        let step = engine_cpu_step(&trace, &inputs);
        common::capture(&mut forward, &trace, &step, &gold);
    }
    compile(&trace, &gold, &prog);
    let reading = Reading::open(&prog).expect("host.json states the reading");
    let mut host = Forward::host(trace.clone());
    let image_rows = grid()[1] * grid()[2] / 4;
    let d = Denoise::of(MINI, 2, Some(image_rows)).expect("the poem states a flow schedule");
    eprintln!("sigmas {:?}", d.sigmas);
    let (latent, _) = d
        .run(&mut inputs, reading.classes.len(), |_, ins| {
            let step = reading.step(&trace, Some(grid()), ins);
            let back = device(&mut forward, &reading, &step);
            let split = reading
                .finish(&mut host, &trace, &step, &back, &["velocity"])?
                .remove(0);
            let whole = velocity(&forward.stream(&step)?);
            assert_eq!(
                split.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                whole.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
            );
            Ok(split)
        })
        .expect("the loop runs");
    let want: Vec<f32> = LOOP_GOLDEN
        .chunks(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let (rel, cos) = rel_cos(&latent, &want);
    eprintln!("flux2-mini 2-step latent vs diffusers: rel L2 {rel:.3e}, cosine {cos:.8}");
    assert!(rel <= 1e-2, "relative L2 {rel}");
    assert!(cos >= 0.999, "cosine {cos}");
}
