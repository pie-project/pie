//! flux2-mini's denoise (models/flux_2.poem at dim 256: 64 text rows and a 16 x 16 latent, sigma
//! 0.5, guidance 4, the weights engine-cpu's flux test and `mk_tiny_flux2.py` draw by name) as
//! the wafer program `wafer-compile stream` makes of it, with no fabric: the arrays a step's
//! inputs make, computed by the engine's host forward from the step's own inputs
//! ([`Reading::entries`]), are the ones the compile writes from engine-cpu's capture of the whole
//! step, bit for bit; the exit's host ops over the rows a program returns give the step's
//! velocity, bit for bit; and the host's denoise loop around such a step lands where diffusers'
//! loop does.

use std::path::Path;

use engine_cerebras::reading::{Denoise, Reading};
use engine_cpu::Forward;
use engine_cpu::forward::{Captured, Output, StreamStep, Values};
use poem_ir::{Operands, Request, Trace};

const MINI: &str = "flux2-mini-bf16-kv-bf16";
const TEXT: usize = 64;
const GRID: [u32; 3] = [1, 16, 16];
const CHANNELS: usize = 32;
const CONTEXT: usize = 192;
/// diffusers' latent after the 2-step loop (`mk_tiny_flux2.py --loop`), f32 [256, 32].
const LOOP_GOLDEN: &[u8] = include_bytes!("flux_2_mini_loop2_latent.f32");

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// `count` values in [-1/2, 1/2), each exact in f32: what mk_tiny_flux2.py draws for `name`.
fn units(name: &str, count: usize) -> Vec<f32> {
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 40) as i64 - (1 << 23)) as f32 / (1u64 << 24) as f32
        })
        .collect()
}

fn bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

fn trace() -> Trace {
    poem_compiler::catalog::repository()
        .parse(MINI)
        .expect("the catalog holds flux2-mini")
        .trace(poem_ir::Platform::Cerebras)
}

/// flux2-mini with mk_tiny_flux2.py's weights, and its step's guest inputs by name.
fn setup(trace: &Trace) -> (Forward, Vec<(String, Vec<f32>)>) {
    let shapes: Vec<(String, Vec<u64>)> = (trace.params.iter())
        .map(|p| (p.name.clone(), p.shape.clone()))
        .collect();
    let plane = |name: &str, count: usize| -> Vec<f32> {
        let u = units(name, count);
        if name.ends_with("norm") {
            return u.iter().map(|u| 1.0 + 0.5 * u).collect();
        }
        let fan_in = (shapes.iter())
            .find(|(n, _)| n == name)
            .and_then(|(_, s)| s.last().copied())
            .unwrap_or(1) as f32;
        let s = (12.0 / fan_in).sqrt();
        u.iter().map(|u| u * s).collect()
    };
    let voxels = (GRID[1] * GRID[2]) as usize;
    let s = 12f32.sqrt();
    let mut context: Vec<f32> = units("input.context", TEXT * CONTEXT)
        .iter()
        .map(|u| bf16(u * s))
        .collect();
    context.resize((TEXT + voxels / 4) * CONTEXT, 0.0);
    let latents: Vec<f32> = units("input.latents", voxels * CHANNELS)
        .iter()
        .map(|u| bf16(u * s))
        .collect();
    let inputs = vec![
        ("timestep".to_string(), vec![500.0; 2]),
        ("guidance".to_string(), vec![4.0; 2]),
        ("context".to_string(), context),
        ("latents".to_string(), latents),
    ];
    let forward = Forward::new(trace.clone(), None, &plane).expect("flux2-mini loads");
    (forward, inputs)
}

/// The step the engine-cpu test runs: its lanes are the reading's classes.
fn engine_cpu_step<'a>(trace: &Trace, inputs: &'a [(String, Vec<f32>)]) -> StreamStep<'a> {
    let word = |stream: &str| {
        let r = Request::new(TEXT as u32, false)
            .switched("reading", "denoise")
            .switched("stream", stream);
        trace.facts.word(&r)
    };
    StreamStep {
        lanes: vec![
            engine_cpu::forward::StreamLane {
                word: word("context"),
                rows: TEXT,
                clip: None,
            },
            engine_cpu::forward::StreamLane {
                word: word("latents"),
                rows: (GRID[1] * GRID[2] / 4) as usize,
                clip: Some(GRID),
            },
        ],
        inputs: inputs
            .iter()
            .map(|(n, v)| (n.as_str(), v.clone()))
            .collect(),
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

fn npy(rows: usize, width: usize, v: &[f32]) -> Vec<u8> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {width}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend(v.iter().flat_map(|x| x.to_le_bytes()));
    out
}

/// Every node's values of the step, as `wafer-compile stream --values` reads a capture
/// (`nodes.json`, `n<node>_{in,out}<k>.npy`; ints as floats).
fn capture(forward: &mut Forward, trace: &Trace, step: &StreamStep<'_>, dir: &Path) {
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
            std::fs::write(dir.join(&name), npy(c.rows, c.width, &v)).expect("a capture file");
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

/// The program's directory, compiled with the capture's values (as M1d's runs were).
fn compile(trace: &Trace, gold: &Path, out: &Path) {
    let o = kernels_cerebras::stream::Options {
        reading: "denoise".into(),
        rows: TEXT as u32,
        round16: true,
        upto: None,
        taps: Vec::new(),
    };
    kernels_cerebras::stream::compile(trace, &o, out, Some(gold)).expect("the program compiles");
}

fn words(path: &Path) -> Vec<u32> {
    let (_, v) = kernels_cerebras::npy::read_f32_or_i32(path).expect("an h2d file");
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
    capture(&mut forward, &trace, &step, &gold);
    compile(&trace, &gold, &prog);

    let reading = Reading::open(&prog).expect("host.json states the reading");
    assert_eq!(reading.model, MINI);
    let mine = reading.step(&trace, Some(GRID), &inputs);
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
        let want = words(&a.file);
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
/// split (host entries, the whole step's rows, host exit), against diffusers' loop over the
/// same sigmas.
#[test]
fn flux_2_minis_denoise_loop_answers_diffusers() {
    let trace = trace();
    let (mut forward, mut inputs) = setup(&trace);
    let tmp = tempfile::tempdir().expect("a scratch directory");
    let (gold, prog) = (tmp.path().join("gold"), tmp.path().join("prog"));
    {
        let step = engine_cpu_step(&trace, &inputs);
        capture(&mut forward, &trace, &step, &gold);
    }
    compile(&trace, &gold, &prog);
    let reading = Reading::open(&prog).expect("host.json states the reading");
    let mut host = Forward::host(trace.clone());
    let image_rows = GRID[1] * GRID[2] / 4;
    let d = Denoise::of(MINI, 2, Some(image_rows)).expect("the poem states a flow schedule");
    eprintln!("sigmas {:?}", d.sigmas);
    let (latent, _) = d
        .run(&mut inputs, reading.classes.len(), |_, ins| {
            let step = reading.step(&trace, Some(GRID), ins);
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

/// What a fabric run of the reading takes and is held to, under `READING_OUT`: the program
/// compiled from a capture (`prog/`), the step's inputs (`inputs/<name>.f32`, as
/// `wafer-bench --reading` reads them), and engine-cpu's answers through the same host loop
/// (`cpu_velocity<i>.f32`, `cpu_latent<n>.f32` after `n` steps, for `READING_STEPS` steps,
/// default 2).
#[test]
#[ignore = "writes a reading's run files; run with READING_OUT"]
fn flux_2_minis_reading_files() {
    let Ok(out) = std::env::var("READING_OUT") else {
        panic!("set READING_OUT");
    };
    let steps: u32 = std::env::var("READING_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let out = Path::new(&out);
    let trace = trace();
    let (mut forward, mut inputs) = setup(&trace);
    let (gold, prog) = (out.join("gold"), out.join("prog"));
    {
        let step = engine_cpu_step(&trace, &inputs);
        capture(&mut forward, &trace, &step, &gold);
    }
    compile(&trace, &gold, &prog);
    std::fs::create_dir_all(out.join("inputs")).expect("inputs");
    for (n, v) in &inputs {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join("inputs").join(format!("{n}.f32")), b).expect("an input");
    }
    let reading = Reading::open(&prog).expect("host.json states the reading");
    let mut host = Forward::host(trace.clone());
    let d = Denoise::of(MINI, steps, Some(GRID[1] * GRID[2] / 4)).expect("a flow schedule");
    let write = |name: String, v: &[f32]| {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(out.join(name), b).expect("a golden");
    };
    let (_, vs) = d
        .run(&mut inputs, reading.classes.len(), |_, ins| {
            let step = reading.step(&trace, Some(GRID), ins);
            let back = device(&mut forward, &reading, &step);
            let v = reading
                .finish(&mut host, &trace, &step, &back, &["velocity"])?
                .remove(0);
            Ok(v)
        })
        .expect("the loop runs");
    let mut x = setup(&trace).1.remove(3).1;
    for (i, v) in vs.iter().enumerate() {
        write(format!("cpu_velocity{i}.f32"), v);
        let dt = d.sigmas[i + 1] - d.sigmas[i];
        for (x, v) in x.iter_mut().zip(v) {
            *x += dt * v;
        }
        write(format!("cpu_latent{}.f32", i + 1), &x);
    }
    eprintln!("sigmas {:?}", d.sigmas);
}
