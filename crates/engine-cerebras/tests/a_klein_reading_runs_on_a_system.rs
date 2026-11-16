//! FLUX.2 [klein] 4B's denoise reading on a system (GENERIC.md M2e): what `wafer-bench
//! --reading` takes, from the real checkpoint, for a program `wafer-compile stream` built.
//!
//! `klein_reading_files` (ignored: it reads `READING_CKPT`, the diffusers snapshot of
//! `black-forest-labs/FLUX.2-klein-4B`) lands the checkpoint through the import contract, as the
//! engine does, and writes under `READING_OUT`: the program's static arrays made from it
//! (`statics/<k>.u32`, the `k`-th array's words: `wafer-bench --statics`), a step's guest inputs
//! (`inputs/<name>.f32`, drawn) and, unless `READING_GOLDEN=0`, engine-cpu's loop over the same
//! inputs at the program's rows (`cpu_velocity<i>.f32`, `cpu_latent<n>.f32`, `READING_STEPS`
//! steps, default 2), the host's ops the engine's; with `READING_ROWS=1`, the values the program
//! returns at that step (`rows/<value>.f32`, as `wafer-bench --rows 1 --save-rows` saves them).

mod common;

use std::path::PathBuf;

use engine_cerebras::reading::{Denoise, Reading};
use engine_cpu::Forward;
use poem_ir::{Def, InputAxis, RuntimeInput};

const KLEIN: &str = "flux2-klein-4b-bf16-kv-bf16";

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore = "reads the klein snapshot at READING_CKPT; set READING_PROGRAM, READING_OUT"]
fn klein_reading_files() {
    let var = |k: &str| PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("set {k}")));
    let (ckpt, prog, out) = (
        var("READING_CKPT"),
        var("READING_PROGRAM"),
        var("READING_OUT"),
    );
    let steps: u32 = std::env::var("READING_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let golden = std::env::var("READING_GOLDEN").map_or(true, |g| g != "0");
    let trace = poem_compiler::catalog::repository()
        .parse(KLEIN)
        .expect("the catalog holds klein")
        .trace(poem_ir::Platform::Cerebras);
    let reading = Reading::open(&prog).expect("the program's host.json");
    let t0 = std::time::Instant::now();
    let contract = common::contract_for(&trace, &ckpt).expect("the contract reads the snapshot");
    let landed = engine_cerebras::prepare::land(&trace, &contract, &ckpt).expect("it lands");
    let of = |n: &str, c: usize| -> Vec<f32> {
        let p =
            <engine_cerebras::prepare::Landed as kernels_cerebras::tiles::Params>::get(&landed, n)
                .unwrap_or_else(|| panic!("{n} landed"));
        if p.bytes.len() == 4 * c {
            p.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        } else {
            p.bytes
                .as_chunks::<2>()
                .0
                .iter()
                .take(c)
                .map(|h| f32::from_bits(u32::from(u16::from_le_bytes(*h)) << 16))
                .collect()
        }
    };
    let mut weights = Forward::new(trace.clone(), None, &of).expect("klein loads");
    drop(landed);
    eprintln!("landed ({:.0} s)", t0.elapsed().as_secs_f64());

    // the static arrays: the weights' alone
    let t0 = std::time::Instant::now();
    let statics = reading
        .statics(&mut weights, &reading.step(&trace, None, &[]))
        .expect("the static arrays");
    let dir = out.join("statics");
    std::fs::create_dir_all(&dir).expect("statics/");
    let mut words = 0usize;
    for (k, w) in statics.iter().enumerate() {
        if let Some(w) = w {
            let b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{k}.u32")), b).expect("an array");
            words += w.len();
        }
    }
    drop(statics);
    eprintln!(
        "static arrays: {words} words ({:.0} s)",
        t0.elapsed().as_secs_f64()
    );

    // a step's guest inputs, drawn: a lane value a lane, a row a lane's row, a voxel a latent's
    let rows = |name: &str| {
        reading
            .classes
            .iter()
            .find(|c| c.name == name)
            .map_or(0, |c| c.rows as usize)
    };
    let image_rows = rows("latents");
    let side = 2 * (image_rows as f64).sqrt().round() as u32;
    let grid = [1, side, side];
    let total: usize = reading.classes.iter().map(|c| c.rows as usize).sum();
    let mut inputs: Vec<(String, Vec<f32>)> = Vec::new();
    for (name, axis, width) in trace.values.iter().filter_map(|d| match &d.def {
        Def::Input(RuntimeInput::Guest {
            id, axis, width, ..
        }) => Some((
            trace.inputs[*id as usize].name.clone(),
            *axis,
            *width as usize,
        )),
        _ => None,
    }) {
        let s = 12f32.sqrt();
        let draw = |n: usize| -> Vec<f32> {
            common::units(&format!("input.{name}"), n)
                .iter()
                .map(|u| common::bf16(u * s))
                .collect()
        };
        let v = match axis {
            InputAxis::Lane if name.contains("timestep") => vec![500.0; reading.classes.len()],
            InputAxis::Lane if name.contains("guidance") => vec![4.0; reading.classes.len()],
            InputAxis::Lane => vec![0.0; reading.classes.len() * width],
            InputAxis::Row => draw(total * width),
            InputAxis::Voxel => draw((side * side) as usize * width),
        };
        eprintln!("input {name}: {axis:?} x {width}, {} values", v.len());
        inputs.push((name, v));
    }
    let idir = out.join("inputs");
    std::fs::create_dir_all(&idir).expect("inputs/");
    for (n, v) in &inputs {
        std::fs::write(idir.join(format!("{n}.f32")), bytes(v)).expect("an input");
    }
    // the entries and the exit's ops as wafer-bench makes them: a host forward of no weights
    let mut host = Forward::host(trace.clone());
    reading
        .entries(&mut host, &reading.step(&trace, Some(grid), &inputs))
        .expect("the entries of no weights");
    std::fs::write(
        out.join("grid.txt"),
        format!("{},{},{}\n", grid[0], grid[1], grid[2]),
    )
    .expect("grid.txt");
    if std::env::var("READING_ROWS").is_ok_and(|r| r == "1") {
        // the values the program returns, engine-cpu's, at the inputs' own step
        // (`rows/<value>.f32`, rows x width: `wafer-bench --rows 1 --save-rows`'s files)
        let t0 = std::time::Instant::now();
        let step = reading.step(&trace, Some(grid), &inputs);
        let rows = out.join("rows");
        std::fs::create_dir_all(&rows).expect("rows/");
        for c in weights
            .stream_values(&step, &[], &reading.out_values())
            .expect("the step's values")
        {
            if let engine_cpu::forward::Values::F32(v) = &c.values {
                std::fs::write(rows.join(format!("{}.f32", c.value.0)), bytes(v)).expect("a row");
            }
        }
        eprintln!("rows ({:.0} s)", t0.elapsed().as_secs_f64());
    }
    if !golden {
        return;
    }
    let d = Denoise::of(KLEIN, steps, Some(image_rows as u32)).expect("a flow schedule");
    let x0 = inputs
        .iter()
        .find(|(n, _)| *n == d.latent)
        .map(|(_, v)| v.clone())
        .expect("a latent input");
    let lanes = reading.classes.len();
    let (_, vs) = d
        .run(&mut inputs, lanes, |i, ins| {
            let t0 = std::time::Instant::now();
            let step = reading.step(&trace, Some(grid), ins);
            let back = weights
                .stream_values(&step, &[], &reading.out_values())
                .map_err(|e| e.to_string())?;
            let v = reading
                .finish(&mut host, &trace, &step, &back, &[d.velocity.as_str()])?
                .remove(0);
            eprintln!("engine-cpu step {i} ({:.0} s)", t0.elapsed().as_secs_f64());
            Ok(v)
        })
        .expect("the loop runs");
    let mut x = x0;
    for (i, v) in vs.iter().enumerate() {
        std::fs::write(out.join(format!("cpu_velocity{i}.f32")), bytes(v)).expect("a golden");
        let dt = d.sigmas[i + 1] - d.sigmas[i];
        for (x, v) in x.iter_mut().zip(v) {
            *x += dt * v;
        }
        std::fs::write(out.join(format!("cpu_latent{}.f32", i + 1)), bytes(&x)).expect("a golden");
    }
    eprintln!("sigmas {:?}", d.sigmas);
}
