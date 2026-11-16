//! `poems`: GENERIC.md G1's survey. Each named deployment through the whole pipeline the engine
//! runs on load, reading by reading: the trace; the decode program (`target::plan`,
//! `Program::of`, `build::emit`) where the trace samples tokens; and every value of its `reading`
//! switch through the stream compiler (`stream::compile_reading`: the stream regime, or a grid
//! reading's tiled / rows regime). Each build is written to a scratch directory and removed.
//! Prints one line a (deployment, reading): `PASS` with what was built, or the stage it stops at
//! and why.
//!
//! ```text
//! cargo run --release -p kernels-cerebras --example poems -- <deployment>... [--out <scratch dir>]
//! ```
//!
//! `ROWS=N`: a stream reading's rows a class (16); `GRID_ROWS=N`: the token rows a grid reading's
//! clip is made from (256); `DECODE=1`: the decode program also for a trace with readings that
//! samples no token.

use std::path::PathBuf;
use std::time::Instant;

use poem_ir::{Operands, Platform};

use kernels_cerebras::build::{self, EMITS};
use kernels_cerebras::emit::params::Levers;
use kernels_cerebras::program::Program;
use kernels_cerebras::target::{self, Ask};

fn short(s: &str) -> String {
    let s = s.replace(['\n', '\t'], " ");
    if s.len() > 400 {
        format!("{}...", &s[..400])
    } else {
        s
    }
}

fn line(name: &str, reading: &str, status: &str, what: &str, t0: Instant) {
    println!(
        "{name}\t{reading}\t{status}\t{}\t{:.1}s",
        short(what),
        t0.elapsed().as_secs_f64()
    );
}

/// The decode program: planned, built and emitted.
fn decode(trace: &poem_ir::Trace, out: &std::path::Path) -> Result<String, (String, String)> {
    let planned = target::plan(trace, &Ask::default()).map_err(|e| ("PLAN".to_string(), e))?;
    let kernels = kernels_cerebras::kernels::KernelSet::of_shape(&planned.plan.shape).name();
    let preset = planned.preset.name.clone();
    let levers = Levers::of(&planned.plan.levers, "").map_err(|e| ("BUILD".to_string(), e))?;
    let prog = Program::of(planned.plan).map_err(|e| ("BUILD".to_string(), e))?;
    let (w, h) = (prog.floor.w, prog.floor.h);
    build::emit(&prog, &levers, None, out, &EMITS).map_err(|e| ("EMIT".to_string(), e))?;
    Ok(format!("kernels={kernels} preset={preset} fabric={w}x{h}"))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let scratch = match args.iter().position(|a| a == "--out") {
        Some(i) => {
            let d = PathBuf::from(args.remove(i + 1));
            args.remove(i);
            d
        }
        None => std::env::temp_dir().join("poems"),
    };
    let rows: u32 = std::env::var("ROWS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(16);
    let grid_rows: u32 = std::env::var("GRID_ROWS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(256);
    let catalog = poem_compiler::catalog::repository();
    for name in &args {
        let t0 = Instant::now();
        let Some(d) = catalog.parse(name) else {
            line(name, "-", "PARSE", "no such deployment", t0);
            continue;
        };
        let trace = match d.try_trace(Platform::Cerebras) {
            Ok(t) => t,
            Err(e) => {
                line(name, "-", "TRACE", &e.0, t0);
                continue;
            }
        };
        if let Ok(list) = std::env::var("NODES") {
            // `NODES=a,b,..`: those nodes and what makes each input, then nothing else
            for at in list.split(',').filter_map(|n| n.parse::<usize>().ok()) {
                let n = &trace.nodes[at];
                println!("node {at}: {:?}", n.op);
                let mut ins = Vec::new();
                n.op.inputs(&mut ins);
                for v in ins {
                    let d = &trace.values[v.0 as usize];
                    let by = match &d.def {
                        poem_ir::Def::Op(m) => {
                            format!("node {m} {}", trace.nodes[*m as usize].op.name())
                        }
                        other => format!("{other:?}"),
                    };
                    println!("  %{} {:?} <- {by}", v.0, d.ty);
                }
            }
            continue;
        }
        let dir = scratch.join(name);
        let samples = trace
            .nodes
            .iter()
            .any(|n| n.op.name().starts_with("sampling."));
        let layers = poem_compiler::analysis::analyze(&trace)
            .map(|a| a.layers.iter().flatten().max().map_or(0, |l| l + 1))
            .unwrap_or(0);
        line(
            name,
            "-",
            "TRACE",
            &format!(
                "nodes={} layers={layers} params={}",
                trace.nodes.len(),
                trace.params.len()
            ),
            t0,
        );
        let decodes =
            samples || std::env::var("DECODE").is_ok() || trace.facts.declared("reading").is_none();
        if decodes {
            let t1 = Instant::now();
            let out = dir.join("decode");
            match decode(&trace, &out) {
                Ok(what) => line(name, "decode", "PASS", &what, t1),
                Err((stage, e)) => line(name, "decode", &stage, &e, t1),
            }
            let _ = std::fs::remove_dir_all(&out);
        }
        if std::env::var("UNITS").is_ok() {
            // the decode step through the units (GENERIC.md G1)
            let t1 = Instant::now();
            let out = dir.join("decode_units");
            let positions = std::env::var("POSITIONS")
                .ok()
                .and_then(|r| r.parse().ok())
                .unwrap_or(4096);
            match kernels_cerebras::stream::compile_decode(&trace, positions, true, &out) {
                Ok(p) => line(
                    name,
                    "decode.units",
                    "PASS",
                    &format!(
                        "units={} fabric={}x{} py={} kv-columns={} cycles~{:.2}M",
                        p.units.len(),
                        p.stream.w,
                        p.stream.h,
                        p.stream.py,
                        p.stream.kv.len(),
                        kernels_cerebras::cost::stream::time(&p).cycles / 1e6
                    ),
                    t1,
                ),
                Err(e) => line(name, "decode.units", "UNITS", &e, t1),
            }
            let _ = std::fs::remove_dir_all(&out);
        }
        let readings = trace
            .facts
            .declared("reading")
            .map(|s| s.values.clone())
            .unwrap_or_default();
        for reading in readings {
            let t1 = Instant::now();
            let grid = kernels_cerebras::stream::grid::is_grid(&trace, &reading);
            let rows = if grid { grid_rows } else { rows };
            let o = kernels_cerebras::stream::Options {
                reading: reading.clone(),
                rows,
                round16: true,
                upto: None,
                taps: Vec::new(),
            };
            let out = dir.join(reading.replace('.', "_"));
            match kernels_cerebras::stream::compile_reading(&trace, &o, &out) {
                Ok((w, h)) => line(
                    name,
                    &reading,
                    "PASS",
                    &format!(
                        "{} rows={rows} fabric={w}x{h}",
                        if grid { "grid" } else { "stream" }
                    ),
                    t1,
                ),
                Err(e) => line(name, &reading, if grid { "GRID" } else { "STREAM" }, &e, t1),
            }
            let _ = std::fs::remove_dir_all(&out);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
