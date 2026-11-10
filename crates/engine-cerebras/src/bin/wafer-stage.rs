//! `wafer-stage`: the checkpoint cut a load makes, made outside the engine (to stage it ahead of a
//! chip session, or compare it): a catalog deployment's trace planned as a load plans it, its
//! checkpoint landed through the import contract and cut into the plan's tiles.
//!
//! ```text
//! wafer-stage --model <deployment> --checkpoint <dir> --out <dir> [--preset <name>]
//!             [--set K=V,...] [--positions N]
//! ```
//!
//! The cut goes to `<out>/<name>.npy`, the plan to `<out>/plan.json`.

use std::path::PathBuf;

use engine_cerebras::spatial::prepare::land;
use kernels_cerebras::spatial::target::{self, Ask};
use kernels_cerebras::spatial::tiles;

const USAGE: &str = "usage: wafer-stage --model <deployment> --checkpoint <dir> --out <dir> \
                     [--preset <name>] [--set K=V,...] [--positions N]";

fn main() {
    if let Err(e) = run() {
        eprintln!("wafer-stage: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut get = std::collections::HashMap::new();
    for pair in args.chunks(2) {
        match pair {
            [k, v] if k.starts_with("--") => {
                get.insert(k.trim_start_matches("--").to_string(), v.clone());
            }
            _ => return Err(USAGE.into()),
        }
    }
    let need = |k: &str| get.get(k).cloned().ok_or_else(|| USAGE.to_string());
    let (model, checkpoint, out) = (
        need("model")?,
        PathBuf::from(need("checkpoint")?),
        PathBuf::from(need("out")?),
    );
    let d = poem_compiler::catalog::repository()
        .parse(&model)
        .ok_or_else(|| format!("--model {model}: no such deployment"))?;
    let trace = d
        .try_trace(poem_ir::Platform::Cerebras)
        .map_err(|e| format!("{model}: {e}"))?;
    let ask = Ask {
        preset: get.get("preset").cloned(),
        set: get.get("set").cloned().unwrap_or_default(),
        positions: get
            .get("positions")
            .map(|n| n.parse().map_err(|e| format!("--positions {n}: {e}")))
            .transpose()?,
    };
    let plan = target::plan(&trace, &ask)?.plan;
    let source = checkpoint::file::snapshot::open(&checkpoint)
        .map_err(|e| format!("{}: {e}", checkpoint.display()))?;
    let contract = d
        .contract(&source, poem_ir::Platform::Cerebras)
        .map_err(|e| format!("the import contract of {model}: {e}"))?;
    let t0 = std::time::Instant::now();
    let landed = land(&trace, &contract, &checkpoint)?;
    eprintln!("landed in {:.1} s", t0.elapsed().as_secs_f64());
    tiles::stage(&plan, &landed, &out)?;
    let text = serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?;
    std::fs::write(out.join("plan.json"), text).map_err(|e| e.to_string())?;
    eprintln!("cut in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
