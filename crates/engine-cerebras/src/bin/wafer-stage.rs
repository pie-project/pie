//! `wafer-stage`: the checkpoint cut a load makes, made outside the engine (to stage it ahead of a
//! chip session, or compare it): a catalog deployment's trace planned as a load plans it, its
//! checkpoint landed through the import contract and cut into the plan's tiles; or a plan
//! already made over params already landed (a test checkpoint's, under the trace's names).
//!
//! ```text
//! wafer-stage --model <deployment> --checkpoint <dir> --out <dir> [--preset <name>]
//!             [--set K=V,...] [--positions N]
//! wafer-stage --plan <plan.json> --landed <params.safetensors> --out <dir>
//! ```
//!
//! The cut goes to `<out>/<name>.npy`, the plan to `<out>/plan.json`.

use std::path::PathBuf;

use engine_cerebras::prepare::{Landed, land};
use kernels_cerebras::plan::Plan;
use kernels_cerebras::target::{self, Ask};
use kernels_cerebras::tiles;

const USAGE: &str = "usage: wafer-stage --model <deployment> --checkpoint <dir> --out <dir> \
                     [--preset <name>] [--set K=V,...] [--positions N]\n       \
                     wafer-stage --plan <plan.json> --landed <params.safetensors> --out <dir>";

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
    let out = PathBuf::from(need("out")?);
    let t0 = std::time::Instant::now();
    let (plan, landed) = match get.get("plan") {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
            let plan: Plan = serde_json::from_str(&text).map_err(|e| format!("{p}: {e}"))?;
            (plan, Landed::read(&PathBuf::from(need("landed")?))?)
        }
        None => {
            let (model, checkpoint) = (need("model")?, PathBuf::from(need("checkpoint")?));
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
            (plan, land(&trace, &contract, &checkpoint)?)
        }
    };
    eprintln!("landed in {:.1} s", t0.elapsed().as_secs_f64());
    tiles::stage(&plan, &landed, &out)?;
    let text = serde_json::to_string_pretty(&plan).map_err(|e| e.to_string())?;
    std::fs::write(out.join("plan.json"), text).map_err(|e| e.to_string())?;
    eprintln!("cut in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}
