//! Builds a wafer program through PIE's own path (generated sources, the plan's
//! parameters, `cslc`, the compile cache, `pie_manifest.json`) and prints the
//! output directory. Runs where `cslc` resolves (inside the SDK image).
//!
//! - `build_program tiny-pipe`: the tiny serve2d_pipe preset (plan + config);
//! - `build_program params <design> <K:V,...> [system]`: explicit parameters,
//!   for checking a design against a build compiled elsewhere.

use engine_cerebras::sdk::Target;
use engine_cerebras::spatial::build;
use engine_cerebras::spatial::codegen::Design;
use engine_cerebras::spatial::program::ProgramSpec;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let (design, params, system) = match args.get(1).map(String::as_str).unwrap_or("tiny-pipe") {
        "tiny-pipe" => {
            let (plan, cfg) = build::presets::pipe_config();
            let (d, p) = build::params(&plan, &cfg)?;
            (d, p, false)
        }
        "params" => {
            let d = Design::parse(args.get(2).ok_or("a design")?).ok_or("unknown design")?;
            let p = args
                .get(3)
                .ok_or("K:V,... params")?
                .split(',')
                .filter_map(|kv| kv.split_once(':'))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            (d, p, args.get(4).is_some_and(|s| s == "system"))
        }
        other => return Err(format!("unknown mode {other}")),
    };
    let program = ProgramSpec::from_params(design, &params)?;
    let out = build::compile(&program, Target::Wse3, system)?;
    println!("{}", out.display());
    Ok(())
}
