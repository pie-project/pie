//! Builds a wafer program through PIE's own path (generated sources, the plan's
//! parameters, `cslc`, the compile cache, `pie_manifest.json`) and prints the
//! output directory: `build_program [tiny-pipe]`. Runs where `cslc` resolves
//! (inside the SDK image).

use engine_cerebras::sdk::Target;
use engine_cerebras::spatial::build;

fn main() -> Result<(), String> {
    let which = std::env::args().nth(1).unwrap_or_else(|| "tiny-pipe".into());
    let (plan, cfg) = match which.as_str() {
        "tiny-pipe" => build::presets::pipe_config(),
        other => return Err(format!("unknown preset {other}")),
    };
    let (design, params) = build::params(&plan, &cfg)?;
    let out = build::compile(design, &params, Target::Wse3, false)?;
    println!("{}", out.display());
    Ok(())
}
