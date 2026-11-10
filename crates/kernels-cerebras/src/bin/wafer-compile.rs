//! `wafer-compile`: the wafer program's compiler. `plan` runs poem's spatial pass over a model's trace
//! (or a shape stated outright) and writes the plan with what a prep tiles the checkpoint by;
//! `build` writes everything a plan builds: the program's sources, the artifact the engine runs,
//! and what the Python tools read.
//!
//! ```text
//! wafer-compile plan (--model <deployment> | --shape <shape.json>) [--preset <name>] [--set K=V,...]
//!                [--positions N] --out <dir>
//! wafer-compile build --plan <plan.json> --out <dir> [--emit floor,rcfg,kernels,layout,host,plan,
//!                 bindings,program] [--levers K:V,...] [--kernels <programs/wafer dir>]
//! wafer-compile link --build <dir>
//! ```
//!
//! `plan` writes `<dir>/plan.json` (the [`Plan`]) and `<dir>/tiling.json` (the family's
//! [`tiling`](kernels_cerebras::spatial::family::Family::tiling): the cut a prep tiles by, with
//! the preset's name).
//! `--model` names a catalog deployment (`qwen35-d9b-bf16-kv-bf16`); `--preset` one of the
//! serving family's presets (else the first that builds); `--set` overrides its keys;
//! `--positions` the positions the program must serve (else as many as the preset holds). The
//! preset carries the kernel levers its program ships with; `build --levers` sets more over them.
//!
//! `build` writes everything the plan builds into `<dir>` (`spatial::build::emit`: the sources,
//! the artifact, what the Python tools read), its kernels the embedded ones unless `--kernels`
//! names a directory; `link` compiles a build directory with cslc into `<dir>/out`, checks every
//! PE image's stack room and puts the artifact beside it (`<dir>/out/artifact`, what the engine
//! opens). `link` runs where the SDK image's toolchain is (`/cb`, `/cbcore`).

use std::path::{Path, PathBuf};

use kernels_cerebras::spatial::build::{self, EMITS};
use kernels_cerebras::spatial::emit::params::Levers;
use kernels_cerebras::spatial::family;
use kernels_cerebras::spatial::plan::{Plan, Shape};
use kernels_cerebras::spatial::program::Program;
use kernels_cerebras::spatial::target::{self, Ask};
use kernels_cerebras::spatial::toolchain::{Arch, Cslc, Toolchain};

const USAGE: &str = "usage: wafer-compile plan (--model <deployment> | --shape <shape.json>) \
     [--preset <name>] [--set K=V,...] [--positions N] --out <dir>\n       wafer-compile build --plan <plan.json> \
     --out <dir> [--emit a,b] [--levers K:V,...] [--kernels <dir>]\n       wafer-compile link \
     --build <dir>";

fn main() {
    if let Err(e) = run() {
        eprintln!("wafer-compile: {e}");
        std::process::exit(1);
    }
}

/// The flags after the subcommand, by name.
fn flags(args: &[String], known: &[&str]) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if !known.contains(&a.as_str()) {
            return Err(format!("{a}?\n{USAGE}"));
        }
        let v = it.next().ok_or_else(|| format!("{a}: a value\n{USAGE}"))?;
        out.push((a.clone(), v.clone()));
    }
    Ok(out)
}

fn get<'a>(f: &'a [(String, String)], k: &str) -> Option<&'a str> {
    f.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str())
}

fn read<T: for<'de> serde::Deserialize<'de>>(p: &str) -> Result<T, String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{p}: {e}"))
}

fn json(dir: &Path, name: &str, v: &impl serde::Serialize) -> Result<(), String> {
    let s = serde_json::to_string_pretty(v).map_err(|e| format!("{name}: {e}"))?;
    let p = dir.join(name);
    std::fs::write(&p, format!("{s}\n")).map_err(|e| format!("{}: {e}", p.display()))
}

fn mkdir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))
}

/// `plan`: the plan of a deployment's trace (or of a shape file) and its tiling.
fn plan(args: &[String]) -> Result<(), String> {
    let f = flags(
        args,
        &[
            "--model",
            "--shape",
            "--preset",
            "--set",
            "--positions",
            "--out",
        ],
    )?;
    let ask = Ask {
        preset: get(&f, "--preset").map(str::to_string),
        set: get(&f, "--set").unwrap_or("").to_string(),
        positions: get(&f, "--positions")
            .map(|n| n.parse().map_err(|e| format!("--positions {n}: {e}")))
            .transpose()?,
    };
    let planned = match (get(&f, "--model"), get(&f, "--shape")) {
        (Some(name), None) => {
            let catalog = poem_compiler::catalog::repository();
            let d = catalog
                .parse(name)
                .ok_or_else(|| format!("--model {name}: no such deployment"))?;
            let trace = d
                .try_trace(poem_ir::Platform::Cerebras)
                .map_err(|e| format!("--model {name}: {e}"))?;
            target::plan(&trace, &ask)?
        }
        (None, Some(path)) => target::plan_shape(read::<Shape>(path)?, &ask)?,
        _ => return Err(format!("--model or --shape, one of them\n{USAGE}")),
    };
    let out = PathBuf::from(get(&f, "--out").ok_or(USAGE)?);
    mkdir(&out)?;
    let plan = &planned.plan;
    let mut tiling = family::of(plan.family).tiling(plan)?;
    tiling["preset"] = planned.preset.name.clone().into();
    json(&out, "plan.json", plan)?;
    json(&out, "tiling.json", &tiling)
}

/// `build`: everything a plan builds.
fn build(args: &[String]) -> Result<(), String> {
    let f = flags(
        args,
        &["--plan", "--out", "--emit", "--levers", "--kernels"],
    )?;
    let plan: Plan = read(get(&f, "--plan").ok_or(USAGE)?)?;
    let out = PathBuf::from(get(&f, "--out").ok_or(USAGE)?);
    let emit: Vec<&str> = get(&f, "--emit").map_or(EMITS.to_vec(), |e| e.split(',').collect());
    let levers = Levers::of(&plan.levers, get(&f, "--levers").unwrap_or(""))?;
    let prog = Program::of(plan)?;
    build::emit(
        &prog,
        &levers,
        get(&f, "--kernels").map(Path::new),
        &out,
        &emit,
    )
}

/// `link`: a build directory compiled, checked and its artifact put beside it.
fn link(args: &[String]) -> Result<(), String> {
    let f = flags(args, &["--build"])?;
    let dir = PathBuf::from(get(&f, "--build").ok_or(USAGE)?);
    let cslc = Cslc::find(&Toolchain::default()).map_err(|e| e.to_string())?;
    let room = build::link(&dir, &cslc, Arch::Wse3)?;
    println!("STACK_ROOM min {room} B; need >= {}: ok", build::STACK_ROOM);
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..]),
        Some("build") => build(&args[1..]),
        Some("link") => link(&args[1..]),
        _ => Err(USAGE.to_string()),
    }
}
