//! `wafer-compile`: the wafer program's compiler. `plan` runs poem's spatial pass over a model's trace
//! (or a shape stated outright) and writes the plan; `build` writes everything a plan builds: the
//! program's sources and the artifact the engine runs.
//!
//! ```text
//! wafer-compile plan (--model <deployment> | --shape <shape.json>) [--preset <name>] [--set K=V,...]
//!                [--positions N] --out <dir>
//! wafer-compile build --plan <plan.json> --out <dir> [--emit floor,rcfg,kernels,layout,host,plan,
//!                 bindings,program] [--levers K:V,...] [--kernels <programs/wafer dir>]
//! wafer-compile link --build <dir> [--fabric simulator|system]
//! wafer-compile cost --plan <plan.json>
//! wafer-compile search --model <deployment> [--positions N] [--set K=V,...] [--keep N]
//! ```
//!
//! `cost` prints a plan's cost features (`cost::Features`) and the bytes it moves on the weight
//! and head PEs; `search` ranks a deployment's partitions by their predicted step (`search`).
//!
//! `plan` writes `<dir>/plan.json` (the [`Plan`]) and names the preset it took.
//! `--model` names a catalog deployment (`qwen35-d9b-bf16-kv-bf16`); `--preset` one of the
//! serving family's presets (else the first that builds); `--set` overrides its keys;
//! `--positions` the positions the program must serve (else as many as the preset holds). The
//! preset carries the kernel levers its program ships with; `build --levers` sets more over them.
//!
//! `build` writes everything the plan builds into `<dir>` (`build::emit`: the sources,
//! the artifact), its kernels the embedded ones unless `--kernels`
//! names a directory; `link` compiles a build directory with cslc into `<dir>/out` (against the
//! smallest fabric that holds it, or with `--fabric system` a CS system's whole wafer), checks
//! every PE image's stack room and puts the artifact beside it (`<dir>/out/artifact`, what the
//! engine opens). `link` runs where the SDK image's toolchain is (`/cb`, `/cbcore`).

use std::path::{Path, PathBuf};

use kernels_cerebras::build::{self, EMITS, Fabric};
use kernels_cerebras::emit::params::Levers;
use kernels_cerebras::plan::{Plan, Shape};
use kernels_cerebras::program::Program;
use kernels_cerebras::target::{self, Ask};
use kernels_cerebras::toolchain::{Arch, Cslc, Toolchain};

const USAGE: &str = "usage: wafer-compile plan (--model <deployment> | --shape <shape.json>) \
     [--preset <name>] [--set K=V,...] [--positions N] --out <dir>\n       wafer-compile build --plan <plan.json> \
     --out <dir> [--emit a,b] [--levers K:V,...] [--kernels <dir>]\n       wafer-compile link \
     --build <dir> [--fabric simulator|system]\n       wafer-compile cost --plan <plan.json>\n       \
     wafer-compile search --model <deployment> [--positions N] [--set K=V,...] [--keep N]";

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

/// `plan`: the plan of a deployment's trace (or of a shape file).
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
    eprintln!(
        "wafer-compile plan: the {} family, preset {}",
        planned.plan.family.name(),
        planned.preset.name
    );
    json(&out, "plan.json", &planned.plan)
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
    let f = flags(args, &["--build", "--fabric"])?;
    let dir = PathBuf::from(get(&f, "--build").ok_or(USAGE)?);
    let fabric = match get(&f, "--fabric").unwrap_or("simulator") {
        "simulator" => Fabric::Simulator,
        "system" => Fabric::System,
        other => return Err(format!("--fabric {other}: simulator or system")),
    };
    let cslc = Cslc::find(&Toolchain::default()).map_err(|e| e.to_string())?;
    let room = build::link(&dir, &cslc, Arch::Wse3, fabric)?;
    println!("STACK_ROOM min {room} B; need >= {}: ok", build::STACK_ROOM);
    Ok(())
}

/// `cost`: what a plan costs (its features, its step time, its images' memory), as JSON.
fn cost(args: &[String]) -> Result<(), String> {
    let f = flags(args, &["--plan"])?;
    let plan: Plan = read(get(&f, "--plan").ok_or(USAGE)?)?;
    let features = kernels_cerebras::cost::Features::of(&plan);
    let moving = kernels_cerebras::cost::memory::Moving::of(&plan);
    println!(
        "{} {} {}",
        serde_json::to_string(&features).map_err(|e| e.to_string())?,
        moving.weight,
        moving.head
    );
    Ok(())
}

/// `search`: a deployment's partitions ranked by their predicted step, with the preset made for
/// it (if any) for comparison.
fn search(args: &[String]) -> Result<(), String> {
    let f = flags(args, &["--model", "--positions", "--set", "--keep"])?;
    let name = get(&f, "--model").ok_or(USAGE)?;
    let catalog = poem_compiler::catalog::repository();
    let d = catalog
        .parse(name)
        .ok_or_else(|| format!("--model {name}: no such deployment"))?;
    let trace = d
        .try_trace(poem_ir::Platform::Cerebras)
        .map_err(|e| format!("--model {name}: {e}"))?;
    let shape = target::shape(&trace)?;
    let ask = kernels_cerebras::search::Asked {
        positions: get(&f, "--positions")
            .map(|n| n.parse().map_err(|e| format!("--positions {n}: {e}")))
            .transpose()?,
        set: get(&f, "--set").unwrap_or("").to_string(),
    };
    let keep = get(&f, "--keep")
        .map_or(Ok(5), str::parse)
        .map_err(|e| format!("--keep: {e}"))?;
    let fam = kernels_cerebras::family::of(kernels_cerebras::family::serving(&shape));
    let made = fam
        .presets()
        .into_iter()
        .find(|p| p.made_for == Some(shape.signature()));
    for (i, c) in kernels_cerebras::search::search(&shape, &ask, keep)?
        .iter()
        .enumerate()
    {
        let moved = made.as_ref().map_or_else(
            || format!("{:?}", c.planned.preset.keys),
            |m| kernels_cerebras::search::moved(m, &c.planned.preset),
        );
        println!(
            "{i} {} {:.4} ms {:.0} B [{}]",
            c.planned.preset.name, c.ms, c.room, moved
        );
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..]),
        Some("build") => build(&args[1..]),
        Some("link") => link(&args[1..]),
        Some("cost") => cost(&args[1..]),
        Some("search") => search(&args[1..]),
        _ => Err(USAGE.to_string()),
    }
}
