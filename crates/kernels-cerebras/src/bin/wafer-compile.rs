//! `wafer-compile`: the wafer program's compiler. `plan` runs poem's spatial pass over a model's
//! trace (or a shape stated outright) and writes the plan; `build` writes everything a plan
//! builds: the program's sources and the artifact the engine runs; `stream` writes a stream
//! reading's program as compile on load chooses it.
//!
//! ```text
//! wafer-compile plan (--model <deployment> | --shape <shape.json>) [--set K=V,...]
//!                [--positions N] --out <dir>
//! wafer-compile build --plan <plan.json> --out <dir> [--emit floor,rcfg,kernels,layout,host,plan,
//!                 bindings,program] [--levers K:V,...] [--kernels <kernels dir>]
//! wafer-compile link --build <dir> [--fabric simulator|system]
//! wafer-compile stream --model <deployment> [--reading R] [--rows N] [--class-rows c=N,..]
//!                 [--ieee] [--positions N] [--lanes B] [--anneal N [--anneal-seed N]
//!                 [--anneal-units 1]] --out <dir>
//! ```
//!
//! `--profile <file>` (any subcommand) prices under the chip's calibration a profile file states
//! (`engine-cerebras/profiles/cs3.toml`), else the defaults.
//!
//! `plan` writes `<dir>/plan.json` (the [`Plan`]) and names the preset it took.
//! `--model` names a catalog deployment (`qwen35-d9b-bf16-kv-bf16`); `--set` pins the search's
//! keys; `--positions` the positions the program must serve. The plan carries the kernel levers
//! its program ships with; `build --levers` sets more over them.
//!
//! `build` writes everything the plan builds into `<dir>` (the sources, the artifact), its
//! kernels the embedded ones unless `--kernels` names a directory. `link` compiles a build
//! directory with cslc into `<dir>/out` (with `--fabric system`, a CS system's whole wafer),
//! checks every PE image's stack room and puts the artifact beside it (`<dir>/out/artifact`). It
//! runs where the SDK image's toolchain is (`/cb`, `/cbcore`).

use std::path::{Path, PathBuf};

use kernels_cerebras::compile::build::{self, EMITS, Fabric};
use kernels_cerebras::compile::emit::params::Levers;
use kernels_cerebras::compile::plan::{Plan, Shape};
use kernels_cerebras::compile::program::Program;
use kernels_cerebras::compile::target::{self, Ask};
use kernels_cerebras::compile::toolchain::{Arch, Cslc, Toolchain};

const USAGE: &str = "usage: wafer-compile plan (--model <deployment> | --shape <shape.json>) \
     [--set K=V,...] [--positions N] --out <dir>\n       wafer-compile build --plan <plan.json> \
     --out <dir> [--emit a,b] [--levers K:V,...] [--kernels <dir>]\n       wafer-compile link \
     --build <dir> [--fabric simulator|system]\n       wafer-compile stream --model <deployment> \
     [--reading R] [--rows N] [--class-rows c=N,..] [--ieee] [--positions N] [--lanes B] \
     [--anneal <moves a block> [--anneal-seed N] [--anneal-units 1]] --out <dir>";

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
        &["--model", "--shape", "--set", "--positions", "--out"],
    )?;
    let ask = Ask {
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
        "wafer-compile plan: the {} kernels, preset {}",
        kernels_cerebras::compile::kernels::KernelSet::of_shape(&planned.plan.shape).name(),
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
    // (a decode program's memory is its model's: its sizes are comptime functions)
    let room = kernels_cerebras::compile::cost::memory::room(
        &plan,
        kernels_cerebras::compile::cost::memory::anchor(&plan.shape),
    );
    let prog = Program::of(plan)?;
    kernels_cerebras::compile::check::keep();
    build::emit(
        &prog,
        &levers,
        get(&f, "--kernels").map(Path::new),
        &out,
        &emit,
    )?;
    // the static check of what was emitted (routes, queues, a PE's memory)
    if let Some(mut rep) = kernels_cerebras::compile::check::last() {
        rep.model_room = Some(room as i64);
        println!("{}", rep.line());
        rep.ok()?;
    }
    Ok(())
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

/// `stream --model <name> [--reading denoise] [--rows 64] [--class-rows name=N,..] [--ieee]
/// --out <dir>`: a stream reading's wafer program as compile on load chooses it; `--reading
/// decode` (a model without one) the decode step through the units, `--positions` long.
fn stream(args: &[String]) -> Result<(), String> {
    let get = |k: &str| {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model = get("--model").ok_or("stream: --model")?;
    let out = get("--out").ok_or("stream: --out")?;
    let reading = get("--reading").unwrap_or_else(|| "denoise".into());
    let rows: u32 = get("--rows")
        .map_or(Ok(64), |r| r.parse())
        .map_err(|e| format!("--rows: {e}"))?;
    let ieee = args.iter().any(|a| a == "--ieee");
    // the annealer after the search: `--anneal <moves a block>`, `--anneal-seed <n>`,
    // `--anneal-units 1` (units move too, not the K / V columns alone)
    let anneal = kernels_cerebras::compile::place::anneal::Anneal {
        per_block: get("--anneal")
            .map_or(Ok(0), |m| m.parse())
            .map_err(|e| format!("--anneal: {e}"))?,
        seed: get("--anneal-seed")
            .map_or(Ok(1), |v| v.parse())
            .map_err(|e| format!("--anneal-seed: {e}"))?,
        units: get("--anneal-units").is_some_and(|v| v == "1"),
    };
    let trace = poem_compiler::catalog::repository()
        .parse(&model)
        .ok_or_else(|| format!("no catalog model {model}"))?
        .trace(poem_ir::Platform::Cerebras);
    let declared = trace
        .facts
        .declared("reading")
        .is_some_and(|d| d.values.contains(&reading));
    if reading == "decode" && !declared {
        let positions: u32 = get("--positions")
            .map_or(Ok(4096), |r| r.parse())
            .map_err(|e| format!("--positions: {e}"))?;
        let lanes: u32 = get("--lanes")
            .map_or(Ok(1), |r| r.parse())
            .map_err(|e| format!("--lanes: {e}"))?;
        let p = kernels_cerebras::compile::stream::compile_decode(
            &trace,
            positions,
            lanes,
            !ieee,
            None,
            &anneal,
            Path::new(&out),
        )?;
        println!(
            "{model} decode: {} units, {} K / V columns of {positions} positions, {} x {} PEs, py {}, {:.3}M cycles a token priced",
            p.units.len(),
            p.stream.kv.len(),
            p.stream.w,
            p.stream.h,
            p.stream.py,
            kernels_cerebras::compile::cost::stream::time(&p).cycles / 1e6
        );
        return Ok(());
    }
    if kernels_cerebras::compile::stream::grid::is_grid(&trace, &reading) {
        // a grid reading (a VAE's): its passes over one unit, on the clip `--rows` token rows make
        let clip = kernels_cerebras::compile::stream::grid::clip_of(&trace, &reading, rows);
        let g = kernels_cerebras::compile::stream::grid::compile_within(
            &trace,
            &reading,
            clip,
            !ieee,
            None,
            kernels_cerebras::compile::stream::grid::WEIGHT_WORDS,
            Path::new(&out),
        )?;
        println!(
            "{model} {reading}: a grid reading on {clip:?}, {} passes over {} x {} roots, {} matvecs, {} x {} PEs, {} weight groups",
            g.passes.len(),
            g.p,
            g.p,
            g.mvs.len(),
            g.stream.w,
            g.stream.h,
            g.groups.len().max(1)
        );
        return Ok(());
    }
    let o = kernels_cerebras::compile::stream::Options {
        reading: reading.clone(),
        rows,
        round16: !ieee,
        upto: None,
        taps: Vec::new(),
        anneal,
    };
    let crow =
        kernels_cerebras::compile::stream::parse_rows(&get("--class-rows").unwrap_or_default())?;
    let (w, h) =
        kernels_cerebras::compile::stream::compile_whole(&trace, &o, &crow, Path::new(&out))?;
    let gemm = std::fs::read_to_string(Path::new(&out).join("stream.json"))
        .is_ok_and(|t| t.contains("\"regime\": \"gemm\""));
    println!(
        "{model} {reading}: the {} regime, {w} x {h} PEs (compile on load's choice)",
        if gemm { "GEMM" } else { "stream" }
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--profile <file>`: the chip's calibration the costs price under (else the defaults)
    if let Some(i) = args.iter().position(|a| a == "--profile") {
        let path = args.get(i + 1).ok_or(USAGE)?.clone();
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
        kernels_cerebras::tuning::set(kernels_cerebras::tuning::load(&text, &path)?)?;
        args.drain(i..i + 2);
    }
    match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..]),
        Some("build") => build(&args[1..]),
        Some("link") => link(&args[1..]),
        Some("stream") => {
            kernels_cerebras::compile::check::keep();
            stream(&args[1..])?;
            kernels_cerebras::compile::check::last().map_or(Ok(()), |rep| {
                println!("{}", rep.line());
                rep.ok()
            })
        }
        _ => Err(USAGE.to_string()),
    }
}
