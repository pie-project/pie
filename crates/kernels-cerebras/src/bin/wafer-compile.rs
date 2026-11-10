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
//! `build` writes, under `<dir>`:
//! - `floor.json`: units (rects, cx / cy, layers), relays, head block, stores, and the kind
//!   rectangles (`Floor::rects_by_kind`);
//! - `rcfg.npy` (u32, [h][w][words], zero-padded to the longest; idle PEs all zero), `rcfg_n.npy`
//!   (u32, [h][w]: each PE's word count) and `rcfg.txt` (`emit::rcfg::dump`);
//! - `kernels.json`: each role kind's source, owned symbols and params (levers applied);
//! - `layout/`: `layout.csl` and the per-kind sources (`emit::layout::write`), with the
//!   `params.txt` and `dims.env` `link.sh` reads;
//! - `host.json`: `emit::host::HostLayout`;
//! - `plan.json`: the plan it built;
//! - `bindings.json`: every copy onto the wafer in placement order (`Program::bindings`), which the
//!   Python runners execute (`tools/wafer/compiler.py`'s `place`);
//! - `program.json`: the artifact's header (`artifact::Header`: the model facts and the plan).
//!   With `host.json`, `bindings.json` and the rcfg arrays it is the `artifact::Artifact` the
//!   engine runs.
//!
//! The kernel sources default to `$WAFER_KERNELS`, else this crate's `programs/wafer`.

use std::path::{Path, PathBuf};

use kernels_cerebras::spatial::artifact::Artifact;
use kernels_cerebras::spatial::emit::params::{Levers, kernels};
use kernels_cerebras::spatial::emit::{layout, rcfg};
use kernels_cerebras::spatial::family;
use kernels_cerebras::spatial::floor::{HeadBlock, Rect, StoreBlock, Unit};
use kernels_cerebras::spatial::plan::{Plan, Shape};
use kernels_cerebras::spatial::program::Program;
use kernels_cerebras::spatial::role::Kind;
use kernels_cerebras::spatial::target::{self, Ask};
use serde::Serialize;

const EMITS: [&str; 8] = [
    "floor", "rcfg", "kernels", "layout", "host", "plan", "bindings", "program",
];

const USAGE: &str = "usage: wafer-compile plan (--model <deployment> | --shape <shape.json>) \
     [--preset <name>] [--set K=V,...] [--positions N] --out <dir>\n       wafer-compile build --plan <plan.json> \
     --out <dir> [--emit a,b] [--levers K:V,...] [--kernels <dir>]";

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

/// `floor.json`: the floor without its per-PE role grid (the kind rectangles stand for it).
#[derive(Serialize)]
struct FloorFile<'a> {
    w: u32,
    h: u32,
    units: &'a [Unit],
    relays: &'a [Rect],
    head: &'a HeadBlock,
    stores: &'a [StoreBlock],
    kinds: Vec<(Kind, Rect)>,
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let p = dir.join(name);
    std::fs::write(&p, bytes).map_err(|e| format!("{}: {e}", p.display()))
}

fn json(dir: &Path, name: &str, v: &impl Serialize) -> Result<(), String> {
    let s = serde_json::to_string_pretty(v).map_err(|e| format!("{name}: {e}"))?;
    write(dir, name, format!("{s}\n").as_bytes())
}

fn mkdir(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))
}

/// `plan`: the plan of a deployment's trace (or of a shape file) and its tiling.
fn plan(args: &[String]) -> Result<(), String> {
    let f = flags(
        args,
        &["--model", "--shape", "--preset", "--set", "--positions", "--out"],
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
    if let Some(e) = emit.iter().find(|e| !EMITS.contains(e)) {
        return Err(format!("--emit {e}: of {}", EMITS.join(",")));
    }
    let levers = Levers::of(&plan.levers, get(&f, "--levers").unwrap_or(""))?;
    let kdir = get(&f, "--kernels")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("WAFER_KERNELS").map(PathBuf::from))
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("programs/wafer"));
    let prog = Program::of(plan)?;
    mkdir(&out)?;
    let want = |w: &str| emit.contains(&w);
    if want("floor") {
        let f = &prog.floor;
        json(
            &out,
            "floor.json",
            &FloorFile {
                w: f.w,
                h: f.h,
                units: &f.units,
                relays: &f.relays,
                head: &f.head,
                stores: &f.stores,
                kinds: f.rects_by_kind(),
            },
        )?;
    }
    if ["rcfg", "host", "bindings", "program"]
        .iter()
        .any(|w| want(w))
    {
        let art = Artifact::of(&prog)?;
        if want("rcfg") {
            art.write_rcfg(&out)?;
            write(&out, "rcfg.txt", rcfg::dump(&prog)?.as_bytes())?;
        }
        if want("host") {
            art.write_host(&out)?;
        }
        if want("bindings") {
            art.write_bindings(&out)?;
        }
        if want("program") {
            art.write_header(&out)?;
        }
    }
    if want("kernels") {
        json(&out, "kernels.json", &kernels(&prog, &levers)?)?;
    }
    if want("layout") {
        layout::write(&prog, &levers, &kdir, &out.join("layout"))?;
    }
    if want("plan") {
        json(&out, "plan.json", &prog.plan)?;
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("plan") => plan(&args[1..]),
        Some("build") => build(&args[1..]),
        _ => Err(USAGE.to_string()),
    }
}
