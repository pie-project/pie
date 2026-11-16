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
//! `--model` names a catalog deployment (`qwen35-d9b-bf16-kv-bf16`); `--preset` a pinned plan
//! (`plans/`; else the one made for the model's shape, else the search's); `--set` overrides its
//! keys; `--positions` the positions the program must serve (else as many as the plan holds). The
//! plan carries the kernel levers its program ships with; `build --levers` sets more over them.
//!
//! `build` writes everything the plan builds into `<dir>` (`build::emit`: the sources,
//! the artifact), its kernels the embedded ones unless `--kernels`
//! names a directory; `link` compiles a build directory with cslc into `<dir>/out` (against the
//! smallest fabric that holds it, or with `--fabric system` a CS system's whole wafer), checks
//! every PE image's stack room and puts the artifact beside it (`<dir>/out/artifact`, what the
//! engine opens). `link` runs where the SDK image's toolchain is (`/cb`, `/cbcore`).

use std::path::{Path, PathBuf};

use kernels_cerebras::build::{self, EMITS, Fabric};
use kernels_cerebras::cost::CLOCK_HZ;
use kernels_cerebras::emit::params::Levers;
use kernels_cerebras::plan::{Plan, Shape};
use kernels_cerebras::program::Program;
use kernels_cerebras::target::{self, Ask};
use kernels_cerebras::toolchain::{Arch, Cslc, Toolchain};

const USAGE: &str = "usage: wafer-compile plan (--model <deployment> | --shape <shape.json>) \
     [--preset <name>] [--set K=V,...] [--positions N] --out <dir>\n       wafer-compile build --plan <plan.json> \
     --out <dir> [--emit a,b] [--levers K:V,...] [--kernels <dir>]\n       wafer-compile link \
     --build <dir> [--fabric simulator|system]\n       wafer-compile cost --plan <plan.json>\n       \
     wafer-compile search --model <deployment> [--positions N] [--set K=V,...] [--keep N]\n       \
     wafer-compile stream --model <deployment> [--reading R] [--rows N] [--ieee] --out <dir> \
     [--regime auto|stream|gemm|tiled] [--gemm C,R,KBM] [--batch B | --batch-most B] [--values <dir>] [--stages N] [--tap v,..] [--py N] [--tile W] [--spill] \
     [--cut THIN,SUB,WIDEN,HELP] [--fabric-width W] [--class-rows context=512,latents=4096] \
     [--price context=512,latents=4096 [--parts]] [--clip t,h,w]";

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
        "wafer-compile plan: the {} kernels, preset {}",
        kernels_cerebras::kernels::KernelSet::of_shape(&planned.plan.shape).name(),
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
    let room = kernels_cerebras::cost::memory::room(
        &plan,
        kernels_cerebras::cost::memory::anchor(&plan.shape),
    );
    let prog = Program::of(plan)?;
    build::emit(
        &prog,
        &levers,
        get(&f, "--kernels").map(Path::new),
        &out,
        &emit,
    )?;
    // the static check of what was emitted (routes, queues, a PE's memory)
    if out.join("layout/layout.csl").is_file() {
        let mut rep = kernels_cerebras::check::dir(&out.join("layout"))?;
        rep.model_room = Some(room as i64);
        println!("{}", rep.line());
        rep.ok()?;
    }
    Ok(())
}

/// The static check of what `stream` wrote in `--out` (where it wrote a program).
fn checked(args: &[String]) -> Result<(), String> {
    let Some(out) = args
        .iter()
        .position(|a| a == "--out")
        .and_then(|i| args.get(i + 1))
    else {
        return Ok(());
    };
    let dir = Path::new(out);
    if !dir.join("layout.csl").is_file() || args.iter().any(|a| a == "--price") {
        return Ok(());
    }
    let rep = kernels_cerebras::check::dir(dir)?;
    println!("{}", rep.line());
    rep.ok()
}

/// `check --dir <program dir>`: the static check of an emitted program (a stream program's
/// directory, or a decode build's `layout/`).
fn check(args: &[String]) -> Result<(), String> {
    let f = flags(args, &["--dir"])?;
    let rep = kernels_cerebras::check::dir(Path::new(get(&f, "--dir").ok_or(USAGE)?))?;
    println!("{}", rep.line());
    for e in rep.errors.iter().take(20) {
        println!("  error: {e}");
    }
    for w in rep.warnings.iter().take(5) {
        println!("  warning: {w}");
    }
    for (f, d, c) in rep.codes.iter().take(8) {
        println!("  {f}: {d} B data + ~{c} B code");
    }
    if !rep.unknown.is_empty() {
        println!(
            "  {} array sizes not evaluated, e.g. {}",
            rep.unknown.len(),
            rep.unknown[0]
        );
    }
    rep.ok()
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
    let made = kernels_cerebras::pins::of(kernels_cerebras::kernels::KernelSet::of_shape(&shape))
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

/// `stream --model <name> [--reading denoise] [--rows 64] [--class-rows name=N,..] [--ieee]
/// [--py N] [--tile W] [--fabric-width W] --out <dir> [--values <dir>]` (`--py 16 --tile 12288`:
/// flux2-mini as M1d built it; `--fabric-width 38`: flux2-mini over three bands):
/// a stream reading's wafer program (kernels-cerebras `stream::compile`); with a step's captured
/// values, the host's files for a run of it (`tools/wafer/stream/run.py`).
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
    let trace = poem_compiler::catalog::repository()
        .parse(&model)
        .ok_or_else(|| format!("no catalog model {model}"))?
        .trace(poem_ir::Platform::Cerebras);
    let declared = trace
        .facts
        .declared("reading")
        .is_some_and(|d| d.values.contains(&reading));
    if reading == "decode" && !declared {
        // the decode step through the units, its caches `--positions` long (GENERIC.md G1)
        let positions: u32 = get("--positions")
            .map_or(Ok(4096), |r| r.parse())
            .map_err(|e| format!("--positions: {e}"))?;
        let lanes: u32 = get("--lanes")
            .map_or(Ok(1), |r| r.parse())
            .map_err(|e| format!("--lanes: {e}"))?;
        let p = kernels_cerebras::stream::compile_decode(
            &trace,
            positions,
            lanes,
            !ieee,
            std::path::Path::new(&out),
        )?;
        println!(
            "{model} decode: {} units, {} K / V columns of {positions} positions, {} x {} PEs, py {}, {:.3}M cycles a token priced",
            p.units.len(),
            p.stream.kv.len(),
            p.stream.w,
            p.stream.h,
            p.stream.py,
            kernels_cerebras::cost::stream::time(&p).cycles / 1e6
        );
        return Ok(());
    }
    if kernels_cerebras::stream::grid::is_grid(&trace, &reading) {
        // a grid reading (a VAE's): its passes over one unit, on the clip `--clip t,h,w` (else
        // the one `--rows` token rows make)
        let clip = match get("--clip") {
            Some(c) => {
                let v: Vec<u32> = c.split(',').filter_map(|x| x.parse().ok()).collect();
                let [t, h, w] = v[..] else {
                    return Err(format!("--clip {c}: t,h,w"));
                };
                [t, h, w]
            }
            None => kernels_cerebras::stream::grid::clip_of(&trace, &reading, rows),
        };
        // `--regime tiled`: the image over the fabric (M2d), `--tile H,W` its top tile pinned
        let tile = match get("--tile") {
            Some(t) => {
                let v: Vec<u32> = t.split(',').filter_map(|x| x.parse().ok()).collect();
                let [a, b] = v[..] else {
                    return Err(format!("--tile {t}: h,w"));
                };
                Some((a, b))
            }
            None => None,
        };
        if get("--regime").as_deref() == Some("tiled") {
            let bounds = kernels_cerebras::floorplan::stream::Bounds::system();
            let p = kernels_cerebras::stream::tiled::program(
                &trace,
                &reading,
                clip,
                !ieee,
                tile,
                Some((bounds.w, bounds.h)),
            )?;
            let c = kernels_cerebras::cost::tiled::time(&p);
            let (fw, fh) = p.fabric();
            println!(
                "{model} {reading} on {clip:?}: tiled, {} ops over a {} x {} array ({fw} x {fh} PEs, tape {}), arena {} + staging {} + block {} words, {:.1}M cycles ({:.2} ms at 750 MHz)",
                p.ops.len(),
                p.gx,
                p.gy,
                p.tape,
                p.arena,
                p.hx / 2,
                p.wb,
                c.cycles / 1e6,
                c.cycles / CLOCK_HZ * 1e3
            );
            if args.iter().any(|a| a == "--parts") {
                let (by_at, by_what) = c.by();
                for (at, cy) in by_at {
                    println!("  {}x{}: {:.2}M", at.0, at.1, cy / 1e6);
                }
                for (w, cy) in by_what {
                    println!("  {w}: {:.2}M", cy / 1e6);
                }
                for o in &c.ops {
                    println!(
                        "  op {} {} {:?}: {:.0} cycles",
                        o.op, o.what, o.at, o.cycles
                    );
                }
                let (at, live) = p.peak();
                println!("  the arena's peak at op {at}: {live:?}");
            }
            if !args.iter().any(|a| a == "--price") {
                kernels_cerebras::stream::tiled::compile_program(&p, std::path::Path::new(&out))?;
            }
            return Ok(());
        }
        if args.iter().any(|a| a == "--price") {
            // the cost model's time for the reading on `clip`, its placement aside
            let g = kernels_cerebras::stream::grid::program_unfit(&trace, &reading, clip, !ieee)?;
            let c = kernels_cerebras::cost::grid::time(&g);
            println!(
                "{model} {reading} on {clip:?}: {} passes over {} x {} roots, {:.1}M cycles ({:.1} ms at 750 MHz)",
                g.passes.len(),
                g.p,
                g.p,
                c.cycles / 1e6,
                c.cycles / CLOCK_HZ * 1e3
            );
            if args.iter().any(|a| a == "--parts") {
                for q in &c.passes {
                    println!(
                        "  pass {} rows {} head {} matvec {:.0} roots {:.0} head row {:.0}: {:.1}M",
                        q.pass,
                        q.rows,
                        q.head_rows,
                        q.matvec_row,
                        q.root_row,
                        q.head_row,
                        q.cycles / 1e6
                    );
                }
            }
            return Ok(());
        }
        // `--weight-words N`: a weight PE's words for tiles and stores (fewer: weight groups)
        let budget = get("--weight-words")
            .map(|w| w.parse::<u32>().map_err(|e| format!("--weight-words: {e}")))
            .transpose()?
            .unwrap_or(kernels_cerebras::stream::grid::WEIGHT_WORDS);
        let g = kernels_cerebras::stream::grid::compile_within(
            &trace,
            &reading,
            clip,
            !ieee,
            None,
            budget,
            std::path::Path::new(&out),
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
    let values = get("--values");
    let taps: Vec<u32> = get("--tap")
        .map(|t| t.split(',').filter_map(|v| v.parse().ok()).collect())
        .unwrap_or_default();
    let upto = get("--stages")
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|e| format!("--stages: {e}"))?;
    let py = get("--py")
        .map(|s| s.parse::<u32>())
        .transpose()
        .map_err(|e| format!("--py: {e}"))?;
    let tile = get("--tile")
        .map(|s| s.parse::<u32>())
        .transpose()
        .map_err(|e| format!("--tile: {e}"))?;
    let o = kernels_cerebras::stream::Options {
        reading: reading.clone(),
        rows,
        round16: !ieee,
        upto,
        taps,
    };
    // with nothing pinned, the regime and its every choice the engine's compile on load makes
    // (`stream::compile_whole`, by the regimes' expected prices); `--regime stream|gemm` and the
    // levers pin them (debugging)
    let pins = [
        "--regime",
        "--gemm",
        "--batch",
        "--batch-most",
        "--py",
        "--tile",
        "--cut",
        "--spill",
        "--fabric-width",
        "--price",
        "--values",
        "--emulate",
    ];
    if get("--regime").as_deref() == Some("auto")
        || !args.iter().any(|a| pins.contains(&a.as_str()))
    {
        let crow = kernels_cerebras::stream::parse_rows(&get("--class-rows").unwrap_or_default())?;
        let (w, h) =
            kernels_cerebras::stream::compile_whole(&trace, &o, &crow, std::path::Path::new(&out))?;
        let gemm = std::fs::read_to_string(std::path::Path::new(&out).join("stream.json"))
            .is_ok_and(|t| t.contains("\"regime\": \"gemm\""));
        println!(
            "{model} {reading}: the {} regime, {w} x {h} PEs (compile on load's choice)",
            if gemm { "GEMM" } else { "stream" }
        );
        return Ok(());
    }
    if get("--regime").as_deref() == Some("gemm") {
        // the GEMM regime (M2f): the reading over the whole fabric, `--gemm C,R,KBM` its array
        // and key block pinned
        let pin = match get("--gemm") {
            Some(g) => {
                let v: Vec<u32> = g.split(',').filter_map(|x| x.parse().ok()).collect();
                let [c, r, k] = v[..] else {
                    return Err(format!("--gemm {g}: C,R,KBM"));
                };
                Some((c, r, k))
            }
            None => None,
        };
        let crow = kernels_cerebras::stream::parse_rows(&get("--class-rows").unwrap_or_default())?;
        let bounds = kernels_cerebras::floorplan::stream::Bounds::system();
        let p = match get("--batch-most") {
            // the batch by price, at most so many requests a launch
            Some(m) if pin.is_none() => kernels_cerebras::stream::gemm::batched(
                &trace,
                &o,
                &crow,
                (bounds.w, bounds.h),
                m.parse().map_err(|e| format!("--batch-most: {e}"))?,
            )?,
            _ => kernels_cerebras::stream::gemm::program(
                &trace,
                &o,
                &crow,
                pin,
                (bounds.w, bounds.h),
                get("--batch")
                    .map_or(Ok(1), |b| b.parse::<u32>())
                    .map_err(|e| format!("--batch: {e}"))?,
            )?,
        };
        let c = kernels_cerebras::cost::gemm::time(&p);
        let m = kernels_cerebras::cost::gemm::memory(&p);
        println!(
            "{model} {reading}: gemm, {} requests a launch, {} ops ({} modules) over {} x {} PEs, {} rows a PE, key blocks of {} rows, stream {} words a column ({} rows of tapes), {} words a PE, {:.2}M cycles ({:.3} ms at 750 MHz)",
            p.l.batch,
            p.l.ops.len(),
            p.modules.len(),
            p.c,
            p.r,
            p.tb,
            p.kbm,
            p.wwords,
            p.tape,
            m.total,
            c.cycles / 1e6,
            c.cycles / CLOCK_HZ * 1e3
        );
        if args.iter().any(|a| a == "--parts") {
            for (w, cy) in c.by() {
                println!("  {w}: {:.3}M", cy / 1e6);
            }
            println!("  memory {m:?}");
            for o in &c.ops {
                println!("  op {} {}: {:.0}", o.op, o.what, o.cycles);
            }
        }
        if args.iter().any(|a| a == "--emulate") {
            // the program on the host over the capture's values, its outputs against it
            let dir = values.as_deref().ok_or("--emulate: --values")?;
            let vals = kernels_cerebras::stream::host::Values::open(std::path::Path::new(dir))?;
            for (v, got) in kernels_cerebras::stream::gemm::emulate(&p.l, &vals)? {
                let (_, want) = vals.get(v)?;
                let (mut a, mut b) = (0f64, 0f64);
                for (x, y) in got.iter().zip(want.iter()) {
                    if x.is_finite() && y.is_finite() {
                        a += f64::from(x - y).powi(2);
                        b += f64::from(*y).powi(2);
                    }
                }
                println!(
                    "EMULATE value %{} rel L2 {:.3e}",
                    v.0,
                    (a / b.max(1e-30)).sqrt()
                );
            }
        }
        if !args.iter().any(|a| a == "--price") {
            kernels_cerebras::stream::gemm::compile_program(
                &p,
                std::path::Path::new(&out),
                values.as_deref().map(std::path::Path::new),
            )?;
        }
        return Ok(());
    }
    // `--cut T,S,W,H`: K / V PEs at 1 / T of their entries over S columns a head, W quarters more
    // rest roots, up to H helper PEs a rest root
    let kv = get("--cut")
        .map(|v| -> Result<_, String> {
            let n: Vec<u32> = v
                .split(',')
                .map(|x| x.parse().map_err(|e| format!("--cut {v}: {e}")))
                .collect::<Result<_, _>>()?;
            let [thin, sub, widen, help] = n[..] else {
                return Err(format!("--cut {v}: THIN,SUB,WIDEN,HELP"));
            };
            Ok(kernels_cerebras::stream::build::KvSplit {
                thin,
                sub,
                widen,
                help,
            })
        })
        .transpose()?;
    let pins = kernels_cerebras::stream::Pins {
        py,
        tile,
        kv,
        spill: args.iter().any(|a| a == "--spill"),
        class_rows: kernels_cerebras::stream::parse_rows(&get("--class-rows").unwrap_or_default())?,
        width: get("--fabric-width")
            .map(|s| s.parse::<u32>())
            .transpose()
            .map_err(|e| format!("--fabric-width: {e}"))?,
    };
    let p = kernels_cerebras::stream::compile_pinned(
        &trace,
        &o,
        &pins,
        std::path::Path::new(&out),
        values.as_deref().map(std::path::Path::new),
    )?;
    if let Some(at) = get("--price") {
        // the program's cost model read at other rows a class (its layout as built)
        let named = kernels_cerebras::stream::parse_rows(&at)?;
        let rows: Vec<u32> = p
            .shape
            .classes
            .iter()
            .map(|(c, r)| named.iter().find(|(n, _)| n == c).map_or(*r, |x| x.1))
            .collect();
        let c = kernels_cerebras::cost::stream::time_at(&p, &rows);
        println!(
            "priced at {at}: {:.1}M cycles ({:.1} ms at 750 MHz)",
            c.cycles / 1e6,
            c.cycles / CLOCK_HZ * 1e3
        );
        for u in &c.units {
            println!(
                "  {} rows {} early {:.0} late {:.0} matvec {:.0} cycles a row",
                u.name, u.rows, u.early_row, u.late_row, u.matvec_row
            );
            if args.iter().any(|a| a == "--parts") {
                for q in &u.parts {
                    println!(
                        "    step {} {}{} roots {} chunk {}: {:.0}",
                        q.step,
                        q.what,
                        if q.late { " (late)" } else { "" },
                        q.roots,
                        q.chunk,
                        q.cycles
                    );
                }
            }
        }
    }
    println!(
        "{model} {reading}: {} units, {} K / V columns, {} x {} PEs, py {}, {} lane words",
        p.units.len(),
        p.stream.kv.len(),
        p.stream.w,
        p.stream.h,
        p.stream.py,
        p.lane_words
    );
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
        Some("stream") => stream(&args[1..]).and_then(|()| checked(&args[1..])),
        Some("check") => check(&args[1..]),
        _ => Err(USAGE.to_string()),
    }
}
