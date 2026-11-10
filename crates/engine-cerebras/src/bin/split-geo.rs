//! `split-geo`: the role-split program's geometry as files, for the Python tools (the preps,
//! `mkfull*.py`, `run*.py`), so Rust is the one geometry implementation.
//!
//! ```text
//! split-geo <prep dir> <prefix> [k=v,...] --out <dir>
//!           [--what floor,rcfg,kernels,layout,host,meta,bindings]
//!           [--levers K:V,...] [--kernels <csl/split dir>]
//! split-geo --plan <plan.json> --out <dir> [...]
//! split-geo --geometry <qwen|gptoss|gemma|gemma12>:<name>
//! ```
//!
//! `--geometry` prints a named geometry as JSON (what a prep needs before its meta exists: the
//! unit shape, the head, Gemma's PLE storage columns, Gemma 12B's row groups); the prep then
//! writes the meta the plan reads.
//!
//! Writes, under `<dir>`:
//! - `floor.json`: units (rects, cx / cy, layers), relays, head block, stores, and the kind
//!   rectangles (`Floor::rects_by_kind`);
//! - `rcfg.npy` (u32, [h][w][words], zero-padded to the longest; idle PEs all zero), `rcfg_n.npy`
//!   (u32, [h][w]: each PE's word count) and `rcfg.txt`
//!   (`emit::rcfg::dump`, the parity tests' format);
//! - `kernels.json`: each role kind's source, owned symbols and params (levers applied);
//! - `layout/`: `layout.csl` and the per-kind sources (`emit::layout::write`), with the
//!   `params.txt` and `geo.env` `link.sh` reads;
//! - `host.json`: `emit::host::HostLayout`;
//! - `meta.json`: the meta the plan states;
//! - `bindings.json`: every copy onto the wafer in placement order (`Family::bindings`), which the
//!   Python runners execute (`tools/split/splitgeo.py`'s `place`) instead of placing on their own.
//!
//! The kernel sources default to `$SPLIT_SRC`, else this crate's `src/spatial/csl/split`.

use std::path::{Path, PathBuf};

use engine_cerebras::spatial::split::emit::host::HostLayout;
use engine_cerebras::spatial::split::emit::meta::Meta;
use engine_cerebras::spatial::split::emit::params::{Levers, kernels};
use engine_cerebras::spatial::split::emit::{layout, rcfg};
use engine_cerebras::spatial::split::family;
use engine_cerebras::spatial::split::floor::{HeadBlock, Rect, StoreBlock, Unit};
use engine_cerebras::spatial::split::plan::{FamilyId, Plan};
use engine_cerebras::spatial::split::program::Program;
use engine_cerebras::spatial::split::role::Kind;
use serde::Serialize;

const WHATS: [&str; 7] = [
    "floor", "rcfg", "kernels", "layout", "host", "meta", "bindings",
];

fn main() {
    if let Err(e) = run() {
        eprintln!("split-geo: {e}");
        std::process::exit(1);
    }
}

struct Args {
    prep: Option<(PathBuf, String, String)>,
    plan: Option<PathBuf>,
    out: PathBuf,
    what: Vec<String>,
    levers: Levers,
    kernels: PathBuf,
}

fn usage() -> String {
    "usage: split-geo <prep dir> <prefix> [k=v,...] --out <dir> [--what a,b] [--levers K:V,...] \
     [--kernels <dir>] | split-geo --plan <plan.json> --out <dir> [...] | split-geo --geometry \
     <family>:<name>"
        .to_string()
}

fn args() -> Result<Args, String> {
    let mut pos = Vec::new();
    let (mut plan, mut out, mut what, mut levers, mut kdir) = (None, None, None, None, None);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .ok_or_else(|| format!("{a}: a value\n{}", usage()))
        };
        match a.as_str() {
            "--plan" => plan = Some(PathBuf::from(val()?)),
            "--out" => out = Some(PathBuf::from(val()?)),
            "--what" => what = Some(val()?),
            "--levers" => levers = Some(val()?),
            "--kernels" => kdir = Some(PathBuf::from(val()?)),
            "-h" | "--help" => return Err(usage()),
            _ if a.starts_with("--") => return Err(format!("{a}?\n{}", usage())),
            _ => pos.push(a),
        }
    }
    let prep = match (plan.is_some(), pos.len()) {
        (true, 0) => None,
        (false, 2) => Some((PathBuf::from(&pos[0]), pos[1].clone(), String::new())),
        (false, 3) => Some((PathBuf::from(&pos[0]), pos[1].clone(), pos[2].clone())),
        _ => return Err(usage()),
    };
    let what: Vec<String> = what
        .map(|w| w.split(',').map(str::to_string).collect())
        .unwrap_or_else(|| WHATS.iter().map(|s| s.to_string()).collect());
    if let Some(w) = what.iter().find(|w| !WHATS.contains(&w.as_str())) {
        return Err(format!("--what {w}: one of {}", WHATS.join(",")));
    }
    let kernels = kdir
        .or_else(|| std::env::var_os("SPLIT_SRC").map(PathBuf::from))
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("src/spatial/csl/split"));
    Ok(Args {
        prep,
        plan,
        out: out.ok_or_else(usage)?,
        what,
        levers: Levers::parse(levers.as_deref().unwrap_or(""))?,
        kernels,
    })
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

/// A little-endian u32 npy of `shape` (version 1.0 header, padded to 64 bytes).
fn npy_u32(shape: &[usize], data: &[u32]) -> Vec<u8> {
    let dims = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut h = format!("{{'descr': '<u4', 'fortran_order': False, 'shape': ({dims},), }}");
    let total = 10 + h.len() + 1;
    h.push_str(&" ".repeat(total.next_multiple_of(64) - total));
    h.push('\n');
    let mut b = Vec::with_capacity(10 + h.len() + data.len() * 4);
    b.extend_from_slice(b"\x93NUMPY\x01\x00");
    b.extend_from_slice(&(h.len() as u16).to_le_bytes());
    b.extend_from_slice(h.as_bytes());
    for w in data {
        b.extend_from_slice(&w.to_le_bytes());
    }
    b
}

/// `--geometry <family>:<name>`: the named geometry (see the module doc).
fn geometry(spec: &str) -> Result<serde_json::Value, String> {
    let (fam, name) = spec
        .split_once(':')
        .ok_or_else(|| format!("--geometry {spec}: <family>:<name>"))?;
    let id = FamilyId::of_name(fam).ok_or_else(|| {
        let names: Vec<_> = FamilyId::ALL.iter().map(|f| f.name()).collect();
        format!("--geometry {spec}: a family of {}", names.join(", "))
    })?;
    family::of(id).geometry(name)
}

fn run() -> Result<(), String> {
    let mut argv = std::env::args().skip(1);
    if let (Some(f), Some(spec)) = (argv.next(), argv.next())
        && f == "--geometry"
    {
        let v = geometry(&spec)?;
        println!("{}", serde_json::to_string(&v).map_err(|e| e.to_string())?);
        return Ok(());
    }
    let a = args()?;
    let prog = match (&a.prep, &a.plan) {
        (Some((dir, prefix, ov)), _) => Program::from_meta(&Meta::read(dir, prefix, ov)?)?,
        (None, Some(p)) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let plan: Plan =
                serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))?;
            Program::build(family::of(plan.shape.family), plan)?
        }
        (None, None) => return Err(usage()),
    };
    std::fs::create_dir_all(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let want = |w: &str| a.what.iter().any(|x| x == w);
    if want("floor") {
        let f = &prog.floor;
        json(
            &a.out,
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
    if want("rcfg") {
        let g = rcfg::all(&prog)?;
        let (w, h) = (prog.floor.w as usize, prog.floor.h as usize);
        let n = g
            .iter()
            .filter_map(|(_, v)| v.as_ref().map(Vec::len))
            .max()
            .unwrap_or(0);
        let mut data = vec![0u32; w * h * n];
        for (c, v) in g.iter() {
            if let Some(v) = v {
                let at = (c.y as usize * w + c.x as usize) * n;
                data[at..at + v.len()].copy_from_slice(v);
            }
        }
        let lens: Vec<u32> = (0..h * w)
            .map(|i| {
                g.get(engine_cerebras::spatial::split::floor::Coord::new(
                    (i % w) as u32,
                    (i / w) as u32,
                ))
                .as_ref()
                .map_or(0, |v| v.len() as u32)
            })
            .collect();
        write(&a.out, "rcfg.npy", &npy_u32(&[h, w, n], &data))?;
        write(&a.out, "rcfg_n.npy", &npy_u32(&[h, w], &lens))?;
        write(&a.out, "rcfg.txt", rcfg::dump(&prog)?.as_bytes())?;
    }
    if want("kernels") {
        json(&a.out, "kernels.json", &kernels(&prog, &a.levers)?)?;
    }
    if want("layout") {
        layout::write(&prog, &a.levers, &a.kernels, &a.out.join("layout"))?;
    }
    if want("host") {
        json(&a.out, "host.json", &HostLayout::of(&prog)?)?;
    }
    if want("meta") {
        json(&a.out, "meta.json", &Meta::of_plan(&prog.plan))?;
    }
    if want("bindings") {
        json(&a.out, "bindings.json", &prog.bindings()?)?;
    }
    Ok(())
}
