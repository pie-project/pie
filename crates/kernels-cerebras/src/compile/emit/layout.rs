//! The program's `layout.csl` and per-kind source files. Each kind's source is its kernel
//! source plus one-word dummies for the exported names it does not own; the layout is written
//! from the floor, every PE's kernel and resolved params ([`super::params::tile`]) grouped into
//! maximal rectangles, every param a literal (levers applied before, so the link takes none).

use std::collections::HashMap;
use std::path::Path;

use crate::compile::abi::{Kernels, Params};
use crate::compile::floor::{Coord, Rect};
use crate::compile::program::Program;
use crate::compile::role::Kind;
use crate::sources;

use super::placed::{At, Placed, Tile};

use super::params::{Levers, kernels, tile};

/// The empty kernel of PEs with no role (and of kinds a layout builds no kernel for).
const FILLER: &str = "param memcpy_params;
const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);
fn init() void { sys_mod.unblock_cmd_stream(); }
fn compute(steps: u32, start_pos: u32) void { sys_mod.unblock_cmd_stream(); }
comptime { @export_symbol(init); @export_symbol(compute); }
";

/// A kind's source as the build directory holds it: `body`, then a one-word dummy for every
/// exported name it does not own.
pub fn kind_source(body: &str, syms: &[(&str, &str)], owns: &[&str]) -> Result<String, String> {
    for o in owns {
        if !syms.iter().any(|(n, _)| n == o) {
            return Err(format!(
                "kernel owns {o}, not an exported name of its layout"
            ));
        }
    }
    let mut d = vec!["// the other programs' exported names (one-word dummies)".to_string()];
    for (n, t) in syms {
        if owns.contains(n) {
            continue;
        }
        d.push(format!(
            "var dd_{n} = @zeros([1]{t});\nvar dd_{n}_ptr: [*]{t} = &dd_{n};"
        ));
    }
    d.push("comptime {".to_string());
    for (n, _) in syms {
        if !owns.contains(n) {
            d.push(format!("  @export_symbol(dd_{n}_ptr, \"{n}\");"));
        }
    }
    d.push("}".to_string());
    Ok(format!("{body}\n{}\n", d.join("\n")))
}

/// Maximal rectangles of equal keys over a `w` x `h` key map (row runs, then runs merged down
/// while their x span and key repeat), in the order their first row appears.
pub fn rects(w: u32, h: u32, key: impl Fn(Coord) -> usize) -> Vec<(usize, Rect)> {
    let mut out: Vec<(usize, Rect)> = Vec::new();
    // (x0, x1, key) -> index in `out` of the rectangle ending at the previous row
    let mut open: HashMap<(u32, u32, usize), usize> = HashMap::new();
    for y in 0..h {
        let mut next = HashMap::new();
        let mut x = 0;
        while x < w {
            let k = key(Coord::new(x, y));
            let mut e = x + 1;
            while e < w && key(Coord::new(e, y)) == k {
                e += 1;
            }
            let i = match open.get(&(x, e, k)) {
                Some(&i) => {
                    out[i].1.h += 1;
                    i
                }
                None => {
                    out.push((
                        k,
                        Rect {
                            x,
                            y,
                            w: e - x,
                            h: 1,
                        },
                    ));
                    out.len() - 1
                }
            };
            next.insert((x, e, k), i);
            x = e;
        }
        open = next;
    }
    out
}

/// Every PE's tile: an index into the returned distinct (kind, params) list; PEs with no kernel
/// build get the filler (`None`).
#[allow(clippy::type_complexity)]
pub fn tiles(
    prog: &Program,
    kinds: &Kernels,
) -> Result<(Vec<Option<(Kind, Params)>>, Vec<usize>), String> {
    let (w, h) = (prog.floor.w, prog.floor.h);
    let mut distinct: Vec<Option<(Kind, Params)>> = vec![None];
    let mut index: HashMap<String, usize> = HashMap::new();
    let mut grid = vec![0usize; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let t = tile(prog, kinds, Coord::new(x, y))?;
            let i = match t {
                None => 0,
                Some((k, p)) => {
                    let key = format!("{}|{}", k.stem, p.render());
                    *index.entry(key).or_insert_with(|| {
                        distinct.push(Some((k, p)));
                        distinct.len() - 1
                    })
                }
            };
            grid[(y * w + x) as usize] = i;
        }
    }
    Ok((distinct, grid))
}

/// The placement of a program: its kinds' rectangles, every param a literal, no routes (a
/// decode program's are its rcfg's, set at its start).
pub fn placed(prog: &Program, kinds: &Kernels) -> Result<Placed, String> {
    let (w, h) = (prog.floor.w, prog.floor.h);
    let (distinct, grid) = tiles(prog, kinds)?;
    let mut pl = Placed::new(
        "param kernel_rows: u16 = 0;\nparam kernel_cols: u16 = 0;\nparam iters: u16 = 0;".into(),
        w,
        h,
    );
    pl.decode = true;
    for (i, r) in rects(w, h, |c| grid[(c.y * w + c.x) as usize]) {
        let (file, params) = match &distinct[i] {
            None => ("filler".to_string(), Vec::new()),
            Some((k, p)) => (
                k.stem.to_string(),
                p.items
                    .iter()
                    .map(|(n, v)| (n.to_string(), v.render()))
                    .collect(),
            ),
        };
        pl.tiles.push(Tile {
            at: At::Region((r.x, r.x + r.w, 1), (r.y, r.y + r.h, 1)),
            file: format!("{file}.csl"),
            params,
        });
    }
    for (n, t) in prog.symbols() {
        pl.export(n, t);
    }
    Ok(pl)
}

/// Writes `<out>/layout.csl`, one `<kind>.csl` a role kind (the kernel source, embedded or from
/// `kernels_dir`, with the other kinds' exported names as one-word dummies) and the op modules
/// under `<out>/ops`.
pub fn write(
    prog: &Program,
    levers: &Levers,
    kernels_dir: Option<&Path>,
    out: &Path,
) -> Result<(), String> {
    let kinds = kernels(prog, levers)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let syms = &prog.symbols();
    let own = std::cell::RefCell::new(Vec::new());
    let wr = |name: &str, text: &str| {
        let p = out.join(name);
        own.borrow_mut().push((name.to_string(), text.to_string()));
        std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))
    };
    for (k, kb) in &kinds {
        // (an empty source: the filler)
        let body = if kb.source.is_empty() {
            FILLER.to_string()
        } else {
            let text = match kernels_dir {
                Some(dir) => sources::read(dir, kb.source)?,
                None => sources::get(kb.source)
                    .ok_or_else(|| format!("no kernel {}", kb.source))?
                    .to_string(),
            };
            prog.source(kb.source, &text)
                .ok_or_else(|| format!("kernel {} not written", kb.source))?
        };
        wr(
            &format!("{}.csl", k.stem),
            &kind_source(&body, syms, &kb.owns)?,
        )?;
    }
    if !kinds.contains_key(&Kind::IDLE) {
        wr("filler.csl", &kind_source(FILLER, syms, &[])?)?;
    }
    // the op modules the kinds import (`ops/<name>`)
    let ops = out.join("ops");
    if !sources::OPS.is_empty() {
        std::fs::create_dir_all(&ops).map_err(|e| format!("{}: {e}", ops.display()))?;
    }
    for (name, src) in sources::OPS {
        let text = match kernels_dir {
            Some(dir) => sources::read(dir, name)?,
            None => (*src).to_string(),
        };
        if let Some(text) = prog.source(name, &text) {
            wr(&format!("ops/{name}"), &text)?;
        }
    }
    let pl = placed(prog, &kinds)?;
    wr("layout.csl", &pl.csl())?;
    if crate::compile::check::keeping() {
        let routes = crate::compile::check::Routes::of_rcfg(&super::rcfg::dump(prog)?, 0, 0)?;
        crate::compile::check::written(&pl, &own.borrow(), Some(routes));
        if prog.plan.seats > 1 {
            let late = crate::compile::check::hazard::late_inputs(prog, &kinds, &own.borrow())?;
            crate::compile::check::found(late);
        }
    }
    Ok(())
}
