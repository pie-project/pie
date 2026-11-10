//! The program's `layout.csl` and per-kind source files: the kind rectangles of the floor, each
//! with its kernel source and params, the memcpy frame. Replaces the layout writers of
//! `build.py` and the family `build_*.py` (their `layout.csl` and `<kind>.csl` outputs).
//!
//! The per-kind sources are the `build*.py` ones byte for byte (the kernel source, then one-word
//! dummies for every exported name the kind does not own, in the family's export order). The
//! layout is written from the floor instead of re-deriving it in CSL: every PE's kernel and
//! resolved params ([`super::params::tile`]) grouped into maximal rectangles, one loop nest a
//! rectangle, every param a literal. The levers are applied before (`wafer-compile build --levers`),
//! so the link takes no params.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

use crate::spatial::abi::{Kernels, Params};
use crate::spatial::floor::{Coord, Rect};
use crate::spatial::program::Program;
use crate::spatial::role::Kind;
use crate::spatial::sources;

use super::params::{Levers, kernels, tile};

/// The empty kernel of PEs with no role (and of kinds a family builds no kernel for).
const FILLER: &str = "param memcpy_params;
const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);
fn init() void { sys_mod.unblock_cmd_stream(); }
fn compute(steps: u32, start_pos: u32) void { sys_mod.unblock_cmd_stream(); }
comptime { @export_symbol(init); @export_symbol(compute); }
";

/// A kind's source as the build directory holds it: `body`, then a one-word dummy for every
/// exported name it does not own (`build*.py`'s text, byte for byte).
pub fn kind_source(body: &str, syms: &[(&str, &str)], owns: &[&str]) -> Result<String, String> {
    for o in owns {
        if !syms.iter().any(|(n, _)| n == o) {
            return Err(format!(
                "kernel owns {o}, not an exported name of its family"
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

/// The `layout.csl` text of a program.
pub fn layout_text(prog: &Program, kinds: &Kernels) -> Result<String, String> {
    let (w, h) = (prog.floor.w, prog.floor.h);
    let (distinct, grid) = tiles(prog, kinds)?;
    let rs = rects(w, h, |c| grid[(c.y * w + c.x) as usize]);
    let mut s = String::new();
    s.push_str("param kernel_rows: u16 = 0;\nparam kernel_cols: u16 = 0;\nparam iters: u16 = 0;\n");
    let _ = writeln!(s, "const W: u16 = {w};\nconst HH: u16 = {h};");
    s.push_str(
        "const memcpy = @import_module(\"<memcpy/get_params>\", .{ .width = W, .height = HH });\n",
    );
    s.push_str("layout {\n  @set_rectangle(W, HH);\n");
    for (i, r) in rs {
        let (file, args) = match &distinct[i] {
            None => ("filler".to_string(), String::new()),
            Some((k, p)) => (
                k.stem.to_string(),
                p.items
                    .iter()
                    .map(|(n, v)| format!(", .{n} = {}", v.render()))
                    .collect::<String>(),
            ),
        };
        let _ = writeln!(
            s,
            "  for (@range(u16, {}, {}, 1)) |x| {{\n    const mp = memcpy.get_params(x);\n    for (@range(u16, {}, {}, 1)) |y| {{\n      @set_tile_code(x, y, \"{file}.csl\", .{{ .memcpy_params = mp{args} }});\n    }}\n  }}",
            r.x,
            r.x + r.w,
            r.y,
            r.y + r.h,
        );
    }
    for (n, t) in prog.family.symbols() {
        let _ = writeln!(s, "  @export_name(\"{n}\", [*]{t}, true);");
    }
    s.push_str(
        "  @export_name(\"init\", fn()void);\n  @export_name(\"compute\", fn(u32, u32)void);\n}\n",
    );
    Ok(s)
}

/// Writes `<out>/layout.csl` and one `<kind>.csl` a role kind (the kernel source, embedded or from
/// `kernels_dir`, with the other kinds' exported names as one-word dummies), plus `dims.env` (the
/// rectangle and fabric dims, for the Python runners).
pub fn write(
    prog: &Program,
    levers: &Levers,
    kernels_dir: Option<&Path>,
    out: &Path,
) -> Result<(), String> {
    let kinds = kernels(prog, levers)?;
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let syms = prog.family.symbols();
    let wr = |name: &str, text: &str| {
        let p = out.join(name);
        std::fs::write(&p, text).map_err(|e| format!("{}: {e}", p.display()))
    };
    for (k, kb) in &kinds {
        // (an empty source: the filler)
        let body = if kb.source.is_empty() {
            FILLER.to_string()
        } else {
            match kernels_dir {
                Some(dir) => {
                    let src = dir.join(kb.source);
                    std::fs::read_to_string(&src)
                        .map_err(|e| format!("{}: {e}", src.display()))?
                }
                None => sources::get(kb.source)
                    .ok_or_else(|| format!("no kernel {}", kb.source))?
                    .to_string(),
            }
        };
        wr(
            &format!("{}.csl", k.stem),
            &kind_source(&body, syms, &kb.owns)?,
        )?;
    }
    if !kinds.contains_key(&Kind::IDLE) {
        wr("filler.csl", &kind_source(FILLER, syms, &[])?)?;
    }
    wr("layout.csl", &layout_text(prog, &kinds)?)?;
    let (w, h) = (prog.floor.w, prog.floor.h);
    wr(
        "dims.env",
        &format!("FW={} FH={} PW={w} PH={h}\n", w + 7, h + 2),
    )?;
    Ok(())
}
