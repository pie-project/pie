//! The placed program every regime's emitter states: the fabric's rectangle, each region of PEs
//! and the code it runs with its parameters, each colour's routes over a region, the symbols the
//! host reads and writes. One printer writes it as `layout.csl` ([`Placed::csl`]); the static
//! check reads it as it is ([`crate::compile::check::layout_ir`]).

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// A PE's code and routes (printed a PE at a time).
type PeEntries<'a> = (Vec<&'a Tile>, Vec<&'a Route>);

/// `[lo, hi)` by `step`.
pub type Span = (u32, u32, u32);

/// Where an entry applies: a region (its columns and rows, each a [`Span`]) or one PE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum At {
    Region(Span, Span),
    Pe(u32, u32),
}

/// A region's code: its file and its parameters past memcpy's (`name`, value as printed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tile {
    pub at: At,
    pub file: String,
    pub params: Vec<(String, String)>,
}

/// A colour's route over a region: what it receives from (`RAMP`, `WEST`, ...) and sends to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub at: At,
    pub colour: u32,
    pub rx: &'static str,
    pub tx: Vec<&'static str>,
}

/// A program's placement.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Placed {
    /// The comment opening `layout.csl` (what the program is).
    pub header: String,
    pub w: u32,
    pub h: u32,
    pub tiles: Vec<Tile>,
    pub routes: Vec<Route>,
    /// The host's symbols: (name, element type), `run` the launch after them.
    pub exports: Vec<(String, String)>,
    /// Printed a PE at a time (rows, then columns: each PE's code, then its routes), every
    /// entry one PE's; else the tiles, then the routes, as listed.
    pub by_pe: bool,
    /// The decode templates' frame (the size as `W` / `HH` consts, a region's loops a line
    /// each, `init` / `compute` the launches); else the stream regimes' (`run`).
    pub decode: bool,
}

impl Placed {
    #[must_use]
    pub fn new(header: String, w: u32, h: u32) -> Placed {
        Placed {
            header,
            w,
            h,
            ..Placed::default()
        }
    }

    /// `file` over `at` with `params`.
    pub fn tile(&mut self, at: At, file: &str, params: &[(&str, String)]) {
        self.tiles.push(Tile {
            at,
            file: file.to_string(),
            params: params
                .iter()
                .map(|(k, v)| ((*k).to_string(), v.clone()))
                .collect(),
        });
    }

    /// Colour `c` over `at` from `rx` to `tx` (none where the region or `tx` is empty).
    pub fn route(&mut self, c: u32, at: At, rx: &'static str, tx: &[&'static str]) {
        if let At::Region(xs, ys) = at
            && (xs.0 >= xs.1 || ys.0 >= ys.1)
        {
            return;
        }
        if tx.is_empty() {
            return;
        }
        self.routes.push(Route {
            at,
            colour: c,
            rx,
            tx: tx.to_vec(),
        });
    }

    pub fn export(&mut self, name: &str, ty: &str) {
        self.exports.push((name.to_string(), ty.to_string()));
    }

    /// `layout.csl`.
    #[must_use]
    pub fn csl(&self) -> String {
        let (w, h) = (self.w, self.h);
        let mut t = if self.decode {
            format!(
                "{}\nconst W: u16 = {w};\nconst HH: u16 = {h};\nconst memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = W, .height = HH }});\nlayout {{\n  @set_rectangle(W, HH);\n",
                self.header
            )
        } else {
            format!(
                "{}\nconst memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {w}, .height = {h} }});\nlayout {{\n  @set_rectangle({w}, {h});\n",
                self.header
            )
        };
        if self.by_pe {
            let mut at: BTreeMap<(u32, u32), PeEntries> = BTreeMap::new();
            for tl in &self.tiles {
                if let At::Pe(x, y) = tl.at {
                    at.entry((y, x)).or_default().0.push(tl);
                }
            }
            for r in &self.routes {
                if let At::Pe(x, y) = r.at {
                    at.entry((y, x)).or_default().1.push(r);
                }
            }
            for (tls, rs) in at.values() {
                for tl in tls {
                    tile_line(&mut t, tl);
                }
                for r in rs {
                    route_line(&mut t, r);
                }
            }
        } else {
            for tl in &self.tiles {
                if self.decode {
                    tile_lines(&mut t, tl);
                } else {
                    tile_line(&mut t, tl);
                }
            }
            for r in &self.routes {
                route_line(&mut t, r);
            }
        }
        for (n, ty) in &self.exports {
            let _ = writeln!(t, "  @export_name(\"{n}\", [*]{ty}, true);");
        }
        t.push_str(if self.decode {
            "  @export_name(\"init\", fn()void);\n  @export_name(\"compute\", fn(u32, u32)void);\n}\n"
        } else {
            "  @export_name(\"run\", fn() void);\n}\n"
        });
        t
    }

    /// Every PE an entry at `at` covers.
    pub fn pes(at: At) -> impl Iterator<Item = (u32, u32)> {
        let (xs, ys) = match at {
            At::Region(xs, ys) => (xs, ys),
            At::Pe(x, y) => ((x, x + 1, 1), (y, y + 1, 1)),
        };
        (xs.0..xs.1)
            .step_by(xs.2.max(1) as usize)
            .flat_map(move |x| {
                (ys.0..ys.1)
                    .step_by(ys.2.max(1) as usize)
                    .map(move |y| (x, y))
            })
    }
}

/// A tile's `@set_tile_code` line.
fn tile_line(t: &mut String, tl: &Tile) {
    let params: String = tl
        .params
        .iter()
        .map(|(k, v)| format!(", .{k} = {v}"))
        .collect();
    let _ = match tl.at {
        At::Region(xs, ys) => writeln!(
            t,
            "  for (@range(u16, {}, {}, {})) |x| {{ const mp = memcpy.get_params(x); for (@range(u16, {}, {}, {})) |y| {{ @set_tile_code(x, y, \"{}\", .{{ .memcpy_params = mp{params} }}); }} }}",
            xs.0, xs.1, xs.2, ys.0, ys.1, ys.2, tl.file
        ),
        At::Pe(x, y) => writeln!(
            t,
            "  @set_tile_code({x}, {y}, \"{}\", .{{ .memcpy_params = memcpy.get_params({x}){params} }});",
            tl.file
        ),
    };
}

/// A region's tile as the decode templates' frame writes it (its loops a line each).
fn tile_lines(t: &mut String, tl: &Tile) {
    let At::Region(xs, ys) = tl.at else {
        tile_line(t, tl);
        return;
    };
    let params: String = tl
        .params
        .iter()
        .map(|(k, v)| format!(", .{k} = {v}"))
        .collect();
    let _ = writeln!(
        t,
        "  for (@range(u16, {}, {}, {})) |x| {{\n    const mp = memcpy.get_params(x);\n    for (@range(u16, {}, {}, {})) |y| {{\n      @set_tile_code(x, y, \"{}\", .{{ .memcpy_params = mp{params} }});\n    }}\n  }}",
        xs.0, xs.1, xs.2, ys.0, ys.1, ys.2, tl.file
    );
}

/// A route's `@set_color_config` line.
fn route_line(t: &mut String, r: &Route) {
    let cfg = format!(
        "@get_color({}), .{{ .routes = .{{ .rx = .{{ {} }}, .tx = .{{ {} }} }} }}",
        r.colour,
        r.rx,
        r.tx.join(", ")
    );
    let _ = match r.at {
        At::Region(xs, ys) => writeln!(
            t,
            "  for (@range(u16, {}, {}, {})) |x| {{ for (@range(u16, {}, {}, {})) |y| {{ @set_color_config(x, y, {cfg}); }} }}",
            xs.0, xs.1, xs.2, ys.0, ys.1, ys.2
        ),
        At::Pe(x, y) => writeln!(t, "  @set_color_config({x}, {y}, {cfg});"),
    };
}
