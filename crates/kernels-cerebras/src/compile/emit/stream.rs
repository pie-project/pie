//! A stream program's CSL ([`crate::compile::stream::build::Program`]): `layout.csl`, a kernel file per
//! PE kind, each root region's printed module, and the arrays the host writes before a step and
//! reads after it ([`HostArray`]). Every root runs its lane vectors in, then its unit's schedule
//! of steps around the barrier; weight PEs run their matvecs a row (ops/smv.csl's `row`).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use poem_ir::ValueId;
use serde::Serialize;

use crate::compile::emit::placed::{At, Placed, Route, Tile as PlacedTile};
use crate::compile::emit::region::{self, ParamAt};
use crate::compile::fabric::{Endpoint, RouteWord};
use crate::compile::floor::{Coord, Dir};
use crate::compile::floorplan::stream::{
    self as fs, COL_CHAIN, HELP_CHAIN, KV_CHAIN, KV_RECV, KV_SEND, LANE_BUS, LANE_GATHER, PATH_IN,
    PATH_OUT, ROW_CHAIN, SEGMENT,
};
use crate::compile::stream::build::{Built, Program};
use crate::compile::stream::place::{self, Seg};
use crate::compile::stream::unit::{Cols, Reader, Side, Step};
use crate::compile::wiring::Slot;

type R<T> = Result<T, String>;

/// What the host writes into (or reads from) one PE's array.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Fill {
    /// Weight tile of unit `unit`'s step `step` at column `col`, weight row `row`.
    Tile {
        unit: usize,
        step: usize,
        col: u32,
        row: u32,
    },
    /// Region table `table` of unit `unit`'s step `step`, at root `r` of the region's roots.
    Table {
        unit: usize,
        step: usize,
        table: usize,
        r: u32,
    },
    /// A region's static resident parameter `param` (a shared weight), its chunk `r`.
    Resident {
        unit: usize,
        step: usize,
        param: usize,
        r: u32,
    },
    /// The scalars each own row of `unit` reads at a root: (step, param) in order.
    Scalars {
        unit: usize,
        of: Vec<(usize, usize)>,
    },
    /// The scalars of the own rows `from, from + step, ...` of `unit`, in a weight PE that adds
    /// them into its column chain for the CY root (row `t` by the PE of row `t % py`).
    RowScalars {
        unit: usize,
        of: Vec<(usize, usize)>,
        from: u32,
        step: u32,
    },
    /// The host rows a root takes in, a row of its own after another: its parts of `values`.
    HostIn {
        unit: usize,
        parts: Vec<(u32, u32, u32)>,
    },
    /// The rows a root leaves to the host: its parts (base, from, len) in order.
    HostOut {
        unit: usize,
        parts: Vec<(u32, u32, u32)>,
    },
    /// A row's argmax over a root's columns of the logits (`(from, len)` spans, in order): two
    /// words a row, the largest logit's bits and its column.
    HostArgmax { unit: usize, parts: Vec<(u32, u32)> },
    /// The concatenation of each `(value, from, len)`'s flat elements (a state column root's
    /// conv taps and decay parameters, its head's of the layer's).
    Slices(Vec<(u32, u32, u32)>),
    /// The stamps.
    Stamps,
    /// A PE's runtime config words (`pcfg`): what differs between PEs sharing one image (a
    /// root's lane drains, a weight column's index in the row store, a chain role).
    Words(Vec<u32>),
}

/// One PE array the host fills or reads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HostArray {
    pub at: Coord,
    pub sym: String,
    pub u16: bool,
    pub len: u32,
    pub fill: Fill,
}

/// The program's files.
#[derive(Clone, Debug)]
pub struct Emitted {
    pub files: Vec<(String, String)>,
    /// Its placement (`layout.csl`'s): what the static check reads.
    pub placed: super::placed::Placed,
    pub arrays: Vec<HostArray>,
    pub w: u32,
    pub h: u32,
}

fn dirname(d: Dir) -> &'static str {
    match d {
        Dir::W => "WEST",
        Dir::E => "EAST",
        Dir::S => "SOUTH",
        Dir::N => "NORTH",
    }
}

/// A route word's receive and sends, as CSL names them.
fn route_sets(w: RouteWord) -> (&'static str, Vec<&'static str>) {
    let rx = if w.from_ramp() {
        "RAMP"
    } else {
        dirname(match w.0 & 0x60 {
            0x00 => Dir::W,
            0x20 => Dir::E,
            0x40 => Dir::S,
            _ => Dir::N,
        })
    };
    let mut tx = Vec::new();
    for (bit, d) in [
        (0x01, Dir::W),
        (0x02, Dir::E),
        (0x04, Dir::S),
        (0x08, Dir::N),
    ] {
        if w.0 & bit != 0 {
            tx.push(dirname(d));
        }
    }
    if w.to_ramp() {
        tx.push("RAMP");
    }
    (rx, tx)
}

/// The placement of a program laid out a PE at a time (the stream and grid regimes): each
/// PE's code (a filler's where it has none) and its routes on `fabric`, the host's symbols
/// `names` (in name order).
pub(crate) fn placed_of(
    header: String,
    (w, h): (u32, u32),
    tiles: &Tiles,
    fabric: &crate::compile::fabric::Fabric,
    names: &BTreeMap<String, &str>,
) -> Placed {
    let mut pl = Placed::new(header, w, h);
    pl.by_pe = true;
    let filler = (String::from("filler.csl"), Vec::new());
    for y in 0..h {
        for x in 0..w {
            let c = Coord::new(x, y);
            let (file, params) = tiles.get(&c).unwrap_or(&filler);
            pl.tiles.push(PlacedTile {
                at: At::Pe(x, y),
                file: file.clone(),
                params: params
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_string()))
                    .collect(),
            });
            for rt in fabric.routes(c) {
                let (rx, tx) = route_sets(rt.word);
                pl.routes.push(Route {
                    at: At::Pe(x, y),
                    colour: u32::from(rt.colour.0),
                    rx,
                    tx,
                });
            }
        }
    }
    for (n, ty) in names {
        pl.export(n, ty);
    }
    pl
}

/// A buffer declaration and its DSD.
pub(crate) fn buf(name: &str, n: u32) -> String {
    let n = n.max(1);
    format!(
        "var {name}: [{n}]f32 align(16) = @zeros([{n}]f32);\nconst D_{name} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{n}}} -> {name}[i] }});\n"
    )
}

/// A drain of `n` words (a `u16` expression) off queue `q` through `D_drn`, 64 at a time, each
/// piece then `then` (with `m` its length).
pub(crate) fn drain(n: &str, q: &str, then: &str) -> String {
    format!(
        "  {{ const nd: u16 = {n}; var k: u16 = 0; while (k < nd) : (k += 64) {{ const m = if (nd - k > 64) 64 else nd - k; @fmovs(@set_dsd_length(D_drn, m), fab.fin({q}, m));{then} }} }}\n"
    )
}

/// The runtime config words' declaration and export (`pcfg`), empty where none.
pub(crate) fn cfg_decl(cfg: &[u32]) -> String {
    if cfg.is_empty() {
        return String::new();
    }
    format!(
        "var pcfg = @zeros([{}]u32);\nvar pcfg_p: [*]u32 = &pcfg;\n",
        cfg.len()
    )
}

/// The host array of a PE's config words, `None` where none.
fn cfg_array(at: Coord, cfg: &[u32]) -> Option<HostArray> {
    (!cfg.is_empty()).then(|| HostArray {
        at,
        sym: "pcfg".into(),
        u16: false,
        len: cfg.len() as u32,
        fill: Fill::Words(cfg.to_vec()),
    })
}

/// `n` words of DSD `d` from `off` (an expression).
pub(crate) fn at(d: &str, off: &str, n: u32) -> String {
    format!("@set_dsd_length(@increment_dsd_offset({d}, @as(i16, {off}), f32), {n})")
}

/// A value's columns where a root keeps them: `[from, from + len)` of `base` at DSD `dsd`
/// from `off` (an expression in the row `t` where held a row).
#[derive(Clone, Debug)]
struct Loc {
    base: u32,
    from: u32,
    len: u32,
    dsd: String,
    off: String,
    /// Read after the barrier (a held row), else before it.
    late: bool,
}

/// One root's code as it is generated.
#[derive(Default)]
struct Gen {
    decls: String,
    imports: String,
    early: String,
    late: String,
    /// The step being generated runs after the barrier.
    now_late: bool,
    lanes: String,
    end: String,
    locs: Vec<Loc>,
    exports: Vec<(String, String, bool)>,
    h16: u32,
    drain: bool,
    /// The row store: held rows cast to the weight PEs, and their fetches up the row chain at
    /// the start of each late row.
    cast: bool,
    fetch: String,
    /// The PE's runtime config words (`pcfg`), so roots that differ only in them share a file.
    cfg: Vec<u32>,
    /// The held buffers by (value, column): their names, numbered a value, not by column, so
    /// every root of a unit names them alike.
    holds: Vec<(u32, u32, String)>,
}

impl Gen {
    fn put(&mut self, late: bool, s: &str) {
        if late {
            self.late.push_str(s);
        } else {
            self.early.push_str(s);
        }
    }

    fn loc(&mut self, base: ValueId, from: u32, len: u32, dsd: &str, off: &str) {
        let late = self.now_late;
        self.locs.push(Loc {
            base: base.0,
            from,
            len,
            dsd: dsd.into(),
            off: off.into(),
            late,
        });
    }

    /// A held row, read after the barrier.
    fn held_loc(&mut self, base: ValueId, from: u32, len: u32, dsd: &str, off: &str) {
        self.locs.push(Loc {
            base: base.0,
            from,
            len,
            dsd: dsd.into(),
            off: off.into(),
            late: true,
        });
    }

    /// The pieces of `[from, from + n)` of `base`: (offset in the range, DSD expression, n).
    fn pieces(&self, base: ValueId, from: u32, n: u32) -> R<Vec<(u32, String, u32)>> {
        let mut out = Vec::new();
        let mut c = from;
        while c < from + n {
            let l = self
                .locs
                .iter()
                .rev()
                .find(|l| {
                    l.late == self.now_late && l.base == base.0 && c >= l.from && c < l.from + l.len
                })
                .ok_or_else(|| {
                    format!(
                        "value %{} column {c} is at no root buffer (late {}; locs {:?})",
                        base.0,
                        self.now_late,
                        self.locs
                            .iter()
                            .map(|l| (l.base, l.from, l.len, l.late))
                            .collect::<Vec<_>>()
                    )
                })?;
            let m = (l.from + l.len).min(from + n) - c;
            let off = if l.off == "0" {
                format!("{}", c - l.from)
            } else {
                format!("{} + {}", l.off, c - l.from)
            };
            out.push((c - from, at(&l.dsd, &off, m), m));
            c += m;
        }
        Ok(out)
    }

    /// A runtime config word: `v` at this PE, a `u16` expression in the file.
    fn word(&mut self, v: u32) -> String {
        self.cfg.push(v);
        format!("@as(u16, pcfg[{}])", self.cfg.len() - 1)
    }

    /// The name of `base`'s held buffer from column `from` (`true` when new).
    fn hold_name(&mut self, base: ValueId, from: u32) -> (String, bool) {
        if let Some(h) = self.holds.iter().find(|h| h.0 == base.0 && h.1 == from) {
            return (h.2.clone(), false);
        }
        let n = self.holds.iter().filter(|h| h.0 == base.0).count();
        let name = format!("hd{}_{n}", base.0);
        self.holds.push((base.0, from, name.clone()));
        (name, true)
    }

    fn round(&mut self, late: bool, d: &str, n: u32) {
        self.h16 = self.h16.max(n);
        self.put(
            late,
            &format!(
                "  @fs2h(@set_dsd_length(H16, {n}), {d}); @fh2s({d}, @set_dsd_length(H16, {n}));\n"
            ),
        );
    }
}

/// The colour a slot's endpoint at `c` sends / receives on.
fn ep(p: &Program, c: Coord, slot: Slot) -> Option<Endpoint> {
    p.fabric.io(c).slots.get(&slot).copied()
}

fn colour_of(p: &Program, name: &str) -> Option<u8> {
    p.palette
        .map
        .iter()
        .find(|(k, _)| k.name == name && k.index == 0)
        .map(|(_, c)| c.0)
}

/// Colours free at `c` (no route there): parks for unused input queues.
fn parks(p: &Program, c: Coord, n: usize, taken: &[u8]) -> Vec<u8> {
    let routed: Vec<u8> = p.fabric.routes(c).iter().map(|r| r.colour.0).collect();
    (0..crate::compile::fabric::COLOURS)
        .filter(|k| !routed.contains(k) && !taken.contains(k))
        .take(n)
        .collect()
}

/// The emitted program.
pub fn emit(p: &Program) -> R<Emitted> {
    let s = &p.stream;
    let mut files: Vec<(String, String)> = Vec::new();
    let mut arrays: Vec<HostArray> = Vec::new();
    // every PE's code: file, params
    let mut tiles: Tiles = BTreeMap::new();
    let trace_reader = |b: &Built| -> Reader<'_> {
        Reader {
            trace: p.shape_trace(),
            word: b.word,
            prologue: b.unit.classes.is_empty(),
        }
    };

    for (ui, b) in p.units.iter().enumerate() {
        let su = &s.units[ui];
        let r = trace_reader(b);
        // the weight columns
        for col in 0..su.cols() {
            let (name, text, mvs, cfg, wsc) = weight_file(p, ui, b, &r, col)?;
            files.push((name.clone(), text));
            for row in 0..su.rows() {
                let c = su.w(col, row);
                // (no row index: a column's PEs differ only in their chains' colours and their
                // config words, so they share a few images)
                put(&mut tiles, c, (name.clone(), weight_params(p, ui, c)))?;
                let mut words = cfg.clone();
                words[0] = weight_role(p, c);
                if !wsc.is_empty() {
                    words[1] = row;
                }
                arrays.extend(cfg_array(c, &words));
                for (sym, len, of) in &wsc {
                    arrays.push(HostArray {
                        at: c,
                        sym: sym.clone(),
                        u16: false,
                        len: *len,
                        fill: Fill::RowScalars {
                            unit: ui,
                            of: of.clone(),
                            from: row,
                            step: su.rows(),
                        },
                    });
                }
                for (k, (step, a, bb)) in mvs.iter().enumerate() {
                    arrays.push(HostArray {
                        at: c,
                        sym: format!("w{k}"),
                        u16: true,
                        len: a * bb + 1,
                        fill: Fill::Tile {
                            unit: ui,
                            step: *step,
                            col,
                            row,
                        },
                    });
                }
            }
        }
        // the CX roots
        // a file a root: its lane words are its own chunk's
        for j in 0..su.rows() {
            let (name, text, cx_arrays, regions) = root_file(p, ui, b, &r, Side::Row, j)
                .map_err(|e| format!("{}: CX root {j}: {e}", b.unit.name))?;
            files.push((name.clone(), text));
            for rg in regions {
                if !files.iter().any(|(n, _)| *n == rg.0) {
                    files.push(rg);
                }
            }
            let c = su.cx(j);
            put(&mut tiles, c, (name, root_params(p, ui, c, Side::Row)))?;
            for a in &cx_arrays {
                arrays.push(rebind(a, c, j, b));
            }
        }
        // the CY roots
        for i in 0..su.cols() {
            let (name, text, cy_arrays, regions) = root_file(p, ui, b, &r, Side::Col, i)
                .map_err(|e| format!("{}: CY root {i}: {e}", b.unit.name))?;
            files.push((name.clone(), text));
            files.extend(regions);
            let c = su.cy(i);
            put(&mut tiles, c, (name, root_params(p, ui, c, Side::Col)))?;
            for a in &cy_arrays {
                arrays.push(rebind(a, c, i, b));
            }
        }
        // the helpers below its rest roots
        if su.helpers > 0 {
            let (name, text) = helper_file(ui, b, &r)?;
            files.push((name.clone(), text));
            for i in su.heads()..su.cols() {
                for k in 0..su.helpers {
                    let c = su.helper(i, k);
                    let (send, recv, role) = chain_cols(p, c, HELP_CHAIN);
                    let hm = colour_of(p, &format!("HM{ui}")).map_or(0xff, u32::from);
                    let params = parked(
                        p,
                        c,
                        vec![
                            ("C_HM".into(), hm),
                            ("C_HS".into(), send),
                            ("C_HR".into(), recv),
                        ],
                    );
                    put(&mut tiles, c, (name.clone(), params))?;
                    arrays.extend(cfg_array(c, &[k + 1, role]));
                }
            }
        }
    }
    // the K / V columns: each its mixer's ([`crate::kernel::Column`])
    for (k, kc) in s.kv.iter().enumerate() {
        let ks = &p.kv[kc.block as usize];
        let col = crate::kernel::column(Some(&kv_joint(p, ks)))?;
        let (rname, rtext) = col.root_file(p, k, ks)?;
        files.push((rname.clone(), rtext));
        let root = kc.root(s);
        put(&mut tiles, root, (rname, kv_root_params(p, kc, root)))?;
        arrays.extend(col.root_arrays(p, kc, ks, root));
        let (pname, ptext) = col.pe_file(p, k, ks)?;
        files.push((pname.clone(), ptext));
        for i in 0..kc.pya {
            let c = kc.pe(s, i);
            let (params, role) = kv_pe_params(p, c);
            put(&mut tiles, c, (pname.clone(), params))?;
            arrays.extend(cfg_array(c, &[i, role]));
            arrays.extend(col.pe_arrays(p, c));
        }
    }
    // the lane root
    if let Some(lb) = &s.lanes {
        let lu = &s.units[lb.unit];
        let c = Coord::new(lu.x, lu.y + 1 + s.py);
        let gin = match ep(p, c, LANE_GATHER) {
            Some(Endpoint::Root(r)) => r.hi.map_or(0, |c| c.0),
            _ => return Err("stream: the lane root gathers nothing".into()),
        };
        let bus = match ep(p, c, LANE_BUS) {
            Some(Endpoint::Send(c)) => c.0,
            _ => return Err("stream: the lane root sends no bus".into()),
        };
        files.push(("lane_root.csl".into(), lane_root_file(p.lane_words)));
        put(
            &mut tiles,
            c,
            (
                "lane_root.csl".into(),
                vec![
                    ("C_GI".into(), u32::from(gin)),
                    ("C_BO".into(), u32::from(bus)),
                ],
            ),
        )?;
    }
    files.push(("filler.csl".into(), filler_file()));
    for y in 0..s.h {
        for x in 0..s.w {
            arrays.push(HostArray {
                at: Coord::new(x, y),
                sym: "tsc".into(),
                u16: false,
                len: 3,
                fill: Fill::Stamps,
            });
        }
    }
    dedup(&mut files, &mut tiles);
    let pl = layout_placed(p, &tiles, &arrays);
    files.push(("layout.csl".into(), pl.csl()));
    Ok(Emitted {
        files,
        placed: pl,
        arrays,
        w: s.w,
        h: s.h,
    })
}

/// Merges files that differ only in comments, regions first, renaming every PE's code: cslc
/// builds an image per distinct (file, params), so this keeps the image count to PE kinds.
pub(crate) fn dedup(files: &mut Vec<(String, String)>, tiles: &mut Tiles) {
    let key = |t: &str| -> String {
        t.lines()
            .filter(|l| !l.starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let merge = |files: &mut Vec<(String, String)>, pick: &dyn Fn(&str) -> bool| {
        let mut first: BTreeMap<String, String> = BTreeMap::new();
        let mut to: BTreeMap<String, String> = BTreeMap::new();
        files.retain(|(n, t)| {
            if !pick(n) {
                return true;
            }
            match first.entry(key(t)) {
                std::collections::btree_map::Entry::Occupied(e) => {
                    to.insert(n.clone(), e.get().clone());
                    false
                }
                std::collections::btree_map::Entry::Vacant(e) => {
                    e.insert(n.clone());
                    true
                }
            }
        });
        to
    };
    let regions = merge(files, &|n| n.starts_with("rg_"));
    if !regions.is_empty() {
        for (n, t) in files.iter_mut() {
            if n.starts_with("rg_") || !t.contains("\"rg_") {
                continue;
            }
            for (from, to) in &regions {
                let q = format!("\"{from}\"");
                if t.contains(&q) {
                    *t = t.replace(&q, &format!("\"{to}\""));
                }
            }
        }
    }
    let kernels = merge(files, &|n| !n.starts_with("rg_"));
    for (f, _) in tiles.values_mut() {
        if let Some(to) = kernels.get(f) {
            f.clone_from(to);
        }
    }
}

/// The chunk root `idx` runs of the region at `step`: its index from the region's first root,
/// or the replicas' chunk for a region whose replicas across roots run the same chunk.
fn chunk_at(b: &Built, step: usize, idx: u32) -> u32 {
    b.regions.get(&step).map_or(idx, |(at, _, _)| {
        let cut = matches!(&b.unit.steps[step], Step::Region { chain, .. } if at.chunk < chain.region.width);
        if cut && at.reps == 1 {
            at.chunk_of(idx - at.first)
        } else {
            idx - at.first
        }
    })
}

/// An array a root file declares, at root `idx` of its side.
fn rebind(a: &HostArray, c: Coord, idx: u32, b: &Built) -> HostArray {
    let mut out = a.clone();
    out.at = c;
    if let Fill::Table {
        unit, step, table, ..
    } = a.fill
    {
        let ri = chunk_at(b, step, idx);
        // a region whole at each root (a head region) has one chunk's tables
        let one = b
            .regions
            .get(&step)
            .is_some_and(|(_, pr, _)| pr.tables.get(table).is_some_and(|t| t.of_chunk.len() == 1));
        out.fill = Fill::Table {
            unit,
            step,
            table,
            r: if one { 0 } else { ri },
        };
    }
    if let Fill::Resident {
        unit, step, param, ..
    } = a.fill
    {
        let ri = chunk_at(b, step, idx);
        out.fill = Fill::Resident {
            unit,
            step,
            param,
            r: ri,
        };
    }
    out
}

fn chain_cols(p: &Program, c: Coord, slot: Slot) -> (u32, u32, u32) {
    // (send, recv, role); recv / send 0xff where none
    match ep(p, c, slot) {
        Some(Endpoint::Member(m)) => (
            u32::from(m.send.0),
            m.recv1.map_or(0xff, |c| u32::from(c.0)),
            m.role,
        ),
        Some(Endpoint::Root(r)) => (0xff, r.lo.or(r.hi).map_or(0xff, |c| u32::from(c.0)), 6),
        _ => (0xff, 0xff, 0),
    }
}

/// Fills `0xff` (none) with park colours free at `c`.
fn parked(p: &Program, c: Coord, mut v: Vec<(String, u32)>) -> Vec<(String, u32)> {
    let taken: Vec<u8> = v
        .iter()
        .filter(|(k, x)| k.starts_with("C_") && *x != 0xff)
        .map(|(_, x)| *x as u8)
        .collect();
    let need = v
        .iter()
        .filter(|(k, x)| k.starts_with("C_") && *x == 0xff)
        .count();
    let mut free = parks(p, c, need, &taken).into_iter();
    for (k, x) in &mut v {
        if k.starts_with("C_") && *x == 0xff {
            *x = u32::from(free.next().unwrap_or(20));
        }
    }
    v
}

/// Weight PE `c` of unit `u`: its chains' colours and its unit's casts (the unit by index, not
/// by `c`'s column: a later band's units share their columns with the first band's).
fn weight_params(p: &Program, u: usize, c: Coord) -> Vec<(String, u32)> {
    let (cs, cr, _) = chain_cols(p, c, COL_CHAIN);
    let (rs, rr, _) = chain_cols(p, c, ROW_CHAIN);
    let xr = colour_of(p, &format!("XR{u}")).map_or(0xff, u32::from);
    let xc = colour_of(p, &format!("XC{u}")).map_or(0xff, u32::from);
    parked(
        p,
        c,
        vec![
            ("C_XR".into(), xr),
            ("C_XC".into(), xc),
            ("C_CS".into(), cs),
            ("C_CR".into(), cr),
            ("C_RS".into(), rs),
            ("C_RR".into(), rr),
        ],
    )
}

/// A weight PE's column-chain role (its `pcfg[0]`).
fn weight_role(p: &Program, c: Coord) -> u32 {
    chain_cols(p, c, COL_CHAIN).2
}

/// A weight PE's matvec: (step, tile rows, tile columns).
type Tile = (usize, u32, u32);

/// A root kernel: (file name, text, the arrays the host fills, its printed regions).
type RootFile = (String, String, Vec<HostArray>, Vec<(String, String)>);

/// Every PE's kernel file and params.
pub(crate) type Tiles = BTreeMap<Coord, (String, Vec<(String, u32)>)>;

/// `tile` at `c`, refused where another block's PE is already there: overlapping blocks route
/// and check clean but leave the earlier block a PE short, which hangs the chip.
fn put(tiles: &mut Tiles, c: Coord, tile: (String, Vec<(String, u32)>)) -> R<()> {
    if let Some((was, _)) = tiles.get(&c) {
        return Err(format!(
            "stream: ({}, {}) is {was}'s and {}'s: two blocks overlap",
            c.x, c.y, tile.0
        ));
    }
    tiles.insert(c, tile);
    Ok(())
}

/// A weight column's file: its matvecs (step, tile rows, tile cols) in step order.
/// The values a step makes that its row roots hold over the barrier (in the order the roots
/// hold them: the row in, a region's results, a matvec's output).
fn held_rows(b: &Built, st: &Step) -> Vec<ValueId> {
    let u = &b.unit;
    let made: Vec<ValueId> = match st {
        Step::In {
            value,
            side: Side::Row,
            host: false,
        } => vec![*value],
        Step::Region {
            side: Side::Row,
            chain,
        } => chain.outs.iter().map(|o| o.value).collect(),
        Step::Matvec {
            from: Side::Col,
            output,
            ..
        } => vec![*output],
        _ => Vec::new(),
    };
    made.into_iter().filter(|v| u.held.contains(v)).collect()
}

/// A root's per-row scalars it holds for every row at most (f32 words); past it the CY roots'
/// scalars come down the column chains with a matvec's output instead ([`carry`]).
pub const SCAL_WORDS: u32 = 2_048;

/// Whether region `printed`'s row reads its scalar parameter `pk` (a scalar it never reads,
/// such as a lane id that picked its lane vector, is not held).
fn reads_scalar(printed: &str, pk: usize) -> bool {
    let Some(at) = printed.find("fn row() void {") else {
        return true;
    };
    let name = format!("sc{pk}");
    printed[at..].match_indices(&name).any(|(i, _)| {
        let after = printed[at + i + name.len()..].chars().next();
        !after.is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            && !printed[..at + i]
                .chars()
                .last()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
    })
}

/// The CY roots' scalars carried down the column chains, a phase each: the matvec whose output
/// carries them and the (step, param)s of the later column regions that read one. Empty where
/// the CY roots hold their scalars within [`SCAL_WORDS`].
fn carry(p: &Program, b: &Built) -> Vec<(usize, Vec<(usize, usize)>)> {
    let u = &b.unit;
    let col_scalars = |lo: usize, hi: usize| -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for (si, step) in u.steps.iter().enumerate().take(hi).skip(lo) {
            let Step::Region {
                side: Side::Col,
                chain,
            } = step
            else {
                continue;
            };
            let Some((_, printed, params)) = b.regions.get(&si) else {
                continue;
            };
            for (pk, at) in params.iter().enumerate().take(chain.ins.len()) {
                if *at == ParamAt::Scalar && reads_scalar(&printed.csl, pk) {
                    v.push((si, pk));
                }
            }
        }
        v
    };
    let all = col_scalars(0, u.steps.len()).len() as u32;
    if all * u.rows <= SCAL_WORDS || u.classes.is_empty() {
        return Vec::new();
    }
    let _ = p;
    let mut out = Vec::new();
    for (lo, hi) in [(0, u.split), (u.split, u.steps.len())] {
        let Some(m) = (lo..hi).find(|&i| {
            matches!(
                u.steps[i],
                Step::Matvec {
                    from: Side::Row,
                    ..
                }
            )
        }) else {
            continue;
        };
        let v = col_scalars(m + 1, hi);
        if !v.is_empty() {
            out.push((m, v));
        }
    }
    out
}

/// A weight column's file, its matvecs and its config words after `pcfg[0]` (the column's
/// index in the row store and its chains' roles), so a unit's columns share a few files.
#[allow(clippy::type_complexity)]
fn weight_file(
    p: &Program,
    ui: usize,
    b: &Built,
    r: &Reader,
    col: u32,
) -> R<(
    String,
    String,
    Vec<Tile>,
    Vec<u32>,
    Vec<(String, u32, Vec<(usize, usize)>)>,
)> {
    let name = format!("w_{ui}_{col}.csl");
    let l = &b.layout;
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// {}: weight column {col} (kernels-cerebras emit::stream). Matvecs over the unit's rows,\n// ops/smv.csl's row a step.",
        b.unit.name
    );
    t.push_str("param memcpy_params;\nparam C_XR: u16;\nparam C_XC: u16;\nparam C_CS: u16;\nparam C_CR: u16;\nparam C_RS: u16;\nparam C_RR: u16;\n");
    t.push_str(COMMON);
    t.push_str("const q_xr = @get_input_queue(2);\nconst q_xc = @get_input_queue(3);\nconst q_cr = @get_input_queue(4);\nconst q_rr = @get_input_queue(5);\nconst q_cs = @get_output_queue(2);\nconst q_rs = @get_output_queue(3);\n");
    t.push_str("const rs = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_cr, .OUT = q_cs });\n");
    t.push_str("const rw = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_rr, .OUT = q_rs });\n");
    let mut mvs = Vec::new();
    let (mut early, mut late) = (String::new(), String::new());
    let (mut drn, mut zer) = (1u32, 1u32);
    // the row store: this column keeps rows t with t % px == col, each its held chunks in order
    let spill = l.spill;
    let mut kept: Vec<u32> = Vec::new();
    let mut fetch = String::new();
    // pcfg[0] the PE's R_COL (and pcfg[1] its row where it carries the CY roots' scalars); the
    // column's words after it
    let carried = carry(p, b);
    let mut cfg: Vec<u32> = if carried.is_empty() {
        vec![0]
    } else {
        vec![0, 0]
    };
    let mut wsc = Vec::new();
    for (ph, (_, of)) in carried.iter().enumerate() {
        let n = of.len() as u32;
        let len = b.unit.rows.div_ceil(l.py) * n;
        let _ = writeln!(
            t,
            "var wsc{ph} = @zeros([{len}]f32);\nvar wsc{ph}_p: [*]f32 = &wsc{ph};"
        );
        wsc.push((format!("wsc{ph}"), len, of.clone()));
    }
    let mut word = |v: u32| {
        cfg.push(v);
        format!("@as(u16, pcfg[{}])", cfg.len() - 1)
    };
    // the columns east of this one on the fabric (the row chain's order: a unit's head columns
    // may sit apart, `StreamUnit::col`)
    let su = &p.stream.units[ui];
    let east: Vec<u32> = (0..l.px()).filter(|&c2| su.col(c2) > su.col(col)).collect();
    let keeper = if spill { word(col) } else { String::new() };
    let fetch_role = if spill {
        word(u32::from(!east.is_empty()))
    } else {
        String::new()
    };
    for (si, st) in b.unit.steps.iter().enumerate() {
        // (kept after the step's own matvec: a held matvec output is cast once summed)
        let mut keep = String::new();
        if spill && si < b.unit.split {
            for v in held_rows(b, st) {
                let len = r.width(v).div_ceil(l.py);
                let off: u32 = kept.iter().sum();
                kept.push(len);
                drn = drn.max(len);
                zer = zer.max(len);
                let _ = writeln!(
                    keep,
                    "  if (t % {px} == {keeper}) {{ @fmovs(@set_dsd_length(@increment_dsd_offset(D_hs, @as(i16, (t / {px}) * HSW + {off}), f32), {len}), fab.fin(q_xr, {len})); }} else {{ @fmovs(@set_dsd_length(D_drn, {len}), fab.fin(q_xr, {len})); }}",
                    px = l.px()
                );
                let _ = writeln!(
                    fetch,
                    "  rw.role = {fetch_role}; if (t % {px} == {keeper}) {{ rw.send(@set_dsd_length(@increment_dsd_offset(D_hs, @as(i16, (t / {px}) * HSW + {off}), f32), {len}), {len}); }} else {{ rw.send(@set_dsd_length(D_zer, {len}), {len}); }}",
                    px = l.px()
                );
            }
        }
        let Step::Matvec {
            from, x32, n, k, ..
        } = st
        else {
            early.push_str(&keep);
            continue;
        };
        let body = if si < b.unit.split {
            &mut early
        } else {
            &mut late
        };
        match place::tile_dims(st, l, col) {
            Some((a, bb)) => {
                let m = mvs.len();
                let (red, q) = if *from == Side::Row {
                    ("rs", "q_xr")
                } else {
                    ("rw", "q_xc")
                };
                let _ = writeln!(
                    t,
                    "const m{m} = @import_module(\"ops/smv.csl\", .{{ .fab = fab, .red = {red}, .XQ = {q}, .Hc = {a}, .Nc = {bb}, .X32 = {x32} }});\nvar w{m}_p: [*]u16 = &m{m}.w;"
                );
                if *from == Side::Col {
                    // the leaf: the easternmost column holding the matvec
                    let leaf = east.iter().all(|&c2| place::tile_dims(st, l, c2).is_none());
                    let _ = writeln!(body, "  rw.role = {};", word(u32::from(!leaf)));
                }
                let _ = writeln!(body, "  m{m}.row(t);");
                if let Some(ph) = carried.iter().position(|c| c.0 == si) {
                    // the row's scalars: this PE's where it keeps row t, zeros from the rest
                    let n = carried[ph].1.len() as u32;
                    zer = zer.max(n);
                    let _ = writeln!(
                        body,
                        "  if (t % {py} == @as(u16, pcfg[1])) {{ rs.send(@set_dsd_length(@increment_dsd_offset(D_wsc{ph}, @as(i16, (t / {py}) * {n}), f32), {n}), {n}); }} else {{ rs.send(@set_dsd_length(D_zer, {n}), {n}); }}",
                        py = l.py
                    );
                    if !t.contains(&format!("const D_wsc{ph}")) {
                        let _ = writeln!(
                            t,
                            "const D_wsc{ph} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{}}} -> wsc{ph}[i] }});",
                            wsc[ph].1
                        );
                    }
                }
                mvs.push((si, a, bb));
            }
            None => match from {
                Side::Row => {
                    let hc = k.div_ceil(l.py);
                    drn = drn.max(hc);
                    let _ = writeln!(
                        body,
                        "  @fmovs(@set_dsd_length(D_drn, {hc}), fab.fin(q_xr, {hc}));"
                    );
                }
                Side::Col => {
                    if east.iter().any(|&c2| place::tile_dims(st, l, c2).is_some()) {
                        let no = n.div_ceil(l.py);
                        zer = zer.max(no);
                        let _ = writeln!(
                            body,
                            "  rw.role = 1; rw.send(@set_dsd_length(D_zer, {no}), {no});"
                        );
                    }
                }
            },
        }
        early.push_str(&keep);
    }
    t.push_str(&buf("drn", drn));
    t.push_str(&buf("zer", zer));
    t.push_str(&cfg_decl(&cfg));
    if !kept.is_empty() {
        let hsw: u32 = kept.iter().sum();
        let _ = writeln!(t, "const HSW: u16 = {hsw};");
        t.push_str(&buf("hs", hsw * b.unit.rows.div_ceil(l.px())));
    }
    let _ = write!(
        t,
        "fn early(t: u16) void {{\n{early}}}\nfn late(t: u16) void {{\n{fetch}{late}}}\n"
    );
    let ne = b.unit.rows;
    let nl = if b.unit.split < b.unit.steps.len() {
        b.unit.rows
    } else {
        0
    };
    let _ = write!(
        t,
        "task go() void {{\n  stamp(0);\n  rs.role = @as(u16, pcfg[0]);\n  var t: u16 = 0;\n  while (t < {ne}) : (t += 1) {{ early(t); }}\n  t = 0;\n  while (t < {nl}) : (t += 1) {{ late(t); }}\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}}\n"
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n");
    for m in 0..mvs.len() {
        let _ = writeln!(t, "  @export_symbol(w{m}_p, \"w{m}\");");
    }
    for (sym, _, _) in &wsc {
        let _ = writeln!(t, "  @export_symbol({sym}_p, \"{sym}\");");
    }
    t.push_str("  @initialize_queue(q_xr, .{ .color = @get_color(C_XR) });\n  @initialize_queue(q_xc, .{ .color = @get_color(C_XC) });\n  @initialize_queue(q_cr, .{ .color = @get_color(C_CR) });\n  @initialize_queue(q_rr, .{ .color = @get_color(C_RR) });\n  @initialize_queue(q_cs, .{ .color = @get_color(C_CS) });\n  @initialize_queue(q_rs, .{ .color = @get_color(C_RS) });\n}\n");
    Ok((name, t, mvs, cfg, wsc))
}

/// Every kernel's opening: memcpy, the stamps, the fabric and math modules, the run task.
pub(crate) const COMMON: &str = "const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);
const timestamp = @import_module(\"<time>\");
const fab = @import_module(\"ops/fabric.csl\");
const math = @import_module(\"ops/math.csl\");
var tsc = @zeros([3]u32);
var ts = @zeros([3]u16);
var tsc_p: [*]u32 = &tsc;
fn stamp(i: u16) void { timestamp.get_timestamp(&ts); tsc[i] = @as(u32, ts[0]) | (@as(u32, ts[1]) << 16); }
const T_GO: local_task_id = @get_local_task_id(8);
fn run() void { timestamp.enable_tsc(); @activate(T_GO); }
";

fn root_params(p: &Program, ui: usize, c: Coord, side: Side) -> Vec<(String, u32)> {
    let opt = |e: Option<Endpoint>, send: bool| -> u32 {
        match e {
            Some(Endpoint::Send(c)) if send => u32::from(c.0),
            Some(Endpoint::Recv(c)) if !send => u32::from(c.0),
            Some(Endpoint::Turn(a, b)) => u32::from(if send { b.0 } else { a.0 }),
            _ => 0xff,
        }
    };
    let lb = opt(ep(p, c, LANE_BUS), false);
    let v = match side {
        Side::Row => {
            let rr = match ep(p, c, ROW_CHAIN) {
                Some(Endpoint::Root(r)) => r.lo.or(r.hi).map_or(0xff, |c| u32::from(c.0)),
                _ => 0xff,
            };
            let xr = colour_of(p, &format!("XR{ui}")).map_or(0xff, u32::from);
            let mut v = vec![
                ("C_PI".into(), opt(ep(p, c, PATH_IN), false)),
                ("C_PO".into(), opt(ep(p, c, PATH_OUT), true)),
                ("C_RR".into(), rr),
                ("C_XR".into(), xr),
                ("C_LB".into(), lb),
            ];
            v.extend(seg_params(p, c, &format!("SM{ui}")));
            v
        }
        Side::Col => {
            let cr = match ep(p, c, COL_CHAIN) {
                Some(Endpoint::Root(r)) => r.lo.or(r.hi).map_or(0xff, |c| u32::from(c.0)),
                _ => 0xff,
            };
            let xc = colour_of(p, &format!("XC{ui}")).map_or(0xff, u32::from);
            let (gs, gr, _) = chain_cols(p, c, LANE_GATHER);
            let mut v = vec![
                ("C_CR".into(), cr),
                ("C_XC".into(), xc),
                ("C_KS".into(), opt(ep(p, c, KV_SEND), true)),
                ("C_KR".into(), opt(ep(p, c, KV_RECV), false)),
                ("C_LB".into(), lb),
                ("C_GS".into(), gs),
                ("C_GR".into(), gr),
            ];
            // a member of the CY roots' segment: its colours as a CX root's
            if ep(p, c, SEGMENT).is_some() {
                v.extend(seg_params(p, c, &format!("YM{ui}")));
            }
            // a rest root over helpers: their chain's cast out, their sums in
            if let Some(Endpoint::Root(r)) = ep(p, c, HELP_CHAIN) {
                let hm = colour_of(p, &format!("HM{ui}")).map_or(0xff, u32::from);
                v.push(("C_HM".into(), hm));
                v.push((
                    "C_HR".into(),
                    r.lo.or(r.hi).map_or(0xff, |c| u32::from(c.0)),
                ));
            }
            v
        }
    };
    parked(p, c, v)
}

/// A segment member's colours and role (ops/seg.csl's): `cast` its multicast's class.
fn seg_params(p: &Program, c: Coord, cast: &str) -> Vec<(String, u32)> {
    let (ss, sa, role) = chain_cols(p, c, SEGMENT);
    let sm = colour_of(p, cast).map_or(0xff, u32::from);
    // the root sends the cast; a member receives it
    let (ss, sm) = if role == 6 { (sm, 0xff) } else { (ss, sm) };
    vec![
        ("C_SA".into(), sa),
        ("C_SM".into(), sm),
        ("C_SS".into(), ss),
        ("S_ROLE".into(), role),
    ]
}

/// A unit's CX roots' file (`side` Row) or column root `idx`'s (Col), its arrays the host
/// fills, its printed regions.
#[allow(clippy::too_many_lines)]
/// The value a decode program's roots reduce to its argmax a row: the widest the host reads
/// back (the logits); none in a stream reading.
fn argmax_of(p: &Program) -> Option<ValueId> {
    p.shape.cache?;
    p.units
        .iter()
        .flat_map(|b| &b.unit.steps)
        .filter_map(|s| match s {
            Step::Out {
                value, host: true, ..
            } => Some(*value),
            _ => None,
        })
        .max_by_key(|c| c.width)
        .map(|c| c.base)
}

fn root_file(p: &Program, ui: usize, b: &Built, r: &Reader, side: Side, idx: u32) -> R<RootFile> {
    let u = &b.unit;
    let l = &b.layout;
    let lane_unit = u.classes.is_empty();
    let name = match side {
        Side::Row => format!("cx_{ui}_{idx}.csl"),
        Side::Col => format!("cy_{ui}_{idx}.csl"),
    };
    let mut g = Gen::default();
    let mut arrays: Vec<HostArray> = Vec::new();
    let mut regions: Vec<(String, String)> = Vec::new();
    // the root's scratch pool: a row a name, its element type and the widest chunk
    let mut pool: BTreeMap<String, (&'static str, u32)> = BTreeMap::new();
    let rows = u.rows;
    let held = |v: ValueId| u.held.contains(&v);
    // a row-side value's chunk at this root
    let hc = |w: u32| w.div_ceil(l.py);
    // this root's columns of `base` (Row: its chunk; Col: its slices)
    let mine = |base: ValueId, width: u32| -> Vec<(u32, u32)> {
        match side {
            Side::Row => vec![(idx * hc(width), hc(width))],
            Side::Col => l
                .parts(
                    idx,
                    Cols {
                        base,
                        from: 0,
                        width: u32::MAX / 2,
                    },
                )
                .iter()
                .map(|(_, s)| (s.from, s.len))
                .collect(),
        }
    };
    let my_cols = |c: Cols| -> Vec<(u32, u32)> {
        match side {
            Side::Row => {
                let w = hc(c.width);
                vec![(c.from + idx * w, w)]
            }
            Side::Col => l
                .parts(idx, c)
                .iter()
                .map(|(_, s)| (s.from, s.len))
                .collect(),
        }
    };
    let _ = my_cols;
    // per-row buffers and held rows
    let spill = side == Side::Row && l.spill;
    let hold = |g: &mut Gen, late: bool, base: ValueId, from: u32, len: u32, src: &str| {
        let (hname, new) = g.hold_name(base, from);
        if spill {
            // the row store: out to the row's weight PEs now, back up the row chain (the
            // keeper's chunk, zeros from the rest) at the start of the late row
            if new {
                g.decls.push_str(&buf(&hname, len));
                let _ = writeln!(g.fetch, "  @fmovs(D_{hname}, fab.fin(q_ci, {len}));");
            }
            g.put(late, &format!("  @fmovs(fab.fout(q_xr, {len}), {src});\n"));
            g.cast = true;
            g.held_loc(base, from, len, &format!("D_{hname}"), "0");
            return;
        }
        if new {
            g.decls.push_str(&buf(&hname, rows * len));
        }
        g.put(
            late,
            &format!(
                "  @fmovs({}, {src});\n",
                at(&format!("D_{hname}"), &format!("t * {len}"), len)
            ),
        );
        g.held_loc(
            base,
            from,
            len,
            &format!("D_{hname}"),
            &format!("t * {len}"),
        );
    };
    let mut hin_w = 0u32;
    let mut hin_parts: Vec<(u32, u32, u32)> = Vec::new();
    let mut hout_w = 0u32;
    let mut hout_parts: Vec<(u32, u32, u32)> = Vec::new();
    // the logits' spans in `hout` (offset, len, column): a decode step's argmax a row
    let mut amax: Vec<(u32, u32, u32)> = Vec::new();
    let mut scal: Vec<(usize, usize)> = Vec::new();
    let carried = if side == Side::Col {
        carry(p, b)
    } else {
        Vec::new()
    };
    let mut lane_pieces: Vec<(u32, u32, String)> = Vec::new(); // (bus off, n, dst dsd)
    // a region over helpers: its row params (index, row of hx) and read scalars
    let mut hrow: Vec<(usize, u32)> = Vec::new();
    let mut hscal: Vec<usize> = Vec::new();
    let mut helped = false;
    // a root kernel's rows (`rowp`) declared at this root
    let mut root_rows = false;
    let mut sends_cast = false;
    let mut kv_io = false;
    let mut chain_in = false;
    let mut path_out = false;
    let mut path_in = false;
    let mut slot = 0usize;
    // the region slots, by step
    let mut slot_of: BTreeMap<usize, usize> = BTreeMap::new();
    // head regions alike at this side's roots (q's heads and k's: one rotation's tables) share
    // one module, each taking its residents in before its heads: step -> the first's step
    let mut alike: BTreeMap<usize, usize> = BTreeMap::new();
    {
        let mut firsts: Vec<usize> = Vec::new();
        for (si, st) in u.steps.iter().enumerate() {
            let Step::Region { side: s2, chain } = st else {
                continue;
            };
            let Some((ra, _, params)) = b.regions.get(&si) else {
                continue;
            };
            let whole = ra.reps > 1 || (ra.chunk == chain.region.width && ra.roots > 1);
            if *s2 != side || !whole {
                continue;
            }
            let same = firsts.iter().copied().find(|&f| {
                let (fa, _, fp) = &b.regions[&f];
                matches!(&u.steps[f], Step::Region { chain: c, .. } if c.region == chain.region)
                    && fa.first == ra.first
                    && fa.roots == ra.roots
                    && fa.chunk == ra.chunk
                    && fp == params
            });
            match same {
                Some(f) => {
                    alike.insert(si, f);
                    alike.insert(f, f);
                }
                None => firsts.push(si),
            }
        }
    }

    for (si, st) in u.steps.iter().enumerate() {
        let late = si >= u.split;
        g.now_late = late;
        match st {
            Step::In {
                value,
                side: s2,
                host,
            } if *s2 == side => {
                let w = r.width(*value);
                if *host {
                    for (from, len) in mine(*value, w) {
                        hin_parts.push((value.0, from, len));
                        g.loc(*value, from, len, "D_hin", &format!("t * HINW + {hin_w}"));
                        if held(*value) {
                            // the host's rows stay whole: read again after the barrier
                            g.held_loc(*value, from, len, "D_hin", &format!("t * HINW + {hin_w}"));
                        }
                        hin_w += len;
                    }
                } else {
                    let n = hc(w);
                    // a buffer a value the row stream brings (one: `pin`)
                    let several = u
                        .steps
                        .iter()
                        .filter(|s| {
                            matches!(
                                s,
                                Step::In {
                                    side: Side::Row,
                                    host: false,
                                    ..
                                }
                            )
                        })
                        .count()
                        > 1;
                    let pin = if several {
                        format!("pin{}", value.0)
                    } else {
                        "pin".to_string()
                    };
                    path_in = true;
                    g.decls.push_str(&buf(&pin, n));
                    g.put(late, &format!("  @fmovs(D_{pin}, fab.fin(q_pi, {n}));\n"));
                    g.loc(*value, idx * n, n, &format!("D_{pin}"), "0");
                    if held(*value) {
                        hold(&mut g, late, *value, idx * n, n, &format!("D_{pin}"));
                    }
                }
            }
            Step::Region { side: s2, chain } if *s2 == side => {
                let (ra, printed, params) = &b.regions[&si];
                if idx < ra.first || idx >= ra.first + ra.roots {
                    continue;
                }
                let ri = ra.chunk_of(idx - ra.first);
                let c = ra.chunk;
                // a head region: a head at a time over the root's `reps` heads
                let reps = ra.reps;
                let mut rowp: Vec<(usize, Cols)> = Vec::new();
                let shared = alike.get(&si).copied();
                // a head at a time (and a shared module's results out of it before its next use)
                let heads = reps > 1 || shared.is_some();
                let later = shared.is_some_and(|f| f != si);
                let k = match shared.filter(|_| later) {
                    Some(f) => slot_of[&f],
                    None => {
                        slot += 1;
                        slot - 1
                    }
                };
                slot_of.insert(si, k);
                // split over this root's helpers: the root computes the chunk's first q
                // columns, helper h the h-th q after them
                let help = if side == Side::Col {
                    b.helper.as_ref().filter(|h| h.0 == si)
                } else {
                    None
                };
                let q = help.map_or(c, |h| c / (h.1 + 1));
                let rname = format!("rg_{ui}_{si}.csl");
                // its scratch rows the root's pool's (the regions at a root run one after
                // another)
                let (text, rows, cc) = region::pooled(help.map_or(&printed.csl, |h| &h.2.csl));
                if !later && !regions.iter().any(|(n, _)| *n == rname) {
                    regions.push((rname.clone(), text));
                }
                let mut ptrs = String::new();
                for (name, ty) in &rows {
                    let e = pool.entry(name.clone()).or_insert((*ty, 0));
                    e.1 = e.1.max(cc);
                    let _ = write!(ptrs, ", .{name} = &pl_{name}");
                }
                let mut nrow = 0u32;
                let mut hsc: Vec<String> = Vec::new();
                if help.is_some() {
                    let np = params.iter().filter(|x| **x == ParamAt::Row).count() as u32;
                    g.decls.push_str(&buf(&format!("hx{k}"), np * c));
                    g.decls
                        .push_str(&buf(&format!("ho{k}"), chain.outs.len() as u32 * c));
                    helped = true;
                }
                let seg = if printed.all_reduces > 0 {
                    "seg"
                } else {
                    "math"
                };
                // a root kernel's module: its arrays the host's, a decode step's rows its own
                let kern = crate::kernel::root_of(r.trace, chain);
                if let Some(kn) = kern {
                    if kn.rows() {
                        let _ = write!(ptrs, ", .rowp = &rowp");
                        if !root_rows {
                            root_rows = true;
                            match rows_of(p) {
                                Some(n) => {
                                    let _ = writeln!(
                                        g.decls,
                                        "var rowp = @zeros([{}]u32);\nvar rowp_p: [*]u32 = &rowp;",
                                        3 * n
                                    );
                                    g.exports.push(("rowp_p".into(), "rowp".into(), false));
                                    arrays.push(rowp_array(Coord::new(0, 0), n));
                                }
                                None => g.decls.push_str("var rowp = @zeros([3]u32);\n"),
                            }
                        }
                    }
                    for a in kn.arrays(r.trace, chain, ri, c) {
                        let sym = format!("k{k}_{}", a.name);
                        let _ = writeln!(g.decls, "var {sym}_p: [*]f32 = &rg{k}.{};", a.name);
                        g.exports.push((format!("{sym}_p"), sym.clone(), false));
                        arrays.push(HostArray {
                            at: Coord::new(0, 0),
                            sym,
                            u16: false,
                            len: a.len,
                            fill: a.fill,
                        });
                    }
                }
                if !later {
                    let _ = writeln!(
                        g.imports,
                        "const rg{k} = @import_module(\"{rname}\", .{{ .math = math, .seg = {seg}{ptrs} }});"
                    );
                }
                let mut nsc = 0;
                for (pk, input) in chain.ins.iter().enumerate() {
                    match params[pk] {
                        ParamAt::Row if heads => {
                            let cc = r.cols(input.value);
                            rowp.push((
                                pk,
                                Cols {
                                    base: cc.base,
                                    from: cc.from + input.from + ri * c * reps,
                                    width: c,
                                },
                            ));
                        }
                        ParamAt::Row => {
                            let cc = r.cols(input.value);
                            let from = cc.from + input.from + ri * c;
                            for (o, src, n) in g.pieces(cc.base, from, c)? {
                                let dst = if help.is_some() {
                                    at(&format!("D_hx{k}"), &(nrow * c + o).to_string(), n)
                                } else {
                                    at(&format!("rg{k}.IN{pk}"), &o.to_string(), n)
                                };
                                g.put(late, &format!("  @fmovs({dst}, {src});\n"));
                            }
                            if help.is_some() {
                                hrow.push((pk, nrow));
                            }
                            nrow += 1;
                        }
                        ParamAt::Scalar if !reads_scalar(&printed.csl, pk) => {}
                        ParamAt::Scalar => {
                            let _ = nsc;
                            nsc += 1;
                            let carried = carried.iter().enumerate().find_map(|(ph, c)| {
                                c.1.iter().position(|&x| x == (si, pk)).map(|i| (ph, i))
                            });
                            if let Some((ph, i)) = carried {
                                if !g.decls.contains(&format!("var csc{ph}:")) {
                                    return Err(format!(
                                        "{}: step {si}'s scalars come with a matvec this root \
                                         does not take",
                                        u.name
                                    ));
                                }
                                g.put(late, &format!("  rg{k}.sc{pk} = csc{ph}[{i}];\n"));
                                hsc.push(format!("csc{ph}[{i}]"));
                            } else {
                                // a row's scalar another region at this root reads too (q's
                                // and k's positions) is held once
                                let same = scal.iter().position(|&(s0, p0)| {
                                    matches!(&u.steps[s0], Step::Region { chain: c0, .. }
                                        if c0.ins[p0].value == input.value && c0.ins[p0].from == input.from)
                                });
                                let at = match same {
                                    Some(i) => i,
                                    None => {
                                        scal.push((si, pk));
                                        scal.len() - 1
                                    }
                                };
                                g.put(late, &format!("  rg{k}.sc{pk} = scal[t * NSC + {at}];\n"));
                                hsc.push(format!("scal[t * NSC + {at}]"));
                            }
                            if help.is_some() {
                                hscal.push(pk);
                            }
                        }
                        ParamAt::Resident => {
                            let lane = input.by.is_some() || r.lane(input.value);
                            if lane && !input.shared {
                                let cc = r.cols(input.value);
                                let from = cc.from + input.from + ri * c;
                                let mut e = 0;
                                while e < c {
                                    let off =
                                        p.lane_offset(cc.base, from + e).ok_or_else(|| {
                                            format!(
                                                "{}: lane %{} column {} is on no bus word",
                                                u.name,
                                                cc.base.0,
                                                from + e
                                            )
                                        })?;
                                    // a run of consecutive bus words
                                    let mut m = 1;
                                    while e + m < c
                                        && p.lane_offset(cc.base, from + e + m) == Some(off + m)
                                    {
                                        m += 1;
                                    }
                                    lane_pieces.push((
                                        off,
                                        m,
                                        at(&format!("rg{k}.IN{pk}"), &e.to_string(), m),
                                    ));
                                    e += m;
                                }
                            } else if shared.is_some() {
                                // its own resident, into the shared module before its heads
                                let sym = format!("r{k}_{pk}_{si}");
                                g.decls.push_str(&buf(&format!("rs{si}_{pk}"), c));
                                let _ = writeln!(g.decls, "var {sym}_p: [*]f32 = &rs{si}_{pk};");
                                g.put(late, &format!("  @fmovs(rg{k}.IN{pk}, D_rs{si}_{pk});\n"));
                                g.exports.push((format!("{sym}_p"), sym.clone(), false));
                                arrays.push(HostArray {
                                    at: Coord::new(0, 0),
                                    sym,
                                    u16: false,
                                    len: c,
                                    fill: Fill::Resident {
                                        unit: ui,
                                        step: si,
                                        param: pk,
                                        r: ri,
                                    },
                                });
                            } else {
                                let sym = format!("r{k}_{pk}");
                                let _ = writeln!(g.decls, "var {sym}_p: [*]f32 = &rg{k}.in{pk};");
                                g.exports.push((format!("{sym}_p"), sym.clone(), false));
                                arrays.push(HostArray {
                                    at: Coord::new(0, 0),
                                    sym,
                                    u16: false,
                                    len: c,
                                    fill: Fill::Resident {
                                        unit: ui,
                                        step: si,
                                        param: pk,
                                        r: ri,
                                    },
                                });
                            }
                        }
                    }
                }
                for (ti, tb) in printed.tables.iter().enumerate().filter(|_| !later) {
                    let n = tb.of_chunk.iter().map(Vec::len).max().unwrap_or(1).max(1) as u32;
                    let ty = if tb.u16 { "u16" } else { "f32" };
                    let sym = format!("t{k}_{}", tb.name);
                    let _ = writeln!(g.decls, "var {sym}_p: [*]{ty} = &rg{k}.T_{}_a;", tb.name);
                    g.exports.push((format!("{sym}_p"), sym.clone(), tb.u16));
                    arrays.push(HostArray {
                        at: Coord::new(0, 0),
                        sym,
                        u16: tb.u16,
                        len: n,
                        fill: Fill::Table {
                            unit: ui,
                            step: si,
                            table: ti,
                            r: if printed.tables[ti].of_chunk.len() == 1 {
                                0
                            } else {
                                ri
                            },
                        },
                    });
                }
                if help.is_some() {
                    // the helpers' parts down their chain, the root's own into the region
                    let mut t2 = String::new();
                    for &(_, r0) in &hrow {
                        let _ = writeln!(
                            t2,
                            "  @fmovs(fab.fout(q_hm, {}), {});",
                            c - q,
                            at(&format!("D_hx{k}"), &(r0 * c + q).to_string(), c - q)
                        );
                    }
                    if !hsc.is_empty() {
                        for (j, e) in hsc.iter().enumerate() {
                            let _ = writeln!(t2, "  hsc{k}[{j}] = {e};");
                        }
                        let _ = writeln!(t2, "  @fmovs(fab.fout(q_hm, {}), D_hsc{k});", hsc.len());
                        g.decls.push_str(&buf(&format!("hsc{k}"), hsc.len() as u32));
                    }
                    for &(pk, r0) in &hrow {
                        let _ = writeln!(
                            t2,
                            "  @fmovs({}, {});",
                            at(&format!("rg{k}.IN{pk}"), "0", q),
                            at(&format!("D_hx{k}"), &(r0 * c).to_string(), q)
                        );
                    }
                    g.put(late, &t2);
                }
                if heads {
                    if help.is_some() {
                        return Err(format!("{}: step {si}: a head region over helpers", u.name));
                    }
                    // each head's row in, the region, its results into the root's heads
                    for o in 0..chain.outs.len() {
                        g.decls.push_str(&buf(&format!("rp{si}_{o}"), c * reps));
                    }
                    for j in 0..reps {
                        for &(pk, cc) in &rowp {
                            for (o, src, n) in g.pieces(cc.base, cc.from + j * c, c)? {
                                let dst = at(&format!("rg{k}.IN{pk}"), &o.to_string(), n);
                                g.put(late, &format!("  @fmovs({dst}, {src});\n"));
                            }
                        }
                        g.put(late, &format!("  rg{k}.row();\n"));
                        for o in 0..chain.outs.len() {
                            let dst = at(&format!("D_rp{si}_{o}"), &(j * c).to_string(), c);
                            let src = at(&format!("rg{k}.OUT{o}"), "0", c);
                            g.put(late, &format!("  @fmovs({dst}, {src});\n"));
                        }
                    }
                    for (o, out) in chain.outs.iter().enumerate() {
                        let b = format!("D_rp{si}_{o}");
                        g.loc(out.value, ri * c * reps, c * reps, &b, "0");
                        if held(out.value) {
                            hold(&mut g, late, out.value, ri * c * reps, c * reps, &b);
                        }
                    }
                    hrow.clear();
                    hscal.clear();
                    continue;
                }
                g.put(late, &format!("  rg{k}.row();\n"));
                // a padded row region: its outputs' padding zeroed (a padded value's padding is
                // zero wherever it is read)
                let valid = chain.region.width.saturating_sub(ri * c).min(c);
                if side == Side::Row && valid < c && help.is_none() {
                    for o in 0..chain.outs.len() {
                        let pad = at(&format!("rg{k}.OUT{o}"), &valid.to_string(), c - valid);
                        g.put(late, &format!("  @fmovs({pad}, @as(f32, 0.0));\n"));
                    }
                }
                for (o, out) in chain.outs.iter().enumerate() {
                    if help.is_some() {
                        let o32 = o as u32;
                        g.put(
                            late,
                            &format!(
                                "  @fmovs({}, {});\n  @fmovs({}, fab.fin(q_hr, {}));\n",
                                at(&format!("D_ho{k}"), &(o32 * c).to_string(), q),
                                at(&format!("rg{k}.OUT{o}"), "0", q),
                                at(&format!("D_ho{k}"), &(o32 * c + q).to_string(), c - q),
                                c - q
                            ),
                        );
                        let off = (o32 * c).to_string();
                        g.loc(out.value, ri * c, c, &format!("D_ho{k}"), &off);
                        if held(out.value) {
                            let src = at(&format!("D_ho{k}"), &off, c);
                            hold(&mut g, late, out.value, ri * c, c, &src);
                        }
                        continue;
                    }
                    g.loc(out.value, ri * c, c, &format!("rg{k}.OUT{o}"), "0");
                    if held(out.value) {
                        hold(&mut g, late, out.value, ri * c, c, &format!("rg{k}.OUT{o}"));
                    }
                }
                hrow.clear();
                hscal.clear();
            }
            Step::Matvec {
                from,
                input,
                output,
                n,
                round,
                ..
            } => {
                if *from == side {
                    // the input out to the tiles
                    let parts: Vec<(u32, u32)> = match side {
                        Side::Row => {
                            let w = hc(input.width);
                            vec![(input.from + idx * w, w)]
                        }
                        Side::Col => l
                            .parts(idx, *input)
                            .iter()
                            .map(|(_, s)| (s.from, s.len))
                            .collect(),
                    };
                    let q = if side == Side::Row { "q_xr" } else { "q_xc" };
                    for (f, len) in parts {
                        for (_, src, m) in g.pieces(input.base, f, len)? {
                            sends_cast = true;
                            g.put(late, &format!("  @fmovs(fab.fout({q}, {m}), {src});\n"));
                        }
                    }
                } else {
                    // the output in from the chain
                    let cols = mine(*output, *n);
                    let tot: u32 = cols.iter().map(|c| c.1).sum();
                    if tot == 0 {
                        continue;
                    }
                    chain_in = true;
                    let bn = format!("mb{si}");
                    g.decls.push_str(&buf(&bn, tot));
                    g.put(late, &format!("  @fmovs(D_{bn}, fab.fin(q_ci, {tot}));\n"));
                    if side == Side::Col
                        && let Some(ph) = carried.iter().position(|c| c.0 == si)
                    {
                        // the row's scalars after it, summed down the column (one PE's, zeros)
                        let n = carried[ph].1.len() as u32;
                        g.decls.push_str(&buf(&format!("csc{ph}"), n));
                        g.put(late, &format!("  @fmovs(D_csc{ph}, fab.fin(q_ci, {n}));\n"));
                    }
                    if *round {
                        g.round(late, &format!("D_{bn}"), tot);
                    }
                    let mut o = 0;
                    for (f, len) in cols {
                        g.loc(*output, f, len, &format!("D_{bn}"), &o.to_string());
                        if held(*output) {
                            hold(
                                &mut g,
                                late,
                                *output,
                                f,
                                len,
                                &at(&format!("D_{bn}"), &o.to_string(), len),
                            );
                        }
                        o += len;
                    }
                }
            }
            Step::Store { q, k, v } if side == Side::Col && idx < l.heads => {
                let j = u.joint.ok_or("a store with no attention")?;
                let qw = j.q_heads / j.kv_heads * j.head_dim;
                let (d, dv) = (j.head_dim, j.v_dim());
                kv_io = true;
                let mut parts = vec![
                    (*q, q.from + j.q_head(idx) * qw, qw),
                    (*k, k.from + j.k_head(idx) * d, d),
                    (*v, v.from + j.v_head(idx) * dv, dv),
                ];
                // what else its column takes a head (a recurrence's beta and decay logits)
                parts.extend(crate::kernel::column(Some(&j))?.head_parts(&j, idx));
                for (c, f, len) in parts {
                    for (_, src, m) in g.pieces(c.base, f, len)? {
                        g.put(late, &format!("  @fmovs(fab.fout(q_ks, {m}), {src});\n"));
                    }
                }
            }
            Step::Attend { out, round } if side == Side::Col && idx < l.heads => {
                let j = u.joint.ok_or("an attend with no attention")?;
                let qw = j.out_dim();
                g.decls.push_str(&buf("ob", qw));
                g.put(late, &format!("  @fmovs(D_ob, fab.fin(q_kr, {qw}));\n"));
                if *round {
                    g.round(late, "D_ob", qw);
                }
                g.loc(*out, idx * qw, qw, "D_ob", "0");
                if held(*out) {
                    hold(&mut g, late, *out, idx * qw, qw, "D_ob");
                }
            }
            Step::Out {
                value,
                side: s2,
                host,
                lane,
            } if *s2 == side && !lane_unit => {
                let _ = lane;
                if *host {
                    let parts: Vec<(u32, u32)> = match side {
                        Side::Row => {
                            let w = hc(value.width);
                            vec![(value.from + idx * w, w)]
                        }
                        Side::Col => l
                            .parts(idx, *value)
                            .iter()
                            .map(|(_, s)| (s.from, s.len))
                            .collect(),
                    };
                    for (f, len) in parts {
                        for (o, src, m) in g.pieces(value.base, f, len)? {
                            let off = format!("t * HOUTW + {}", hout_w + o);
                            g.put(
                                late,
                                &format!("  @fmovs({}, {src});\n", at("D_hout", &off, m)),
                            );
                        }
                        if argmax_of(p) == Some(value.base) {
                            amax.push((hout_w, len, f));
                        }
                        hout_parts.push((value.base.0, f, len));
                        hout_w += len;
                    }
                } else if side == Side::Row {
                    path_out = true;
                    let w = hc(value.width);
                    for (_, src, m) in g.pieces(value.base, value.from + idx * w, w)? {
                        g.put(late, &format!("  @fmovs(fab.fout(q_po, {m}), {src});\n"));
                    }
                }
            }
            _ => {}
        }
    }
    // the lane prologue's column roots: their lane words to the lane root (own, then the
    // farther roots')
    let mut gather = false;
    g.now_late = false;
    if lane_unit && side == Side::Col {
        let mut own = 0;
        for s in &l.cols[idx as usize] {
            if p.lane_at
                .iter()
                .any(|&(bb, f, _, _)| bb == s.base && f == s.from)
            {
                for (_, src, m) in g.pieces(ValueId(s.base), s.from, s.len)? {
                    g.early
                        .push_str(&format!("  @fmovs(fab.fout(q_gs, {m}), {src});\n"));
                }
                own += s.len;
            }
        }
        let farther: u32 = (idx + 1..l.px())
            .map(|i| {
                l.cols[i as usize]
                    .iter()
                    .filter(|s| {
                        p.lane_at
                            .iter()
                            .any(|&(bb, f, _, _)| bb == s.base && f == s.from)
                    })
                    .map(|s| s.len)
                    .sum::<u32>()
            })
            .sum();
        if own > 0 || farther > 0 {
            gather = true;
            // (a runtime count, zero at the farthest root: every root of the unit one file)
            g.drain = true;
            let n = g.word(farther);
            let d = drain(
                &n,
                "q_gr",
                " @fmovs(fab.fout(q_gs, m), @set_dsd_length(D_drn, m));",
            );
            g.early.push_str(&d);
        }
    }
    // the lane words off the bus: drained between, copied where two read one
    let mut lanes_in = !lane_pieces.is_empty();
    if lanes_in {
        lane_pieces.sort_by_key(|x| x.0);
        let mut cur = 0u32;
        let mut got: Vec<(u32, u32, String)> = Vec::new();
        for (off, n, dst) in &lane_pieces {
            if *off >= cur {
                // the gap before it a runtime count (zero where none): a unit's roots differ
                // in their lane words' offsets only, so they share one file
                g.drain = true;
                let gap = g.word(off - cur);
                let d = drain(&gap, "q_lb", "");
                g.lanes.push_str(&d);
                let _ = writeln!(g.lanes, "  @fmovs({dst}, fab.fin(q_lb, {n}));");
                got.push((*off, *n, dst.clone()));
                cur = off + n;
            } else {
                let (o0, _, src) = got
                    .iter()
                    .find(|(o, m, _)| *off >= *o && off + n <= o + m)
                    .ok_or("stream: a lane piece straddles two read before it")?
                    .clone();
                let src = format!("@increment_dsd_offset({src}, @as(i16, {}), f32)", off - o0);
                let _ = writeln!(g.lanes, "  @fmovs({dst}, @set_dsd_length({src}, {n}));");
            }
        }
        g.drain = true;
        let n = g.word(p.lane_words - cur);
        let d = drain(&n, "q_lb", "");
        g.lanes.push_str(&d);
    }
    let _ = &mut lanes_in;

    // the file
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// {}: {} (kernels-cerebras emit::stream).",
        u.name,
        match side {
            Side::Row => "the CX roots".to_string(),
            Side::Col => format!("CY root {idx}"),
        }
    );
    t.push_str("param memcpy_params;\n");
    let qs: Vec<(&str, &str, bool)> = match side {
        Side::Row => vec![
            ("C_PI", "q_pi", true),
            ("C_RR", "q_ci", true),
            ("C_LB", "q_lb", true),
            ("C_PO", "q_po", false),
            ("C_XR", "q_xr", false),
        ],
        Side::Col => vec![
            ("C_CR", "q_ci", true),
            ("C_KR", "q_kr", true),
            ("C_LB", "q_lb", true),
            ("C_GR", "q_gr", true),
            ("C_XC", "q_xc", false),
            ("C_KS", "q_ks", false),
            ("C_GS", "q_gs", false),
            ("C_HR", "q_hr", true),
            ("C_HM", "q_hm", false),
        ],
    };
    let segment = s_segment(p, ui, side, idx);
    let mut pnames: Vec<&str> = match side {
        Side::Row => vec![
            "C_PI", "C_PO", "C_RR", "C_XR", "C_LB", "C_SA", "C_SM", "C_SS", "S_ROLE",
        ],
        Side::Col if helped => vec![
            "C_CR", "C_XC", "C_KS", "C_KR", "C_LB", "C_GS", "C_GR", "C_HM", "C_HR",
        ],
        Side::Col => vec!["C_CR", "C_XC", "C_KS", "C_KR", "C_LB", "C_GS", "C_GR"],
    };
    if side == Side::Col && segment {
        pnames.extend(["C_SA", "C_SM", "C_SS", "S_ROLE"]);
    }
    for pn in &pnames {
        let _ = writeln!(t, "param {pn}: u16;");
    }
    t.push_str(COMMON);
    let (mut iq, mut oq) = (2, 2);
    sends_cast |= g.cast;
    chain_in |= !g.fetch.is_empty();
    let used = |q: &str| -> bool {
        match q {
            "q_pi" => path_in,
            "q_ci" => chain_in,
            "q_lb" => lanes_in,
            "q_gr" | "q_gs" => gather,
            "q_po" => path_out,
            "q_xr" | "q_xc" => sends_cast,
            "q_kr" | "q_ks" => kv_io,
            "q_hr" | "q_hm" => helped,
            _ => false,
        }
    };
    let mut inits = String::new();
    for (cname, q, input) in &qs {
        // q_gr is used only by a root with farther roots; it is declared with its pair
        if !used(q) {
            continue;
        }
        if *input {
            let _ = writeln!(t, "const {q} = @get_input_queue({iq});");
            iq += 1;
        } else {
            let _ = writeln!(t, "const {q} = @get_output_queue({oq});");
            oq += 1;
        }
        let _ = writeln!(
            inits,
            "  @initialize_queue({q}, .{{ .color = @get_color({cname}) }});"
        );
    }
    if segment {
        // (ops/seg.csl binds iq6, iq7 and oq4 itself: a root's own queues stop short of them)
        if iq > 6 || oq > 4 {
            return Err(format!(
                "a segment root's own queues reach ops/seg.csl's (input {iq}, output {oq})"
            ));
        }
        t.push_str("var rcfg = [18]u32 { 0, 0, 0, 0, @as(u32, C_SM), @as(u32, C_SA), 0, 0, 0, 0, 0, 0, 0, @as(u32, C_SA), 0, @as(u32, C_SS), @as(u32, C_SM), @as(u32, S_ROLE) };\n");
        t.push_str("const seg = @import_module(\"ops/seg.csl\", .{ .fab = fab, .rcfg = &rcfg, .RQ = 0, .XM = @get_input_queue(6), .YM = @get_input_queue(7), .AO = @get_output_queue(4), .UT = @get_ut_id(5) });\n");
        inits.push_str("  @initialize_queue(@get_input_queue(6), .{ .color = @get_color(C_SM) });\n  @initialize_queue(@get_input_queue(7), .{ .color = @get_color(C_SA) });\n  @initialize_queue(@get_output_queue(4), .{ .color = @get_color(C_SS) });\n");
    }
    t.push_str(&pool_decls(&pool));
    t.push_str(&g.imports);
    if hin_w > 0 {
        let _ = writeln!(t, "const HINW: u16 = {hin_w};");
        t.push_str(&buf("hin", rows * hin_w));
        t.push_str("var hin_p: [*]f32 = &hin;\n");
        g.exports.push(("hin_p".into(), "hin".into(), false));
        arrays.push(HostArray {
            at: Coord::new(0, 0),
            sym: "hin".into(),
            u16: false,
            len: rows * hin_w,
            fill: Fill::HostIn {
                unit: ui,
                parts: hin_parts,
            },
        });
    }
    if hout_w > 0 {
        let _ = writeln!(t, "const HOUTW: u16 = {hout_w};");
        t.push_str(&buf("hout", rows * hout_w));
        t.push_str("var hout_p: [*]f32 = &hout;\n");
        g.exports.push(("hout_p".into(), "hout".into(), false));
        arrays.push(HostArray {
            at: Coord::new(0, 0),
            sym: "hout".into(),
            u16: false,
            len: rows * hout_w,
            fill: Fill::HostOut {
                unit: ui,
                parts: hout_parts,
            },
        });
    }
    if !amax.is_empty() {
        // a row's (largest logit, its column): the first column on a tie, torch.argmax's
        let _ = writeln!(
            t,
            "var hmax = @zeros([{}]u32);\nvar hmax_p: [*]u32 = &hmax;",
            2 * rows
        );
        let mut f =
            String::from("fn amax(t: u16) void {\n  var m: f32 = -3.0e38;\n  var k: u32 = 0;\n");
        for &(o, len, from) in &amax {
            // (its column a config word: the roots one image)
            g.cfg.push(from);
            let w = g.cfg.len() - 1;
            let _ = writeln!(
                f,
                "  {{ var i: u16 = 0; while (i < {len}) : (i += 1) {{ const v = hout[t * HOUTW + {o} + i]; if (v > m) {{ m = v; k = pcfg[{w}] + @as(u32, i); }} }} }}"
            );
        }
        f.push_str("  hmax[2 * t] = @bitcast(u32, m);\n  hmax[2 * t + 1] = k;\n}\n");
        t.push_str(&f);
        g.exports.push(("hmax_p".into(), "hmax".into(), false));
        arrays.push(HostArray {
            at: Coord::new(0, 0),
            sym: "hmax".into(),
            u16: false,
            len: 2 * rows,
            fill: Fill::HostArgmax {
                unit: ui,
                parts: amax.iter().map(|&(_, len, from)| (from, len)).collect(),
            },
        });
        g.end.push_str(&format!(
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ amax(t); }} }}\n"
        ));
    }
    if !scal.is_empty() {
        let ns = scal.len() as u32;
        let _ = writeln!(t, "const NSC: u16 = {ns};");
        let _ = writeln!(
            t,
            "var scal = @zeros([{}]f32);\nvar scal_p: [*]f32 = &scal;",
            rows * ns
        );
        g.exports.push(("scal_p".into(), "scal".into(), false));
        arrays.push(HostArray {
            at: Coord::new(0, 0),
            sym: "scal".into(),
            u16: false,
            len: rows * ns,
            fill: Fill::Scalars { unit: ui, of: scal },
        });
    }
    if g.h16 > 0 {
        let _ = writeln!(
            t,
            "var h16 = @zeros([{0}]@fp16());\nconst H16 = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{0}}} -> h16[i] }});",
            g.h16
        );
    }
    if g.drain {
        t.push_str(&buf("drn", 64));
    }
    t.push_str(&cfg_decl(&g.cfg));
    if let Some(a) = cfg_array(Coord::new(0, 0), &g.cfg) {
        g.exports.push(("pcfg_p".into(), "pcfg".into(), false));
        arrays.push(a);
    }
    t.push_str(&g.decls);
    let _ = write!(
        t,
        "fn early(t: u16) void {{\n{}}}\nfn late(t: u16) void {{\n{}{}}}\nfn lanes() void {{\n{}}}\n",
        g.early, g.fetch, g.late, g.lanes
    );
    // the schedule
    let mut body = String::new();
    body.push_str("  stamp(0);\n");
    for k in 0..slot {
        let _ = writeln!(body, "  rg{k}.init();");
    }
    if segment {
        body.push_str("  seg.role = S_ROLE;\n  seg.qmode(0);\n");
    }
    body.push_str("  lanes();\n");
    let segs: Vec<Seg> = match &b.sched {
        Some(s) => s.segs.clone(),
        None => vec![Seg::Early {
            t0: 0,
            n: rows,
            host: true,
        }],
    };
    let (mut ne, mut nl) = (0u32, 0u32);
    for sg in &segs {
        match *sg {
            Seg::Early { t0, n, .. } => {
                ne += n;
                let _ = writeln!(
                    body,
                    "  {{ var t: u16 = {t0}; while (t < {}) : (t += 1) {{ early(t); }} }}",
                    t0 + n
                );
            }
            Seg::Late { t0, n } => {
                nl += n;
                let _ = writeln!(
                    body,
                    "  {{ var t: u16 = {t0}; while (t < {}) : (t += 1) {{ late(t); }} }}",
                    t0 + n
                );
            }
            Seg::Fwd { n } if side == Side::Row => {
                let w = hc(p.stream_hidden());
                let _ = writeln!(
                    body,
                    "  {{ var t: u16 = 0; while (t < {n}) : (t += 1) {{ @fmovs(D_fw, fab.fin(q_pi, {w})); @fmovs(fab.fout(q_po, {w}), D_fw); }} }}"
                );
                if !t.contains("var fw:") {
                    t.push_str(&buf("fw", w));
                }
            }
            Seg::Drop { n } if side == Side::Row => {
                let w = hc(p.stream_hidden());
                let _ = writeln!(
                    body,
                    "  {{ var t: u16 = 0; while (t < {n}) : (t += 1) {{ @fmovs(D_fw, fab.fin(q_pi, {w})); }} }}"
                );
                if !t.contains("var fw:") {
                    t.push_str(&buf("fw", w));
                }
            }
            _ => {}
        }
    }
    let _ = (ne, nl);
    body.push_str(&g.end);
    body.push_str("  stamp(1);\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n");
    for (ptr, sym, _) in &g.exports {
        let _ = writeln!(t, "  @export_symbol({ptr}, \"{sym}\");");
    }
    t.push_str(&inits);
    t.push_str("}\n");
    // the path's queues where the schedule forwards but the steps read no stream
    if (segs
        .iter()
        .any(|s| matches!(s, Seg::Fwd { .. } | Seg::Drop { .. }))
        && side == Side::Row)
        && !t.contains("const q_pi")
    {
        t = t.replacen(
            "const sys_mod",
            "const q_pi = @get_input_queue(5);\nconst sys_mod",
            1,
        );
        t = t.replacen(
            "comptime {\n",
            "comptime {\n  @initialize_queue(q_pi, .{ .color = @get_color(C_PI) });\n",
            1,
        );
    }
    if segs.iter().any(|s| matches!(s, Seg::Fwd { .. }))
        && side == Side::Row
        && !t.contains("const q_po")
    {
        t = t.replacen(
            "const sys_mod",
            "const q_po = @get_output_queue(5);\nconst sys_mod",
            1,
        );
        t = t.replacen(
            "comptime {\n",
            "comptime {\n  @initialize_queue(q_po, .{ .color = @get_color(C_PO) });\n",
            1,
        );
    }
    Ok((name, t, arrays, regions))
}

/// A unit's helper PE (below a rest root): each own row, its part (`pcfg[0]`) of the region at
/// `q` columns, its results summed up the helpers' chain to the root (`pcfg[1]` its role).
fn helper_file(ui: usize, b: &Built, r: &Reader) -> R<(String, String)> {
    let Some((si, h, printed)) = &b.helper else {
        return Err(format!("{}: helpers of no region", b.unit.name));
    };
    let Step::Region { chain, .. } = &b.unit.steps[*si] else {
        return Err(format!("{}: helpers of no region", b.unit.name));
    };
    let (ra, full, params) = &b.regions[si];
    let c = ra.chunk;
    let q = c / (h + 1);
    let cq = c - q;
    let name = format!("hp_{ui}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// {}: the helpers below its rest roots, step {si}'s region over {q} of a root's {c} columns \
         (kernels-cerebras emit::stream).",
        b.unit.name
    );
    t.push_str("param memcpy_params;\nparam C_HM: u16;\nparam C_HS: u16;\nparam C_HR: u16;\n");
    t.push_str(COMMON);
    t.push_str("const q_hm = @get_input_queue(2);\nconst q_hr = @get_input_queue(3);\nconst q_hs = @get_output_queue(2);\n");
    t.push_str("const red = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_hr, .OUT = q_hs });\n");
    let (_, rows, cc) = region::pooled(&printed.csl);
    let pool: BTreeMap<String, (&'static str, u32)> =
        rows.iter().map(|(n, ty)| (n.clone(), (*ty, cc))).collect();
    t.push_str(&pool_decls(&pool));
    let ptrs: String = rows
        .iter()
        .map(|(n, _)| format!(", .{n} = &pl_{n}"))
        .collect();
    let _ = writeln!(
        t,
        "const rg = @import_module(\"rg_{ui}_{si}.csl\", .{{ .math = math, .seg = math{ptrs} }});"
    );
    t.push_str(&cfg_decl(&[0, 0]));
    t.push_str(&buf("drn", 64));
    let nout = chain.outs.len() as u32;
    t.push_str(&buf("ov", nout * cq));
    let scalars: Vec<usize> = (0..chain.ins.len())
        .filter(|&pk| params[pk] == ParamAt::Scalar && reads_scalar(&full.csl, pk))
        .collect();
    if !scalars.is_empty() {
        t.push_str(&buf("hsc", scalars.len() as u32));
    }
    let _ = r;
    let mut row = String::from("  const hk: u16 = @as(u16, pcfg[0]);\n");
    let _ = writeln!(
        row,
        "  const before: u16 = (hk - 1) * {q};\n  const after: u16 = {cq} - before - {q};"
    );
    for (pk, _) in params
        .iter()
        .enumerate()
        .filter(|(_, x)| **x == ParamAt::Row)
    {
        row.push_str(&drain("before", "q_hm", ""));
        let _ = writeln!(
            row,
            "  @fmovs({}, fab.fin(q_hm, {q}));",
            at(&format!("rg.IN{pk}"), "0", q)
        );
        row.push_str(&drain("after", "q_hm", ""));
    }
    if !scalars.is_empty() {
        let _ = writeln!(row, "  @fmovs(D_hsc, fab.fin(q_hm, {}));", scalars.len());
        for (j, pk) in scalars.iter().enumerate() {
            let _ = writeln!(row, "  rg.sc{pk} = hsc[{j}];");
        }
    }
    row.push_str("  rg.row();\n");
    for o in 0..nout {
        let _ = writeln!(
            row,
            "  @fmovs(@set_dsd_length(@increment_dsd_offset(D_ov, @as(i16, before + {}), f32), {q}), {});",
            o * cq,
            at(&format!("rg.OUT{o}"), "0", q)
        );
    }
    let _ = writeln!(row, "  red.send(D_ov, {});", nout * cq);
    let _ = write!(
        t,
        "fn row() void {{\n{row}}}\ntask go() void {{\n  stamp(0);\n  rg.init();\n  red.role = @as(u16, pcfg[1]);\n  {{ var t: u16 = 0; while (t < {}) : (t += 1) {{ row(); }} }}\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}}\n",
        b.unit.rows
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @initialize_queue(q_hm, .{ .color = @get_color(C_HM) });\n  @initialize_queue(q_hr, .{ .color = @get_color(C_HR) });\n  @initialize_queue(q_hs, .{ .color = @get_color(C_HS) });\n}\n");
    let _ = printed;
    Ok((name, t))
}

/// A root's scratch pool ([`region::pooled`]): a row a name, as wide as its widest module's.
fn pool_decls(pool: &BTreeMap<String, (&'static str, u32)>) -> String {
    let mut t = String::new();
    for (name, (ty, n)) in pool {
        let _ = writeln!(t, "var pl_{name}: [{n}]{ty} align(16) = @zeros([{n}]{ty});");
    }
    t
}

/// Whether root `idx` of unit `ui`'s `side` all-reduces over its side's segment.
fn s_segment(p: &Program, ui: usize, side: Side, idx: u32) -> bool {
    let u = &p.stream.units[ui];
    match side {
        Side::Row => u.segment,
        Side::Col => u
            .col_segment
            .is_some_and(|(first, roots)| idx >= first && idx < first + roots),
    }
}

fn kv_root_params(p: &Program, kc: &fs::KvColumn, c: Coord) -> Vec<(String, u32)> {
    let mut v = Vec::new();
    for k in 0..kc.units.len() {
        let i = match ep(p, c, fs::kv_in(k)) {
            Some(Endpoint::Recv(c)) => u32::from(c.0),
            _ => 0xff,
        };
        let o = match ep(p, c, fs::kv_out(k)) {
            Some(Endpoint::Send(c)) => u32::from(c.0),
            _ => 0xff,
        };
        v.push((format!("C_I{k}"), i));
        v.push((format!("C_O{k}"), o));
    }
    let (_, ar, _) = chain_cols(p, c, KV_CHAIN);
    let am = colour_of(p, &format!("AM{}_{}", kc.block, kc.head)).map_or(0xff, u32::from);
    v.push(("C_AR".into(), ar));
    v.push(("C_AM".into(), am));
    parked(p, c, v)
}

/// A K / V PE's colours and its chain role (a config word).
fn kv_pe_params(p: &Program, c: Coord) -> (Vec<(String, u32)>, u32) {
    let (s, r, role) = chain_cols(p, c, KV_CHAIN);
    // the cast colour: the column's
    let am = p
        .stream
        .kv
        .iter()
        .find(|k| (0..k.pya).any(|i| k.pe(&p.stream, i) == c))
        .and_then(|k| colour_of(p, &format!("AM{}_{}", k.block, k.head)))
        .map_or(0xff, u32::from);
    (
        parked(
            p,
            c,
            vec![("C_AM".into(), am), ("C_S".into(), s), ("C_R".into(), r)],
        ),
        role,
    )
}

/// A K / V column's attention: its block's first unit's (the reading's first where none).
pub(crate) fn kv_joint(p: &Program, ks: &place::KvSched) -> crate::compile::stream::unit::Joint {
    let a = &p.shape.attn;
    ks.units
        .first()
        .and_then(|&u| p.units.get(u))
        .and_then(|b| b.unit.joint)
        .unwrap_or(crate::compile::stream::unit::Joint {
            q_heads: a.q_heads,
            kv_heads: a.kv_heads,
            head_dim: a.head_dim,
            scale: a.sm_scale,
            causal: a.causal,
            window: None,
            state: None,
            sink: None,
            kv_rep: 1,
        })
}

pub(crate) fn kv_consts(p: &Program, ks: &place::KvSched) -> (u32, u32, u32, f32) {
    let a = kv_joint(p, ks);
    let qh = a.q_heads / a.kv_heads;
    (qh, a.head_dim, qh * a.head_dim, a.scale)
}

/// A decode program's rows a step (its lanes): none in a stream reading's.
pub(crate) fn rows_of(p: &Program) -> Option<u32> {
    p.shape.cache?;
    Some(p.class_rows.iter().map(|c| c.0).sum::<u32>().max(1))
}

/// The array of a decode step's `n` rows' (lane slot, state slot, position) at `c`.
pub(crate) fn rowp_array(c: Coord, n: u32) -> HostArray {
    HostArray {
        at: c,
        sym: "rowp".into(),
        u16: false,
        len: 3 * n,
        fill: Fill::Words(vec![0; 3 * n as usize]),
    }
}

/// A decode program's paging: its K / V columns' slots in pages of `size` positions (at least
/// [`PAGE_MIN`]), `pages` a column (the last a scratch page), `lanes` lane slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Paging {
    pub size: u32,
    pub pages: u32,
    pub lanes: u32,
}

/// A decode program's state root ([`super::state`]): each K / V column a region the host
/// addresses as `arrays` place it; none in a stream reading's.
#[must_use]
pub fn state_root(p: &Program, arrays: &[HostArray]) -> Option<super::state::StateRoot> {
    use super::state::{Pages, Pe, Region, StateRoot};
    let lanes = rows_of(p)?;
    let s = &p.stream;
    let g = paging(p);
    let mut regions = Vec::new();
    let mut advances = true;
    for kc in &s.kv {
        let ks = &p.kv[kc.block as usize];
        let j = kv_joint(p, ks);
        let col = crate::kernel::column(Some(&j)).ok()?;
        let mut at: Vec<Coord> = vec![kc.root(s)];
        at.extend((0..kc.pya).map(|i| kc.pe(s, i)));
        let of = |sym: &str| -> Vec<Pe> {
            let mut v: Vec<Pe> = (arrays.iter())
                .filter(|a| a.sym == sym && at.contains(&a.at))
                .map(|a| Pe {
                    x: a.at.x,
                    y: a.at.y,
                })
                .collect();
            v.sort_unstable();
            v
        };
        regions.push(Region {
            kind: col.region(&j).into(),
            block: kc.block,
            head: kc.head,
            slots: lanes,
            paging: match (col.paged(), g) {
                (true, Some(g)) => Some(Pages {
                    size: g.size,
                    pages: g.pages,
                }),
                _ => None,
            },
            pos: of("pos"),
            rows: of("rowp"),
            tables: of("ptab"),
        });
        if regions.last().is_some_and(|r| !r.rows.is_empty()) && !col.advances() {
            advances = false;
        }
    }
    // the roots whose kernels keep a lane slot's state (a root kernel's `rowp`): one region
    let columns: Vec<Coord> = (s.kv.iter())
        .flat_map(|kc| std::iter::once(kc.root(s)).chain((0..kc.pya).map(|i| kc.pe(s, i))))
        .collect();
    let mut roots: Vec<Pe> = (arrays.iter())
        .filter(|a| a.sym == "rowp" && !columns.contains(&a.at))
        .map(|a| Pe {
            x: a.at.x,
            y: a.at.y,
        })
        .collect();
    if !roots.is_empty() {
        advances = false;
        roots.sort_unstable();
        regions.push(Region {
            kind: "rows".into(),
            block: 0,
            head: 0,
            slots: lanes,
            paging: None,
            pos: Vec::new(),
            rows: roots,
            tables: Vec::new(),
        });
    }
    let advances = advances && regions.iter().any(|r| !r.rows.is_empty());
    Some(StateRoot {
        lanes,
        regions,
        advances,
    })
}

/// The fewest positions a page holds.
pub const PAGE_MIN: u32 = 64;

/// The paging of `p`'s attention columns: none in a stream reading's or a recurrence's.
#[must_use]
pub fn paging(p: &Program) -> Option<Paging> {
    let cache = p.shape.cache?;
    let attn: Vec<u32> = (p.stream.kv.iter())
        .filter(|kc| {
            crate::kernel::column(Some(&kv_joint(p, &p.kv[kc.block as usize])))
                .is_ok_and(|c| c.paged())
        })
        .map(|kc| kc.pya)
        .collect();
    if attn.is_empty() {
        return None;
    }
    let gcd = |mut a: u32, mut b: u32| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let l = attn.iter().fold(1u32, |l, &y| l / gcd(l, y) * y);
    let size = l * PAGE_MIN.div_ceil(l);
    let lanes = p.class_rows.iter().map(|c| c.0).sum::<u32>().max(1);
    Some(Paging {
        size,
        // (and the scratch an idle row stores into)
        pages: cache.div_ceil(size) + 1,
        lanes,
    })
}

fn lane_root_file(words: u32) -> String {
    let mut t = String::from(
        "// The lane root (kernels-cerebras emit::stream): the gathered lane words onto the bus.\nparam memcpy_params;\nparam C_GI: u16;\nparam C_BO: u16;\n",
    );
    t.push_str(COMMON);
    t.push_str("const q_gi = @get_input_queue(2);\nconst q_bo = @get_output_queue(2);\n");
    t.push_str(&buf("lw", 64));
    let _ = write!(
        t,
        "task go() void {{\n  stamp(0);\n  {{ var k: u16 = 0; while (k < {words}) : (k += 64) {{ const m = if ({words} - k > 64) 64 else {words} - k; @fmovs(@set_dsd_length(D_lw, m), fab.fin(q_gi, m)); @fmovs(fab.fout(q_bo, m), @set_dsd_length(D_lw, m)); }} }}\n  stamp(1);\n  sys_mod.unblock_cmd_stream();\n}}\n"
    );
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @initialize_queue(q_gi, .{ .color = @get_color(C_GI) });\n  @initialize_queue(q_bo, .{ .color = @get_color(C_BO) });\n}\n");
    t
}

fn filler_file() -> String {
    let mut t = String::from(
        "// An idle PE (kernels-cerebras emit::stream): routes only.\nparam memcpy_params;\n",
    );
    t.push_str(COMMON);
    t.push_str("task go() void { sys_mod.unblock_cmd_stream(); }\ncomptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n}\n");
    t
}

/// The placement: every PE's code and params, every route, every exported name.
fn layout_placed(p: &Program, tiles: &Tiles, arrays: &[HostArray]) -> Placed {
    let s = &p.stream;
    // an array's element type: words (config, positions, stamps, an argmax's) u32
    let mut names: BTreeMap<String, &str> = BTreeMap::new();
    for a in arrays {
        let ty = match a.fill {
            Fill::Words(_) | Fill::Stamps | Fill::HostArgmax { .. } => "u32",
            _ if a.u16 => "u16",
            _ => "f32",
        };
        names.insert(a.sym.clone(), ty);
    }
    let header = format!(
        "// A stream reading's wafer program (kernels-cerebras emit::stream): {} x {} PEs.",
        s.w, s.h
    );
    placed_of(header, (s.w, s.h), tiles, &p.fabric, &names)
}
