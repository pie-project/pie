//! A stream program's CSL ([`crate::stream::build::Program`]): `layout.csl` (every PE's code
//! and routes, the routes from the laid fabric), a kernel file a PE kind (a unit's weight
//! column, its CX roots, each CY root; a K / V column's root and PEs; the lane root; the filler)
//! and each root region's printed module, plus what the host writes into each PE before a step
//! and reads after it ([`HostArray`]).
//!
//! Every root runs the same loop: the lane vectors it holds off the bus, then its unit's
//! schedule (each run of rows: its own rows' steps before the barrier, others' rows on, its own
//! rows' steps after it). A step at a root is a handful of moves: a row in (from the stream, a
//! chain, the K / V column, the host), its region's parameters copied in and `row()`, a vector
//! out (a multicast to the tiles, the stream, the K / V column, the host). Weight PEs run their
//! matvecs a row (ops/smv.csl's `row`), in the unit's step order.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use poem_ir::ValueId;
use serde::Serialize;

use crate::emit::region::ParamAt;
use crate::fabric::{Colour, Endpoint, RouteWord};
use crate::floor::{Coord, Dir};
use crate::floorplan::stream::{
    self as fs, COL_CHAIN, HELP_CHAIN, KV_CHAIN, KV_RECV, KV_SEND, LANE_BUS, LANE_GATHER, PATH_IN,
    PATH_OUT, ROW_CHAIN, SEGMENT,
};
use crate::stream::build::{Built, Program};
use crate::stream::place::{self, Seg};
use crate::stream::unit::{Cols, Reader, Side, Step};
use crate::wiring::{Class, Slot};

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

/// A route word as CSL's `.rx` / `.tx` sets.
fn routes(w: RouteWord) -> String {
    let rx = if w.from_ramp() {
        "RAMP".to_string()
    } else {
        let d = match w.0 & 0x60 {
            0x00 => Dir::W,
            0x20 => Dir::E,
            0x40 => Dir::S,
            _ => Dir::N,
        };
        dirname(d).to_string()
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
    format!(
        ".{{ .routes = .{{ .rx = .{{ {rx} }}, .tx = .{{ {} }} }} }}",
        tx.join(", ")
    )
}

/// A buffer declaration and its DSD.
fn buf(name: &str, n: u32) -> String {
    let n = n.max(1);
    format!(
        "var {name}: [{n}]f32 align(16) = @zeros([{n}]f32);\nconst D_{name} = @get_dsd(mem1d_dsd, .{{ .tensor_access = |i|{{{n}}} -> {name}[i] }});\n"
    )
}

/// A drain of `n` words (a `u16` expression) off queue `q` through `D_drn`, 64 at a time, each
/// piece then `then` (with `m` its length).
fn drain(n: &str, q: &str, then: &str) -> String {
    format!(
        "  {{ const nd: u16 = {n}; var k: u16 = 0; while (k < nd) : (k += 64) {{ const m = if (nd - k > 64) 64 else nd - k; @fmovs(@set_dsd_length(D_drn, m), fab.fin({q}, m));{then} }} }}\n"
    )
}

/// The runtime config words' declaration and export (`pcfg`), empty where none.
fn cfg_decl(cfg: &[u32]) -> String {
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
fn at(d: &str, off: &str, n: u32) -> String {
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
                .ok_or_else(|| format!("value %{} column {c} is at no root buffer", base.0))?;
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
    (0..crate::fabric::COLOURS)
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
                tiles.insert(c, (name.clone(), weight_params(p, c)));
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
            tiles.insert(c, (name, root_params(p, ui, c, Side::Row)));
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
            tiles.insert(c, (name, root_params(p, ui, c, Side::Col)));
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
                    tiles.insert(c, (name.clone(), params));
                    arrays.extend(cfg_array(c, &[k + 1, role]));
                }
            }
        }
    }
    // the K / V columns
    for (k, kc) in s.kv.iter().enumerate() {
        let ks = &p.kv[kc.block as usize];
        let (rname, rtext) = kv_root_file(p, k, ks)?;
        files.push((rname.clone(), rtext));
        let root = kc.root(s);
        tiles.insert(root, (rname, kv_root_params(p, kc, root)));
        let (pname, ptext) = kv_pe_file(p, k, ks)?;
        files.push((pname.clone(), ptext));
        for i in 0..kc.pya {
            let c = kc.pe(s, i);
            let (params, role) = kv_pe_params(p, c);
            tiles.insert(c, (pname.clone(), params));
            arrays.extend(cfg_array(c, &[i, role]));
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
        tiles.insert(
            c,
            (
                "lane_root.csl".into(),
                vec![
                    ("C_GI".into(), u32::from(gin)),
                    ("C_BO".into(), u32::from(bus)),
                ],
            ),
        );
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
    files.push(("layout.csl".into(), layout_file(p, &tiles, &arrays)));
    Ok(Emitted {
        files,
        arrays,
        w: s.w,
        h: s.h,
    })
}

/// One file a text: files that differ only in their comments (a unit's name, a column's index)
/// become the first of them, the regions first (the roots import them by name), then the PE
/// kernels, every PE's code renamed. cslc builds an image a distinct (file, params), so this is
/// what keeps a large program's image count to its kinds of PE, not its PEs.
fn dedup(files: &mut Vec<(String, String)>, tiles: &mut Tiles) {
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

/// An array a root file declares, at root `idx` of its side.
fn rebind(a: &HostArray, c: Coord, idx: u32, b: &Built) -> HostArray {
    let mut out = a.clone();
    out.at = c;
    if let Fill::Table {
        unit, step, table, ..
    } = a.fill
    {
        let first = b.regions.get(&step).map_or(0, |(at, _, _)| at.first);
        out.fill = Fill::Table {
            unit,
            step,
            table,
            r: idx - first,
        };
    }
    if let Fill::Resident {
        unit, step, param, ..
    } = a.fill
    {
        let first = b.regions.get(&step).map_or(0, |(at, _, _)| at.first);
        out.fill = Fill::Resident {
            unit,
            step,
            param,
            r: idx - first,
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

fn weight_params(p: &Program, c: Coord) -> Vec<(String, u32)> {
    let u = p
        .stream
        .units
        .iter()
        .position(|u| c.x > u.x && c.x <= u.x + u.cols())
        .unwrap_or(0);
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
type Tiles = BTreeMap<Coord, (String, Vec<(String, u32)>)>;

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

/// The CY roots' scalars carried down the column chains, a phase each (before the barrier,
/// after it): the matvec whose output carries them (the phase's first from the row side) and
/// the (step, param)s of the column regions after it that read one, in order. Empty where the
/// unit's CY roots hold their scalars for every row within [`SCAL_WORDS`].
fn carry(p: &Program, b: &Built) -> Vec<(usize, Vec<(usize, usize)>)> {
    let u = &b.unit;
    let col_scalars = |lo: usize, hi: usize| -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        for si in lo..hi {
            let Step::Region {
                side: Side::Col,
                chain,
            } = &u.steps[si]
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

/// A weight column's file, its matvecs and its config words after the first (`pcfg[0]` is each
/// PE's column-chain role, [`weight_params`]'s `R_COL`): the column's index in the row store and
/// its chains' roles, so the columns of a unit share a few files.
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
    let keeper = if spill { word(col) } else { String::new() };
    let fetch_role = if spill {
        word(u32::from(col + 1 < l.px()))
    } else {
        String::new()
    };
    for (si, st) in b.unit.steps.iter().enumerate() {
        // (kept after the step's own matvec: a held matvec output is cast once summed)
        let mut keep = String::new();
        if spill && si < b.unit.split {
            for v in held_rows(b, st) {
                let len = r.width(v) / l.py;
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
                    let leaf = (col + 1..l.px()).all(|c2| place::tile_dims(st, l, c2).is_none());
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
                    let hc = k / l.py;
                    drn = drn.max(hc);
                    let _ = writeln!(
                        body,
                        "  @fmovs(@set_dsd_length(D_drn, {hc}), fab.fin(q_xr, {hc}));"
                    );
                }
                Side::Col => {
                    let east = (col + 1..l.px()).any(|c2| place::tile_dims(st, l, c2).is_some());
                    if east {
                        let no = n / l.py;
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
const COMMON: &str = "const sys_mod = @import_module(\"<memcpy/memcpy>\", memcpy_params);
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
            let (ss, sa, role) = chain_cols(p, c, SEGMENT);
            let sm = colour_of(p, &format!("SM{ui}")).map_or(0xff, u32::from);
            // the root sends the cast; a member receives it
            let (ss, sm) = if role == 6 { (sm, 0xff) } else { (ss, sm) };
            vec![
                ("C_PI".into(), opt(ep(p, c, PATH_IN), false)),
                ("C_PO".into(), opt(ep(p, c, PATH_OUT), true)),
                ("C_RR".into(), rr),
                ("C_XR".into(), xr),
                ("C_LB".into(), lb),
                ("C_SA".into(), sa),
                ("C_SM".into(), sm),
                ("C_SS".into(), ss),
                ("S_ROLE".into(), role),
            ]
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

/// A unit's CX roots' file (`side` Row) or column root `idx`'s (Col), its arrays the host
/// fills, its printed regions.
#[allow(clippy::too_many_lines)]
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
    let rows = u.rows;
    let held = |v: ValueId| u.held.contains(&v);
    // a row-side value's chunk at this root
    let hc = |w: u32| w / l.py;
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
    let mut sends_cast = false;
    let mut kv_io = false;
    let mut chain_in = false;
    let mut path_out = false;
    let mut slot = 0usize;
    // the region slots, by step
    let mut slot_of: BTreeMap<usize, usize> = BTreeMap::new();

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
                    g.decls.push_str(&buf("pin", n));
                    g.put(late, &format!("  @fmovs(D_pin, fab.fin(q_pi, {n}));\n"));
                    g.loc(*value, idx * n, n, "D_pin", "0");
                    if held(*value) {
                        hold(&mut g, late, *value, idx * n, n, "D_pin");
                    }
                }
            }
            Step::Region { side: s2, chain } if *s2 == side => {
                let (ra, printed, params) = &b.regions[&si];
                if idx < ra.first || idx >= ra.first + ra.roots {
                    continue;
                }
                let ri = idx - ra.first;
                let c = ra.chunk;
                let k = slot;
                slot += 1;
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
                if !regions.iter().any(|(n, _)| *n == rname) {
                    regions.push((
                        rname.clone(),
                        help.map_or(&printed.csl, |h| &h.2.csl).clone(),
                    ));
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
                let _ = writeln!(
                    g.imports,
                    "const rg{k} = @import_module(\"{rname}\", .{{ .math = math, .seg = {seg} }});"
                );
                let mut nsc = 0;
                for (pk, input) in chain.ins.iter().enumerate() {
                    match params[pk] {
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
                                scal.push((si, pk));
                                g.put(
                                    late,
                                    &format!(
                                        "  rg{k}.sc{pk} = scal[t * NSC + {}];\n",
                                        scal.len() - 1
                                    ),
                                );
                                hsc.push(format!("scal[t * NSC + {}]", scal.len() - 1));
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
                for (ti, tb) in printed.tables.iter().enumerate() {
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
                            r: ri,
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
                g.put(late, &format!("  rg{k}.row();\n"));
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
                let d = j.head_dim;
                kv_io = true;
                for (c, f, len) in [
                    (q, q.from + idx * qw, qw),
                    (k, k.from + idx * d, d),
                    (v, v.from + idx * d, d),
                ] {
                    for (_, src, m) in g.pieces(c.base, f, len)? {
                        g.put(late, &format!("  @fmovs(fab.fout(q_ks, {m}), {src});\n"));
                    }
                }
            }
            Step::Attend { out, round } if side == Side::Col && idx < l.heads => {
                let j = u.joint.ok_or("an attend with no attention")?;
                let qw = j.q_heads / j.kv_heads * j.head_dim;
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
    let pnames: Vec<&str> = match side {
        Side::Row => vec![
            "C_PI", "C_PO", "C_RR", "C_XR", "C_LB", "C_SA", "C_SM", "C_SS", "S_ROLE",
        ],
        Side::Col if helped => vec![
            "C_CR", "C_XC", "C_KS", "C_KR", "C_LB", "C_GS", "C_GR", "C_HM", "C_HR",
        ],
        Side::Col => vec!["C_CR", "C_XC", "C_KS", "C_KR", "C_LB", "C_GS", "C_GR"],
    };
    for pn in &pnames {
        let _ = writeln!(t, "param {pn}: u16;");
    }
    t.push_str(COMMON);
    let (mut iq, mut oq) = (2, 2);
    sends_cast |= g.cast;
    chain_in |= !g.fetch.is_empty();
    let used = |q: &str| -> bool {
        match q {
            "q_pi" => g.decls.contains("var pin:"),
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
    if side == Side::Row && s_segment(p, ui) {
        t.push_str("var rcfg = [18]u32 { 0, 0, 0, 0, @as(u32, C_SM), @as(u32, C_SA), 0, 0, 0, 0, 0, 0, 0, @as(u32, C_SA), 0, @as(u32, C_SS), @as(u32, C_SM), @as(u32, S_ROLE) };\n");
        t.push_str("const seg = @import_module(\"ops/seg.csl\", .{ .fab = fab, .rcfg = &rcfg, .RQ = 0, .XM = @get_input_queue(6), .YM = @get_input_queue(7), .AO = @get_output_queue(4), .UT = @get_ut_id(5) });\n");
        inits.push_str("  @initialize_queue(@get_input_queue(6), .{ .color = @get_color(C_SM) });\n  @initialize_queue(@get_input_queue(7), .{ .color = @get_color(C_SA) });\n  @initialize_queue(@get_output_queue(4), .{ .color = @get_color(C_SS) });\n");
    }
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
    if side == Side::Row && s_segment(p, ui) {
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

/// A unit's helper PE (below a rest root): each own row, its part of the region's row
/// parameters off the root's cast (`pcfg[0]` its place `k`: columns `k q .. (k + 1) q` of the
/// root's chunk), the row's scalars after them, the region at `q` columns, its results into
/// its part of a zero vector summed up the helpers' chain to the root (`pcfg[1]` its role).
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
    let _ = writeln!(
        t,
        "const rg = @import_module(\"rg_{ui}_{si}.csl\", .{{ .math = math, .seg = math }});"
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

/// Whether unit `ui`'s CX roots all-reduce.
fn s_segment(p: &Program, ui: usize) -> bool {
    p.stream.units[ui].segment
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

fn kv_consts(p: &Program) -> (u32, u32, u32, f32) {
    let a = &p.shape.attn;
    let qh = a.q_heads / a.kv_heads;
    (qh, a.head_dim, qh * a.head_dim, a.sm_scale)
}

/// A K / V column's root: each unit's rows' [q | k | v] down the column in the store order,
/// then each query's output back to the unit that asked it.
fn kv_root_file(p: &Program, k: usize, ks: &place::KvSched) -> R<(String, String)> {
    let (qh, dc, qd, _) = kv_consts(p);
    let (lo, hi) = if p.round16 {
        ("1.0e-30", "1.0e30")
    } else {
        ("0.00390625", "32768.0")
    };
    let name = format!("kr_{k}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// K / V column {k}'s root (kernels-cerebras emit::stream; ops/cy_satt.csl)."
    );
    t.push_str("param memcpy_params;\n");
    for u in 0..ks.units.len() {
        let _ = writeln!(t, "param C_I{u}: u16;\nparam C_O{u}: u16;");
    }
    t.push_str("param C_AR: u16;\nparam C_AM: u16;\n");
    t.push_str(COMMON);
    let n = ks.units.len();
    for u in 0..n {
        let _ = writeln!(
            t,
            "const q_i{u} = @get_input_queue({});\nconst q_o{u} = @get_output_queue({});",
            2 + u,
            3 + u
        );
    }
    let _ = writeln!(
        t,
        "const q_ar = @get_input_queue({});\nconst q_am = @get_output_queue(2);",
        2 + n
    );
    let _ = writeln!(
        t,
        "const cy = @import_module(\"ops/cy_satt.csl\", .{{ .fab = fab, .math = math, .YO = q_am, .IN = q_ar, .Qh = {qh}, .Dc = {dc}, .LO = {lo}, .HI = {hi} }});"
    );
    let sw = qd + 2 * dc;
    t.push_str(&buf("sx", sw));
    t.push_str(&buf("ob", qd));
    let mut body = String::from("  stamp(0);\n");
    for &(u, rows, _) in &ks.stores {
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ @fmovs(D_sx, fab.fin(q_i{u}, {sw})); @fmovs(fab.fout(q_am, {sw}), D_sx); }} }}"
        );
    }
    for &(u, rows) in &ks.queries {
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ cy.query_q(D_ob); @fmovs(fab.fout(q_o{u}, {qd}), D_ob); }} }}"
        );
    }
    body.push_str("  stamp(1);\n  tsc[2] = cy.passes;\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n");
    for u in 0..n {
        let _ = writeln!(
            t,
            "  @initialize_queue(q_i{u}, .{{ .color = @get_color(C_I{u}) }});\n  @initialize_queue(q_o{u}, .{{ .color = @get_color(C_O{u}) }});"
        );
    }
    t.push_str("  @initialize_queue(q_ar, .{ .color = @get_color(C_AR) });\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n}\n");
    Ok((name, t))
}

/// A K / V column's attention PE: the rows it keeps (q too), then every query.
fn kv_pe_file(p: &Program, k: usize, ks: &place::KvSched) -> R<(String, String)> {
    let (qh, dc, _, sc) = kv_consts(p);
    let total: u32 = ks.stores.iter().map(|s| s.1).sum();
    let pya = p.stream.kv[k].pya;
    let ne = total.div_ceil(pya);
    let pmax = if p.round16 { "1.0e30" } else { "65504.0" };
    let name = format!("kp_{k}.csl");
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// K / V column {k}'s attention PEs (kernels-cerebras emit::stream; ops/satt.csl)."
    );
    t.push_str("param memcpy_params;\nparam C_AM: u16;\nparam C_S: u16;\nparam C_R: u16;\n");
    t.push_str(COMMON);
    // pcfg: the PE's index in the column, its chain role (the column's PEs one image)
    t.push_str(&cfg_decl(&[0, 0]));
    t.push_str("const q_am = @get_input_queue(2);\nconst q_r = @get_input_queue(3);\nconst q_s = @get_output_queue(2);\n");
    t.push_str("const red = @import_module(\"ops/schain.csl\", .{ .fab = fab, .IN1 = q_r, .OUT = q_s });\n");
    let _ = writeln!(
        t,
        "const att = @import_module(\"ops/satt.csl\", .{{ .fab = fab, .math = math, .red = red, .YM = q_am, .Qh = {qh}, .Dc = {dc}, .NE = {ne}, .PYA = {}, .SC = {sc:?}, .PMAX = {pmax}, .QH = true }});",
        pya
    );
    let mut body = String::from(
        "  stamp(0);\n  att.init(@as(u16, pcfg[0]));\n  red.role = @as(u16, pcfg[1]);\n",
    );
    for &(_, rows, p0) in &ks.stores {
        let _ = writeln!(
            body,
            "  {{ var t: u16 = 0; while (t < {rows}) : (t += 1) {{ att.store_q({p0} + t); }} }}"
        );
    }
    let _ = writeln!(
        body,
        "  {{ var t: u16 = 0; while (t < {total}) : (t += 1) {{ att.query_q(t); }} }}"
    );
    body.push_str("  stamp(1);\n  sys_mod.unblock_cmd_stream();\n");
    let _ = write!(t, "task go() void {{\n{body}}}\n");
    t.push_str("comptime {\n  @bind_local_task(go, T_GO);\n  @export_symbol(run);\n  @export_symbol(tsc_p, \"tsc\");\n  @export_symbol(pcfg_p, \"pcfg\");\n  @initialize_queue(q_am, .{ .color = @get_color(C_AM) });\n  @initialize_queue(q_r, .{ .color = @get_color(C_R) });\n  @initialize_queue(q_s, .{ .color = @get_color(C_S) });\n}\n");
    Ok((name, t))
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

/// `layout.csl`: every PE's code and params, every route, every exported name.
fn layout_file(p: &Program, tiles: &Tiles, arrays: &[HostArray]) -> String {
    let s = &p.stream;
    let mut t = String::new();
    let _ = writeln!(
        t,
        "// A stream reading's wafer program (kernels-cerebras emit::stream): {} x {} PEs.\nconst memcpy = @import_module(\"<memcpy/get_params>\", .{{ .width = {}, .height = {} }});\nlayout {{\n  @set_rectangle({}, {});",
        s.w, s.h, s.w, s.h, s.w, s.h
    );
    for y in 0..s.h {
        for x in 0..s.w {
            let c = Coord::new(x, y);
            let (file, params) = tiles
                .get(&c)
                .cloned()
                .unwrap_or_else(|| ("filler.csl".into(), Vec::new()));
            let mut ps = format!(".memcpy_params = memcpy.get_params({x})");
            for (k, v) in &params {
                let _ = write!(ps, ", .{k} = {v}");
            }
            let _ = writeln!(t, "  @set_tile_code({x}, {y}, \"{file}\", .{{ {ps} }});");
            for rt in p.fabric.routes(c) {
                let _ = writeln!(
                    t,
                    "  @set_color_config({x}, {y}, @get_color({}), {});",
                    rt.colour.0,
                    routes(rt.word)
                );
            }
        }
    }
    let mut names: BTreeMap<String, bool> = BTreeMap::new();
    for a in arrays {
        names.insert(a.sym.clone(), a.u16);
    }
    for (n, u16) in &names {
        let ty = if n == "tsc" || n == "pcfg" {
            "u32"
        } else if *u16 {
            "u16"
        } else {
            "f32"
        };
        let _ = writeln!(t, "  @export_name(\"{n}\", [*]{ty}, true);");
    }
    t.push_str("  @export_name(\"run\", fn() void);\n}\n");
    let _ = Colour(0);
    let _ = Class::new("");
    t
}
