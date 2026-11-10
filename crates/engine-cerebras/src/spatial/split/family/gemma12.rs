//! Gemma 4 12B: explicit unit origins, row roles (attention rows r < RA with Hc A dims and the
//! scores segments, plain rows with Hc W dims), 128 attention rows a unit so the 1024 window is
//! an exact ring (`RING`), handoffs of 2 Hc(r) + 1 words, and the head groups: head column c
//! takes the dims of the CX rows in its group, chained on `CH` to the group root on the last
//! unit's CX column; the token loop's embedding returns the same way into unit 0.
//!
//! Kernels: `weight_g12.csl`, `ctl_cx_g12.csl`, `ctl_cy_g12.csl`, `head_g12.csl`, `head_ctl.csl`.
//!
//! The floor: units of PY + 1 rows by PX + 1 columns in two unit columns, the first visited
//! down, the second up (unit 0 and the last unit side by side on row 0: the token loop's
//! embedding turns west on the rows of the head's group roots); the head (HC columns, one a head
//! group, around the head control column) east of them, HR rows.
//!
//! Chain member roles and root queues come from the fabric ([`PeIo`] slots `RowReduce`,
//! `ColReduce`, `Segment`, `HeadRowReduce`, `LoopReduce`); everything else a word needs (the
//! handoffs' relay turns, the CX queues, the head groups' chains, the head's relay roles and the
//! token loop's turns) is a function of the floor and the plan, worked out here.

use std::collections::{BTreeMap, BTreeSet};

use crate::spatial::split::abi::{KernelBuild, Kernels, Overrides};
use crate::spatial::split::block::head::{HeadCopies, Hemb};
use crate::spatial::split::block::{self, Block, Legacy, Whole, head, segments, token_loop, units};
use crate::spatial::split::emit::meta::Meta;
use crate::spatial::split::fabric::Palette;
use crate::spatial::split::family::Family;
use crate::spatial::split::floor::{Coord, Floor, FloorBuilder, Rect, UnitGrid, UnitLayout, skip};
use crate::spatial::split::kernels::Pe;
use crate::spatial::split::kernels::common::{self as abi, CT6, CxIo, HeadIo};
use crate::spatial::split::plan::{
    Act, AttnSpec, Contract, FamilyId, HeadPartition, HeadSpec, KvEntries, LayerKind, MlpSpec,
    NormSpec, Partition, Plan, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::split::program::Program;
use crate::spatial::split::role::{Kind, Place};
use crate::spatial::split::weights::{Binding, Copies, From, Slice, Transform, chunks, from};
use crate::spatial::split::wiring::{self, Chain, Class, Multicast, Path, Slot, Wiring};

pub struct Gemma12;

type R<T> = Result<T, String>;

// ---------------------------------------------------------------- colours

/// The head-group chains (`CH`), alternating by a member's place in its group.
pub const HEAD_GROUP: [Class; 2] = Class::pair("CH");

/// Gemma 12B's own colours past the base: the head-group chains (`CH`).
const OWN: [(Class, u8); 2] = [(HEAD_GROUP[0], 4), (HEAD_GROUP[1], 5)];

/// A class's colour as an rcfg word.
fn cw(c: Class) -> u32 {
    let own = OWN.iter().find(|(k, _)| *k == c).map(|&(_, v)| v);
    u32::from(
        own.or_else(|| block::standard_colour(c))
            .expect("gemma12: every class it reads has a colour"),
    )
}

// ---------------------------------------------------------------- geometries

/// What a geometry name fixes beyond the meta:
/// the scores segment width and the head groups' sizes (attention rows in groups of `ga`, plain
/// rows in groups of `gw`, never across the CY row).
struct Geometry {
    px: u32,
    py: u32,
    hr: u32,
    hca: u32,
    hcw: u32,
    ra: u32,
    swa: u32,
    ga: u32,
    gw: u32,
    k: u32,
    units: u32,
}

fn geometry(name: &str) -> R<Geometry> {
    match name {
        "tiny" => Ok(Geometry {
            px: 8,
            py: 8,
            hr: 18,
            hca: 2,
            hcw: 3,
            ra: 3,
            swa: 4,
            ga: 3,
            gw: 2,
            k: 3,
            units: 4,
        }),
        // attention rows 0-127: the 1024 window an exact ring of 8 x 128
        "g12" => Ok(Geometry {
            px: 344,
            py: 283,
            hr: 1153,
            hca: 12,
            hcw: 15,
            ra: 128,
            swa: 43,
            ga: 6,
            gw: 5,
            k: 6,
            units: 8,
        }),
        _ => Err(format!("split g12: no geometry {name}")),
    }
}

/// A named geometry as the prep reads it (`split-geo --geometry gemma12:<name>`), before any meta
/// exists: the unit shape and count, the attention rows and both row kinds' dims, the scores
/// segment width, the head (one column a row group; `hc`: every row's dims).
pub fn geometry_json(name: &str) -> R<serde_json::Value> {
    let g = geometry(name)?;
    let groups = groups_split(g.py, g.py / 2, g.ra, g.ga, g.gw);
    let hc: Vec<u32> = (0..g.py)
        .map(|r| if r < g.ra { g.hca } else { g.hcw })
        .collect();
    Ok(serde_json::json!({
        "PX": g.px, "PY": g.py, "RA": g.ra, "SWA": g.swa, "HcA": g.hca, "HcW": g.hcw, "K": g.k,
        "NU": g.units, "HC": groups.len(), "HR": g.hr, "groups": groups, "hc": hc,
    }))
}

/// Head groups: the attention rows in groups of `ga`, the plain rows above the CY row and those
/// below it in groups of `gw`.
pub fn groups_split(py: u32, yc: u32, ra: u32, ga: u32, gw: u32) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    for (lo, hi, gs) in [(0, ra.min(yc), ga), (ra.min(yc), yc, gw), (yc, py, gw)] {
        let mut r = lo;
        while r < hi {
            out.push((r..(r + gs).min(hi)).collect());
            r += gs;
        }
    }
    out
}

/// Runs of rows of one kind on one side of the CY row: (r0, r1, attention rows?)
///.
pub fn rowblocks(py: u32, yc: u32, ra: u32) -> Vec<(u32, u32, bool)> {
    let mut out = Vec::new();
    for (a, b) in [(0, yc), (yc, py)] {
        for (k0, k1, att) in [(0, ra, true), (ra, py, false)] {
            let (lo, hi) = (a.max(k0), b.min(k1));
            if lo < hi {
                out.push((lo, hi, att));
            }
        }
    }
    out
}

/// A head dim's columns a segment PE (rotated pairs side by side): `prep_g12.py::dcf`.
fn dcf(hd: u32, swa: u32) -> u32 {
    2 * (hd / 2).div_ceil(swa)
}

// ---------------------------------------------------------------- the plan's numbers

/// The plan's numbers as the kernels and the geometry use them (the prep's meta keys).
struct Dims {
    px: u32,
    py: u32,
    xc: u32,
    yc: u32,
    ra: u32,
    hca: u32,
    hcw: u32,
    swa: u32,
    k: u32,
    km: u32,
    qh: u32,
    hds: u32,
    hdg: u32,
    dcs: u32,
    dcg: u32,
    ic: u32,
    nes: u32,
    neg: u32,
    win: u32,
    h: u32,
    hc: u32,
    hr: u32,
    hxc: u32,
    vr: u32,
    hdc: u32,
    groups: Vec<Vec<u32>>,
}

impl Dims {
    fn of(plan: &Plan) -> R<Dims> {
        let p = &plan.partition;
        let s = &plan.shape;
        let (px, py) = (p.cols, p.rows);
        let (a, b) = match p.row_dims.as_slice() {
            [a, b] if a.attention && !b.attention && a.rows == (0, b.rows.0) && b.rows.1 == py => {
                (a, b)
            }
            _ => {
                return Err(
                    "split g12: row runs want attention rows 0..RA then plain rows RA..PY".into(),
                );
            }
        };
        let k = p.layers_per_unit;
        if s.layers.len() < (p.units * k) as usize {
            return Err(format!(
                "split g12: {} layers for {} units of {k}",
                s.layers.len(),
                p.units
            ));
        }
        let unit_kinds = |u: u32| -> Vec<bool> {
            s.layers[(u * k) as usize..((u + 1) * k) as usize]
                .iter()
                .map(|l| matches!(l, LayerKind::Global))
                .collect()
        };
        let first = unit_kinds(0);
        for u in 1..p.units {
            if unit_kinds(u) != first {
                return Err(format!(
                    "split g12: unit {u}'s layer kinds differ from unit 0's"
                ));
            }
        }
        let km = first
            .iter()
            .enumerate()
            .filter(|(_, g)| **g)
            .fold(0u32, |m, (l, _)| m | (1 << l));
        let window = s
            .layers
            .iter()
            .find_map(|l| match l {
                LayerKind::Sliding { window } => Some(*window),
                _ => None,
            })
            .ok_or("split g12: no sliding layer")?;
        let swa = p.segments.scores;
        let nseg = px / swa;
        if nseg == 0 || s.attn.kv_heads != nseg || !s.attn.q_heads.is_multiple_of(nseg) {
            return Err(format!(
                "split g12: {} q / {} kv heads over {nseg} segments",
                s.attn.q_heads, s.attn.kv_heads
            ));
        }
        let hds = s.attn.head_dim;
        let hdg = s.attn.global_head_dim.unwrap_or(hds);
        let ra = a.rows.1;
        let nes = p
            .kv_entries
            .sliding
            .ok_or("split g12: no sliding entries (NES)")?;
        let neg = p
            .kv_entries
            .global
            .ok_or("split g12: no global entries (NEG)")?;
        let groups = p.head.groups.clone().ok_or("split g12: no head groups")?;
        let hc_of = |r: u32| if r < ra { a.dims } else { b.dims };
        let hdc = groups
            .iter()
            .map(|g| g.iter().map(|&r| hc_of(r)).sum::<u32>())
            .max()
            .unwrap_or(0);
        Ok(Dims {
            px,
            py,
            xc: px / 2,
            yc: py / 2,
            ra,
            hca: a.dims,
            hcw: b.dims,
            swa,
            k,
            km,
            qh: s.attn.q_heads / nseg,
            hds,
            hdg,
            dcs: dcf(hds, swa),
            dcg: dcf(hdg, swa),
            ic: s.mlp.intermediate.div_ceil(px),
            nes,
            neg,
            win: if window == nes * ra { 0 } else { window },
            h: s.hidden,
            hc: p.head.cols,
            hr: p.head.rows,
            hxc: p.head.cols / 2,
            vr: p.head.vocab_rows,
            hdc,
            groups,
        })
    }

    /// Row `r`'s dims.
    fn hc_of(&self, r: u32) -> u32 {
        if r < self.ra { self.hca } else { self.hcw }
    }

    /// Head column `c`'s dims (its group's).
    fn nx(&self, c: usize) -> u32 {
        self.groups[c].iter().map(|&r| self.hc_of(r)).sum()
    }

    /// The group holding CX row `r`, and `r`'s index in it.
    fn group_of(&self, r: u32) -> Option<(usize, usize)> {
        self.groups
            .iter()
            .enumerate()
            .find_map(|(g, rows)| rows.iter().position(|&x| x == r).map(|k| (g, k)))
    }

    /// The dims a group's rows past index `k` hold (the words a chain member forwards).
    fn after(&self, g: usize, k: usize) -> u32 {
        self.groups[g][k + 1..].iter().map(|&r| self.hc_of(r)).sum()
    }
}

// ---------------------------------------------------------------- the floor's geometry

/// Where things are, from the floor: unit origins, the head block.
struct Geo<'a> {
    floor: &'a Floor,
    d: Dims,
}

impl Geo<'_> {
    fn units(&self) -> usize {
        self.floor.units.len()
    }

    fn origin(&self, u: usize) -> (u32, u32) {
        let r = self.floor.units[u].rect;
        (r.x, r.y)
    }

    /// Wafer y of unit `u`'s weight row `r`.
    fn ry(&self, u: usize, r: u32) -> u32 {
        self.origin(u).1 + skip(r, self.d.yc)
    }

    fn cx_x(&self, u: usize) -> u32 {
        self.origin(u).0 + self.d.xc
    }

    fn hx0(&self) -> u32 {
        self.floor.head.rect.x
    }

    /// Wafer x of head column `c`.
    fn hx(&self, c: u32) -> u32 {
        self.hx0() + c + u32::from(c >= self.d.hxc)
    }

    fn hctl_x(&self) -> u32 {
        self.hx0() + self.d.hxc
    }
}

// ---------------------------------------------------------------- Family

impl Gemma12 {
    /// The plan from `prep_g12.py`'s meta, the geometry its `geo` names, and the build's `G2` /
    /// `G2W` (default 16 / 24; the tiny 2 / 2).
    fn plan(&self, meta: &Meta) -> R<Plan> {
        let name = meta.str("geo").ok_or("split g12 meta: no geo")?.to_string();
        let g = geometry(&name)?;
        for (k, v) in [
            ("PX", g.px),
            ("PY", g.py),
            ("RA", g.ra),
            ("HcA", g.hca),
            ("HcW", g.hcw),
            ("K", g.k),
            ("HR", g.hr),
        ] {
            if let Some(m) = meta.int(k).filter(|&m| m != v) {
                return Err(format!("split g12 meta: {k} {m}, geometry {name} {v}"));
            }
        }
        let units = meta.int("NU").unwrap_or(g.units);
        if units != g.units {
            return Err(format!(
                "split g12 meta: NU {units}, geometry {name} {}",
                g.units
            ));
        }
        let groups = groups_split(g.py, g.py / 2, g.ra, g.ga, g.gw);
        let tiny = name == "tiny";
        let g2 = meta.int("G2").unwrap_or(if tiny { 2 } else { 16 });
        let g2w = meta.int("G2W").unwrap_or(if tiny { 2 } else { 24 });
        let km = meta.need("KM")?;
        let nes = meta.need("NES")?;
        let neg = meta.need("NEG")?;
        let win = meta.int("WIN").unwrap_or(0);
        let window = if win == 0 { nes * g.ra } else { win };
        let hds = meta.need("HDS")?;
        let nseg = g.px / g.swa;
        let qh = meta.need("Qh")?;
        let ic = meta.need("Ic")?;
        let layers = (0..units * g.k)
            .map(|l| {
                if km >> (l % g.k) & 1 == 1 {
                    LayerKind::Global
                } else {
                    LayerKind::Sliding { window }
                }
            })
            .collect();
        let plan = Plan {
            shape: Shape {
                gdn: None,
                family: FamilyId::Gemma12,
                hidden: meta.need("H")?,
                hidden_placed: meta.need("HP")?,
                vocab: meta.need("V")?,
                layers,
                attn: AttnSpec {
                    q_heads: qh * nseg,
                    kv_heads: nseg,
                    head_dim: hds,
                    global_head_dim: Some(meta.need("HDG")?),
                    k_eq_v: true,
                    qk_norm: true,
                    ..Default::default()
                },
                // (the prep's placed width: Ic a column)
                mlp: MlpSpec {
                    intermediate: ic * g.px,
                    act: Act::GeluTanh,
                    layer_widths: Vec::new(),
                },
                moe: None,
                ple: None,
                head: HeadSpec {
                    tied: true,
                    final_softcap: Some(meta.float("SOFTCAP").unwrap_or(30.0) as f32),
                },
                // (the RoPE tables are staged, `rot`; the build reads no theta)
                rope: RopeSpec {
                    theta: meta.float("ROPE_THETA").unwrap_or(0.0) as f32,
                    rotary_dims: hds,
                },
                norm: NormSpec {
                    eps: meta.float("EPS").unwrap_or(1e-6) as f32,
                    sandwich: true,
                    ..Default::default()
                },
            },
            partition: Partition {
                unit_grid: None,
                units,
                layers_per_unit: g.k,
                rows: g.py,
                cols: g.px,
                row_dims: vec![
                    RowRun {
                        rows: (0, g.ra),
                        dims: g.hca,
                        attention: true,
                    },
                    RowRun {
                        rows: (g.ra, g.py),
                        dims: g.hcw,
                        attention: false,
                    },
                ],
                kv_rows: g.ra,
                kv_entries: KvEntries {
                    per_row: neg,
                    global: Some(neg),
                    sliding: Some(nes),
                },
                reduce: ReduceGroups {
                    row: g2,
                    col: g2,
                    head: g2,
                    token_loop: g2w,
                },
                segments: Segments {
                    scores: g.swa,
                    ..Default::default()
                },
                attn_cols: None,
                head: HeadPartition {
                    cols: groups.len() as u32,
                    rows: g.hr,
                    vocab_rows: meta.need("VR")?,
                    groups: Some(groups),
                },
            },
            contract: Contract {
                positions: neg * g.ra,
                rope_levels: 4,
                rope_bits: 3,
                cap: meta.int("CAP").unwrap_or(256),
                loop_len: meta.int("LOOP").unwrap_or(32),
                token_loop: true,
            },
        };
        // the derived kernel shape must be the prep's
        let d = Dims::of(&plan)?;
        for (k, v) in [
            ("KM", d.km),
            ("Qh", d.qh),
            ("DcS", d.dcs),
            ("DcG", d.dcg),
            ("Ic", d.ic),
            ("HDc", d.hdc),
            ("HC", d.hc),
            ("WIN", d.win),
        ] {
            if let Some(m) = meta.int(k).filter(|&m| m != v) {
                return Err(format!("split g12 meta: {k} {m}, the plan makes {v}"));
            }
        }
        Ok(plan)
    }

    /// Units in two unit columns (down the first, up the second), the head east of them.
    fn floor(&self, plan: &Plan) -> R<Floor> {
        let d = Dims::of(plan)?;
        let n = plan.partition.units;
        if n == 0 || !n.is_multiple_of(2) {
            return Err(format!(
                "split g12: {n} units (two unit columns, unit 0 and the last on row 0)"
            ));
        }
        let (uw, uh) = (d.px + 1, d.py + 1);
        let uny = n / 2;
        // two unit columns: column 0 top down, column 1 bottom up
        let origins: Vec<(u32, u32)> = (0..n)
            .map(|u| {
                if u < uny {
                    (0, uh * u)
                } else {
                    (uw, uh * (n - 1 - u))
                }
            })
            .collect();
        let grid = UnitGrid {
            cols: d.px,
            rows: d.py,
            cx: d.xc,
            cy: d.yc,
            layout: UnitLayout::Explicit { origins },
            layers_per_unit: plan.partition.layers_per_unit,
        };
        let mut fb = FloorBuilder::units(&grid, (0, 0))?;
        let hx0 = fb.w;
        let head = fb.head_east(hx0, d.hc, d.hr, true);
        Ok(fb.finish(head))
    }

    /// The units' chains (scores segments on the attention rows only, no GDN, no o-norm), the
    /// handoffs, the head groups' input and token loop, the head rows.
    fn wire(&self, plan: &Plan, floor: &Floor) -> R<Wiring> {
        let g = Geo {
            floor,
            d: Dims::of(plan)?,
        };
        let d = &g.d;
        if d.swa == 0 || !d.xc.is_multiple_of(d.swa) {
            return Err(format!(
                "split geometry: SWA {} does not divide XC {} (a segment would span the control column)",
                d.swa, d.xc
            ));
        }
        check_groups(d)?;
        let mut w = Wiring::default();
        units::unit_chains(plan, floor, &mut w)?;
        for u in &floor.units {
            for r in 0..d.ra {
                let pts: Vec<Coord> = (0..d.px)
                    .map(|c| Coord::new(u.col_x(c), u.row_y(r)))
                    .collect();
                let l1 = [segments::SCORE_L1[0], segments::SCORE_L1[1]];
                wiring::segments(
                    &pts,
                    d.swa,
                    true,
                    (l1, segments::SCORE_CAST, segments::SEGMENT),
                    &mut w,
                )?;
            }
        }
        head_in(&g, &mut w)?;
        head::head_rows(plan, floor, &mut w)?;
        token_loop(&g, plan.partition.reduce.token_loop, &mut w)?;
        Ok(w)
    }

    fn palette(&self, _plan: &Plan) -> Palette {
        block::standard_palette(&OWN)
    }

    /// `mkfull_g12.py`'s kinds: each kind's source, owned symbols and tile params (in the
    /// layout's order). A weight / CX PE's `Hc` and a weight PE's `ATT` follow its row run
    /// (attention rows `Hc A`, `ATT 1`; plain rows `Hc W`, `ATT 2`): the params state the plain
    /// rows', [`crate::spatial::split::block::legacy_overrides`] sets each PE's.
    fn kinds(&self, prog: &Program) -> R<BTreeMap<Kind, KernelBuild>> {
        let d = Dims::of(&prog.plan)?;
        let ring = u32::from(d.win == 0);
        let mut seen = BTreeSet::new();
        let mut out = BTreeMap::new();
        for (_, role) in prog.floor.roles.iter() {
            let kind = role.kind;
            if !seen.insert(kind) {
                continue;
            }
            let kb = match kind {
                Kind::WEIGHT => KernelBuild::ints(
                    "weight_g12.csl",
                    &["w", "rcfg", "tdbg"],
                    &[
                        ("K", d.k),
                        ("Hc", d.hcw),
                        ("Ic", d.ic),
                        ("MV", 1),
                        ("KM", d.km),
                        ("ATT", 2),
                        ("Qh", d.qh),
                        ("DcS", d.dcs),
                        ("DcG", d.dcg),
                        ("NES", d.nes),
                        ("NEG", d.neg),
                        ("PYW", d.ra),
                        ("HDS", d.hds),
                        ("HDG", d.hdg),
                        ("WIN", d.win),
                        ("WSPEC", 0),
                        ("RING", ring),
                        ("DIAG", 0),
                        ("FIFOW", 32),
                        ("TSTAMP", 0),
                    ],
                ),
                Kind::CX => KernelBuild::ints(
                    "ctl_cx_g12.csl",
                    &["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
                    &[
                        ("K", d.k),
                        ("Hc", d.hcw),
                        ("DIAG", 0),
                        ("FIFOW", 32),
                        ("XKEEP", 0),
                    ],
                ),
                Kind::CY => KernelBuild::ints(
                    "ctl_cy_g12.csl",
                    &["qkw", "rot", "tab", "rcfg", "tdbg"],
                    &[
                        ("K", d.k),
                        ("KM", d.km),
                        ("Qh", d.qh),
                        ("DcS", d.dcs),
                        ("DcG", d.dcg),
                        ("Ic", d.ic),
                        ("HR", d.h),
                        ("DIAG", 0),
                        ("FIFOW", 32),
                        ("ATTL", 2),
                        ("TSTAMP", 0),
                        ("HDS", d.hds),
                        ("HDG", d.hdg),
                    ],
                ),
                Kind::HEAD => KernelBuild::ints(
                    "head_g12.csl",
                    &["hw", "hg", "rcfg", "hemb", "tdbg"],
                    &[("VR", d.vr), ("HDc", d.hdc), ("NPX", 1), ("HACC16", 0)],
                ),
                Kind::HEAD_CTL => KernelBuild::ints(
                    "head_ctl.csl",
                    &["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"],
                    &[("VR", d.vr), ("FIFOW", 32), ("WFORCE", 0), ("WFN", d.hr)],
                ),
                // the filler: the empty kernel the layout writes itself
                Kind::IDLE => continue,
                other => return Err(format!("split g12: no {} PEs", other.name)),
            };
            out.insert(kind, kb);
        }
        Ok(out)
    }

    /// The placement, in order: per unit and row block (attention / plain
    /// rows, each side of the CY row), each half of the columns: the weights' rcfg and their
    /// tiles 16 rows a copy; the CX column's rcfg and gammas; then each half of the CY row: rcfg,
    /// q / k norm weights, RoPE tables; the head by halves (`hw0` / `hw1`) 64 rows a copy: rcfg,
    /// tiles, the final norm's gammas (`hg`, the same every row); the head control column's rcfg.
    fn bindings(&self, prog: &Program) -> R<Vec<Binding>> {
        let g = Geo {
            floor: &prog.floor,
            d: Dims::of(&prog.plan)?,
        };
        let d = &g.d;
        let mut out = Copies::default();
        let f16 = |names: &[&str]| -> From {
            names
                .iter()
                .map(|n| (n.to_string(), vec![Transform::Bf16ToF16]))
                .collect()
        };
        let (px, xc, yc) = (d.px, d.xc, d.yc);
        for u in 0..g.units() {
            let (x0u, y0u) = g.origin(u);
            let un = u as u32;
            // each row run: its two halves' weight blocks, then its CX column
            for (r0, r1, att) in rowblocks(d.py, yc, d.ra) {
                let (wname, rb0) = if att {
                    (format!("w{u}a"), 0)
                } else {
                    (format!("w{u}w"), d.ra)
                };
                for (c0, c1) in [(0, xc), (xc, px)] {
                    let x0 = x0u + skip(c0, xc);
                    out.rcfg(Rect::new(x0, y0u + skip(r0, yc), c1 - c0, r1 - r0));
                    for (rb, re) in chunks(r0, r1, 16) {
                        out.slice(
                            "w",
                            Rect::new(x0, y0u + skip(rb, yc), c1 - c0, re - rb),
                            Slice::of(&wname, None, vec![(rb - rb0, re - rb0), (c0, c1)]),
                            true,
                            f16(&[
                                "self_attn.{q,k,v,o}_proj.weight",
                                "mlp.{gate,up,down}_proj.weight",
                            ]),
                        );
                    }
                }
                let cx = Rect::new(x0u + xc, y0u + skip(r0, yc), 1, r1 - r0);
                out.rcfg(cx);
                let gam = Slice::of(
                    if att { "gama" } else { "gamw" },
                    None,
                    vec![(un, un + 1), (r0 - rb0, r1 - rb0)],
                );
                out.slice("gam", cx, gam, false, Vec::new());
            }
            for (c0, c1) in [(0, xc), (xc, px)] {
                let r = Rect::new(x0u + skip(c0, xc), y0u + yc, c1 - c0, 1);
                out.rcfg(r);
                let qkw = Slice::of("qkw", None, vec![(un, un + 1), (c0, c1)]);
                out.slice("qkw", r, qkw, false, Vec::new());
                out.slice(
                    "rot",
                    r,
                    Slice::of("rot", None, vec![(c0, c1)]),
                    false,
                    Vec::new(),
                );
            }
        }
        // the head, in its two halves' files (one is over the 2 GiB a staged file may take)
        HeadCopies {
            cols: d.hc,
            rows: d.hr,
            hw: |i, rows, (c0, c1)| Slice::of(&format!("hw{i}"), None, vec![rows, (0, c1 - c0)]),
            hw_from: vec![(
                "embed_tokens.weight".into(),
                vec![
                    Transform::PadLastRow { to: d.hr * d.vr },
                    Transform::Bf16ToF16,
                ],
            )],
            hg_from: from("norm.weight"),
            hemb: Hemb::None,
        }
        .put(&mut out, g.hx0());
        Ok(out.0)
    }
}

/// Every group is consecutive rows on one side of the CY row, and the groups cover the rows once
///.
fn check_groups(d: &Dims) -> R<()> {
    if d.groups.len() as u32 != d.hc {
        return Err(format!(
            "split g12: {} head groups for {} head columns",
            d.groups.len(),
            d.hc
        ));
    }
    let mut all: Vec<u32> = d.groups.iter().flatten().copied().collect();
    all.sort_unstable();
    if all != (0..d.py).collect::<Vec<_>>() {
        return Err("split g12: the head groups do not cover every row once".into());
    }
    for g in &d.groups {
        let (a, b) = (g[0], *g.last().unwrap());
        if g.iter().enumerate().any(|(i, &r)| r != a + i as u32) || (a < d.yc && d.yc <= b) {
            return Err(format!(
                "split g12: head group {g:?} not consecutive on one side"
            ));
        }
    }
    Ok(())
}

fn head_in(g: &Geo, w: &mut Wiring) -> R<()> {
    let d = &g.d;
    let ul = g.units() - 1;
    let xcx = g.cx_x(ul);
    for (c, grp) in d.groups.iter().enumerate() {
        let y = g.ry(ul, grp[0]);
        let hx = g.hx(c as u32);
        w.path(Path {
            pts: (xcx..=hx).map(|x| Coord::new(x, y)).collect(),
            class: units::HANDOFF_ACROSS,
            send: units::X_OUT,
            recv: head::HEAD_IN,
        });
        w.multicast(Multicast {
            pts: (0..d.hr).map(|yy| Coord::new(hx, yy)).collect(),
            src: y as usize,
            class: head::HEAD_COL_CAST,
            slot: Slot::io("g12.hm"),
        });
        for k in 1..grp.len() {
            w.path(Path {
                pts: vec![
                    Coord::new(xcx, g.ry(ul, grp[k])),
                    Coord::new(xcx, g.ry(ul, grp[k - 1])),
                ],
                class: HEAD_GROUP[k % 2],
                send: Slot::io("g12.ch.out"),
                recv: Slot::io("g12.ch.in"),
            });
        }
    }
    Ok(())
}

/// The token loop: the max-reduce down the head control column
/// into its middle row (`WA` / `WB`, `WC2` in groups of `G2W`; the winner multicast `WM`), and the
/// winner's embedding back: up each head column (`VA`), down to the row of its group root on
/// unit 0 (`VS`), west into unit 0's CX of that row (`HE`), down the group's chain (`CH`).
fn token_loop(g: &Geo, g2w: u32, w: &mut Wiring) -> R<()> {
    let d = &g.d;
    let hr = d.hr;
    let hcol = g.hctl_x();
    w.chain(Chain {
        pts: (0..hr).map(|y| Coord::new(hcol, y)).collect(),
        root: (hr / 2) as usize,
        l1: [token_loop::LOOP_L1[0], token_loop::LOOP_L1[1]],
        l2: wiring::level2(token_loop::LOOP_L2, g2w),
        cast: token_loop::LOOP_CAST,
        loopback: false,
        slot: token_loop::LOOP_REDUCE,
    });
    if g.origin(0).1 != g.origin(g.units() - 1).1 {
        return Err("split g12: unit 0 shares the last unit's rows".into());
    }
    let xcx = g.cx_x(0);
    for (c, grp) in d.groups.iter().enumerate() {
        let x = g.hx(c as u32);
        let te = g.ry(0, grp[0]);
        w.path(Path {
            pts: (0..hr).rev().map(|y| Coord::new(x, y)).collect(),
            class: token_loop::EMB_UP,
            send: Slot::io("g12.va.out"),
            recv: Slot::io("g12.va.in"),
        });
        if te > 0 {
            w.path(Path {
                pts: (0..=te).map(|y| Coord::new(x, y)).collect(),
                class: token_loop::EMB_DOWN,
                send: Slot::io("g12.vs.out"),
                recv: Slot::io("g12.vs.in"),
            });
        }
        w.path(Path {
            pts: (xcx..=x).rev().map(|xx| Coord::new(xx, te)).collect(),
            class: token_loop::EMB_WEST,
            send: Slot::io("g12.he.out"),
            recv: token_loop::EMB_IN,
        });
        for k in 1..grp.len() {
            w.path(Path {
                pts: vec![
                    Coord::new(xcx, g.ry(0, grp[k - 1])),
                    Coord::new(xcx, g.ry(0, grp[k])),
                ],
                class: HEAD_GROUP[k % 2],
                send: Slot::io("g12.ch.out"),
                recv: Slot::io("g12.ch.in"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- rcfg words

/// A CX's head-group chain: input / output colours, the words it forwards
/// (x to the head, the embedding back), whether it is a member of the x / embedding chains.
#[derive(Default)]
struct CxChain {
    ci: Option<u32>,
    co: Option<u32>,
    nfx: u32,
    nfe: u32,
    member: bool,
    emember: bool,
}

/// A CX's head-group chain: on the last unit the x chain north into the group root
/// (`head_in`), on unit 0 the embedding's chain south from it (`wloop`, applied after).
fn cx_chain(g: &Geo, cx: &CxPe) -> CxChain {
    let d = &g.d;
    let mut ch = CxChain::default();
    let Some((gi, k)) = d.group_of(cx.row) else {
        return ch;
    };
    let len = d.groups[gi].len();
    let hcol = |k: usize| cw(HEAD_GROUP[k % 2]);
    if cx.last {
        ch.nfx = d.after(gi, k);
        if k > 0 {
            ch.co = Some(hcol(k));
            ch.member = true;
        }
        if k + 1 < len {
            ch.ci = Some(hcol(k + 1));
        }
    }
    if cx.first {
        ch.nfe = d.after(gi, k);
        if k > 0 {
            ch.ci = Some(hcol(k));
            ch.emember = true;
        }
        if k + 1 < len {
            ch.co = Some(hcol(k + 1));
        }
    }
    ch
}

impl Gemma12 {
    /// The weight kernel's 18 route slots, the others' 12.
    fn route_slots(kind: Kind) -> usize {
        match kind {
            Kind::WEIGHT => 18,
            Kind::RELAY => 4,
            _ => 12,
        }
    }

    /// The shared words ([`abi`]) with the 12B's: a weight PE's handoff of 2 Hc(r) + 1 words and
    /// its flags (column 0 1, the flush unit 4); a CX's head-group chain (the flush unit's 8 |
    /// 16, a member of the x chain 64 | 2, of the embedding's 4 | 128; its in / out colours, the
    /// words it forwards); the CY's flush flag; a head PE's group (its root takes the group's x
    /// and multicasts it, no relay north) and its group's x words.
    fn rcfg(&self, prog: &Program, at: Coord) -> R<Vec<u32>> {
        let g = Geo {
            floor: &prog.floor,
            d: Dims::of(&prog.plan)?,
        };
        let d = &g.d;
        let pe = Pe::new(prog, at);
        let flush = |u: u32| u as usize == g.units() - 1;
        match prog.floor.roles.get(at).place {
            Place::Weight { unit, row, col } => {
                let (pre, post) = (pe.turn(units::RELAY_PRE), pe.turn(units::RELAY_POST));
                let fl = u32::from(col == 0) | if flush(unit) { 4 } else { 0 };
                let hn = 2 * d.hc_of(row) + 1;
                abi::weight(&pe, 18, row, [fl, 0], hn, [pre, post])
            }
            Place::Cx {
                row, first, last, ..
            } => {
                let cx = CxPe { row, first, last };
                let pk = pe.parks(4)?;
                let ch = cx_chain(&g, &cx);
                let mut fl = 0;
                if cx.last {
                    fl |= 8 | 16 | if ch.member { 64 | 2 } else { 0 };
                }
                if cx.first && ch.emember {
                    fl |= 4 | 128;
                }
                let (ci, co) = (ch.ci.unwrap_or(pk[3]), ch.co.unwrap_or(pk[3]));
                let x = CxIo {
                    first: cx.first,
                    inn: pe.colour(units::X_IN),
                    out: pe.colour(units::X_OUT),
                    emb: pe.colour(token_loop::EMB_IN),
                    flags: fl,
                    finals: vec![ci],
                    tail: vec![
                        ci,
                        co,
                        if cx.last { ch.nfx } else { 0 },
                        if cx.first { ch.nfe } else { 0 },
                    ],
                };
                abi::cx(&pe, x, 4, &CT6)
            }
            Place::Cy { unit, .. } => {
                let mut w = abi::cy(&pe, None)?;
                w.extend_from_slice(&[if flush(unit) { 4 } else { 0 }, 0, 0, 0, 0]);
                Ok(w)
            }
            Place::Head { row, col } => {
                let grp0 = d.groups[col as usize][0];
                let xr = if row == g.ry(g.units() - 1, grp0) {
                    2
                } else {
                    0
                };
                let tr = u32::from(row == 0) | if row == g.ry(0, grp0) { 2 } else { 0 };
                let h = HeadIo {
                    row,
                    role: xr,
                    relay_north: false,
                    turns: tr,
                    token_loop: true,
                };
                let mut w = abi::head(&pe, h)?;
                w.push(d.nx(col as usize));
                Ok(w)
            }
            Place::HeadCtl { row, .. } => abi::hctl(&pe, row * d.vr, true),
            other => Err(format!("{at:?}: no {other:?} in Gemma 12B")),
        }
    }
}

/// A CX PE's place, as [`cx_chain`] reads it.
struct CxPe {
    row: u32,
    first: bool,
    last: bool,
}

impl Whole for Gemma12 {
    fn floor(&self, plan: &Plan) -> R<Floor> {
        Gemma12::floor(self, plan)
    }

    fn wire(&self, plan: &Plan, floor: &Floor) -> R<Wiring> {
        Gemma12::wire(self, plan, floor)
    }

    fn palette(&self, plan: &Plan) -> Palette {
        Gemma12::palette(self, plan)
    }

    /// Attention rows below `RA` (`ATT 1`), plain rows past it (`ATT 2`).
    fn overrides(&self, prog: &Program, o: &mut Overrides) -> R<()> {
        let ra = Dims::of(&prog.plan)?.ra;
        let floor = &prog.floor;
        block::legacy_overrides(
            prog,
            &|at| match floor.roles.get(at).place {
                Place::Weight { row, .. } => Some(if row < ra { 1 } else { 2 }),
                _ => None,
            },
            o,
        )
    }

    fn bindings(&self, prog: &Program) -> R<Vec<Binding>> {
        Gemma12::bindings(self, prog)
    }
}

impl Family for Gemma12 {
    fn id(&self) -> FamilyId {
        FamilyId::Gemma12
    }

    fn plan(&self, meta: &Meta) -> R<Plan> {
        Gemma12::plan(self, meta)
    }

    fn blocks(&self, plan: &Plan) -> R<Vec<Box<dyn Block>>> {
        Ok(vec![Box::new(Legacy::new(Gemma12, plan))])
    }

    fn kernels(&self, prog: &Program) -> R<Kernels> {
        Ok(block::legacy_kernels(
            self.kinds(prog)?,
            Gemma12::route_slots,
            |pe| Gemma12.rcfg(pe.prog, pe.at),
        ))
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        SYMBOLS
    }
}

/// Every name the kernels export, in export order (`mkfull_g12.py`'s `NAMES`).
const SYMBOLS: &[(&str, &str)] = &[
    ("w", "u16"),
    ("rcfg", "u32"),
    ("tdbg", "u32"),
    ("x", "f32"),
    ("xdbg", "f32"),
    ("xred", "f32"),
    ("gam", "f32"),
    ("qkw", "f32"),
    ("rot", "f32"),
    ("tab", "u32"),
    ("hw", "u16"),
    ("hg", "f32"),
    ("hemb", "u16"),
    ("res", "f32"),
    ("ring", "u32"),
    ("lgt", "f32"),
    ("force", "u32"),
    ("nforce", "u32"),
];
