//! gpt-oss: units of AX attention + MX MoE columns (CX after the attention columns), PY rows,
//! a snake that starts upward (even unit columns), the head as a band below the units with its
//! control row, router segments (`SWR`), the band's handoff / embedding classes (`HV2`, `HE`,
//! `V1`, `W2`).
//!
//! Kernels: `weight_gptoss_att.csl` / the MoE weight kernel, its CY halves (`cya` / `cym`), CX,
//! head, head control (see `tools/split/gptoss/mkfull_gptoss.py`).
//!
//! Layout. A unit is [AX attention columns | CX | MX MoE columns] by PY weight rows, its CY row at
//! local y = PY / 2; two unit columns, unit column 0 visited from its bottom unit up, column 1 top
//! down, so the first and the last unit sit on the band. The band below the units: PY dims-rows
//! (dims-row i takes the last unit's CX row i) and the head control row in their middle, by HVC
//! vocabulary columns. The last unit's x reaches the band through its MoE columns (row r turns down
//! at MoE column r, `BandVert`), one multicast along the dims-row (`HeadColCast`); the token loop
//! max-reduces along the control row, and the winner's embedding returns west along each dims-row
//! (`EmbWest`, the band bus), up MoE column i of unit 0 (`BandV1`) and west into its CX (`BandW2`).

use std::collections::BTreeMap;

use crate::spatial::split::emit::meta::Meta;
use crate::spatial::split::emit::params::{KernelBuild, Value};
use crate::spatial::split::emit::rcfg::RcfgAbi;
use crate::spatial::split::fabric::Palette;
use crate::spatial::split::family::Family;
use crate::spatial::split::family::abi::{self, CT5, CT6, Pe, word};
use crate::spatial::split::floor::{
    Coord, CtlLine, Floor, FloorBuilder, HeadBlock, Rect, UnitGrid, UnitLayout, skip,
};
use crate::spatial::split::plan::{
    Act, AttnSpec, Contract, FamilyId, HeadPartition, HeadSpec, KvEntries, LayerKind, MlpSpec,
    MoeSpec, NormSpec, Partition, Plan, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::split::program::Program;
use crate::spatial::split::role::{Cx, Cy, Head, HeadCtl, PeRole, RoleKind, Weight, WeightPart};
use crate::spatial::split::weights::{Binding, Copies, Slice, Source, Transform, chunks};
use crate::spatial::split::wiring::{self, Chain, Class, Half, Multicast, Path, Slot, Wiring};

pub struct GptOss;

/// (AX, MX, PY, UNX, UNY, Hc, SWA, SWR, HVC) of a geometry name.
fn table(name: &str) -> Result<[u32; 9], String> {
    match name {
        "tiny" => Ok([4, 8, 4, 2, 1, 8, 2, 8, 26]),
        "full" => Ok([64, 288, 160, 2, 6, 18, 8, 32, 748]),
        _ => Err(format!("gpt-oss geometry {name}?")),
    }
}

/// A named geometry as the prep reads it (`split-geo --geometry gptoss:<name>`), before any meta
/// exists.
pub fn geometry(name: &str) -> Result<serde_json::Value, String> {
    let [ax, mx, py, unx, uny, hc, swa, swr, hvc] = table(name)?;
    Ok(serde_json::json!({
        "AX": ax, "MXc": mx, "PY": py, "UNX": unx, "UNY": uny, "NU": unx * uny, "Hc": hc,
        "SWA": swa, "SWR": swr, "HVC": hvc,
    }))
}

// The family's queue slots past the generic ones.
const ROUTER: Slot = Slot::Family("router");
/// The last unit's x leaving its MoE row for the band (the turn's in / out).
const BAND_IN: Slot = Slot::Family("band_in");
const BAND_OUT: Slot = Slot::Family("band_out");
const BAND_CAST: Slot = Slot::Family("band_cast");
/// The band bus (`HE`): its east end's send, dims-row 0's take.
const BUS_TX: Slot = Slot::Family("bus_tx");
const BUS: Slot = Slot::Family("bus");
/// The embedding's return: up unit 0's MoE column (`V1`) and its turn west (`W2`).
const V1_TX: Slot = Slot::Family("v1_tx");
const EMB_IN: Slot = Slot::Family("emb_in");
const EMB_OUT: Slot = Slot::Family("emb_out");

/// gpt-oss's own colours past the base: the band's handoff (`HV2`), its bus (`HE`, the
/// embedding's return west), the return up unit 0's MoE column (`V1`) and west into its CX (`W2`).
const HE: u8 = 10;
const OWN: [(Class, u8); 4] = [
    (Class::EmbWest, HE),
    (Class::BandVert, 9),
    (Class::BandV1, 8),
    (Class::BandW2, 13),
];

/// Every physical quantity the family derives from a plan.
struct Geom {
    ax: u32,
    mx: u32,
    px: u32,
    py: u32,
    /// Local CX column (= AX), local CY row.
    cx: u32,
    cy: u32,
    uw: u32,
    uh: u32,
    unx: u32,
    uny: u32,
    /// The band's first row, its control row's local index, its columns.
    hb0: u32,
    hxc: u32,
    hvc: u32,
    hcc: u32,
    swa: u32,
    swr: u32,
    vr: u32,
    wl: bool,
    g2: ReduceGroups,
    w: u32,
    h: u32,
}

impl Geom {
    fn of(plan: &Plan) -> Result<Geom, String> {
        let p = &plan.partition;
        let ax = p.attn_cols.ok_or("gpt-oss plan: no attention columns")?;
        let (px, py) = (p.cols, p.rows);
        if ax >= px {
            return Err("gpt-oss plan: no MoE columns".into());
        }
        if !p.units.is_multiple_of(2) {
            return Err(
                "gpt-oss: an even unit-column count (the last unit next to the band)".into(),
            );
        }
        let (unx, uny) = (2, p.units / 2);
        let (uw, uh) = (px + 1, py + 1);
        let hvc = p.head.cols;
        let hb0 = uny * uh;
        let hcc = p.row_dims.first().ok_or("gpt-oss plan: no row dims")?.dims;
        Ok(Geom {
            ax,
            mx: px - ax,
            px,
            py,
            cx: ax,
            cy: py / 2,
            uw,
            uh,
            unx,
            uny,
            hb0,
            hxc: py / 2,
            hvc,
            hcc,
            swa: p.segments.scores,
            swr: p
                .segments
                .router
                .ok_or("gpt-oss plan: no router segments")?,
            vr: p.head.vocab_rows,
            wl: plan.contract.token_loop,
            g2: p.reduce,
            w: (unx * uw).max(hvc),
            h: hb0 + py + 1,
        })
    }

    /// Dims-row i's y.
    fn by(&self, i: u32) -> u32 {
        self.hb0 + skip(i, self.hxc)
    }

    /// The wafer y of unit row `r` of the unit at `y0`.
    fn row_y(&self, y0: u32, r: u32) -> u32 {
        y0 + skip(r, self.cy)
    }

    /// Every PE's role.
    fn role(&self, units: &[(u32, u32)], c: Coord) -> PeRole {
        let (x, y) = (c.x, c.y);
        if y >= self.hb0 {
            if x >= self.hvc {
                return PeRole::Idle;
            }
            let ly = y - self.hb0;
            if ly == self.hxc {
                return PeRole::HeadCtl(HeadCtl {
                    row: x,
                    root: x == self.hvc / 2,
                });
            }
            return PeRole::Head(Head {
                row: ly - u32::from(ly > self.hxc),
                col: x,
            });
        }
        let (ux, uy) = (x / self.uw, y / self.uh);
        if ux >= self.unx || uy >= self.uny {
            return PeRole::Idle;
        }
        let o = (ux * self.uw, uy * self.uh);
        let Some(unit) = units.iter().position(|&u| u == o) else {
            return PeRole::Idle;
        };
        let unit = unit as u32;
        let (lx, ly) = (x - o.0, y - o.1);
        let row = ly - u32::from(ly > self.cy);
        let col = lx - u32::from(lx > self.cx);
        let part = if lx < self.cx {
            WeightPart::Attn
        } else {
            WeightPart::Moe
        };
        match (lx == self.cx, ly == self.cy) {
            (true, true) => PeRole::Idle,
            (true, false) => PeRole::Cx(Cx {
                unit,
                row,
                first: unit == 0,
                last: unit as usize == units.len() - 1,
            }),
            (false, true) => PeRole::Cy(Cy { unit, col, part }),
            _ => PeRole::Weight(Weight {
                unit,
                row,
                col,
                part,
            }),
        }
    }
}

/// The derived kernel dims (`prep_gptoss.py`'s `Qh`, `Dc`, `NROT`, `Ic`, `NQA`, `QO`, `XE`).
struct Dims {
    k: u32,
    km: u32,
    hcc: u32,
    qh: u32,
    dc: u32,
    nrot: u32,
    ic: u32,
    nex: u32,
    tk: u32,
    nqa: u32,
    qo: u32,
    xe: u32,
    nlv: u32,
    hr: u32,
    vr: u32,
    ne: u32,
    win: u32,
}

impl Dims {
    fn of(plan: &Plan) -> Result<Dims, String> {
        let (s, p) = (&plan.shape, &plan.partition);
        let moe = s.moe.ok_or("gpt-oss plan: no MoE")?;
        let ax = p.attn_cols.ok_or("gpt-oss plan: no attention columns")?;
        let mx = p.cols - ax;
        let swa = p.segments.scores;
        let k = p.layers_per_unit;
        let hcc = p.row_dims.first().ok_or("gpt-oss plan: no row dims")?.dims;
        let qh = s.attn.q_heads / s.attn.kv_heads;
        let dc = s.attn.head_dim / swa;
        let ic = moe.expert_intermediate / mx;
        let km = (0..k)
            .filter(|&l| matches!(s.layers.get(l as usize), Some(LayerKind::Global)))
            .fold(0, |m, l| m | (1 << l));
        let win = s
            .layers
            .iter()
            .find_map(|l| match l {
                LayerKind::Sliding { window } => Some(*window),
                _ => None,
            })
            .unwrap_or(128);
        Ok(Dims {
            k,
            km,
            hcc,
            qh,
            dc,
            nrot: dc / 2,
            ic,
            nex: moe.experts,
            tk: moe.top_k,
            nqa: qh * dc + 2 * dc,
            qo: qh * dc,
            xe: (hcc / 2) * 2 * ic + (ic / 2) * hcc,
            nlv: plan.contract.rope_levels,
            hr: s.hidden,
            vr: p.head.vocab_rows,
            ne: p.kv_entries.per_row,
            win,
        })
    }
}

impl Family for GptOss {
    fn id(&self) -> FamilyId {
        FamilyId::GptOss
    }

    /// The plan from a `prep_gptoss.py` meta: the geometry name's table (`geo`), the checkpoint
    /// dims the prep wrote (`HD`, `K`, `KM`, `Hc`, `Qh`, `Dc`, `Ic`, `NEX`, `TK`, `HR`, `NLV`, `VR`,
    /// `V`, `WIN`), the build's `G2` / `G2W` / `NE` / `CAP` / `LOOP` / `WL`.
    fn plan(&self, meta: &Meta) -> Result<Plan, String> {
        let name = meta.str("geo").ok_or("split gpt-oss meta: no geo")?;
        let [ax, mx, py, unx, uny, hcc, swa, swr, hvc] = table(name)?;
        let tiny = name == "tiny";
        for (k, v) in [
            ("AX", ax),
            ("MX", mx),
            ("PY", py),
            ("Hc", hcc),
            ("HVC", hvc),
            ("NU", unx * uny),
        ] {
            if meta.int(k).is_some_and(|m| m != v) {
                return Err(format!(
                    "split gpt-oss meta: {k} {:?} vs geo {name}'s {v}",
                    meta.int(k)
                ));
            }
        }
        let units = unx * uny;
        let k = meta.need("K")?;
        let km = meta.need("KM")?;
        let qh = meta.need("Qh")?;
        let hd = meta.need("HD")?;
        let ic = meta.need("Ic")?;
        let win = meta.int("WIN").unwrap_or(128);
        if !ax.is_multiple_of(swa) {
            return Err("gpt-oss: SWA must divide the attention columns".into());
        }
        let kv_heads = ax / swa;
        if meta.int("Dc").is_some_and(|d| d * swa != hd) {
            return Err("split gpt-oss meta: Dc x SWA != HD".into());
        }
        let layers = (0..units * k)
            .map(|l| {
                if km & (1 << (l % k)) != 0 {
                    LayerKind::Global
                } else {
                    LayerKind::Sliding { window: win }
                }
            })
            .collect();
        let g2 = meta.int("G2").unwrap_or(if tiny { 2 } else { 16 });
        let g2w = meta.int("G2W").unwrap_or(if tiny { 2 } else { 24 });
        let ne = meta.int("NE").unwrap_or(128);
        let shape = Shape {
            gdn: None,
            family: FamilyId::GptOss,
            hidden: meta.need("HR")?,
            hidden_placed: py * hcc,
            vocab: meta.need("V")?,
            layers,
            attn: AttnSpec {
                q_heads: kv_heads * qh,
                kv_heads,
                head_dim: hd,
                sinks: true,
                ..Default::default()
            },
            mlp: MlpSpec {
                layer_widths: Vec::new(),
                intermediate: mx * ic,
                act: Act::SwigluOai,
            },
            moe: Some(MoeSpec {
                experts: meta.need("NEX")?,
                top_k: meta.need("TK")?,
                expert_intermediate: mx * ic,
            }),
            ple: None,
            head: HeadSpec {
                ..Default::default()
            },
            rope: RopeSpec {
                theta: meta.float("THETA").unwrap_or(150_000.0) as f32,
                rotary_dims: hd,
            },
            norm: NormSpec {
                eps: 1e-5,
                ..Default::default()
            },
        };
        let partition = Partition {
            unit_grid: Some((unx, uny)),
            units,
            layers_per_unit: k,
            rows: py,
            cols: ax + mx,
            row_dims: vec![RowRun {
                rows: (0, py),
                dims: hcc,
                attention: true,
            }],
            kv_rows: py,
            kv_entries: KvEntries {
                per_row: ne,
                ..Default::default()
            },
            reduce: ReduceGroups {
                row: g2,
                col: g2,
                head: g2,
                token_loop: g2w,
            },
            segments: Segments {
                scores: swa,
                router: Some(swr),
                ..Default::default()
            },
            attn_cols: Some(ax),
            head: HeadPartition {
                cols: hvc,
                rows: py,
                vocab_rows: meta.need("VR")?,
                ..Default::default()
            },
        };
        let contract = Contract {
            positions: ne * py,
            rope_levels: meta.int("NLV").unwrap_or(5),
            rope_bits: 3,
            cap: meta.int("CAP").unwrap_or(256),
            loop_len: meta.int("LOOP").unwrap_or(32),
            token_loop: meta.int("WL").is_none_or(|v| v != 0),
        };
        Ok(Plan {
            shape,
            partition,
            contract,
        })
    }

    /// Two unit columns as a snake that starts upward, the band below them.
    fn floor(&self, plan: &Plan) -> Result<Floor, String> {
        let g = Geom::of(plan)?;
        let grid = UnitGrid {
            cols: g.px,
            rows: g.py,
            cx: g.cx,
            cy: g.cy,
            layout: UnitLayout::Snake {
                unx: g.unx,
                uny: g.uny,
                up_first: true,
                gaps: Vec::new(),
            },
            layers_per_unit: plan.partition.layers_per_unit,
        };
        let mut b = FloorBuilder::units(&grid, (g.w - g.unx * g.uw, g.py + 1))?;
        if (b.w, b.h) != (g.w, g.h) {
            return Err(format!(
                "gpt-oss floor: {} x {} wafer, want {} x {}",
                b.w, b.h, g.w, g.h
            ));
        }
        let origins: Vec<(u32, u32)> = b.units.iter().map(|u| (u.rect.x, u.rect.y)).collect();
        for (i, &o) in origins.iter().enumerate() {
            let (ux, uy) = (o.0 / g.uw, o.1 / g.uh);
            let want = if ux % 2 == 0 {
                (ux, g.uny - 1 - (i as u32 % g.uny))
            } else {
                (ux, i as u32 % g.uny)
            };
            if (ux, uy) != want || ux != i as u32 / g.uny {
                return Err(format!("gpt-oss floor: unit {i} at slot ({ux}, {uy})"));
            }
        }
        for y in 0..g.h {
            for x in 0..g.w {
                let c = Coord::new(x, y);
                b.set_role(c, g.role(&origins, c));
            }
        }
        let ctl = g.hb0 + g.hxc;
        Ok(b.finish(HeadBlock {
            rect: Rect {
                x: 0,
                y: g.hb0,
                w: g.hvc,
                h: g.py + 1,
            },
            ctl: CtlLine::Row(ctl),
            root: Coord::new(g.hvc / 2, ctl),
        }))
    }

    /// The units' generic wiring, the router segments, the band's head input and the token loop.
    fn wire(&self, plan: &Plan, floor: &Floor) -> Result<Wiring, String> {
        let g = Geom::of(plan)?;
        if !g.ax.is_multiple_of(g.swa) || !g.mx.is_multiple_of(g.swr) {
            return Err("gpt-oss: segments must divide their halves".into());
        }
        let mut w = Wiring::default();
        let n = floor.units.len();
        for unit in &floor.units {
            let (x0, y0) = (unit.rect.x, unit.rect.y);
            for r in 0..g.py {
                let y = g.row_y(y0, r);
                w.chain(Chain {
                    pts: (0..=g.px).map(|lx| Coord::new(x0 + lx, y)).collect(),
                    root: g.cx as usize,
                    l1: [Class::RowL1(Half::Lo), Class::RowL1(Half::Hi)],
                    l2: wiring::level2(Class::RowL2, g.g2.row),
                    cast: Class::RowCast,
                    loopback: false,
                    slot: Slot::RowReduce,
                });
                w.turn(wiring::loopback(Coord::new(x0 + g.cx, y)));
                // the scores segments on the attention columns
                let wpts: Vec<Coord> = (0..g.ax).map(|c| Coord::new(x0 + c, y)).collect();
                for seg in wpts.chunks(g.swa as usize) {
                    w.chain(Chain {
                        pts: seg.to_vec(),
                        root: seg.len() / 2,
                        l1: [Class::ScoreL1(Half::Lo), Class::ScoreL1(Half::Hi)],
                        l2: None,
                        cast: Class::ScoreCast,
                        loopback: false,
                        slot: Slot::Segment,
                    });
                }
            }
            for c in 0..g.px {
                let x = x0 + skip(c, g.cx);
                w.chain(Chain {
                    pts: (0..=g.py).map(|ly| Coord::new(x, y0 + ly)).collect(),
                    root: g.cy as usize,
                    l1: [Class::ColL1(Half::Lo), Class::ColL1(Half::Hi)],
                    l2: wiring::level2(Class::ColL2, g.g2.col),
                    cast: Class::ColCast,
                    loopback: false,
                    slot: Slot::ColReduce,
                });
                w.turn(wiring::loopback(Coord::new(x, y0 + g.cy)));
            }
            // the router segments on the MoE half of the CY row
            let y = y0 + g.cy;
            let cpts: Vec<Coord> = (0..g.mx)
                .map(|c| Coord::new(x0 + g.cx + 1 + c, y))
                .collect();
            for seg in cpts.chunks(g.swr as usize) {
                w.chain(Chain {
                    pts: seg.to_vec(),
                    root: seg.len() / 2,
                    l1: [Class::OnormL1(Half::Lo), Class::OnormL1(Half::Hi)],
                    l2: None,
                    cast: Class::OnormCast,
                    loopback: false,
                    slot: ROUTER,
                });
            }
        }
        for u in 0..n.saturating_sub(1) {
            wiring::handoff(floor, u, u + 1, &mut w)?;
        }
        self.head_in(&g, floor, &mut w)?;
        // the band's dims-columns: chains into the control row
        for j in 0..g.hvc {
            w.chain(Chain {
                pts: (0..=g.py).map(|ly| Coord::new(j, g.hb0 + ly)).collect(),
                root: g.hxc as usize,
                l1: [Class::RowL1(Half::Lo), Class::RowL1(Half::Hi)],
                l2: wiring::level2(Class::RowL2, g.g2.head),
                cast: Class::HeadRowWord,
                loopback: false,
                slot: Slot::HeadRowReduce,
            });
            w.turn(wiring::loopback(Coord::new(j, g.hb0 + g.hxc)));
        }
        if g.wl {
            self.token_loop(&g, floor, &mut w)?;
        }
        Ok(w)
    }

    fn palette(&self, _plan: &Plan) -> Palette {
        Palette::base(&OWN)
    }

    /// `mkfull_gptoss.py`'s kinds: sources, owned symbols and each kind's `@set_tile_code` params
    /// in its order, with the serving defaults (`DIAG` / `TSTAMP` / `XKEEP` / `XDLY` / `PFTZ` /
    /// `WFORCE` 0, `GLUV` 1, `CYV` 2, `FIFOW` 32, `NE` / `WIN` from the plan).
    fn kernels(&self, prog: &Program) -> Result<BTreeMap<RoleKind, KernelBuild>, String> {
        let d = Dims::of(&prog.plan)?;
        let g = Geom::of(&prog.plan)?;
        let qsc = 1.0f64 / f64::from(prog.plan.shape.attn.head_dim).sqrt();
        let i = |v: u32| Value::Int(i64::from(v));
        let (diag, ts, fifow) = (i(0), i(0), i(32));
        let mut out = BTreeMap::new();
        out.insert(
            RoleKind::WeightAttn,
            KernelBuild::new(
                "weight_gptoss_att.csl",
                &["w", "rcfg", "tdbg"],
                vec![
                    ("K", i(d.k)),
                    ("KM", i(d.km)),
                    ("Hc", i(d.hcc)),
                    ("NQA", i(d.nqa)),
                    ("QO", i(d.qo)),
                    ("Qh", i(d.qh)),
                    ("Dc", i(d.dc)),
                    ("NE", i(d.ne)),
                    ("PYW", i(g.py)),
                    ("WIN", i(d.win)),
                    ("PFTZ", i(0)),
                    ("DIAG", diag.clone()),
                    ("TSTAMP", ts.clone()),
                    ("FIFOW", fifow.clone()),
                ],
            ),
        );
        out.insert(
            RoleKind::WeightMoe,
            KernelBuild::new(
                "weight_gptoss_moe.csl",
                &["wx", "wr", "rcfg", "tdbg"],
                vec![
                    ("K", i(d.k)),
                    ("Hc", i(d.hcc)),
                    ("Ic", i(d.ic)),
                    ("NEX", i(d.nex)),
                    ("TK", i(d.tk)),
                    ("DIAG", diag.clone()),
                    ("TSTAMP", ts.clone()),
                    ("FIFOW", fifow.clone()),
                ],
            ),
        );
        out.insert(
            RoleKind::Cx,
            KernelBuild::new(
                "ctl_cx_gptoss.csl",
                &["x", "gam", "bo", "bd", "rcfg", "xdbg", "tdbg", "xred"],
                vec![
                    ("K", i(d.k)),
                    ("Hc", i(d.hcc)),
                    ("NEX", i(d.nex)),
                    ("TK", i(d.tk)),
                    ("DIAG", diag.clone()),
                    ("XKEEP", i(0)),
                    ("XDLY", i(0)),
                ],
            ),
        );
        out.insert(
            RoleKind::CyAttn,
            KernelBuild::new(
                "ctl_cy_gptoss_att.csl",
                &["qb", "snk", "rot", "rcfg", "tdbg"],
                vec![
                    ("K", i(d.k)),
                    ("NQA", i(d.nqa)),
                    ("Qh", i(d.qh)),
                    ("Dc", i(d.dc)),
                    ("NROT", i(d.nrot)),
                    ("HR", i(d.hr)),
                    ("NLV", i(d.nlv)),
                    ("QSC", Value::Float(py_float(qsc))),
                    ("CYV", i(2)),
                    ("DIAG", diag.clone()),
                    ("TSTAMP", ts.clone()),
                    ("FIFOW", fifow.clone()),
                ],
            ),
        );
        out.insert(
            RoleKind::CyMoe,
            KernelBuild::new(
                "ctl_cy_gptoss_moe.csl",
                &["rb", "gb", "ps", "rcfg", "tdbg"],
                vec![
                    ("K", i(d.k)),
                    ("Ic", i(d.ic)),
                    ("NEX", i(d.nex)),
                    ("TK", i(d.tk)),
                    ("HR", i(d.hr)),
                    ("GLUV", i(1)),
                    ("DIAG", diag.clone()),
                    ("TSTAMP", ts.clone()),
                    ("FIFOW", fifow.clone()),
                ],
            ),
        );
        out.insert(
            RoleKind::Head,
            KernelBuild::new(
                "head_pe_gptoss.csl",
                &["hw", "hg", "rcfg", "hemb", "tdbg"],
                vec![("VR", i(d.vr)), ("HDc", i(d.hcc))],
            ),
        );
        out.insert(
            RoleKind::HeadCtl,
            KernelBuild::new(
                "head_ctl.csl",
                &["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"],
                vec![
                    ("VR", i(d.vr)),
                    ("FIFOW", fifow),
                    ("WFORCE", i(0)),
                    ("WFN", i(g.hvc)),
                ],
            ),
        );
        // the filler: memcpy's unblock only, no source file (the layout writes it inline)
        out.insert(RoleKind::Idle, KernelBuild::filler());
        Ok(out)
    }

    /// The placement, in order: per unit, by row halves in blocks
    /// of 16 rows, the attention columns' rcfg and `w`, the MoE columns' rcfg, `wx`, `wr`; then
    /// each half's CX rcfg, `gam`, `bo`, `bd`; the CY row's attention half (rcfg, `qb`, `snk`,
    /// `rot`) and MoE half (rcfg, `rb`, `gb`, `ps`). Then the band by dims-row halves in blocks of
    /// 16 rows and 64 columns (rcfg, `hw`, `hemb`, `hg` a dims-row), and the control row's rcfg.
    fn bindings(&self, prog: &Program) -> Result<Vec<Binding>, String> {
        let g = Geom::of(&prog.plan)?;
        let d = Dims::of(&prog.plan)?;
        let mut out = Copies::default();
        let rows = |name: String, r: (u32, u32), words: u32| Source::Staged {
            name,
            index: r,
            words,
        };
        let lwa = d.hcc * d.nqa + d.qo * d.hcc;
        let experts = vec![(
            "model.layers.*.mlp.experts.{gate_up,down}_proj_blocks".to_string(),
            vec![Transform::Mxfp4ToE5m2],
        )];
        for (u, unit) in prog.floor.units.iter().enumerate() {
            let (x0, y0u) = (unit.rect.x, unit.rect.y);
            for (r0, r1) in [(0, g.cy), (g.cy, g.py)] {
                for (rb, re) in chunks(r0, r1, 16) {
                    let y0 = g.row_y(y0u, rb);
                    let ra = Rect::new(x0, y0, g.ax, re - rb);
                    out.rcfg(ra);
                    let wa = rows(format!("wa{u}"), (rb, re), d.k * lwa);
                    out.push("w", ra, wa, true, Vec::new());
                    let rm = Rect::new(x0 + g.cx + 1, y0, g.mx, re - rb);
                    out.rcfg(rm);
                    let wx = rows(format!("wx{u}"), (rb, re), d.k * d.nex * d.xe);
                    out.push("wx", rm, wx, true, experts.clone());
                    let wr = rows(format!("wr{u}"), (rb, re), d.k * d.hcc);
                    out.push("wr", rm, wr, false, Vec::new());
                }
                let rc = Rect::new(x0 + g.cx, g.row_y(y0u, r0), 1, r1 - r0);
                out.rcfg(rc);
                for (s, n) in [
                    ("gam", 2 * d.k * d.hcc),
                    ("bo", d.k * d.hcc),
                    ("bd", d.k * d.nex * d.hcc),
                ] {
                    out.push(
                        s,
                        rc,
                        rows(format!("{s}{u}"), (r0, r1), n),
                        false,
                        Vec::new(),
                    );
                }
            }
            let y = y0u + g.cy;
            let ca = Rect::new(x0, y, g.ax, 1);
            out.rcfg(ca);
            for (s, name, n) in [
                ("qb", format!("qb{u}"), d.k * d.nqa),
                ("snk", format!("snk{u}"), d.k * d.qh),
                ("rot", "rot".to_string(), d.nlv * 8 * 2 * d.nrot),
            ] {
                out.push(s, ca, rows(name, (0, g.ax), n), false, Vec::new());
            }
            let cm = Rect::new(x0 + g.cx + 1, y, g.mx, 1);
            out.rcfg(cm);
            for (s, n) in [
                ("rb", d.k),
                ("gb", d.k * d.nex * 2 * d.ic),
                ("ps", d.k * d.nex * 2),
            ] {
                out.push(
                    s,
                    cm,
                    rows(format!("{s}{u}"), (0, g.mx), n),
                    false,
                    Vec::new(),
                );
            }
        }
        let pad = |name: &str| {
            vec![(
                name.to_string(),
                vec![Transform::PadLastRow { to: g.hvc * g.vr }],
            )]
        };
        for (i0, i1) in [(0, g.hxc), (g.hxc, g.py)] {
            for (ib, ie) in chunks(i0, i1, 16) {
                for (cb, ce) in chunks(0, g.hvc, 64) {
                    let r = Rect::new(cb, g.by(ib), ce - cb, ie - ib);
                    let block = |n: &str| Slice::of(n, None, vec![(ib, ie), (cb, ce)]);
                    out.rcfg(r);
                    out.slice("hw", r, block("hw"), true, pad("lm_head.weight"));
                    out.slice(
                        "hemb",
                        r,
                        block("hemb"),
                        true,
                        pad("model.embed_tokens.weight"),
                    );
                    let hg = Slice {
                        across: r.w,
                        ..Slice::of("hg", None, vec![(ib, ie)])
                    };
                    out.slice("hg", r, hg, false, Vec::new());
                }
            }
        }
        out.rcfg(Rect::new(0, g.hb0 + g.hxc, g.hvc, 1));
        Ok(out.0)
    }
}

impl GptOss {
    /// The last unit's x into the band: CX row r east to MoE column r
    /// (`HandoffOut`), down it to dims-row r (`BandVert`), multicast along the dims-row
    /// (`HeadColCast`).
    fn head_in(&self, g: &Geom, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
        let ul = floor.units.last().ok_or("gpt-oss: no units")?;
        let (x0, y0) = (ul.rect.x, ul.rect.y);
        if y0 + g.uh != g.hb0 || g.py > g.mx {
            return Err("gpt-oss: the last unit sits on the band, PY <= MX".into());
        }
        for r in 0..g.py {
            let y = g.row_y(y0, r);
            let xr = x0 + g.cx + 1 + r;
            if xr >= g.hvc {
                return Err("gpt-oss: the x path's column past the band".into());
            }
            w.path(Path {
                pts: (x0 + g.cx..=xr).map(|x| Coord::new(x, y)).collect(),
                class: Class::HandoffOut,
                send: Slot::XOut,
                recv: BAND_IN,
            });
            let yb = g.by(r);
            w.path(Path {
                pts: (y..=yb).map(|yy| Coord::new(xr, yy)).collect(),
                class: Class::BandVert,
                send: BAND_OUT,
                recv: Slot::HeadIn,
            });
            w.multicast(Multicast {
                pts: (0..g.hvc).map(|j| Coord::new(j, yb)).collect(),
                src: xr as usize,
                class: Class::HeadColCast,
                slot: BAND_CAST,
            });
        }
        Ok(())
    }

    /// The token loop: the max-reduce along the control row, then each
    /// dims-row's band bus west to column 0 (`EmbWest`), east along the dims-row and up unit 0's
    /// MoE column i (`BandV1`), west into its CX row i (`BandW2`).
    fn token_loop(&self, g: &Geom, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
        let yc = g.hb0 + g.hxc;
        w.chain(Chain {
            pts: (0..g.hvc).map(|j| Coord::new(j, yc)).collect(),
            root: (g.hvc / 2) as usize,
            l1: [Class::LoopL1(Half::Lo), Class::LoopL1(Half::Hi)],
            l2: wiring::level2(Class::LoopL2, g.g2.token_loop),
            cast: Class::LoopCast,
            loopback: false,
            slot: Slot::LoopReduce,
        });
        let u0 = floor.units.first().ok_or("gpt-oss: no units")?;
        let (ux, uy) = (u0.rect.x, u0.rect.y);
        if uy + g.uh != g.hb0 {
            return Err("gpt-oss: unit 0 sits on the band".into());
        }
        if g.hvc < 2 {
            return Err("gpt-oss: a band of one column".into());
        }
        for i in 0..g.py {
            let yb = g.by(i);
            w.path(Path {
                pts: (0..g.hvc).rev().map(|j| Coord::new(j, yb)).collect(),
                class: Class::EmbWest,
                send: BUS_TX,
                recv: BUS,
            });
            let ci = ux + g.cx + 1 + i;
            let yu = g.row_y(uy, i);
            if ci >= g.hvc {
                return Err("gpt-oss: the return column past the band".into());
            }
            let mut pts: Vec<Coord> = (0..=ci).map(|x| Coord::new(x, yb)).collect();
            pts.extend((yu..yb).rev().map(|yy| Coord::new(ci, yy)));
            w.path(Path {
                pts,
                class: Class::BandV1,
                send: V1_TX,
                recv: EMB_IN,
            });
            w.path(Path {
                pts: (ux + g.cx..=ci).rev().map(|x| Coord::new(x, yu)).collect(),
                class: Class::BandW2,
                send: EMB_OUT,
                recv: Slot::EmbIn,
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- rcfg

impl RcfgAbi for GptOss {
    fn route_slots(&self, kind: RoleKind) -> usize {
        match kind {
            RoleKind::WeightAttn | RoleKind::WeightMoe | RoleKind::Weight => 16,
            RoleKind::Relay => 4,
            RoleKind::Idle => 0,
            _ => 12,
        }
    }

    /// The shared words ([`abi`]) with gpt-oss's: a weight PE's router column flag (the MoE
    /// half's first column), its relays (the handoffs', or the embedding's / band's turn) and
    /// whether it takes the embedding; the CY's router segment and member index; the band's CX,
    /// head and head control PEs (never parked on the bus, which counts as final).
    fn rcfg(&self, prog: &Program, at: Coord) -> Result<Vec<u32>, String> {
        let g = Geom::of(&prog.plan)?;
        let pal = self.palette(&prog.plan);
        let mut pe = Pe::new(prog, &pal, at);
        match prog.floor.roles.get(at) {
            PeRole::Weight(wt) => {
                let rwc = u32::from(wt.part == WeightPart::Moe && wt.col == g.ax);
                let turn = |s: Slot, inn: Slot, out: Slot| {
                    pe.turn(s).or_else(|| pe.colour(inn).zip(pe.colour(out)))
                };
                let relays = [
                    turn(Slot::RelayPre, EMB_IN, EMB_OUT),
                    turn(Slot::RelayPost, BAND_IN, BAND_OUT),
                ];
                let mx = pe.c(Class::RowCast)?;
                let mut w = abi::weight(&pe, 16, wt.row, [rwc, mx], g.hcc, relays)?;
                w.push(u32::from(pe.colour(EMB_IN).is_some()));
                Ok(w)
            }
            PeRole::Cx(cx) => rcfg_cx(&pe, g.wl, cx.first),
            PeRole::Cy(cy) => {
                let mut w = abi::cy(&pe, Some((ROUTER, Class::OnormCast)))?;
                // a router segment's member index: MoE column c's place in its segment
                w.push(match cy.part {
                    WeightPart::Moe => (cy.col - g.ax) % g.swr,
                    _ => 0,
                });
                Ok(w)
            }
            PeRole::Head(_) => {
                pe.band = Some(HE);
                rcfg_head(&pe, g.wl)
            }
            PeRole::HeadCtl(_) => {
                pe.band = Some(HE);
                abi::hctl(&pe, at.x * g.vr, g.wl)
            }
            r => Err(format!("{at:?}: gpt-oss has no rcfg for {r:?}")),
        }
    }
}

fn distinct(at: Coord, what: &str, q: &[u32]) -> Result<(), String> {
    let mut s = q.to_vec();
    s.sort_unstable();
    s.dedup();
    if s.len() != q.len() {
        return Err(format!("{at:?}: {what} queues share a colour {q:?}"));
    }
    Ok(())
}

/// A CX PE's words: its row root's halves parked where absent, a parked third queue, x in, the
/// loop-back, the multicast, x out, the flags, a spare colour, the embedding queue.
fn rcfg_cx(pe: &Pe, wl: bool, first: bool) -> Result<Vec<u32>, String> {
    let pk = pe.parks(4)?;
    let (inn, out, emb) = (
        pe.colour(Slot::XIn),
        pe.colour(Slot::XOut),
        pe.colour(Slot::EmbIn),
    );
    let rq = pe.root(Slot::RowReduce)?;
    let q = [
        rq.lo.map_or(pk[0], word),
        rq.hi.map_or(pk[1], word),
        pk[2],
        inn.unwrap_or(pk[0]),
        pe.c(Class::Loopback)?,
        pe.c(Class::RowCast)?,
        out.unwrap_or(pk[1]),
    ];
    let fl = u32::from(first || inn.is_none())
        | if out.is_none() { 2 } else { 0 }
        | if wl && emb.is_some() { 4 } else { 0 };
    let q6 = emb.unwrap_or(pk[3]);
    let fin = [q[0], q[1], q[2], q[3], q6];
    distinct(pe.at, "CX", &fin)?;
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[fl, pe.spare(&fin, &CT5)?, q6]);
    Ok(w)
}

/// A band head PE's words: its dims-row's queues, the band's x in (`HV2`, its source), the bus,
/// the dims-row multicast, the row word, the send, the role, the source flag, a spare colour,
/// the flags (the loop runs, on the bus), the embedding's queues.
fn rcfg_head(pe: &Pe, wl: bool) -> Result<Vec<u32>, String> {
    let pk = pe.parks(6)?;
    let ro = pe.member(Slot::HeadRowReduce)?;
    let src = pe.colour(Slot::HeadIn).is_some();
    let bus = wl && pe.colour(BUS).is_some();
    let (hv2, hm, hr3) = (
        pe.c(Class::BandVert)?,
        pe.c(Class::HeadColCast)?,
        pe.c(Class::HeadRowWord)?,
    );
    let he = u32::from(HE);
    let q = [
        ro.recv1.map_or(pk[0], word),
        ro.recv2.map_or(pk[1], word),
        if src { hv2 } else { pk[2] },
        if bus { he } else { pk[3] },
        if src { pk[5] } else { hm },
        if wl { hr3 } else { pk[4] },
    ];
    distinct(pe.at, "head", &q)?;
    let mut w = pe.routes(12)?;
    w.extend_from_slice(&q);
    w.extend_from_slice(&[
        word(ro.send),
        hm,
        if ro.role == 3 { 1 } else { ro.role },
        if src { 2 } else { 0 },
        pe.spare(&q, &CT6)?,
        u32::from(wl) | if bus { 2 } else { 0 },
        0,
        if wl { he } else { pk[5] },
        if wl { pe.c(Class::BandV1)? } else { pk[5] },
    ]);
    Ok(w)
}

/// Python's `repr(float)` for the layout's `QSC` literal (shortest round-trip, at least one
/// decimal: `0.125`, `0.25`).
fn py_float(v: f64) -> String {
    let s = format!("{v:?}");
    if s.contains('.') || s.contains('e') {
        s
    } else {
        format!("{s}.0")
    }
}
