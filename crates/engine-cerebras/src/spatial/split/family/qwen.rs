//! Qwen3.5 (GDN layers + gated full attention: 9B / 4B / 0.8B) and Qwen3 (dense attention, no
//! output gate: Qwen3-8B), on the snake of PX x PY units with the head east of them.
//!
//! Owns: the snake with gap slots and their relay PEs (`GAPX` / `GAPY`, `relay.csl`); the scores
//! segments (`SWA`), GDN's (k, q) multicast (`SWG`, rows < `DKH`, the `0xC000` / `0x8000` /
//! `0x4000` source words), the CY row's o-norm segments (`SWV`); the RoPE levels (`RLV` /
//! `RBITS`) in the contract; Qwen3's hidden padding (`hidden_placed`). Kernels: `weight_q35.csl`,
//! `ctl_cx.csl`, `ctl_cy_q35.csl`, `head_pe.csl`, `head_ctl.csl`, `relay.csl`.
//!
//! The generic layers lay the rest (`wiring::units`, `wiring::head_input`) and write the
//! shared rcfg words ([`abi`]); the GDN source words, the head PEs' relay roles and the token
//! loop's turns are computed here from the plan and the floor.

use std::collections::BTreeMap;

use crate::spatial::split::emit::meta::Meta;
use crate::spatial::split::emit::params::KernelBuild;
use crate::spatial::split::emit::rcfg::RcfgAbi;
use crate::spatial::split::fabric::Palette;
use crate::spatial::split::family::Family;
use crate::spatial::split::family::abi::{self, CT5, CxIo, HeadIo, Pe};
use crate::spatial::split::family::tiles::{HeadCopies, Hemb, UnitCopies};
use crate::spatial::split::floor::{Coord, Floor, FloorBuilder, UnitGrid, UnitLayout, skip};
use crate::spatial::split::plan::{
    Act, AttnSpec, Contract, FamilyId, GdnSpec, HeadPartition, HeadSpec, KvEntries, LayerKind,
    MlpSpec, NormSpec, Partition, Plan, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::split::program::Program;
use crate::spatial::split::role::{PeRole, RoleKind};
use crate::spatial::split::weights::{Binding, Copies, From, Slice, Transform, from};
use crate::spatial::split::wiring::{
    Chain, Class, Half, Multicast, Slot, Wiring, head_input, units,
};

type R<T> = Result<T, String>;

pub struct Qwen;

/// The physical facts the floor, the wiring and the words share, from the plan.
struct Geom {
    px: u32,
    py: u32,
    xc: u32,
    yc: u32,
    unx: u32,
    uny: u32,
    gap: Option<(u32, u32)>,
    hc: u32,
    hr: u32,
    hx0: u32,
    hxc: u32,
    hcc: u32,
    token_loop: bool,
    att: bool,
    gdn: bool,
    swa: u32,
    swg: u32,
    swv: u32,
    /// GDN's key dims a column (`DKG`) and the rows that hold a key head (`DKH`).
    dkg: u32,
    dkh: u32,
}

impl Geom {
    fn of(plan: &Plan) -> R<Geom> {
        let p = &plan.partition;
        let (unx, uny) = p
            .unit_grid
            .ok_or("qwen: the plan has no unit grid (UNX, UNY)")?;
        let slots = unx * uny;
        let gap = match slots.checked_sub(p.units) {
            Some(0) => None,
            // one slot short of the grid: the last column's row-1 slot stays out of the snake, so
            // the last unit still ends on row 0 (the token loop's return); its PEs relay
            Some(1) if unx % 2 == 0 && uny >= 3 => Some((unx - 1, 1)),
            _ => {
                return Err(format!(
                    "qwen: {} units on {unx} x {uny} slots (at most one gap slot, on an even number of unit columns, below the turn row)",
                    p.units
                ));
            }
        };
        let hcc = p.row_dims.first().ok_or("qwen: no row dims")?.dims;
        let att = plan
            .shape
            .layers
            .iter()
            .any(|k| !matches!(k, LayerKind::Gdn));
        let gdn = plan
            .shape
            .layers
            .iter()
            .any(|k| matches!(k, LayerKind::Gdn));
        let (dkg, dkh) = match plan.shape.gdn {
            Some(g) => (g.key_dim / p.segments.gdn, g.key_dim),
            // (Qwen3: GDN's fields only shape unused buffers; no GDN segment is laid)
            None => (
                plan.shape.attn.head_dim / p.segments.gdn,
                plan.shape.attn.head_dim,
            ),
        };
        Ok(Geom {
            px: p.cols,
            py: p.rows,
            xc: p.cols / 2,
            yc: p.rows / 2,
            unx,
            uny,
            gap,
            hc: p.head.cols,
            hr: p.head.rows,
            hx0: unx * (p.cols + 1),
            hxc: p.head.cols / 2,
            hcc,
            token_loop: plan.contract.token_loop,
            att,
            gdn,
            swa: p.segments.scores,
            swg: p.segments.gdn,
            swv: p.segments.onorm,
            dkg,
            dkh,
        })
    }

    /// The head column `at.x` is, if any.
    fn head_col(&self, x: u32) -> Option<u32> {
        let l = x.checked_sub(self.hx0)?;
        (l <= self.hc && l != self.hxc).then(|| l - u32::from(l > self.hxc))
    }

    /// A head PE's relay role in the head input: 1 on an odd CX row (it
    /// relays its x north), 2 on the even row above (it multicasts the pair down its column).
    fn head_role(&self, floor: &Floor, at: Coord) -> R<u32> {
        let Some(c) = self.head_col(at.x) else {
            return Ok(0);
        };
        let last = floor.units.last().ok_or("qwen: no units")?;
        let y0 = last.rect.y;
        for r in [2 * c, 2 * c + 1] {
            if r < self.py && y0 + skip(r, self.yc) == at.y {
                return Ok(if r % 2 == 1 { 1 } else { 2 });
            }
        }
        Ok(0)
    }

    /// The token loop's turns at a head PE: 1 the embedding's turn up at row
    /// 0, 2 its turn down onto the even CX row, 4 its split onto the odd one.
    fn head_turns(&self, floor: &Floor, at: Coord) -> u32 {
        if !self.token_loop || self.head_col(at.x).is_none() {
            return 0;
        }
        let c = self.head_col(at.x).unwrap_or(0);
        let ty = floor.units.first().map_or(0, |u| u.rect.y);
        let (te, to) = (ty + skip(2 * c, self.yc), ty + skip(2 * c + 1, self.yc));
        let mut t = 0;
        if at.y == 0 {
            t |= 1;
        }
        if at.y == te {
            t |= 2;
        }
        if at.y == to {
            t |= 4;
        }
        t
    }

    /// GDN's source word at weight row `r`, column `c`: 0xC000 | d a one-PE
    /// segment's own key dim, 0x8000 | d the multicast's source, 0 a receiver, 0x4000 none.
    fn gdn_source(&self, r: u32, c: u32) -> u32 {
        if !self.gdn || r >= self.dkh {
            return 0x4000;
        }
        let s0 = c / self.swg * self.swg;
        let len = self.swg.min(self.px - s0);
        let d = r % self.dkg;
        if len == 1 {
            0xC000 | d
        } else if c - s0 == r / self.dkg {
            0x8000 | d
        } else {
            0
        }
    }
}

/// A segment chain (scores, o-norm): root in its middle, single level, no loop-back
///.
fn segment(pts: &[Coord], l1: [Class; 2], cast: Class, slot: Slot) -> Chain {
    Chain {
        pts: pts.to_vec(),
        root: pts.len() / 2,
        l1,
        l2: None,
        cast,
        loopback: false,
        slot,
    }
}

impl Family for Qwen {
    fn id(&self) -> FamilyId {
        FamilyId::Qwen
    }

    fn plan(&self, meta: &Meta) -> R<Plan> {
        let px = meta.need("PX")?;
        let py = meta.need("PY")?;
        let (unx, uny) = (meta.need("UNX")?, meta.need("UNY")?);
        let k = meta.need("K")?;
        let km = meta.int("KM").unwrap_or((1 << k) - 1);
        let gap = meta.int("GAPX").zip(meta.int("GAPY"));
        let units = unx * uny - u32::from(gap.is_some());
        if let Some(g) = gap
            && (unx * uny - units != 1 || g != (unx - 1, 1))
        {
            return Err(format!(
                "qwen meta: gap slot {g:?} is not the snake's ({}, 1)",
                unx - 1
            ));
        }
        let pattern: Vec<LayerKind> = (0..k)
            .map(|l| {
                if km >> l & 1 != 0 {
                    LayerKind::Global
                } else {
                    LayerKind::Gdn
                }
            })
            .collect();
        let layers: Vec<LayerKind> = (0..units).flat_map(|_| pattern.clone()).collect();
        let (swa, swg, swv) = (meta.need("SWA")?, meta.need("SWG")?, meta.need("SWV")?);
        let gate = meta.int("GATE").unwrap_or(1);
        let head_dim = meta.need("HD")?;
        let kv_heads = px / swa;
        let qh = meta.need("Qh")?;
        let hc = meta.need("Hc")?;
        let hidden = meta.int("H").unwrap_or(hc * py);
        let gdn = if km == (1 << k) - 1 {
            None
        } else {
            let key_dim = meta.need("HK")?;
            if meta.int("DKH").is_some_and(|d| d != key_dim)
                || meta.int("DKG").is_some_and(|d| d * swg != key_dim)
            {
                return Err("qwen meta: DKG / DKH disagree with HK / SWG".into());
            }
            Some(GdnSpec {
                key_heads: px / swg,
                value_heads: px / swv,
                key_dim,
                value_dim: meta.need("HV")?,
            })
        };
        let g2 = meta.int("G2").unwrap_or(0);
        let rope_levels = meta.int("RLV").unwrap_or(4);
        let rope_bits = meta.int("RBITS").unwrap_or(3);
        let ne = meta.need("NE")?;
        let shape = Shape {
            family: FamilyId::Qwen,
            hidden,
            hidden_placed: hc * py,
            vocab: meta.int("V").unwrap_or(0),
            layers,
            attn: AttnSpec {
                q_heads: qh * kv_heads,
                kv_heads,
                head_dim,
                output_gate: gate != 0,
                qk_norm: true,
                ..Default::default()
            },
            mlp: MlpSpec {
                layer_widths: Vec::new(),
                intermediate: meta.need("Ic")? * px,
                act: Act::Silu,
            },
            moe: None,
            gdn,
            ple: None,
            head: HeadSpec {
                tied: meta.int("tied").unwrap_or(0) != 0,
                ..Default::default()
            },
            rope: RopeSpec {
                // (the preps write the tables, not theta: 0 unless the meta records it)
                theta: meta.float("THETA").unwrap_or(0.0) as f32,
                // Qwen3: full RoPE; Qwen3.5: a quarter of the head (the preps' partial factor)
                rotary_dims: meta.int("ROTD").unwrap_or(if gate == 0 {
                    head_dim
                } else {
                    head_dim / 4
                }),
            },
            norm: NormSpec {
                eps: meta.float("EPS").unwrap_or(1e-6) as f32,
                unit_offset: gate != 0,
                ..Default::default()
            },
        };
        let partition = Partition {
            units,
            layers_per_unit: k,
            unit_grid: Some((unx, uny)),
            rows: py,
            cols: px,
            row_dims: vec![RowRun {
                rows: (0, py),
                dims: hc,
                attention: km != 0,
            }],
            kv_rows: py,
            kv_entries: KvEntries {
                per_row: ne,
                ..Default::default()
            },
            reduce: ReduceGroups {
                row: meta.int("G2X").unwrap_or(g2),
                col: meta.int("G2Y").unwrap_or(g2),
                head: meta.int("G2H").unwrap_or(g2),
                token_loop: meta.int("G2W").unwrap_or(24),
            },
            segments: Segments {
                scores: swa,
                gdn: swg,
                onorm: swv,
                ..Default::default()
            },
            attn_cols: None,
            head: HeadPartition {
                cols: meta.need("HC")?,
                rows: meta.need("HR")?,
                vocab_rows: meta.need("VR")?,
                ..Default::default()
            },
        };
        let contract = Contract {
            // bounded by the CY's RoPE tables: RLV digits of RBITS bits
            positions: (ne * py).min(1 << (rope_bits * rope_levels).min(31)),
            rope_levels,
            rope_bits,
            cap: meta.int("CAP").unwrap_or(256),
            loop_len: meta.int("LOOP").unwrap_or(32),
            token_loop: meta.int("WL").unwrap_or(0) != 0,
        };
        Ok(Plan {
            shape,
            partition,
            contract,
        })
    }

    fn floor(&self, plan: &Plan) -> R<Floor> {
        let g = Geom::of(plan)?;
        let grid = UnitGrid {
            cols: g.px,
            rows: g.py,
            cx: g.xc,
            cy: g.yc,
            layout: UnitLayout::Snake {
                unx: g.unx,
                uny: g.uny,
                up_first: false,
                gaps: g.gap.into_iter().collect(),
            },
            layers_per_unit: plan.partition.layers_per_unit,
        };
        let mut b = FloorBuilder::units(&grid, (0, 0))?;
        let head = b.head_east(g.hx0, g.hc, g.hr, g.token_loop);
        Ok(b.finish(head))
    }

    fn wire(&self, plan: &Plan, floor: &Floor) -> R<Wiring> {
        let g = Geom::of(plan)?;
        for (n, sw) in [("SWA", g.swa), ("SWG", g.swg), ("SWV", g.swv)] {
            if sw == 0 || g.xc % sw != 0 {
                return Err(format!(
                    "qwen: {n} {sw} does not divide XC {} (a segment would span the control column)",
                    g.xc
                ));
            }
        }
        let mut w = Wiring::default();
        units(plan, floor, &mut w)?;
        for u in &floor.units {
            let (x0, y0) = (u.rect.x, u.rect.y);
            for r in 0..g.py {
                let y = y0 + skip(r, g.yc);
                let wpts: Vec<Coord> = (0..g.px)
                    .map(|c| Coord::new(x0 + skip(c, g.xc), y))
                    .collect();
                if g.att {
                    for s in wpts.chunks(g.swa as usize) {
                        w.chain(segment(
                            s,
                            [Class::ScoreL1(Half::Lo), Class::ScoreL1(Half::Hi)],
                            Class::ScoreCast,
                            Slot::Segment,
                        ));
                    }
                }
                if g.gdn && r < g.dkh {
                    for s in wpts.chunks(g.swg as usize) {
                        if s.len() == 1 {
                            continue;
                        }
                        let src = (r / g.dkg) as usize;
                        if src >= s.len() {
                            return Err(format!(
                                "qwen: GDN row {r}'s source {src} past its segment"
                            ));
                        }
                        w.multicast(Multicast {
                            pts: s.to_vec(),
                            src,
                            class: Class::GdnKq,
                            slot: Slot::Gdn,
                        });
                    }
                }
            }
            if g.gdn {
                let y = y0 + g.yc;
                let cpts: Vec<Coord> = (0..g.px)
                    .map(|c| Coord::new(x0 + skip(c, g.xc), y))
                    .collect();
                for s in cpts.chunks(g.swv as usize).filter(|s| s.len() > 1) {
                    w.chain(segment(
                        s,
                        [Class::OnormL1(Half::Lo), Class::OnormL1(Half::Hi)],
                        Class::OnormCast,
                        Slot::Onorm,
                    ));
                }
            }
        }
        if g.hc * 2 != g.py {
            return Err("qwen: head columns = CX rows / 2".into());
        }
        head_input(plan, floor, &mut w)?;
        Ok(w)
    }

    fn palette(&self, _plan: &Plan) -> Palette {
        Palette::base(&[])
    }

    fn kernels(&self, prog: &Program) -> R<BTreeMap<RoleKind, KernelBuild>> {
        let plan = &prog.plan;
        let (s, p) = (&plan.shape, &plan.partition);
        let k = p.layers_per_unit;
        let km: u32 = s.layers[..k as usize]
            .iter()
            .enumerate()
            .filter(|(_, l)| !matches!(l, LayerKind::Gdn))
            .map(|(i, _)| 1 << i)
            .sum();
        let px = p.cols;
        let hcc = p.row_dims.first().ok_or("qwen: no row dims")?.dims;
        let gate = u32::from(s.attn.output_gate);
        let (hd, kvh) = (s.attn.head_dim, s.attn.kv_heads);
        let qh = s.attn.q_heads / kvh;
        let dc = hd / p.segments.scores;
        // GDN's dims (Qwen3: the stand-ins that shape its unused buffers, DV = Qh Dc)
        let (hk, hv) = match s.gdn {
            Some(g) => (g.key_dim, g.value_dim),
            None => (hd, qh * hd),
        };
        let (dk, dv) = (hk / p.segments.gdn, hv / p.segments.onorm);
        let ic = s.mlp.intermediate / px;
        let nqa = (1 + gate) * qh * dc + 2 * dc;
        let nqg = 2 * dk + 2 * dv + 2;
        let mut out = BTreeMap::new();

        out.insert(
            RoleKind::Weight,
            KernelBuild::ints(
                "weight_q35.csl",
                &["w", "rcfg", "tdbg"],
                &[
                    ("K", k),
                    ("Hc", hcc),
                    ("QO", qh * dc),
                    ("Ic", ic),
                    ("MV", 1),
                    ("KM", km),
                    ("NQA", nqa),
                    ("NQG", nqg),
                    ("GDN", u32::from(km != (1 << k) - 1)),
                    ("ATT", u32::from(km != 0)),
                    ("DV", dv),
                    ("DK", dk),
                    ("Qh", qh),
                    ("Dc", dc),
                    ("NE", p.kv_entries.per_row),
                    ("PYW", p.rows),
                    ("HD", hd),
                    ("DIAG", 0),
                    ("FIFOW", 32),
                    ("GDEC", 1),
                    ("ATTL", 2),
                    ("TSTAMP", 0),
                ],
            ),
        );

        out.insert(
            RoleKind::Cx,
            KernelBuild::ints(
                "ctl_cx.csl",
                &["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
                &[
                    ("K", k),
                    ("Hc", hcc),
                    ("HOSTX", 0),
                    ("XG16", 0),
                    ("DIAG", 0),
                    ("FIFOW", 32),
                    ("XKEEP", 0),
                ],
            ),
        );

        out.insert(
            RoleKind::Cy,
            KernelBuild::ints(
                "ctl_cy_q35.csl",
                &["cw", "gpar", "gnw", "qkw", "rot", "tab", "rcfg", "tdbg"],
                &[
                    ("K", k),
                    ("KM", km),
                    ("NQA", nqa),
                    ("NQG", nqg),
                    ("Qh", qh),
                    ("Dc", dc),
                    ("DK", dk),
                    ("DV", dv),
                    ("Ic", ic),
                    ("HR", s.hidden),
                    ("CONV", 4),
                    ("NROT", dc / 2),
                    ("HK", hk),
                    ("HV", hv),
                    ("DIAG", 0),
                    ("FIFOW", 32),
                    ("SILU16", 1),
                    ("CONVQK", 1),
                    ("ATTV", 1),
                    ("GDEC", 1),
                    ("ATTL", 2),
                    ("TSTAMP", 0),
                    ("GATE", gate),
                    ("SLW", 0),
                    ("RLV", plan.contract.rope_levels),
                    ("RBITS", plan.contract.rope_bits),
                ],
            ),
        );

        out.insert(
            RoleKind::Head,
            KernelBuild::ints(
                "head_pe.csl",
                &["hw", "hg", "rcfg", "hemb", "tdbg"],
                &[
                    ("VR", p.head.vocab_rows),
                    ("HDc", 2 * hcc),
                    ("NPX", 1),
                    ("HACC16", 0),
                ],
            ),
        );

        out.insert(
            RoleKind::HeadCtl,
            KernelBuild::ints(
                "head_ctl.csl",
                &["res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce"],
                &[
                    ("VR", p.head.vocab_rows),
                    ("FIFOW", 32),
                    ("WFORCE", 0),
                    ("WFN", p.head.rows),
                ],
            ),
        );

        if !prog.floor.relays.is_empty() {
            out.insert(
                RoleKind::Relay,
                KernelBuild::ints("relay.csl", &["rcfg"], &[]),
            );
        }
        out.insert(RoleKind::Idle, KernelBuild::filler());
        Ok(out)
    }

    fn bindings(&self, prog: &Program) -> R<Vec<Binding>> {
        let g = Geom::of(&prog.plan)?;
        let shape = &prog.plan.shape;
        let pad = |t: &str| -> From {
            let tr = if shape.hidden_placed != shape.hidden {
                vec![Transform::PadZero {
                    dim: 1,
                    to: shape.hidden_placed,
                }]
            } else {
                vec![]
            };
            vec![(t.to_string(), tr)]
        };
        let mut out = Copies::default();
        let units = UnitCopies {
            cols: g.px,
            rows: g.py,
            w_from: pad("layers.*.{q,k,v,o,gate,up,down}_proj, linear_attn.*"),
            gam_b16: true,
            gam_from: pad("layers.*.{input,post_attention}_layernorm, norm"),
            cy: &[
                ("cw", true),
                ("gpar", true),
                ("gnw", true),
                ("qkw", true),
                ("rot", true),
            ],
            cy_optional: true,
        };
        for unit in &prog.floor.units {
            units.put(&mut out, unit);
        }
        for r in &prog.floor.relays {
            out.rcfg(*r);
        }
        let vocab = shape.vocab;
        HeadCopies {
            x0: g.hx0,
            cols: g.hc,
            rows: g.hr,
            hw: &|_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
            hw_from: vec![(
                "lm_head".into(),
                vec![Transform::PadLastRow {
                    to: g.hr * prog.plan.partition.head.vocab_rows,
                }],
            )],
            hg_from: from("norm"),
            hemb: if g.token_loop {
                Hemb::After(from(&format!("embed_tokens ({vocab} rows)")))
            } else {
                Hemb::None
            },
        }
        .put(&mut out);
        Ok(out.0)
    }
}

impl RcfgAbi for Qwen {
    fn route_slots(&self, kind: RoleKind) -> usize {
        match kind {
            RoleKind::Weight => 16,
            RoleKind::Relay => 4,
            _ => 12,
        }
    }

    fn rcfg(&self, prog: &Program, at: Coord) -> R<Vec<u32>> {
        let g = Geom::of(&prog.plan)?;
        let pal = self.palette(&prog.plan);
        let pe = Pe::new(prog, &pal, at);
        match prog.floor.roles.get(at) {
            PeRole::Weight(w) => {
                let gdn = match pe.colour(Slot::Gdn) {
                    Some(c) => c,
                    None => pe.c(Class::RowCast)?,
                };
                let relays = [pe.turn(Slot::RelayPre), pe.turn(Slot::RelayPost)];
                let mid = [g.gdn_source(w.row, w.col), gdn];
                abi::weight(&pe, 16, w.row, mid, g.hcc, relays)
            }
            PeRole::Cx(cx) => {
                let x = CxIo {
                    first: cx.first,
                    inn: pe.colour(Slot::XIn),
                    out: pe.colour(Slot::XOut),
                    emb: pe.colour(Slot::EmbIn),
                    ..CxIo::default()
                };
                abi::cx(&pe, x, 3, &CT5)
            }
            PeRole::Cy(_) => abi::cy(&pe, Some((Slot::Onorm, Class::OnormCast))),
            PeRole::Head(_) => abi::head(
                &pe,
                HeadIo {
                    row: at.y,
                    xr: g.head_role(&prog.floor, at)?,
                    relay_north: true,
                    tr: g.head_turns(&prog.floor, at),
                    token_loop: g.token_loop,
                },
            ),
            PeRole::HeadCtl(_) => abi::hctl(
                &pe,
                at.y * prog.plan.partition.head.vocab_rows,
                g.token_loop,
            ),
            PeRole::Relay => pe.routes(4),
            r => Err(format!("{at:?}: no rcfg for {r:?}")),
        }
    }
}
