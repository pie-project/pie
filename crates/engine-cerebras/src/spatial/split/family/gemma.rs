//! Gemma 4 E2B / E4B: explicit unit origins in unit columns, the PLE storage blocks above each
//! unit column (token word `TB`, per-column bus `PB`, store PEs' levels), the shared-KV line
//! (`KVS`) from the KV source unit to its sharers, handoffs of 2 Hc + 1 words, the sliding /
//! global KV entries (`NES` / `NEG`) and the KV rows (`PYW`: E4B keeps KV on 128 of 160 rows so
//! the 512 window is an exact ring).
//!
//! Kernels: `weight_gemma.csl` (18 route slots), `ctl_cx_gemma.csl`, `ctl_cy_gemma.csl`,
//! `head_pe.csl`, `head_ctl_gemma.csl`, `ple_gemma.csl` (the store kernel: storage PEs and the
//! route-only PEs).

use std::collections::BTreeMap;

use crate::spatial::split::emit::meta::Meta;
use crate::spatial::split::emit::params::KernelBuild;
use crate::spatial::split::emit::rcfg::RcfgAbi;
use crate::spatial::split::fabric::{Colour, Palette};
use crate::spatial::split::family::Family;
use crate::spatial::split::family::abi::{self, CT5, CT6, CxIo, HeadIo, Pe};
use crate::spatial::split::family::tiles::{HeadCopies, Hemb, UnitCopies};
use crate::spatial::split::floor::{
    Coord, CtlLine, Dir, Floor, FloorBuilder, Rect, StoreBlock, Unit, UnitGrid, UnitLayout, skip,
};
use crate::spatial::split::plan::{
    Act, AttnSpec, Contract, FamilyId, HeadPartition, HeadSpec, KvEntries, LayerKind, MlpSpec,
    NormSpec, Partition, Plan, PleSpec, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::split::program::Program;
use crate::spatial::split::role::{self, PeRole, RoleKind};
use crate::spatial::split::weights::{Binding, Copies, Slice, Transform, chunks, from};
use crate::spatial::split::wiring::{self, Class, Half, Link, Port, Slot, Turn, Wiring};

pub struct Gemma;

// ------------------------------------------------------------------ arrangements

/// A PLE storage block's size: rows above its column's top unit, tokens a storage PE, bus colour.
struct StoreSpec {
    rows: u32,
    tokens: u32,
    bus: u8,
}

/// A hand-made Gemma floorplan: unit column A at
/// x 0 holds `a_units` top to bottom, stacked from row `a_top`; unit column B, east of it, holds
/// each of `b_units` level with a column-A unit; above each column's top unit its PLE storage
/// block. Unit index order is the handoff order (unit 0 takes the embedding). The head sits east
/// of column B.
struct Arrangement {
    a_top: u32,
    a_units: &'static [u32],
    /// (unit, the column-A unit it sits level with), top to bottom.
    b_units: &'static [(u32, u32)],
    /// Column A's storage, column B's.
    stores: [StoreSpec; 2],
}

/// The arrangements, by unit shape (weight columns, rows, units). A new Gemma size adds one.
const ARRANGEMENTS: &[((u32, u32, u32), Arrangement)] = &[
    // the tiny simulation: E2B's arrangement in small (unit 0 shares the last unit's rows, as the
    // head's turns need)
    (
        (8, 8, 5),
        Arrangement {
            a_top: 4,
            a_units: &[4, 3, 2],
            b_units: &[(0, 4), (1, 2)],
            stores: [
                StoreSpec {
                    rows: 4,
                    tokens: 128,
                    bus: 17,
                },
                StoreSpec {
                    rows: 2,
                    tokens: 256,
                    bus: 10,
                },
            ],
        },
    ),
    // E2B: column A units 6 .. 2 down to the wafer's last row (1153), column B units 0 (level
    // with 6) and 1 (level with 2)
    (
        (256, 128, 7),
        Arrangement {
            a_top: 1153 - 129 * 5,
            a_units: &[6, 5, 4, 3, 2],
            b_units: &[(0, 6), (1, 2)],
            stores: [
                StoreSpec {
                    rows: 345,
                    tokens: 760,
                    bus: 17,
                },
                StoreSpec {
                    rows: 138,
                    tokens: 1900,
                    bus: 10,
                },
            ],
        },
    ),
    // E4B: column A units 6 .. 3, column B units 0, 1, 2 level with 6, 5, 3
    (
        (256, 160, 7),
        Arrangement {
            a_top: 1153 - 161 * 4,
            a_units: &[6, 5, 4, 3],
            b_units: &[(0, 6), (1, 5), (2, 3)],
            stores: [
                StoreSpec {
                    rows: 332,
                    tokens: 791,
                    bus: 17,
                },
                StoreSpec {
                    rows: 249,
                    tokens: 1055,
                    bus: 10,
                },
            ],
        },
    ),
];

impl Arrangement {
    fn of(plan: &Plan) -> Result<&'static Arrangement, String> {
        let p = &plan.partition;
        let key = (p.cols, p.rows, p.units);
        ARRANGEMENTS
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, a)| a)
            .ok_or_else(|| {
                format!(
                    "split gemma: no arrangement for {} x {} units x {}",
                    p.cols, p.rows, p.units
                )
            })
    }

    /// Unit origins in unit order, for units of `uw` x `uh` PEs.
    fn origins(&self, uw: u32, uh: u32) -> Result<Vec<(u32, u32)>, String> {
        let n = self.a_units.len() + self.b_units.len();
        let mut o = vec![None; n];
        let mut ya = BTreeMap::new();
        for (i, &u) in self.a_units.iter().enumerate() {
            let y = self.a_top + i as u32 * uh;
            ya.insert(u, y);
            o[u as usize] = Some((0, y));
        }
        for &(u, level) in self.b_units {
            let y = *ya
                .get(&level)
                .ok_or_else(|| format!("split gemma: unit {u} level with {level}, not in A"))?;
            o[u as usize] = Some((uw, y));
        }
        o.into_iter()
            .enumerate()
            .map(|(u, v)| v.ok_or_else(|| format!("split gemma: unit {u} unplaced")))
            .collect()
    }

    /// The unit columns: (x, units top to bottom, storage).
    fn columns(&self, uw: u32) -> [(u32, Vec<u32>, &StoreSpec); 2] {
        [
            (0, self.a_units.to_vec(), &self.stores[0]),
            (
                uw,
                self.b_units.iter().map(|&(u, _)| u).collect(),
                &self.stores[1],
            ),
        ]
    }
}

/// A named Gemma geometry the prep builds for (`prep_gemma.py`'s `geo`): the unit shape that picks
/// the arrangement (weight columns, rows, units), the head's columns and rows, and the dims a
/// weight row holds.
struct Named {
    name: &'static str,
    shape: (u32, u32, u32),
    head_cols: u32,
    head_rows: u32,
    dims: u32,
}

const NAMED: &[Named] = &[
    Named {
        name: "tiny",
        shape: (8, 8, 5),
        head_cols: 4,
        head_rows: 32,
        dims: 3,
    },
    Named {
        name: "e2b",
        shape: (256, 128, 7),
        head_cols: 64,
        head_rows: 1130,
        dims: 12,
    },
    Named {
        name: "e4b",
        shape: (256, 160, 7),
        head_cols: 80,
        head_rows: 1130,
        dims: 16,
    },
];

/// A named geometry as the prep reads it (`split-geo --geometry gemma:<name>`), before any meta
/// exists: the unit shape and count, the head, a row's dims, and each unit column's x, units (top
/// to bottom) and PLE storage (rows `srows`, `nt` tokens a storage PE).
pub fn geometry(name: &str) -> Result<serde_json::Value, String> {
    let n = NAMED
        .iter()
        .find(|n| n.name == name)
        .ok_or_else(|| format!("split gemma: no geometry {name}"))?;
    let (px, py, nu) = n.shape;
    let (_, arr) = ARRANGEMENTS
        .iter()
        .find(|(k, _)| *k == (px, py, nu))
        .ok_or_else(|| format!("split gemma: no arrangement for {name}"))?;
    let (uw, uh) = (px + 1, py + 1);
    let origins = arr.origins(uw, uh)?;
    let cols = arr
        .columns(uw)
        .iter()
        .map(|(x, units, st)| {
            let top = origins[units[0] as usize].1;
            serde_json::json!({
                "x": x,
                "units": units,
                "srows": [top - st.rows, top],
                "nt": st.tokens,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "PX": px, "PY": py, "XC": px / 2, "NU": nu, "Hc": n.dims, "HC": n.head_cols,
        "HR": n.head_rows,
        "origins": origins, "cols": cols,
    }))
}

// ------------------------------------------------------------------ the plan's Gemma facts

/// The KV-sharing units: the source unit (its CY row sends KVS) and the sharers (units whose
/// layers read another layer's KV).
fn kv_units(plan: &Plan) -> (Option<u32>, Vec<u32>) {
    let k = plan.partition.layers_per_unit;
    let layers = &plan.shape.layers;
    let dst: Vec<u32> = (0..plan.partition.units)
        .filter(|&u| matches!(layers[(u * k) as usize], LayerKind::SharedKv { .. }))
        .collect();
    let src = layers.iter().find_map(|l| match *l {
        LayerKind::SharedKv { from, .. } => Some(from / k),
        _ => None,
    });
    (src, dst)
}

/// Handoff words: x | the pending term p | its row's |h|^2.
fn hand_n(plan: &Plan) -> u32 {
    2 * plan.partition.row_dims[0].dims + 1
}

/// Store bus words of a column of `units` units.
fn levels(plan: &Plan, units: usize) -> u32 {
    plan.partition.layers_per_unit * units as u32
}

/// The largest staged PLE file (`prep_gemma.py`'s `MAXB`).
const PLE_FILE_MAX: u64 = 1900 * 1024 * 1024;

/// A storage block's staged files `pt<col>_<n>`: their row ranges (`prep_gemma.py`'s split).
fn ple_files(rows: u32, px: u32, tokens: u32, lv: u32) -> Vec<(u32, u32)> {
    let per = (PLE_FILE_MAX / (px as u64 * tokens as u64 * lv as u64 * 2)).max(1) as u32;
    (0..rows)
        .step_by(per as usize)
        .map(|r0| (r0, (r0 + per).min(rows)))
        .collect()
}

fn unit_containing(floor: &Floor, c: Coord) -> Option<&Unit> {
    floor.units.iter().find(|u| u.rect.contains(c))
}

/// The store block a PE of the storage area belongs to (its index is its bus).
fn store_of(floor: &Floor, c: Coord) -> Option<usize> {
    floor.stores.iter().position(|s| s.rect.contains(c))
}

/// The store column serving unit `u`: (its index, the unit's place top to bottom, bus words).
fn column_of(floor: &Floor, plan: &Plan, u: u32) -> Option<(usize, u32, u32)> {
    floor.stores.iter().enumerate().find_map(|(i, s)| {
        s.units
            .iter()
            .position(|&v| v == u)
            .map(|ui| (i, ui as u32, levels(plan, s.units.len())))
    })
}

fn bus(i: usize) -> Class {
    Class::PleBus(i as u8)
}

impl Family for Gemma {
    fn id(&self) -> FamilyId {
        FamilyId::Gemma
    }

    /// From `prep_gemma.py`'s meta: units of K layers (S .. S G, KM's bits the global layers),
    /// units at or past NO the shared-KV units (their layers read unit NO - 1's last layer of the
    /// same kind); the sliding window NES PYW exactly (WIN 0) or WIN with every row a KV row.
    /// `geo` (the arrangement's name) only picks the reduction groups' defaults, as before.
    fn plan(&self, meta: &Meta) -> Result<Plan, String> {
        let px = meta.need("PX")?;
        let py = meta.need("PY")?;
        let hcc = meta.need("Hc")?;
        let k = meta.need("K")?;
        let km = meta.need("KM")?;
        let no = meta.need("NO")?;
        let nu = meta.need("NU")?;
        let nes = meta.need("NES")?;
        let neg = meta.need("NEG")?;
        let win = meta.int("WIN").unwrap_or(0);
        let pyw = meta.int("PYW").unwrap_or(py);
        let tiny = meta.str("geo") == Some("tiny");
        let g2 = meta.int("G2").unwrap_or(if tiny { 2 } else { 16 });
        let g2w = meta.int("G2W").unwrap_or(if tiny { 2 } else { 24 });
        let window = if win > 0 { win } else { nes * pyw };
        let global = |l: u32| (km >> (l % k)) & 1 == 1;
        let owned = k * no;
        let layers = (0..k * nu)
            .map(|l| {
                if l < owned {
                    return Ok(if global(l) {
                        LayerKind::Global
                    } else {
                        LayerKind::Sliding { window }
                    });
                }
                // a shared layer reads the last owner layer of its kind
                let from = (0..owned)
                    .rev()
                    .find(|&f| global(f) == global(l))
                    .ok_or_else(|| format!("split gemma: no owner layer for shared layer {l}"))?;
                Ok(LayerKind::SharedKv {
                    from,
                    sliding: !global(l),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let (ic0, ic1) = (meta.need("Ic0")?, meta.need("Ic1")?);
        let shape = Shape {
            gdn: None,
            family: FamilyId::Gemma,
            hidden: meta.need("H")?,
            hidden_placed: meta.need("H")?,
            vocab: meta.need("V")?,
            layers,
            attn: AttnSpec {
                q_heads: 2 * meta.need("Qh")?,
                // the prep's kv heads (1 or 2) are not in the meta; nothing physical reads them
                kv_heads: meta.int("NKV").unwrap_or(1),
                head_dim: meta.need("HDS")?,
                global_head_dim: Some(meta.need("HDG")?),
                qk_norm: true,
                ..Default::default()
            },
            // the KV-owning layers' width; the sharing layers' (Ic1 a column) differs
            mlp: MlpSpec {
                intermediate: ic0 * px,
                act: Act::GeluTanh,
                layer_widths: (0..k * nu)
                    .map(|l| if l < owned { ic0 * px } else { ic1 * px })
                    .collect(),
            },
            moe: None,
            ple: Some(PleSpec { width: px }),
            head: HeadSpec {
                tied: true,
                final_softcap: Some(meta.float("SOFTCAP").unwrap_or(30.0) as f32),
            },
            rope: RopeSpec {
                // the prep bakes the tables (`rot`); the meta records no theta
                theta: meta.float("ROPE_THETA").unwrap_or(0.0) as f32,
                rotary_dims: meta.need("HDS")?,
            },
            norm: NormSpec {
                eps: meta.float("EPS").unwrap_or(1e-6) as f32,
                sandwich: true,
                ..Default::default()
            },
        };
        let partition = Partition {
            unit_grid: None,
            units: nu,
            layers_per_unit: k,
            rows: py,
            cols: px,
            row_dims: vec![RowRun {
                rows: (0, py),
                dims: hcc,
                attention: true,
            }],
            kv_rows: pyw,
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
            // a scores segment a half row
            segments: Segments {
                scores: px / 2,
                gdn: px / 2,
                onorm: px / 2,
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
            positions: neg * pyw,
            rope_levels: 4,
            rope_bits: 3,
            cap: meta.int("CAP").unwrap_or(256),
            loop_len: meta.int("LOOP").unwrap_or(32),
            token_loop: true,
        };
        Ok(Plan {
            shape,
            partition,
            contract,
        })
    }

    /// The arrangement's units, the head east of them (HC + 1 columns, HR rows), the storage
    /// blocks above each unit column; every non-unit PE west of the head that a route crosses
    /// runs the store kernel with no storage (`pass_pes`), the rest are idle.
    fn floor(&self, plan: &Plan) -> Result<Floor, String> {
        let arr = Arrangement::of(plan)?;
        let p = &plan.partition;
        let (px, py) = (p.cols, p.rows);
        let (uw, uh) = (px + 1, py + 1);
        let (xc, yc) = (px / 2, py / 2);
        let origins = arr.origins(uw, uh)?;
        let (hc, hr) = (p.head.cols, p.head.rows);
        let grid = UnitGrid {
            cols: px,
            rows: py,
            cx: xc,
            cy: yc,
            layout: UnitLayout::Explicit {
                origins: origins.clone(),
            },
            layers_per_unit: p.layers_per_unit,
        };
        let mut fb = FloorBuilder::units(&grid, (0, 0))?;
        let hx0 = fb.w;
        let head = fb.head_east(hx0, hc, hr, true);
        // the storage blocks, directly above each column's top unit
        for (x, units, st) in arr.columns(uw) {
            let top = origins[units[0] as usize].1;
            let y0 = top
                .checked_sub(st.rows)
                .ok_or("split gemma: the storage runs off the wafer's top")?;
            let rect = Rect {
                x,
                y: y0,
                w: uw,
                h: st.rows,
            };
            let lv = levels(plan, units.len());
            for y in y0..top {
                for lx in 0..uw {
                    let store = if lx == xc {
                        // the block's CX column: routes only
                        role::Store {
                            first_token: 0,
                            tokens: 0,
                            level: 0,
                            top: false,
                        }
                    } else {
                        role::Store {
                            first_token: (y - y0) * st.tokens,
                            tokens: st.tokens,
                            level: lv,
                            top: y == y0,
                        }
                    };
                    fb.set_role(Coord::new(x + lx, y), PeRole::Store(store));
                }
            }
            fb.stores.push(StoreBlock {
                rect,
                units,
                tokens_per_pe: st.tokens,
            });
        }
        let mut floor = fb.finish(head);
        // the route-only PEs: non-unit, non-storage PEs west of the head a link crosses
        let wiring = self.wire(plan, &floor)?;
        for c in touched(&wiring) {
            if c.x < hx0
                && matches!(floor.roles.get(c), PeRole::Idle)
                && unit_containing(&floor, c).is_none()
                && store_of(&floor, c).is_none()
            {
                *floor.roles.get_mut(c) = PeRole::Store(role::Store {
                    first_token: 0,
                    tokens: 0,
                    level: 0,
                    top: false,
                });
            }
        }
        Ok(floor)
    }

    /// The generic unit wiring (chains, loop-backs, handoffs, the head rows, the token loop and
    /// the embedding's return), the scores segments on every weight row, the head input from the
    /// last unit, then Gemma's own: the PLE token word (TB) along each storage row, the PLE bus (PB)
    /// down each weight column of a unit column, the shared-KV line (KVS) north from the source
    /// unit's CY row through its sharers.
    fn wire(&self, plan: &Plan, floor: &Floor) -> Result<Wiring, String> {
        let mut w = Wiring::default();
        wiring::units(plan, floor, &mut w)?;
        let p = &plan.partition;
        for unit in &floor.units {
            for r in 0..p.rows {
                let pts: Vec<Coord> = (0..p.cols)
                    .map(|c| Coord::new(unit.col_x(c), unit.row_y(r)))
                    .collect();
                wiring::segments(
                    &pts,
                    p.segments.scores,
                    [Class::ScoreL1(Half::Lo), Class::ScoreL1(Half::Hi)],
                    Class::ScoreCast,
                    Slot::Segment,
                    &mut w,
                )?;
            }
        }
        wiring::head_input(plan, floor, &mut w)?;
        ple_token(floor, &mut w)?;
        ple_bus(plan, floor, &mut w)?;
        shared_kv(plan, floor, &mut w)?;
        Ok(w)
    }

    /// The base colours, the PLE token word (`TB`), the shared-KV line (`KVS`) and the
    /// arrangement's PLE buses.
    fn palette(&self, plan: &Plan) -> Palette {
        let mut own = vec![(Class::PleToken, 0), (Class::SharedKv, 10)];
        if let Ok(arr) = Arrangement::of(plan) {
            own.extend(
                arr.stores
                    .iter()
                    .enumerate()
                    .map(|(i, st)| (bus(i), st.bus)),
            );
        }
        Palette::base(&own)
    }

    /// `mkfull_gemma.py`'s kernels: each kind's source, the symbols it owns, its tile-code params
    /// as the plan states them, plus the lever defaults (`PD`). `SH` and `Ic` are the KV-owning
    /// units'; the KV-sharing units' (SH 1, their wider MLP) and the route-only store PEs' NTW 1
    /// follow from their roles ([`crate::spatial::split::emit::params::role_params`]).
    fn kernels(&self, prog: &Program) -> Result<BTreeMap<RoleKind, KernelBuild>, String> {
        let plan = &prog.plan;
        let p = &plan.partition;
        let s = &plan.shape;
        let k = p.layers_per_unit;
        let km: u32 = (0..k)
            .filter(|&l| matches!(s.layers[l as usize], LayerKind::Global))
            .map(|l| 1 << l)
            .sum();
        let hcc = p.row_dims[0].dims;
        let qh = s.attn.q_heads / 2;
        let dcs = s.attn.head_dim / (p.cols / 2);
        let hdg = s.attn.global_head_dim.unwrap_or(s.attn.head_dim);
        let dcg = hdg / (p.cols / 2);
        let ic = s.mlp.intermediate / p.cols;
        let window = s
            .layers
            .iter()
            .find_map(|l| match *l {
                LayerKind::Sliding { window } => Some(window),
                _ => None,
            })
            .unwrap_or(0);
        let nes = p.kv_entries.sliding.unwrap_or(0);
        let win = if nes * p.kv_rows == window { 0 } else { window };
        let neg = p.kv_entries.global.unwrap_or(0);
        let ntw = prog
            .floor
            .stores
            .iter()
            .map(|st| st.tokens_per_pe * levels(plan, st.units.len()))
            .max()
            .unwrap_or(0);
        let mut out = BTreeMap::new();
        out.insert(
            RoleKind::Weight,
            KernelBuild::ints(
                "weight_gemma.csl",
                &["w", "rcfg", "tdbg", "kvlog"],
                &[
                    ("K", k),
                    ("Hc", hcc),
                    ("Ic", ic),
                    ("MV", 1),
                    ("KM", km),
                    ("SH", 0),
                    ("Qh", qh),
                    ("DcS", dcs),
                    ("DcG", dcg),
                    ("NES", nes),
                    ("NEG", neg),
                    ("PYW", p.kv_rows),
                    ("HDS", s.attn.head_dim),
                    ("HDG", hdg),
                    ("WIN", win),
                    ("WSPEC", 0),
                    ("PLOG", 0),
                    ("DIAG", 0),
                    ("FIFOW", 32),
                    ("ATTL", 2),
                    ("TSTAMP", 0),
                ],
            ),
        );
        out.insert(
            RoleKind::Cx,
            KernelBuild::ints(
                "ctl_cx_gemma.csl",
                &["x", "gam", "rcfg", "xdbg", "tdbg", "xred"],
                &[
                    ("K", k),
                    ("Hc", hcc),
                    ("DIAG", 0),
                    ("FIFOW", 32),
                    ("XKEEP", 0),
                ],
            ),
        );
        out.insert(
            RoleKind::Cy,
            KernelBuild::ints(
                "ctl_cy_gemma.csl",
                &["qkw", "rot", "tab", "pl", "plog", "rcfg", "tdbg"],
                &[
                    ("K", k),
                    ("KM", km),
                    ("SH", 0),
                    ("Qh", qh),
                    ("DcS", dcs),
                    ("DcG", dcg),
                    ("Ic", ic),
                    ("HR", s.hidden),
                    ("DIAG", 0),
                    ("FIFOW", 32),
                    ("ATTL", 2),
                    ("TSTAMP", 0),
                    ("HDS", s.attn.head_dim),
                    ("HDG", hdg),
                    ("PLOG", 0),
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
                "head_ctl_gemma.csl",
                &[
                    "res", "rcfg", "tdbg", "ring", "lgt", "force", "nforce", "tin",
                ],
                &[
                    ("VR", p.head.vocab_rows),
                    ("FIFOW", 32),
                    ("WFORCE", 0),
                    ("WFN", p.head.rows),
                ],
            ),
        );
        out.insert(
            RoleKind::Store,
            KernelBuild::ints(
                "ple_gemma.csl",
                &["ptab", "rcfg", "psend"],
                &[("NTW", ntw), ("DLY", 1500)],
            ),
        );
        Ok(out)
    }

    /// The placement, in order: per unit its four weight quadrants (rcfg, then the
    /// weights in blocks of 16 rows), its CX halves (rcfg, gammas), its CY halves (rcfg, q / k
    /// norm weights, RoPE tables); the PLE storage per column and staged file (rcfg, then the
    /// tiles padded to NTW words, in blocks of 16 rows) and the route-only PEs' rcfg by row runs;
    /// the head halves in blocks of 64 rows (rcfg, hw, hg, hemb); the head control column's rcfg.
    fn bindings(&self, prog: &Program) -> Result<Vec<Binding>, String> {
        let plan = &prog.plan;
        let floor = &prog.floor;
        let p = &plan.partition;
        let (px, xc) = (p.cols, p.cols / 2);
        let mut out = Copies::default();
        let units = UnitCopies {
            cols: px,
            rows: p.rows,
            w_from: Vec::new(),
            gam_b16: false,
            gam_from: Vec::new(),
            cy: &[("qkw", true), ("rot", false)],
            cy_optional: false,
        };
        for unit in &floor.units {
            units.put(&mut out, unit);
        }
        // PLE storage, the tiles padded to NTW words
        let ntw = floor
            .stores
            .iter()
            .map(|st| st.tokens_per_pe * levels(plan, st.units.len()))
            .max()
            .unwrap_or(0);
        for (ci, st) in floor.stores.iter().enumerate() {
            let lv = levels(plan, st.units.len());
            let files = ple_files(st.rect.h, px, st.tokens_per_pe, lv);
            for (n, (fr0, fr1)) in files.into_iter().enumerate() {
                for (c0, c1) in [(0, xc), (xc, px)] {
                    for (rb, re) in chunks(fr0, fr1, 16) {
                        let rect =
                            Rect::new(st.rect.x + skip(c0, xc), st.rect.y + rb, c1 - c0, re - rb);
                        out.rcfg(rect);
                        let dims = vec![(rb - fr0, re - fr0), (c0, c1)];
                        out.slice(
                            "ptab",
                            rect,
                            Slice::of(&format!("pt{ci}_{n}"), None, dims).pad_to(ntw),
                            true,
                            from("embed_tokens_per_layer, per_layer_model_projection"),
                        );
                    }
                }
            }
        }
        // the route-only store PEs (storage blocks' CX columns, the pass PEs), by row runs
        let mut rows: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (c, r) in floor.roles.iter() {
            if matches!(r, PeRole::Store(s) if s.tokens == 0) {
                rows.entry(c.y).or_default().push(c.x);
            }
        }
        for (y, mut xs) in rows {
            xs.sort_unstable();
            for run in xs.chunk_by(|a, b| b == &(a + 1)) {
                out.rcfg(Rect::new(run[0], y, run.len() as u32, 1));
            }
        }
        let hr = p.head.rows;
        HeadCopies {
            x0: floor.head.rect.x,
            cols: p.head.cols,
            rows: hr,
            hw: &|_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
            hw_from: vec![(
                "embed_tokens".into(),
                vec![Transform::PadLastRow {
                    to: hr * p.head.vocab_rows,
                }],
            )],
            hg_from: from("norm"),
            hemb: Hemb::Inline(Vec::new()),
        }
        .put(&mut out);
        Ok(out.0)
    }
}

// ------------------------------------------------------------------ Gemma's links

/// Every PE a link crosses.
fn touched(w: &Wiring) -> Vec<Coord> {
    let mut out = Vec::new();
    for l in &w.links {
        match l {
            Link::Chain(c) => out.extend_from_slice(&c.pts),
            Link::Multicast(m) => out.extend_from_slice(&m.pts),
            Link::Path(p) => out.extend_from_slice(&p.pts),
            Link::Turn(t) => out.push(t.at),
        }
    }
    out
}

fn turn(w: &mut Wiring, at: Coord, class: Class, from: Port, to: Vec<Port>) {
    w.turn(Turn {
        at,
        class,
        from,
        to,
    });
}

/// TB: one stream a storage row, west from the head control column to the westmost storage block
/// on the row; every storage PE (not a block's CX column) takes a copy.
fn ple_token(floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let CtlLine::Column(ctl) = floor.head.ctl else {
        return Err("split gemma: no head control column".into());
    };
    let mut rows: BTreeMap<u32, Vec<&StoreBlock>> = BTreeMap::new();
    for st in &floor.stores {
        for y in st.rect.y..st.rect.y + st.rect.h {
            rows.entry(y).or_default().push(st);
        }
    }
    for (y, blocks) in rows {
        let xw = blocks.iter().map(|b| b.rect.x).min().unwrap_or(0);
        let xs: Vec<u32> = (xw..=ctl).rev().collect();
        for (i, &x) in xs.iter().enumerate() {
            let from = if i == 0 {
                Port::Ramp
            } else {
                Port::Dir(Dir::E)
            };
            let mut to = Vec::new();
            if i + 1 < xs.len() {
                to.push(Port::Dir(Dir::W));
            }
            let takes = x < floor.head.rect.x
                && blocks.iter().any(|b| {
                    let cx = b.rect.x + floor_xc(floor);
                    b.rect.x <= x && x < b.rect.x + b.rect.w && x != cx
                });
            if takes {
                to.push(Port::Ramp);
            }
            turn(w, Coord::new(x, y), Class::PleToken, from, to);
        }
    }
    Ok(())
}

/// The units' CX column (local x), the same in every unit.
fn floor_xc(floor: &Floor) -> u32 {
    floor.units.first().map_or(0, |u| u.cx)
}

/// PB: a bus a unit column, down each weight column from the storage block through the column's
/// units (and the gaps between them); each unit's CY PE takes a copy, the last unit's ends it.
fn ple_bus(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let p = &plan.partition;
    for (i, st) in floor.stores.iter().enumerate() {
        let units: Vec<&Unit> =
            st.units
                .iter()
                .map(|&u| {
                    floor.units.iter().find(|v| v.index == u).ok_or_else(|| {
                        format!("split gemma: store column unit {u} not on the floor")
                    })
                })
                .collect::<Result<_, _>>()?;
        let (y0, y1) = (st.rect.y, st.rect.y + st.rect.h);
        if units.first().map(|u| u.rect.y) != Some(y1) {
            return Err(
                "split gemma: the storage sits directly above the column's top unit".into(),
            );
        }
        for c in 0..p.cols {
            let x = st.rect.x + skip(c, floor_xc(floor));
            for y in y0..y1 {
                let from = if y == y0 {
                    Port::Ramp
                } else {
                    Port::Dir(Dir::N)
                };
                turn(w, Coord::new(x, y), bus(i), from, vec![Port::Dir(Dir::S)]);
            }
            let mut prev_end = y1;
            for (ui, unit) in units.iter().enumerate() {
                let uy = unit.rect.y;
                let lastu = ui + 1 == units.len();
                for y in prev_end..uy {
                    turn(
                        w,
                        Coord::new(x, y),
                        bus(i),
                        Port::Dir(Dir::N),
                        vec![Port::Dir(Dir::S)],
                    );
                }
                prev_end = uy + unit.rect.h;
                for ly in 0..unit.rect.h {
                    let at = Coord::new(x, uy + ly);
                    if ly == unit.cy {
                        let mut to = vec![Port::Ramp];
                        if !lastu {
                            to.push(Port::Dir(Dir::S));
                        }
                        turn(w, at, bus(i), Port::Dir(Dir::N), to);
                        if lastu {
                            break;
                        }
                    } else {
                        turn(w, at, bus(i), Port::Dir(Dir::N), vec![Port::Dir(Dir::S)]);
                    }
                }
            }
        }
    }
    Ok(())
}

/// KVS: the source unit's CY row north through the sharer units above it in its column (sharer
/// weight PEs take a copy; the source's weight PEs and the sharers' CY PEs pass it).
fn shared_kv(plan: &Plan, floor: &Floor, w: &mut Wiring) -> Result<(), String> {
    let (Some(src), dst) = kv_units(plan) else {
        return Ok(());
    };
    let unit = |u: u32| {
        floor
            .units
            .iter()
            .find(|v| v.index == u)
            .ok_or_else(|| format!("split gemma: KV unit {u} not on the floor"))
    };
    let s = unit(src)?;
    let mut ytop = u32::MAX;
    for &u in &dst {
        let d = unit(u)?;
        if d.rect.x != s.rect.x || d.rect.y >= s.rect.y {
            return Err("split gemma: the KVS consumers sit above the source in its column".into());
        }
        ytop = ytop.min(d.rect.y);
    }
    if dst.is_empty() {
        return Ok(());
    }
    for c in 0..plan.partition.cols {
        let x = s.col_x(c);
        let ysrc = s.cy_y();
        turn(
            w,
            Coord::new(x, ysrc),
            Class::SharedKv,
            Port::Ramp,
            vec![Port::Dir(Dir::N)],
        );
        for y in (ytop..ysrc).rev() {
            let at = Coord::new(x, y);
            let u = unit_containing(floor, at)
                .ok_or_else(|| format!("split gemma: KVS crosses a gap at ({x}, {y})"))?;
            let take = dst.contains(&u.index) && y - u.rect.y != u.cy;
            let mut to = Vec::new();
            if take {
                to.push(Port::Ramp);
            }
            if y > ytop {
                to.push(Port::Dir(Dir::N));
            }
            turn(w, at, Class::SharedKv, Port::Dir(Dir::S), to);
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ rcfg

impl RcfgAbi for Gemma {
    fn route_slots(&self, kind: RoleKind) -> usize {
        match kind {
            RoleKind::Weight => 18,
            RoleKind::Store => 8,
            RoleKind::Relay => 4,
            _ => 12,
        }
    }

    /// The shared words ([`abi`]) with Gemma's: a weight PE's flags (column 0 sends the deferred
    /// scalar 1, a KVS consumer 2, the flush unit 4) and the KVS colour; the last CX's flush
    /// flags (8 | 16); a CY's PLE flags (on a bus 1, the KVS source 2, the flush unit 4), its bus
    /// colour, place in its column, bus words, the KVS colour; a head control row's PLE token
    /// flag and the TB colour; the store PEs.
    fn rcfg(&self, prog: &Program, at: Coord) -> Result<Vec<u32>, String> {
        let plan = &prog.plan;
        let pal = self.palette(plan);
        let pe = Pe::new(prog, &pal, at);
        let last = plan.partition.units - 1;
        let kvs = pe.c(Class::SharedKv)?;
        match prog.floor.roles.get(at) {
            PeRole::Weight(wt) => {
                let fl = u32::from(wt.col == 0)
                    | if kv_units(plan).1.contains(&wt.unit) {
                        2
                    } else {
                        0
                    }
                    | if wt.unit == last { 4 } else { 0 };
                let relays = [pe.turn(Slot::RelayPre), pe.turn(Slot::RelayPost)];
                abi::weight(&pe, 18, wt.row, [fl, kvs], hand_n(plan), relays)
            }
            PeRole::Cx(cx) => {
                let x = CxIo {
                    first: cx.first,
                    inn: pe.colour(Slot::XIn),
                    out: pe.colour(Slot::XOut),
                    emb: pe.colour(Slot::EmbIn),
                    flags: if cx.last { 8 | 16 } else { 0 },
                    ..CxIo::default()
                };
                abi::cx(&pe, x, 3, &CT5)
            }
            PeRole::Cy(cy) => {
                let mut w = abi::cy(&pe, None)?;
                let col = column_of(&prog.floor, plan, cy.unit);
                let fl = u32::from(col.is_some())
                    | if kv_units(plan).0 == Some(cy.unit) {
                        2
                    } else {
                        0
                    }
                    | if cy.unit == last { 4 } else { 0 };
                let (pb, ui, lv) = match col {
                    Some((i, ui, lv)) => (pe.c(bus(i))?, ui, lv),
                    None => (0, 0, 0),
                };
                w.extend_from_slice(&[fl, pb, ui, lv, kvs]);
                Ok(w)
            }
            PeRole::Head(hd) => {
                let (u0, ul) = match (prog.floor.units.first(), prog.floor.units.last()) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err("split gemma: no units".into()),
                };
                let c = hd.col;
                let xr = if at.y == ul.row_y(2 * c) {
                    2
                } else if at.y == ul.row_y(2 * c + 1) {
                    1
                } else {
                    0
                };
                let tr = u32::from(at.y == 0)
                    | if at.y == u0.row_y(2 * c) { 2 } else { 0 }
                    | if at.y == u0.row_y(2 * c + 1) { 4 } else { 0 };
                let h = HeadIo {
                    row: at.y,
                    xr,
                    relay_north: true,
                    tr,
                    token_loop: true,
                };
                abi::head(&pe, h)
            }
            PeRole::HeadCtl(_) => {
                let mut w = abi::hctl(&pe, at.y * plan.partition.head.vocab_rows, true)?;
                let tb = prog
                    .floor
                    .stores
                    .iter()
                    .any(|s| at.y >= s.rect.y && at.y < s.rect.y + s.rect.h);
                w.extend_from_slice(&[u32::from(tb), pe.c(Class::PleToken)?]);
                Ok(w)
            }
            PeRole::Store(st) => store(&pe, st),
            PeRole::Idle | PeRole::Relay => Err(format!("{at:?}: no Gemma rcfg for this PE")),
        }
    }
}

/// A store PE's words: the route table (8 slots), the flags (a storage PE 1, takes TB 2, the
/// column's top 4), its first token, tokens, bus words, bus colour, TB's queue, a spare colour.
fn store(pe: &Pe, st: &role::Store) -> Result<Vec<u32>, String> {
    let tb = pe.c(Class::PleToken)?;
    let mut w = pe.routes(8)?;
    let stored = st.tokens > 0;
    let tb_in = pe
        .prog
        .fabric
        .route(pe.at, Colour(tb as u8))
        .is_some_and(|r| r.to_ramp());
    let pk = pe.parks(2)?;
    let fl = u32::from(stored) | if tb_in { 2 } else { 0 } | if stored && st.top { 4 } else { 0 };
    let tbq = if tb_in { tb } else { pk[0] };
    let pb = if stored {
        let i = store_of(&pe.prog.floor, pe.at)
            .ok_or_else(|| format!("{:?}: a storage PE off the storage", pe.at))?;
        pe.c(bus(i))?
    } else {
        pk[1]
    };
    let (first, tokens, level) = if stored {
        (st.first_token, st.tokens, st.level)
    } else {
        (0, 0, 0)
    };
    w.extend_from_slice(&[fl, first, tokens, level, pb, tbq]);
    w.push(pe.spare(&[tbq], &CT6)?);
    Ok(w)
}
