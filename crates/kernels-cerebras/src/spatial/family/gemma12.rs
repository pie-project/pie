//! Gemma 4 12B: units of PY + 1 rows by PX + 1 columns at explicit origins in two unit
//! columns (the first visited down, the second up: unit 0 and the last unit side by side on row
//! 0), their rows in two runs (attention rows `r < RA` with `Hc A` dims and the scores
//! segments, 128 of them so the 1024 window is an exact ring; plain rows with `Hc W` dims), and
//! the head east of them fed by groups of CX rows, the token loop's winner coming back the same
//! way. Blocks: [`RowRuns`], [`HeadGroups`]. Kernels: [`gemma12`].

use crate::spatial::abi::Kernels;
use crate::spatial::block::Block;
use crate::spatial::block::head::{HeadCopies, Hemb};
use crate::spatial::block::head_groups::HeadGroups;
use crate::spatial::block::row_runs::{RowRuns, RunCopies};
use crate::spatial::emit::meta::Meta;
use crate::spatial::family::Family;
use crate::spatial::floor::{UnitGrid, UnitLayout};
use crate::spatial::kernels::gemma12::{self, Dims};
use crate::spatial::plan::{
    Act, AttnSpec, Contract, FamilyId, HeadPartition, HeadSpec, KvEntries, LayerKind, MlpSpec,
    NormSpec, Partition, Plan, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::program::Program;
use crate::spatial::weights::{Slice, Transform, from};

pub struct Gemma12;

type R<T> = Result<T, String>;

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

impl Gemma12 {
    /// The plan from `prep_g12.py`'s meta, the geometry its `geo` names, and the build's `G2` /
    /// `G2W` (default 16 / 24; the tiny 2 / 2).
    fn read_plan(meta: &Meta) -> R<Plan> {
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
            ("HC", plan.partition.head.cols),
            ("WIN", d.win),
        ] {
            if let Some(m) = meta.int(k).filter(|&m| m != v) {
                return Err(format!("split g12 meta: {k} {m}, the plan makes {v}"));
            }
        }
        Ok(plan)
    }

    /// Units in two unit columns (down the first, up the second).
    fn grid(plan: &Plan) -> R<UnitGrid> {
        let p = &plan.partition;
        let n = p.units;
        if n == 0 || !n.is_multiple_of(2) {
            return Err(format!(
                "split g12: {n} units (two unit columns, unit 0 and the last on row 0)"
            ));
        }
        let (uw, uh, uny) = (p.cols + 1, p.rows + 1, n / 2);
        let origins = (0..n)
            .map(|u| {
                if u < uny {
                    (0, uh * u)
                } else {
                    (uw, uh * (n - 1 - u))
                }
            })
            .collect();
        Ok(UnitGrid {
            cols: p.cols,
            rows: p.rows,
            cx: p.cols / 2,
            cy: p.rows / 2,
            layout: UnitLayout::Explicit { origins },
            layers_per_unit: p.layers_per_unit,
        })
    }
}

impl Family for Gemma12 {
    /// A named geometry as the prep reads it (`split-geo --geometry gemma12:<name>`), before any meta
    /// exists: the unit shape and count, the attention rows and both row kinds' dims, the scores
    /// segment width, the head (one column a row group; `hc`: every row's dims).
    fn geometry(&self, name: &str) -> R<serde_json::Value> {
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

    fn plan(&self, meta: &Meta) -> R<Plan> {
        Gemma12::read_plan(meta)
    }

    /// The units with their row runs, the grouped head. Copies: per unit and run block its
    /// weights (bf16 -> f16) and gammas, its CY row's q / k norms and RoPE tables; the head in
    /// its two halves' files (one is over the 2 GiB a staged file may take), its rows padded with
    /// the embedding's last.
    fn blocks(&self, plan: &Plan) -> R<Vec<Box<dyn Block>>> {
        let p = &plan.partition;
        let f16 = vec![
            "self_attn.{q,k,v,o}_proj.weight",
            "mlp.{gate,up,down}_proj.weight",
        ]
        .into_iter()
        .map(|n| (n.to_string(), vec![Transform::Bf16ToF16]))
        .collect();
        Ok(vec![
            Box::new(RowRuns {
                grid: Gemma12::grid(plan)?,
                copies: RunCopies {
                    w_from: f16,
                    cy: &[("qkw", true), ("rot", false)],
                },
            }),
            Box::new(HeadGroups {
                cols: p.head.cols,
                rows: p.head.rows,
                copies: HeadCopies {
                    cols: p.head.cols,
                    rows: p.head.rows,
                    hw: |i, rows, (c0, c1)| {
                        Slice::of(&format!("hw{i}"), None, vec![rows, (0, c1 - c0)])
                    },
                    hw_from: vec![(
                        "embed_tokens.weight".into(),
                        vec![
                            Transform::PadLastRow {
                                to: p.head.rows * p.head.vocab_rows,
                            },
                            Transform::Bf16ToF16,
                        ],
                    )],
                    hg_from: from("norm.weight"),
                    hemb: Hemb::None,
                },
            }),
        ])
    }

    fn kernels(&self, prog: &Program) -> R<Kernels> {
        gemma12::kernels(prog)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        gemma12::SYMBOLS
    }
}
