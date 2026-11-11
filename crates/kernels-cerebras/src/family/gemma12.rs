//! Gemma 4 12B: units of PY + 1 rows by PX + 1 columns at explicit origins in two unit
//! columns (the first visited down, the second up: unit 0 and the last unit side by side on row
//! 0), their rows in two runs (attention rows `r < RA` with `Hc A` dims and the scores
//! segments, 128 of them so the 1024 window is an exact ring; plain rows with `Hc W` dims), and
//! the head east of them fed by groups of CX rows, the token loop's winner coming back the same
//! way. Blocks: [`RowRuns`], [`HeadGroups`]. Kernels: [`gemma12`].

use crate::abi::Kernels;
use crate::block::Block;
use crate::block::head::{HeadCopies, Hemb};
use crate::block::head_groups::HeadGroups;
use crate::block::row_runs::{RowRuns, RunCopies};
use crate::family::{Family, MadeFor, Row, ceil_to, table};
use crate::floor::{UnitGrid, UnitLayout};
use crate::kernels::gemma12::{self, Dims};
use crate::plan::{
    Contract, FamilyId, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset, ReduceGroups,
    RowRun, Segments, Shape,
};
use crate::program::Program;
use crate::tiles::spec::Spec;
use crate::weights::{Slice, Transform, from};

pub struct Gemma12;

mod cut;

type R<T> = Result<T, String>;

/// The presets: weight columns and rows a unit (PX, PY), the head's rows (HR), the attention rows'
/// and the plain rows' dims (HcA, HcW), the attention rows (RA: the KV rows; 128 make the 1024
/// window an exact ring of 8 x 128), the scores segment width (SWA), the head groups' sizes
/// (attention rows in groups of GA, plain rows in groups of GW, never across the CY row), the
/// units (NU), the global KV entries a KV row (NEG), the reduction groups (G2, G2W).
const PRESETS: &[Row<13>] = &[
    // name, [PX, PY, HR, HcA, HcW, RA, SWA, GA, GW, NU, NEG, G2, G2W], overrides, levers
    // (requests 0920 / 0921: the exact ring, 54 head columns)
    (
        "g12",
        [344, 283, 1153, 12, 15, 128, 43, 6, 5, 8, 32, 16, 24],
        "",
        "DIAG:0",
    ),
    ("tiny", [8, 8, 18, 2, 3, 3, 4, 3, 2, 4, 12, 2, 2], "", ""),
];

/// The model each shipping preset was measured on.
const MADE_FOR: &[MadeFor] = &[("g12", [3840, 48, 15360])];

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
    /// Units in two unit columns (down the first, up the second).
    fn grid(plan: &Plan) -> R<UnitGrid> {
        let p = &plan.partition;
        let n = p.units;
        if n == 0 || !n.is_multiple_of(2) {
            return Err(format!(
                "gemma12: {n} units (two unit columns, unit 0 and the last on row 0)"
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
    fn presets(&self) -> Vec<Preset> {
        let keys = [
            "PX", "PY", "HR", "HcA", "HcW", "RA", "SWA", "GA", "GW", "NU", "NEG", "G2", "G2W",
        ];
        table(keys, PRESETS, &[("LOOP", 64.0)], MADE_FOR)
    }

    /// Units of K = layers / NU (S .. S G) in two unit columns; rows in two runs (the attention
    /// rows `r < RA` with HcA dims and the scores segments, the plain rows with HcW); the sliding
    /// window over the RA KV rows (NES entries a row: exact where RA divides it); the head one
    /// column a row group, VR vocabulary rows a head PE (a multiple of 8). Overrides: `NEG`,
    /// `G2`, `G2W`, `CAP`, `LOOP`.
    fn plan(&self, s: &Shape, pr: &Preset) -> R<Plan> {
        let (px, py, ra) = (pr.need("PX")?, pr.need("PY")?, pr.need("RA")?);
        let (hca, hcw, swa) = (pr.need("HcA")?, pr.need("HcW")?, pr.need("SWA")?);
        let (hr, units, neg) = (pr.need("HR")?, pr.need("NU")?, pr.need("NEG")?);
        let n = s.layers.len() as u32;
        if units == 0 || !n.is_multiple_of(units) {
            return Err(format!("gemma12: {n} layers on {units} units"));
        }
        let window = s
            .layers
            .iter()
            .find_map(|l| match l {
                LayerKind::Sliding { window } => Some(*window),
                _ => None,
            })
            .ok_or("gemma12: no sliding layer")?;
        let groups = groups_split(py, py / 2, ra, pr.need("GA")?, pr.need("GW")?);
        let g2 = pr.need("G2")?;
        let plan = Plan {
            family: FamilyId::Gemma12,
            shape: s.clone(),
            partition: Partition {
                unit_grid: None,
                units,
                layers_per_unit: n / units,
                rows: py,
                cols: px,
                row_dims: vec![
                    RowRun {
                        rows: (0, ra),
                        dims: hca,
                        attention: true,
                    },
                    RowRun {
                        rows: (ra, py),
                        dims: hcw,
                        attention: false,
                    },
                ],
                kv_rows: ra,
                kv_entries: KvEntries {
                    per_row: neg,
                    global: Some(neg),
                    sliding: Some(window.div_ceil(ra)),
                },
                reduce: ReduceGroups {
                    row: g2,
                    col: g2,
                    head: g2,
                    token_loop: pr.need("G2W")?,
                },
                segments: Segments {
                    scores: swa,
                    ..Default::default()
                },
                attn_cols: None,
                head: HeadPartition {
                    cols: groups.len() as u32,
                    rows: hr,
                    vocab_rows: ceil_to(s.vocab, hr, 8),
                    groups: Some(groups),
                },
            },
            contract: Contract {
                positions: neg * ra,
                rope_levels: 4,
                rope_bits: 3,
                cap: pr.int("CAP").unwrap_or(256),
                loop_len: pr.need("LOOP")?,
                token_loop: true,
                logit_scale: 1.0,
            },
            levers: pr.levers.clone(),
        };
        // the kernels' shape of it (q heads a segment, the MLP's columns) must hold
        Dims::of(&plan)?;
        Ok(plan)
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

    fn tiles(&self, plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
        cut::tiles(plan, shapes)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        gemma12::SYMBOLS
    }
}
