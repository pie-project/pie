//! Gemma 4 E2B / E4B: units at explicit origins in two unit columns (the arrangement a unit
//! shape names), the scores segments a half row, the PLE storage above each unit column
//! ([`Ple`]), the shared-KV line from the KV source unit to its sharers ([`SharedKv`]), the head
//! east of the units and the token loop. The sliding / global KV entries (`NES` / `NEG`) and the
//! KV rows (`PYW`: E4B keeps KV on 128 of 160 rows so the 512 window is an exact ring) are in the
//! plan. Kernels: [`gemma`].

use crate::tiles::spec::Spec;
use std::collections::BTreeMap;

use crate::abi::Kernels;
use crate::block::Block;
use crate::block::head::{HeadCopies, HeadEast, Hemb};
use crate::block::ple::{Ple, PleColumn};
use crate::block::segments::Scores;
use crate::block::shared_kv::SharedKv;
use crate::block::token_loop::TokenLoop;
use crate::block::units::{UnitCopies, Units};
use crate::family::{Family, MadeFor, Row, ceil_to, table};
use crate::floor::{UnitGrid, UnitLayout};
use crate::kernels::gemma;
use crate::plan::{
    Contract, FamilyId, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset, ReduceGroups,
    RowRun, Segments, Shape,
};
use crate::program::Program;
use crate::weights::{Slice, Transform, from};

mod cut;

type R<T> = Result<T, String>;

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
    fn of(plan: &Plan) -> R<&'static Arrangement> {
        let p = &plan.partition;
        let key = (p.cols, p.rows, p.units);
        ARRANGEMENTS
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, a)| a)
            .ok_or_else(|| {
                format!(
                    "gemma: no arrangement for {} x {} units x {}",
                    p.cols, p.rows, p.units
                )
            })
    }

    /// Unit origins in unit order, for units of `uw` x `uh` PEs.
    fn origins(&self, uw: u32, uh: u32) -> R<Vec<(u32, u32)>> {
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
                .ok_or_else(|| format!("gemma: unit {u} level with {level}, not in A"))?;
            o[u as usize] = Some((uw, y));
        }
        o.into_iter()
            .enumerate()
            .map(|(u, v)| v.ok_or_else(|| format!("gemma: unit {u} unplaced")))
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

/// The presets: the unit shape that picks the arrangement (weight columns, rows, units), the head's
/// rows, the global KV entries a KV row (NEG; the context is NEG x the KV rows), the reduction
/// groups and the posted loop.
const PRESETS: &[Row<7>] = &[
    // name, [PX, PY, NU, HR, NEG, G2, G2W], overrides, levers
    ("e2b", [256, 128, 7, 1130, 32, 16, 24], "", "WSPEC:2"),
    // E4B as on the chip (request 0974): PYW 128, NES 4, NEG 20
    (
        "e4b",
        [256, 160, 7, 1130, 20, 16, 24],
        "",
        "DIAG:0,WSPEC:0,w.FIFOW:16",
    ),
    // E4B at its first staged params (PYW 160, NEG 16)
    (
        "e4b-meta",
        [256, 160, 7, 1130, 20, 16, 24],
        "PYW=160,NEG=16",
        "w.FIFOW:16",
    ),
    ("tiny", [8, 8, 5, 32, 4, 2, 2], "", ""),
];

/// The model each shipping preset was measured on.
const MADE_FOR: &[MadeFor] = &[("e2b", [1536, 35, 6144]), ("e4b", [2560, 42, 10240])];

impl Family for Gemma {
    fn presets(&self) -> Vec<Preset> {
        table(
            ["PX", "PY", "NU", "HR", "NEG", "G2", "G2W"],
            PRESETS,
            &[("LOOP", 64.0)],
            MADE_FOR,
        )
    }

    /// Units of K = layers / NU (S .. S G); a row's dims the hidden over PY; the sliding window
    /// over the KV rows (`PYW`: the window over the fewest sliding entries a row, NES, where that
    /// divides it, so the ring is exact; else every row, and the kernel masks the older entries);
    /// the head PY / 2 columns of VR vocabulary rows (a multiple of 8). Overrides: `PYW`, `NEG`,
    /// `G2`, `G2W`, `CAP`, `LOOP`.
    fn plan(&self, s: &Shape, pr: &Preset) -> R<Plan> {
        let (px, py, nu) = (pr.need("PX")?, pr.need("PY")?, pr.need("NU")?);
        let n = s.layers.len() as u32;
        if nu == 0 || !n.is_multiple_of(nu) {
            return Err(format!("gemma: {n} layers on {nu} units"));
        }
        let k = n / nu;
        if !s.hidden.is_multiple_of(py) {
            return Err(format!("gemma: {} hidden over {py} rows", s.hidden));
        }
        let window = s
            .layers
            .iter()
            .find_map(|l| match l {
                LayerKind::Sliding { window } => Some(*window),
                _ => None,
            })
            .ok_or("gemma: no sliding layer")?;
        let nes = window.div_ceil(py);
        let pyw = pr.int("PYW").unwrap_or(if window.is_multiple_of(nes) {
            window / nes
        } else {
            py
        });
        let nes = window.div_ceil(pyw);
        let neg = pr.need("NEG")?;
        let hr = pr.need("HR")?;
        let partition = Partition {
            unit_grid: None,
            units: nu,
            layers_per_unit: k,
            rows: py,
            cols: px,
            row_dims: vec![RowRun {
                rows: (0, py),
                dims: s.hidden / py,
                attention: true,
            }],
            kv_rows: pyw,
            kv_entries: KvEntries {
                per_row: neg,
                global: Some(neg),
                sliding: Some(nes),
            },
            reduce: ReduceGroups {
                row: pr.need("G2")?,
                col: pr.need("G2")?,
                head: pr.need("G2")?,
                token_loop: pr.need("G2W")?,
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
                cols: py / 2,
                rows: hr,
                vocab_rows: ceil_to(s.vocab, hr, 8),
                ..Default::default()
            },
        };
        let contract = Contract {
            positions: neg * pyw,
            rope_levels: 4,
            rope_bits: 3,
            cap: pr.int("CAP").unwrap_or(256),
            loop_len: pr.need("LOOP")?,
            token_loop: true,
            logit_scale: 1.0,
        };
        Ok(Plan {
            family: FamilyId::Gemma,
            shape: s.clone(),
            partition,
            contract,
            levers: pr.levers.clone(),
        })
    }

    /// The arrangement's units, the scores segments a half row, the PLE storage of each unit
    /// column, the shared-KV line, the head east of the units (fed the embedding inline), the
    /// token loop.
    fn blocks(&self, plan: &Plan) -> R<Vec<Box<dyn Block>>> {
        let arr = Arrangement::of(plan)?;
        let p = &plan.partition;
        let (px, py) = (p.cols, p.rows);
        let (hr, vr) = (p.head.rows, p.head.vocab_rows);
        Ok(vec![
            Box::new(Units {
                grid: UnitGrid {
                    cols: px,
                    rows: py,
                    cx: px / 2,
                    cy: py / 2,
                    layout: UnitLayout::Explicit {
                        origins: arr.origins(px + 1, py + 1)?,
                    },
                    layers_per_unit: p.layers_per_unit,
                },
                copies: UnitCopies {
                    cols: px,
                    rows: py,
                    w_from: Vec::new(),
                    gam_b16: false,
                    gam_from: Vec::new(),
                    cy: &[("qkw", true), ("rot", false)],
                    cy_optional: false,
                },
            }),
            Box::new(Scores {
                width: p.segments.scores,
            }),
            Box::new(Ple {
                columns: arr
                    .columns(px + 1)
                    .into_iter()
                    .map(|(_, units, st)| PleColumn {
                        units,
                        rows: st.rows,
                        tokens: st.tokens,
                        bus: st.bus,
                    })
                    .collect(),
            }),
            Box::new(SharedKv),
            Box::new(HeadEast {
                cols: p.head.cols,
                rows: hr,
                token_loop: true,
                copies: HeadCopies {
                    cols: p.head.cols,
                    rows: hr,
                    hw: |_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
                    hw_from: vec![(
                        "embed_tokens".into(),
                        vec![Transform::PadLastRow { to: hr * vr }],
                    )],
                    hg_from: from("norm"),
                    hemb: Hemb::Inline(Vec::new()),
                },
            }),
            Box::new(TokenLoop),
        ])
    }

    fn kernels(&self, prog: &Program) -> R<Kernels> {
        gemma::kernels(prog)
    }

    fn tiles(&self, plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
        cut::tiles(plan, shapes)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        gemma::SYMBOLS
    }
}
