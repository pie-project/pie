//! Gemma 4 E2B / E4B: units at explicit origins in two unit columns (the arrangement a unit
//! shape names), the scores segments a half row, the PLE storage above each unit column
//! ([`Ple`]), the shared-KV line from the KV source unit to its sharers ([`SharedKv`]), the head
//! east of the units and the token loop. The sliding / global KV entries (`NES` / `NEG`) and the
//! KV rows (`PYW`: E4B keeps KV on 128 of 160 rows so the 512 window is an exact ring) are in the
//! plan. Kernels: [`gemma`].

use std::collections::BTreeMap;

use crate::spatial::split::abi::Kernels;
use crate::spatial::split::block::Block;
use crate::spatial::split::block::head::{HeadCopies, HeadEast, Hemb};
use crate::spatial::split::block::ple::{Ple, PleColumn};
use crate::spatial::split::block::segments::Scores;
use crate::spatial::split::block::shared_kv::SharedKv;
use crate::spatial::split::block::token_loop::TokenLoop;
use crate::spatial::split::block::units::{UnitCopies, Units};
use crate::spatial::split::emit::meta::Meta;
use crate::spatial::split::family::Family;
use crate::spatial::split::floor::{UnitGrid, UnitLayout};
use crate::spatial::split::kernels::gemma;
use crate::spatial::split::plan::{
    Act, AttnSpec, Contract, FamilyId, HeadPartition, HeadSpec, KvEntries, LayerKind, MlpSpec,
    NormSpec, Partition, Plan, PleSpec, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::split::program::Program;
use crate::spatial::split::weights::{Slice, Transform, from};

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
                    "split gemma: no arrangement for {} x {} units x {}",
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

impl Family for Gemma {
    fn id(&self) -> FamilyId {
        FamilyId::Gemma
    }

    /// From `prep_gemma.py`'s meta: units of K layers (S .. S G, KM's bits the global layers),
    /// units at or past NO the shared-KV units (their layers read unit NO - 1's last layer of the
    /// same kind); the sliding window NES PYW exactly (WIN 0) or WIN with every row a KV row.
    /// `geo` (the arrangement's name) only picks the reduction groups' defaults, as before.
    fn plan(&self, meta: &Meta) -> R<Plan> {
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

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        gemma::SYMBOLS
    }
}
