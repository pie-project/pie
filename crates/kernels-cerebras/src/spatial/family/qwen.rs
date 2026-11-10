//! Qwen3.5 (GDN layers + gated full attention: 9B / 4B / 0.8B) and Qwen3 (dense attention, no
//! output gate: Qwen3-8B): a snake of PX x PY units (at most one gap slot, whose PEs relay), the
//! scores segments on every weight row, GDN's (k, q) multicast and the o-norm segments where
//! there are GDN layers, the head east of the units, the token loop. Kernels: [`qwen`].

use crate::spatial::abi::Kernels;
use crate::spatial::block::Block;
use crate::spatial::block::gdn::Gdn;
use crate::spatial::block::head::{HeadCopies, HeadEast, Hemb};
use crate::spatial::block::segments::{Onorm, Scores};
use crate::spatial::block::token_loop::TokenLoop;
use crate::spatial::block::units::{UnitCopies, Units};
use crate::spatial::family::{Family, units_of};
use crate::spatial::floor::{UnitGrid, UnitLayout};
use crate::spatial::kernels::qwen;
use crate::spatial::plan::{
    Contract, FamilyId, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset, ReduceGroups,
    RowRun, Segments, Shape,
};
use crate::spatial::program::Program;
use crate::spatial::weights::{From, Slice, Transform, from};

type R<T> = Result<T, String>;

pub struct Qwen;

/// The unit slots' gap: none when the units fill the grid; one slot short, the last column's
/// row-1 slot stays out of the snake (the last unit still ends on row 0, the token loop's
/// return), on an even number of unit columns below the turn row.
fn gap(units: u32, (unx, uny): (u32, u32)) -> R<Option<(u32, u32)>> {
    match (unx * uny).checked_sub(units) {
        Some(0) => Ok(None),
        Some(1) if unx % 2 == 0 && uny >= 3 => Ok(Some((unx - 1, 1))),
        _ => Err(format!(
            "qwen: {units} units on {unx} x {uny} slots (at most one gap slot, on an even number of unit columns, below the turn row)"
        )),
    }
}

/// The presets: weight columns and rows a unit (PX, PY), the unit slots (UNX x UNY; the units are
/// the layers over K, one short of the slots leaves a gap slot), layers a unit (K), the head's rows
/// and vocabulary rows a head PE (HR, VR), KV entries a row (NE). A new size adds one; the tiny
/// ones are the simulator's.
const PRESETS: &[(&str, [u32; 8])] = &[
    // name, [PX, PY, UNX, UNY, K, HR, VR, NE]
    ("9b", [256, 256, 2, 4, 4, 1153, 216, 64]),
    ("4b", [256, 256, 2, 4, 4, 1153, 216, 64]),
    ("0.8b", [256, 128, 2, 3, 4, 1153, 216, 128]),
    // Qwen3-8B: 36 layers on 9 units, one gap slot
    ("q3-8b", [256, 228, 2, 5, 4, 1153, 132, 17]),
    ("tiny-t08", [8, 8, 2, 3, 4, 10, 8, 4]),
    ("tiny-t4", [16, 16, 2, 2, 4, 18, 8, 4]),
    ("tiny-q3", [16, 8, 2, 2, 3, 10, 8, 4]),
    ("tiny-q3-gap", [16, 8, 2, 3, 4, 10, 8, 4]),
];

impl Family for Qwen {
    fn presets(&self) -> Vec<Preset> {
        PRESETS
            .iter()
            .map(|(name, g)| {
                let keys = ["PX", "PY", "UNX", "UNY", "K", "HR", "VR", "NE"];
                let kv: Vec<(&str, f64)> = keys
                    .iter()
                    .zip(g)
                    .map(|(k, v)| (*k, f64::from(*v)))
                    .collect();
                Preset::new(name, &kv)
            })
            .collect()
    }

    /// Units of K layers, every unit the same kinds (GDN and gated attention, or Qwen3's plain
    /// attention); a row's dims the hidden over PY (zero-padded past it); a scores segment a KV
    /// head's columns, GDN's a key head's, the o-norm's a value head's (Qwen3: the scores'); the
    /// head PY / 2 columns; the RoPE tables enough levels for the context. Overrides: the
    /// reduction groups (`G2`, `G2X` / `G2Y` / `G2H`, `G2W`), the RoPE tables' levels and digit
    /// bits (`RLV`, `RBITS`), `CAP`, `LOOP`, the token loop (`WL`).
    fn plan(&self, s: &Shape, pr: &Preset) -> R<Plan> {
        let (px, py, k) = (pr.need("PX")?, pr.need("PY")?, pr.need("K")?);
        let unit_grid = (pr.need("UNX")?, pr.need("UNY")?);
        let units = units_of(&s.layers, k, "qwen")?;
        gap(units, unit_grid)?;
        let pattern = &s.layers[..k as usize];
        if s.layers.chunks(k as usize).any(|u| u != pattern) {
            return Err("qwen: every unit the same layer kinds".into());
        }
        let per = |heads: u32, what: &str| {
            if heads == 0 || !px.is_multiple_of(heads) {
                return Err(format!("qwen: {heads} {what} over {px} columns"));
            }
            Ok(px / heads)
        };
        let scores = per(s.attn.kv_heads, "KV heads")?;
        let (gdn, onorm) = match s.gdn {
            Some(g) => (
                per(g.key_heads, "key heads")?,
                per(g.value_heads, "value heads")?,
            ),
            None => (scores, scores),
        };
        let g2 = pr.int("G2").unwrap_or(0);
        let ne = pr.need("NE")?;
        // a position is RLV digits of RBITS bits: enough levels for the context NE PY (at least 4)
        let rope_bits = pr.int("RBITS").unwrap_or(3);
        let rope_levels = pr
            .int("RLV")
            .unwrap_or(4.max((32 - (ne * py - 1).leading_zeros()).div_ceil(rope_bits)));
        let partition = Partition {
            units,
            layers_per_unit: k,
            unit_grid: Some(unit_grid),
            rows: py,
            cols: px,
            row_dims: vec![RowRun {
                rows: (0, py),
                dims: s.hidden.div_ceil(py),
                attention: pattern.iter().any(|l| !matches!(l, LayerKind::Gdn)),
            }],
            kv_rows: py,
            kv_entries: KvEntries {
                per_row: ne,
                ..Default::default()
            },
            reduce: ReduceGroups {
                row: pr.int("G2X").unwrap_or(g2),
                col: pr.int("G2Y").unwrap_or(g2),
                head: pr.int("G2H").unwrap_or(g2),
                token_loop: pr.int("G2W").unwrap_or(24),
            },
            segments: Segments {
                scores,
                gdn,
                onorm,
                ..Default::default()
            },
            attn_cols: None,
            head: HeadPartition {
                cols: py / 2,
                rows: pr.need("HR")?,
                vocab_rows: pr.need("VR")?,
                ..Default::default()
            },
        };
        let contract = Contract {
            // bounded by the CY's RoPE tables: RLV digits of RBITS bits
            positions: (ne * py).min(1 << (rope_bits * rope_levels).min(31)),
            rope_levels,
            rope_bits,
            cap: pr.int("CAP").unwrap_or(256),
            loop_len: pr.int("LOOP").unwrap_or(32),
            token_loop: pr.int("WL").unwrap_or(0) != 0,
            logit_scale: 1.0,
        };
        Ok(Plan {
            family: FamilyId::Qwen,
            shape: s.clone(),
            partition,
            contract,
        })
    }

    /// `qwen/prep_qwen.py`'s cut: PX, PY, UNX, UNY, K, HR, VR, NE, the units (NU) and the RoPE tables'
    /// levels and digit bits (RLV, RBITS).
    fn tiling(&self, plan: &Plan) -> R<serde_json::Value> {
        let p = &plan.partition;
        let (unx, uny) = p.unit_grid.ok_or("qwen: no unit grid")?;
        Ok(serde_json::json!({
            "PX": p.cols, "PY": p.rows, "UNX": unx, "UNY": uny, "NU": p.units,
            "K": p.layers_per_unit, "HR": p.head.rows, "VR": p.head.vocab_rows,
            "NE": p.kv_entries.per_row, "RLV": plan.contract.rope_levels,
            "RBITS": plan.contract.rope_bits,
        }))
    }

    fn blocks(&self, plan: &Plan) -> R<Vec<Box<dyn Block>>> {
        let (s, p) = (&plan.shape, &plan.partition);
        let ug = p
            .unit_grid
            .ok_or("qwen: the plan has no unit grid (UNX, UNY)")?;
        let (cols, rows) = (p.cols, p.rows);
        for (n, sw) in [
            ("SWA", p.segments.scores),
            ("SWG", p.segments.gdn),
            ("SWV", p.segments.onorm),
        ] {
            if sw == 0 || (cols / 2) % sw != 0 {
                return Err(format!(
                    "qwen: {n} {sw} does not divide XC {} (a segment would span the control column)",
                    cols / 2
                ));
            }
        }
        if p.head.cols * 2 != rows {
            return Err("qwen: head columns = CX rows / 2".into());
        }
        // Qwen3-8B's hidden zero-padded to the row split
        let placed = p.hidden_placed();
        let pad = |t: &str| -> From {
            let tr = if placed != s.hidden {
                vec![Transform::PadZero { dim: 1, to: placed }]
            } else {
                vec![]
            };
            vec![(t.to_string(), tr)]
        };
        let mut out: Vec<Box<dyn Block>> = vec![Box::new(Units {
            grid: UnitGrid {
                cols,
                rows,
                cx: cols / 2,
                cy: rows / 2,
                layout: UnitLayout::Snake {
                    unx: ug.0,
                    uny: ug.1,
                    up_first: false,
                    gaps: gap(p.units, ug)?.into_iter().collect(),
                },
                layers_per_unit: p.layers_per_unit,
            },
            copies: UnitCopies {
                cols,
                rows,
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
            },
        })];
        if s.layers.iter().any(|k| !matches!(k, LayerKind::Gdn)) {
            out.push(Box::new(Scores {
                width: p.segments.scores,
            }));
        }
        if let Some(g) = Gdn::of(plan) {
            out.push(Box::new(g));
            out.push(Box::new(Onorm {
                width: p.segments.onorm,
            }));
        }
        let token_loop = plan.contract.token_loop;
        out.push(Box::new(HeadEast {
            cols: p.head.cols,
            rows: p.head.rows,
            token_loop,
            copies: HeadCopies {
                cols: p.head.cols,
                rows: p.head.rows,
                hw: |_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
                hw_from: vec![(
                    "lm_head".into(),
                    vec![Transform::PadLastRow {
                        to: p.head.rows * p.head.vocab_rows,
                    }],
                )],
                hg_from: from("norm"),
                hemb: if token_loop {
                    Hemb::After(from(&format!("embed_tokens ({} rows)", s.vocab)))
                } else {
                    Hemb::None
                },
            },
        }));
        if token_loop {
            out.push(Box::new(TokenLoop));
        }
        Ok(out)
    }

    fn kernels(&self, prog: &Program) -> R<Kernels> {
        qwen::kernels(prog)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        qwen::SYMBOLS
    }
}
