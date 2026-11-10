//! Qwen3.5 (GDN layers + gated full attention: 9B / 4B / 0.8B) and Qwen3 (dense attention, no
//! output gate: Qwen3-8B): a snake of PX x PY units (at most one gap slot, whose PEs relay), the
//! scores segments on every weight row, GDN's (k, q) multicast and the o-norm segments where
//! there are GDN layers, the head east of the units, the token loop. Kernels: [`q35`].

use crate::spatial::abi::Kernels;
use crate::spatial::block::Block;
use crate::spatial::block::gdn::Gdn;
use crate::spatial::block::head::{HeadCopies, HeadEast, Hemb};
use crate::spatial::block::segments::{Onorm, Scores};
use crate::spatial::block::token_loop::TokenLoop;
use crate::spatial::block::units::{UnitCopies, Units};
use crate::spatial::emit::meta::Meta;
use crate::spatial::family::Family;
use crate::spatial::floor::{UnitGrid, UnitLayout};
use crate::spatial::kernels::q35;
use crate::spatial::plan::{
    Act, AttnSpec, Contract, FamilyId, GdnSpec, HeadPartition, HeadSpec, KvEntries, LayerKind,
    MlpSpec, NormSpec, Partition, Plan, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
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

/// The named geometries (`prep_q35.py <model> <out> <golden> <name> [prefix]`): weight columns
/// and rows a unit (PX, PY), the unit slots (UNX x UNY) and units (NU; one short: a gap slot),
/// layers a unit (K), the head's rows and vocabulary rows a head PE (HR, VR), KV entries a row
/// (NE). A new size adds one.
const GEOMETRIES: &[(&str, [u32; 9])] = &[
    // name, [PX, PY, UNX, UNY, NU, K, HR, VR, NE]
    ("9b", [256, 256, 2, 4, 8, 4, 1153, 216, 64]),
    ("4b", [256, 256, 2, 4, 8, 4, 1153, 216, 64]),
    ("0.8b", [256, 128, 2, 3, 6, 4, 1153, 216, 128]),
    // Qwen3-8B: 36 layers on 9 units, one gap slot
    ("q3-8b", [256, 228, 2, 5, 9, 4, 1153, 132, 17]),
];

impl Family for Qwen {
    fn geometry(&self, name: &str) -> R<serde_json::Value> {
        let [px, py, unx, uny, nu, k, hr, vr, ne] = GEOMETRIES
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, g)| *g)
            .ok_or_else(|| format!("qwen: no geometry {name}"))?;
        gap(nu, (unx, uny))?;
        Ok(serde_json::json!({
            "PX": px, "PY": py, "UNX": unx, "UNY": uny, "NU": nu, "K": k, "HR": hr, "VR": vr,
            "NE": ne,
        }))
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
        let pad = |t: &str| -> From {
            let tr = if s.hidden_placed != s.hidden {
                vec![Transform::PadZero {
                    dim: 1,
                    to: s.hidden_placed,
                }]
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
        q35::kernels(prog)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        q35::SYMBOLS
    }
}
