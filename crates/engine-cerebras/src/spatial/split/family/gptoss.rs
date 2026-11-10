//! gpt-oss: units of AX attention + MX MoE columns (CX after the attention columns), PY rows, a
//! snake of two unit columns that starts upward (so the first and the last unit sit on the
//! band), the scores segments on the attention columns, the router's on the CY row's MoE
//! columns, the head as a band below the units with its control row, the token loop over the
//! band. Blocks: [`MoeUnits`], [`AttnScores`], [`Router`], [`Band`], [`BandLoop`]; kernels:
//! [`gptoss`].

use crate::spatial::split::abi::Kernels;
use crate::spatial::split::block::Block;
use crate::spatial::split::block::band::{Band, BandLoop};
use crate::spatial::split::block::moe::{AttnScores, MoeCopies, MoeUnits, Router};
use crate::spatial::split::emit::meta::Meta;
use crate::spatial::split::family::Family;
use crate::spatial::split::floor::{UnitGrid, UnitLayout};
use crate::spatial::split::kernels::gptoss::{self, Dims};
use crate::spatial::split::plan::{
    Act, AttnSpec, Contract, FamilyId, HeadPartition, HeadSpec, KvEntries, LayerKind, MlpSpec,
    MoeSpec, NormSpec, Partition, Plan, ReduceGroups, RopeSpec, RowRun, Segments, Shape,
};
use crate::spatial::split::program::Program;
use crate::spatial::split::weights::Transform;

type R<T> = Result<T, String>;

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

impl Family for GptOss {
    fn id(&self) -> FamilyId {
        FamilyId::GptOss
    }

    /// The plan from a `prep_gptoss.py` meta: the geometry name's table (`geo`), the checkpoint
    /// dims the prep wrote (`HD`, `K`, `KM`, `Hc`, `Qh`, `Dc`, `Ic`, `NEX`, `TK`, `HR`, `NLV`, `VR`,
    /// `V`, `WIN`), the build's `G2` / `G2W` / `NE` / `CAP` / `LOOP` / `WL`.
    fn plan(&self, meta: &Meta) -> R<Plan> {
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

    fn blocks(&self, plan: &Plan) -> R<Vec<Box<dyn Block>>> {
        let (p, d) = (&plan.partition, Dims::of(plan)?);
        let ax = p.attn_cols.ok_or("gpt-oss plan: no attention columns")?;
        if ax >= p.cols {
            return Err("gpt-oss plan: no MoE columns".into());
        }
        if !p.units.is_multiple_of(2) {
            return Err("gpt-oss: an even unit count (the last unit next to the band)".into());
        }
        let k = d.k;
        let lwa = d.hcc * d.nqa + d.qo * d.hcc;
        let mut out: Vec<Box<dyn Block>> = vec![
            Box::new(MoeUnits {
                grid: UnitGrid {
                    cols: p.cols,
                    rows: p.rows,
                    cx: ax,
                    cy: p.rows / 2,
                    layout: UnitLayout::Snake {
                        unx: 2,
                        uny: p.units / 2,
                        up_first: true,
                        gaps: Vec::new(),
                    },
                    layers_per_unit: k,
                },
                copies: MoeCopies {
                    w: k * lwa,
                    wx: k * d.nex * d.xe,
                    wr: k * d.hcc,
                    cx: [2 * k * d.hcc, k * d.hcc, k * d.nex * d.hcc],
                    cya: [k * d.nqa, k * d.qh, d.nlv * 8 * 2 * d.nrot],
                    cym: [k, k * d.nex * 2 * d.ic, k * d.nex * 2],
                    wx_from: vec![(
                        "model.layers.*.mlp.experts.{gate_up,down}_proj_blocks".to_string(),
                        vec![Transform::Mxfp4ToE5m2],
                    )],
                },
            }),
            Box::new(AttnScores {
                width: p.segments.scores,
            }),
            Box::new(Router {
                width: p
                    .segments
                    .router
                    .ok_or("gpt-oss plan: no router segments")?,
            }),
            Box::new(Band {
                cols: p.head.cols,
                rows: p.head.rows,
                vocab_rows: p.head.vocab_rows,
            }),
        ];
        if plan.contract.token_loop {
            out.push(Box::new(BandLoop));
        }
        Ok(out)
    }

    fn kernels(&self, prog: &Program) -> R<Kernels> {
        gptoss::kernels(prog)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        gptoss::SYMBOLS
    }
}
