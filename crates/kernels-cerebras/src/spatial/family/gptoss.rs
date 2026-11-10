//! gpt-oss: units of AX attention + MX MoE columns (CX after the attention columns), PY rows, a
//! snake of two unit columns that starts upward (so the first and the last unit sit on the
//! band), the scores segments on the attention columns, the router's on the CY row's MoE
//! columns, the head as a band below the units with its control row, the token loop over the
//! band. Blocks: [`MoeUnits`], [`AttnScores`], [`Router`], [`Band`], [`BandLoop`]; kernels:
//! [`gptoss`].

use crate::spatial::abi::Kernels;
use crate::spatial::block::Block;
use crate::spatial::block::band::{Band, BandLoop};
use crate::spatial::block::moe::{AttnScores, MoeCopies, MoeUnits, Router};
use crate::spatial::family::{Family, MadeFor, Row, ceil_to, table};
use crate::spatial::floor::{UnitGrid, UnitLayout};
use crate::spatial::kernels::gptoss::{self, Dims};
use crate::spatial::plan::{
    Contract, FamilyId, HeadPartition, KvEntries, Partition, Plan, Preset, ReduceGroups, RowRun,
    Segments, Shape,
};
use crate::spatial::program::Program;
use crate::spatial::tiles::spec::Spec;
use crate::spatial::weights::Transform;

mod cut;

type R<T> = Result<T, String>;

pub struct GptOss;

/// The presets: attention and MoE columns a unit (AX, MX), rows (PY), the unit slots (UNX x UNY,
/// every slot a unit), the scores and router segments (SWA, SWR), the band's vocabulary columns
/// (HVC), KV entries a row (NE), the RoPE tables' levels (NLV), the reduction groups (G2, G2W);
/// the logits' prescale (HSCALE: the band head's 2^15) and the posted loop (LOOP).
const PRESETS: &[Row<11>] = &[
    // name, [AX, MX, PY, UNX, UNY, SWA, SWR, HVC, NE, G2, G2W], overrides, levers
    // (request 0887: CYV 2, GLUV 1)
    (
        "full",
        [64, 288, 160, 2, 6, 8, 32, 748, 128, 16, 24],
        "",
        "DIAG:0,GLUV:1,CYV:2",
    ),
    ("tiny", [4, 8, 4, 2, 1, 2, 8, 26, 8, 2, 2], "", ""),
];

/// The model each shipping preset was measured on.
const MADE_FOR: &[MadeFor] = &[("full", [2880, 24, 2880])];

impl Family for GptOss {
    fn presets(&self) -> Vec<Preset> {
        let keys = [
            "AX", "MX", "PY", "UNX", "UNY", "SWA", "SWR", "HVC", "NE", "G2", "G2W",
        ];
        table(
            keys,
            PRESETS,
            &[("NLV", 5.0), ("HSCALE", 32768.0), ("LOOP", 64.0)],
            MADE_FOR,
        )
    }

    /// Units of K = layers / units, a row's dims the hidden over PY; a scores segment a KV head's
    /// attention columns; the band head HVC columns of VR vocabulary rows (a multiple of 8, the
    /// head control's argmax). Overrides: `NE`, `G2`, `G2W`, `CAP`, `LOOP`, `WL` (the token loop;
    /// on unless 0).
    fn plan(&self, s: &Shape, pr: &Preset) -> R<Plan> {
        let (ax, mx, py) = (pr.need("AX")?, pr.need("MX")?, pr.need("PY")?);
        let (unx, uny) = (pr.need("UNX")?, pr.need("UNY")?);
        let (swa, hvc) = (pr.need("SWA")?, pr.need("HVC")?);
        let units = unx * uny;
        let n = s.layers.len() as u32;
        if units == 0 || !n.is_multiple_of(units) {
            return Err(format!("gpt-oss: {n} layers on {units} units"));
        }
        let k = n / units;
        if s.layers
            .chunks(k as usize)
            .any(|u| u != &s.layers[..k as usize])
        {
            return Err("gpt-oss: every unit the same layer kinds".into());
        }
        if !ax.is_multiple_of(swa) || ax / swa != s.attn.kv_heads {
            return Err(format!(
                "gpt-oss: {ax} attention columns in segments of {swa} for {} KV heads",
                s.attn.kv_heads
            ));
        }
        if !s.hidden.is_multiple_of(py) {
            return Err(format!("gpt-oss: {} hidden over {py} rows", s.hidden));
        }
        let ne = pr.need("NE")?;
        let partition = Partition {
            unit_grid: Some((unx, uny)),
            units,
            layers_per_unit: k,
            rows: py,
            cols: ax + mx,
            row_dims: vec![RowRun {
                rows: (0, py),
                dims: s.hidden / py,
                attention: true,
            }],
            kv_rows: py,
            kv_entries: KvEntries {
                per_row: ne,
                ..Default::default()
            },
            reduce: ReduceGroups {
                row: pr.need("G2")?,
                col: pr.need("G2")?,
                head: pr.need("G2")?,
                token_loop: pr.need("G2W")?,
            },
            segments: Segments {
                scores: swa,
                router: Some(pr.need("SWR")?),
                ..Default::default()
            },
            attn_cols: Some(ax),
            head: HeadPartition {
                cols: hvc,
                rows: py,
                vocab_rows: ceil_to(s.vocab, hvc, 8),
                ..Default::default()
            },
        };
        let contract = Contract {
            positions: ne * py,
            rope_levels: pr.need("NLV")?,
            rope_bits: 3,
            cap: pr.int("CAP").unwrap_or(256),
            loop_len: pr.need("LOOP")?,
            token_loop: pr.int("WL").is_none_or(|v| v != 0),
            logit_scale: pr.float("HSCALE").unwrap_or(1.0) as f32,
        };
        Ok(Plan {
            family: FamilyId::GptOss,
            shape: s.clone(),
            partition,
            contract,
            levers: pr.levers.clone(),
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

    fn tiles(&self, plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
        cut::tiles(plan, shapes)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        gptoss::SYMBOLS
    }
}
