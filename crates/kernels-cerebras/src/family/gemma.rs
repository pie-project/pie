//! Gemma 4 E2B / E4B (composed by [`crate::compose`]): units at explicit origins in two unit columns (the arrangement a unit
//! shape names), the scores segments a half row, the PLE storage above each unit column
//! ([`crate::block::ple::Ple`]), the shared-KV line from the KV source unit to its sharers ([`crate::block::shared_kv::SharedKv`]), the head
//! east of the units and the token loop. The sliding / global KV entries (`NES` / `NEG`) and the
//! KV rows (`PYW`: E4B keeps KV on 128 of 160 rows so the 512 window is an exact ring) are in the
//! plan. Kernels: [`gemma`].

use crate::tiles::spec::Spec;

use crate::abi::Kernels;
use crate::block::head::{HeadCopies, Hemb};
use crate::block::units::UnitCopies;
use crate::compose::KernelCopies;
use crate::family::{Family, MadeFor, Row, ceil_to, table};
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
    fn copies(&self, plan: &Plan) -> R<KernelCopies> {
        let p = &plan.partition;
        let hr = p.head.rows;
        Ok(KernelCopies {
            units: Some(UnitCopies {
                cols: p.cols,
                rows: p.rows,
                w_from: Vec::new(),
                gam_b16: false,
                gam_from: Vec::new(),
                cy: &[("qkw", true), ("rot", false)],
                cy_optional: false,
            }),
            head: Some(HeadCopies {
                cols: p.head.cols,
                rows: hr,
                hw: |_, rows, cols| Slice::of("hw", None, vec![rows, cols]),
                hw_from: vec![(
                    "embed_tokens".into(),
                    vec![Transform::PadLastRow {
                        to: hr * p.head.vocab_rows,
                    }],
                )],
                hg_from: from("norm"),
                hemb: Hemb::Inline(Vec::new()),
            }),
            ..Default::default()
        })
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
