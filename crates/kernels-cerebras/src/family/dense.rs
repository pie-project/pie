//! Qwen3.5 (GDN layers + gated full attention: 9B / 4B / 0.8B), Qwen3 and Llama 3 (dense
//! attention, no output gate: Qwen3-8B, Llama 3.1 8B): a snake of PX x PY units (at most one gap slot, whose PEs relay), the
//! scores segments on every weight row, GDN's (k, q) multicast and the o-norm segments where
//! there are GDN layers, the head east of the units, the token loop. Kernels: [`dense`].

use crate::abi::Kernels;
use crate::compose::snake_gap;
use crate::family::{Family, MadeFor, Row, table, units_of};
use crate::kernels::dense;
use crate::plan::{
    Contract, FamilyId, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset, ReduceGroups,
    RowRun, Segments, Shape,
};
use crate::program::Program;
use crate::tiles::spec::Spec;

mod cut;

type R<T> = Result<T, String>;

pub struct Dense;

/// The presets: weight columns and rows a unit (PX, PY), the unit slots (UNX x UNY; the units are
/// the layers over K, one short of the slots leaves a gap slot), layers a unit (K), the head's rows
/// and vocabulary rows a head PE (HR, VR), KV entries a row (NE). A new size adds one; the tiny
/// ones are the simulator's.
/// The shipping levers of the unit snake's kernels (requests 0900 .. 0905).
const SHIP: &str = "NPX:1,DIAG:0,TSTAMP:0,XKEEP:0,HACC16:0,ATTL:2";

const PRESETS: &[Row<8>] = &[
    // name, [PX, PY, UNX, UNY, K, HR, VR, NE], overrides, levers
    ("9b", [256, 256, 2, 4, 4, 1153, 216, 64], "G2=16,WL=1", SHIP),
    ("4b", [256, 256, 2, 4, 4, 1153, 216, 64], "G2=16,WL=1", SHIP),
    (
        "0.8b",
        [256, 128, 2, 3, 4, 1153, 216, 128],
        "G2=16,G2Y=8,WL=1",
        SHIP,
    ),
    // Qwen3-8B: 36 layers on 9 units, one gap slot (requests 0961 / 0963)
    (
        "q3-8b",
        [256, 228, 2, 5, 4, 1153, 132, 17],
        "G2=12,WL=1",
        "NPX:1,DIAG:0,TSTAMP:0,XKEEP:0,HACC16:0,ATTL:2,SLW:1",
    ),
    // Llama 3.1 8B: the 9B's units; every layer attends, so 32 KV entries a row (8,192
    // positions) keep the weight PEs' stack room (3,104 B; 64 entries do not link)
    (
        "l3-8b",
        [256, 256, 2, 4, 4, 1153, 216, 32],
        "G2=16,WL=1",
        SHIP,
    ),
    ("tiny-t08", [8, 8, 2, 3, 4, 10, 8, 4], "G2=2,WL=1,G2W=2", ""),
    (
        "tiny-t4",
        [16, 16, 2, 2, 4, 18, 8, 4],
        "G2=2,WL=1,G2W=2",
        "",
    ),
    ("tiny-q3", [16, 8, 2, 2, 3, 10, 8, 4], "G2=2,WL=1,G2W=2", ""),
    (
        "tiny-q3-gap",
        [16, 8, 2, 3, 4, 10, 8, 4],
        "G2=2,WL=1,G2W=2",
        "",
    ),
];

/// The model each shipping preset was measured on.
const MADE_FOR: &[MadeFor] = &[
    ("9b", [4096, 32, 12288]),
    ("4b", [2560, 32, 9216]),
    ("0.8b", [1024, 24, 3584]),
    ("q3-8b", [4096, 36, 12288]),
    ("l3-8b", [4096, 32, 14336]),
];

impl Family for Dense {
    fn presets(&self) -> Vec<Preset> {
        table(
            ["PX", "PY", "UNX", "UNY", "K", "HR", "VR", "NE"],
            PRESETS,
            &[],
            MADE_FOR,
        )
    }

    /// Units of K layers, every unit the same kinds (GDN and gated attention, or Qwen3's plain
    /// attention); a row's dims the hidden over PY (zero-padded past it); a scores segment a KV
    /// head's columns, GDN's a key head's, the o-norm's a value head's (Qwen3: the scores'); the
    /// head PY / 2 columns; the RoPE tables enough levels for the context (`CTX`, the positions
    /// asked for, where fewer than NE PY). Overrides: the
    /// reduction groups (`G2`, `G2X` / `G2Y` / `G2H`, `G2W`), the RoPE tables' levels and digit
    /// bits (`RLV`, `RBITS`), `CAP`, `LOOP`, the token loop (`WL`).
    fn plan(&self, s: &Shape, pr: &Preset) -> R<Plan> {
        let (px, py, k) = (pr.need("PX")?, pr.need("PY")?, pr.need("K")?);
        let unit_grid = (pr.need("UNX")?, pr.need("UNY")?);
        let units = units_of(&s.layers, k, "dense")?;
        snake_gap(units, unit_grid)?;
        let pattern = &s.layers[..k as usize];
        if s.layers.chunks(k as usize).any(|u| u != pattern) {
            return Err("dense: every unit the same layer kinds".into());
        }
        let per = |heads: u32, what: &str| {
            if heads == 0 || !px.is_multiple_of(heads) {
                return Err(format!("dense: {heads} {what} over {px} columns"));
            }
            Ok(px / heads)
        };
        // a head's dims split evenly over its columns
        let split = |dims: u32, cols: u32, what: &str| {
            if dims.is_multiple_of(cols) {
                Ok(())
            } else {
                Err(format!("dense: {what}'s {dims} dims over {cols} columns"))
            }
        };
        let scores = per(s.attn.kv_heads, "KV heads")?;
        split(s.attn.head_dim, scores, "an attention head")?;
        let (gdn, onorm) = match s.gdn {
            Some(g) => {
                let (gdn, onorm) = (
                    per(g.key_heads, "key heads")?,
                    per(g.value_heads, "value heads")?,
                );
                split(g.key_dim, gdn, "a key head")?;
                split(g.value_dim, onorm, "a value head")?;
                (gdn, onorm)
            }
            None => (scores, scores),
        };
        // the MLP's columns split evenly over the unit's (the cut has no padded column)
        if !s.mlp.intermediate.is_multiple_of(px) {
            return Err(format!(
                "dense: the MLP's {} over {px} columns",
                s.mlp.intermediate
            ));
        }
        let g2 = pr.int("G2").unwrap_or(0);
        let ne = pr.need("NE")?;
        // a position is RLV digits of RBITS bits: enough levels for the context NE PY (at least 4)
        let rope_bits = pr.int("RBITS").unwrap_or(3);
        let reach = pr.int("CTX").map_or(ne * py, |c| c.min(ne * py));
        let rope_levels = pr
            .int("RLV")
            .unwrap_or(4.max((32 - (reach - 1).leading_zeros()).div_ceil(rope_bits)));
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
            arrangement: None,
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
            family: FamilyId::Dense,
            shape: s.clone(),
            partition,
            contract,
            levers: pr.levers.clone(),
        })
    }

    fn kernels(&self, prog: &Program) -> R<Kernels> {
        dense::kernels(prog)
    }

    fn tiles(&self, plan: &Plan, shapes: &dyn Fn(&str) -> Option<Vec<u64>>) -> R<Spec> {
        cut::tiles(plan, shapes)
    }

    fn symbols(&self) -> &'static [(&'static str, &'static str)] {
        dense::SYMBOLS
    }
}
