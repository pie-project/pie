//! Row runs: Gemma 4 12B as it plans: units of PY + 1 rows by PX + 1 columns at explicit
//! origins in two unit columns (the first visited down, the second up: unit 0 and the last unit
//! side by side on row 0), their rows in two runs (attention rows `r < RA` with `Hc A` dims and
//! the scores segments, 128 of them so the 1024 window is an exact ring; plain rows with `Hc W`
//! dims), and the head east of them fed by groups of CX rows, the token loop's winner coming back
//! the same way. Blocks (by [`crate::compose`]): `RowRuns`, `HeadGroups`. Kernels:
//! [`crate::kernels::runs`]; cut: [`crate::tiles::runs`].

use crate::kernels::runs::Dims;
use crate::partition::ceil_to;
use crate::plan::{
    Contract, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset, ReduceGroups, RowRun,
    Segments, Shape,
};

type R<T> = Result<T, String>;

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

/// The keys a plan's preset gives: weight columns and rows a unit (PX, PY), the head's rows (HR),
/// the attention rows' and the plain rows' dims (HcA, HcW), the attention rows (RA: the KV rows;
/// 128 make the 1024 window an exact ring of 8 x 128), the scores segment width (SWA), the head
/// groups' sizes (attention rows in groups of GA, plain rows in groups of GW, never across the CY
/// row), the units (NU), the global KV entries a KV row (NEG), the reduction groups (G2, G2W).
///
/// Units of K = layers / NU (S .. S G) in two unit columns; rows in two runs (the attention
/// rows `r < RA` with HcA dims and the scores segments, the plain rows with HcW); the sliding
/// window over the RA KV rows (NES entries a row: exact where RA divides it); the head one
/// column a row group, VR vocabulary rows a head PE (a multiple of 8). Overrides: `NEG`,
/// `G2`, `G2W`, `G3W`, `CAP`, `LOOP`.
pub fn plan(s: &Shape, pr: &Preset) -> R<Plan> {
    let (px, py, ra) = (pr.need("PX")?, pr.need("PY")?, pr.need("RA")?);
    let (hca, hcw, swa) = (pr.need("HcA")?, pr.need("HcW")?, pr.need("SWA")?);
    let (hr, units, neg) = (pr.need("HR")?, pr.need("NU")?, pr.need("NEG")?);
    let n = s.layers.len() as u32;
    if units == 0 || !n.is_multiple_of(units) {
        return Err(format!("runs: {n} layers on {units} units"));
    }
    let window = s
        .layers
        .iter()
        .find_map(|l| match l {
            LayerKind::Sliding { window } => Some(*window),
            _ => None,
        })
        .ok_or("runs: no sliding layer")?;
    let groups = groups_split(py, py / 2, ra, pr.need("GA")?, pr.need("GW")?);
    let g2 = pr.need("G2")?;
    let plan = Plan {
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
                token_loop3: pr.int("G3W").unwrap_or(0),
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
            arrangement: None,
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
