//! The stacked columns constructor ([`super::KernelSet::Shared`]'s pinned plans and the
//! search's moves): Gemma 4 E2B / E4B as they plan: units at the explicit origins of the
//! plan's arrangement in two unit columns, the scores segments a half row, the PLE storage above
//! each unit column ([`crate::block::ple::Ple`]), the shared-KV line from the KV source unit to
//! its sharers ([`crate::block::shared_kv::SharedKv`]), the head east of the units and the token
//! loop. The sliding / global KV entries (`NES` / `NEG`) and the KV rows (`PYW`: E4B keeps KV on
//! 128 of 160 rows so the 512 window is an exact ring) are in the plan. Kernels:
//! [`crate::kernels::stacked`]; cut: [`crate::tiles::stacked`].

use std::collections::BTreeMap;

use crate::floorplan::ceil_to;
use crate::plan::{
    Arrangement, Contract, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset,
    ReduceGroups, RowRun, Segments, Shape, StoreSpec,
};

type R<T> = Result<T, String>;

/// The keys a plan's preset gives: the unit shape the arrangement is made for (weight columns,
/// rows, units: PX, PY, NU), the head's rows (HR), the global KV entries a KV row (NEG; the context
/// is NEG x the KV rows), the reduction groups (G2, G2W) and the posted loop (LOOP).
///
/// Units of K = layers / NU (S .. S G); a row's dims the hidden over PY; the sliding window
/// over the KV rows (`PYW`: the window over the fewest sliding entries a row, NES, where that
/// divides it, so the ring is exact; else every row, and the kernel masks the older entries);
/// the head PY / 2 columns of VR vocabulary rows (a multiple of 8). Overrides: `PYW`, `NEG`,
/// `G2`, `G2W`, `G3W`, `CAP`, `LOOP`.
pub fn plan(s: &Shape, pr: &Preset) -> R<Plan> {
    let (px, py, nu) = (pr.need("PX")?, pr.need("PY")?, pr.need("NU")?);
    let n = s.layers.len() as u32;
    if nu == 0 || !n.is_multiple_of(nu) {
        return Err(format!("stacked: {n} layers on {nu} units"));
    }
    let k = n / nu;
    if !s.hidden.is_multiple_of(py) {
        return Err(format!("stacked: {} hidden over {py} rows", s.hidden));
    }
    let window = s
        .layers
        .iter()
        .find_map(|l| match l {
            LayerKind::Sliding { window } => Some(*window),
            _ => None,
        })
        .ok_or("stacked: no sliding layer")?;
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
            token_loop3: pr.int("G3W").unwrap_or(0),
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
        arrangement: pr.arrangement.clone().filter(|a| a.unit == (px, py, nu)),
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
        shape: s.clone(),
        partition,
        contract,
        levers: pr.levers.clone(),
    })
}

impl Arrangement {
    /// The plan's arrangement, made for its unit shape (weight columns, rows, units).
    pub fn of(plan: &Plan) -> R<&Arrangement> {
        let p = &plan.partition;
        let key = (p.cols, p.rows, p.units);
        p.arrangement
            .as_ref()
            .filter(|a| a.unit == key)
            .ok_or_else(|| {
                format!(
                    "compose: no stacked arrangement for {} x {} units x {}",
                    p.cols, p.rows, p.units
                )
            })
    }

    /// Unit origins in unit order, for units of `uw` x `uh` PEs.
    pub fn origins(&self, uw: u32, uh: u32) -> R<Vec<(u32, u32)>> {
        let n = self.a_units.len() + self.b_units.len();
        let mut o = vec![None; n];
        let mut ya = BTreeMap::new();
        for (i, &u) in self.a_units.iter().enumerate() {
            let y = self.a_top + i as u32 * uh;
            ya.insert(u, y);
            o[u as usize] = Some((0, y));
        }
        for &(u, level) in &self.b_units {
            let y = *ya
                .get(&level)
                .ok_or_else(|| format!("compose: unit {u} level with {level}, not in A"))?;
            o[u as usize] = Some((uw, y));
        }
        o.into_iter()
            .enumerate()
            .map(|(u, v)| v.ok_or_else(|| format!("compose: unit {u} unplaced")))
            .collect()
    }

    /// The unit columns: (x, units top to bottom, storage).
    pub fn columns(&self, uw: u32) -> [(u32, Vec<u32>, &StoreSpec); 2] {
        [
            (0, self.a_units.clone(), &self.stores[0]),
            (
                uw,
                self.b_units.iter().map(|&(u, _)| u).collect(),
                &self.stores[1],
            ),
        ]
    }
}
