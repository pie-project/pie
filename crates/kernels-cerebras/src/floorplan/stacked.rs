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
/// rows, units: PX, PY, NU; a unit shape the preset's arrangement is not made for takes the
/// derived one, [`Arrangement::derive`]), the head's rows (HR), the global KV entries a KV row (NEG; the context
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
        arrangement: Some(
            pr.arrangement
                .clone()
                .filter(|a| a.unit == (px, py, nu))
                .unwrap_or_else(|| Arrangement::derive(s, (px, py, nu))),
        ),
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

/// The words a storage PE holds of its column's side input (every layer's row of `tokens`
/// tokens): the pinned plans' columns hold 18,984 to 19,000.
pub const STORE_WORDS: u32 = 19_000;

/// The colours the storage columns' buses prefer (the pinned plans').
pub const STORE_BUS: [u8; 2] = [17, 10];

impl Arrangement {
    /// The arrangement of `nu` units of `px` x `py` weight PEs for `shape`'s layers, where no
    /// pinned plan states one: column A holds the units from the shared caches' source unit on
    /// (the sharers sit above the source in its column: the KV line runs north) and at least
    /// half of them, the last unit on top; column B the rest, unit 0 on top, each level with a
    /// column-A unit from the top down and its last level with column A's bottom (the handoff
    /// across). Each column's storage holds the whole vocabulary's rows of its layers'
    /// side input, [`STORE_WORDS`] words a storage PE; column A starts below the taller storage.
    /// The pinned E2B and E4B columns are this arrangement's but for column A's top row (the
    /// pins sit it lower).
    #[must_use]
    pub fn derive(shape: &Shape, (px, py, nu): (u32, u32, u32)) -> Arrangement {
        let n = nu.max(1);
        let k = (shape.layers.len() as u32).div_ceil(n).max(1);
        let source = shape.layers.iter().find_map(|l| match *l {
            LayerKind::SharedKv { from, .. } => Some(from / k),
            _ => None,
        });
        let na = n.div_ceil(2).max(source.map_or(0, |s| n.saturating_sub(s)));
        let nb = n - na;
        let a_units: Vec<u32> = (nb..n).rev().collect();
        let b_units = (0..nb)
            .map(|u| (u, if u + 1 == nb { nb } else { n - 1 - u }))
            .collect();
        let store = |units: u32, bus: u8| {
            let tokens = (STORE_WORDS / (k * units).max(1)).max(1);
            StoreSpec {
                rows: if units == 0 {
                    0
                } else {
                    shape.vocab.div_ceil(tokens)
                },
                tokens,
                bus,
            }
        };
        let stores = [store(na, STORE_BUS[0]), store(nb, STORE_BUS[1])];
        Arrangement {
            unit: (px, py, nu),
            a_top: stores[0].rows.max(stores[1].rows),
            a_units,
            b_units,
            stores,
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::{Ask, plan};

    fn trace(name: &str) -> poem_ir::Trace {
        poem_compiler::catalog::repository()
            .parse(name)
            .unwrap_or_else(|| panic!("{name} parses"))
            .trace(poem_ir::Platform::Cerebras)
    }

    /// The derived arrangement is the pinned E2B and E4B ones but for column A's top row: the
    /// same columns, levels and storage.
    #[test]
    fn the_derived_arrangement_is_the_pinned_ones() {
        for (name, preset) in [
            ("gemma4-e2b-bf16-kv-bf16", "e2b"),
            ("gemma4-e4b-bf16-kv-bf16", "e4b"),
        ] {
            let p = plan(&trace(name), &Ask::preset(preset)).unwrap().plan;
            let pinned = p.partition.arrangement.clone().unwrap();
            let derived = Arrangement::derive(&p.shape, pinned.unit);
            assert_eq!(
                Arrangement {
                    a_top: pinned.a_top,
                    ..derived.clone()
                },
                pinned,
                "{name}"
            );
            assert!(derived.a_top <= pinned.a_top, "{name}");
        }
    }

    /// A Gemma 4 model no pinned arrangement is made for plans by the search, on the derived
    /// arrangement: the one-layer E4B mini.
    #[test]
    fn a_shape_with_no_pinned_arrangement_plans() {
        let p = plan(&trace("gemma4-e4b-mini-l1-bf16-kv-bf16"), &Ask::default())
            .unwrap()
            .plan;
        crate::program::Program::of(p).unwrap();
    }
}
