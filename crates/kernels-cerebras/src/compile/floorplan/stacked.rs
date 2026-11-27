//! The stacked columns constructor (Gemma 4 E2B / E4B): units at the arrangement's origins in two
//! unit columns, the scores segments a half row, PLE storage above each unit column
//! ([`crate::compile::block::ple::Ple`]), the shared-KV line ([`crate::compile::block::shared_kv::SharedKv`]), the
//! head east of the units and the token loop.
//! Kernels: [`crate::compile::kernels::stacked`]; cut: [`crate::compile::tiles::stacked`].

use std::collections::BTreeMap;

use crate::compile::floorplan::ceil_to;
use crate::compile::plan::{
    Arrangement, Contract, HeadPartition, KvEntries, LayerKind, Partition, Plan, Preset,
    ReduceGroups, RowRun, Segments, Shape, StoreSpec,
};

type R<T> = Result<T, String>;

/// Preset keys: the unit shape (PX, PY, NU; its arrangement [`Arrangement::derive`]), head rows (HR), global KV entries a KV row (NEG), reduction groups
/// (G2, G2W), posted loop (LOOP). The sliding window runs over `PYW` KV rows (exact ring where
/// NES divides it, else every row with older entries masked). Overrides: `PYW`, `NEG`, `G2`,
/// `G2W`, `G3W`, `CAP`, `LOOP`, `SEATS`.
pub fn plan(s: &Shape, pr: &Preset) -> R<Plan> {
    let (px, py, nu) = (pr.need("PX")?, pr.need("PY")?, pr.need("NU")?);
    let n = s.layers.len() as u32;
    if nu == 0 || !n.is_multiple_of(nu) {
        return Err(format!("stacked: {n} layers on {nu} units"));
    }
    let k = n / nu;
    // the kernels take one unit's layer kinds for every unit: unit 0's global mask (`KM`), a
    // sharer's whole unit (`SH`) and its first layer's MLP width (`Ic`)
    let global = |l: &LayerKind| {
        matches!(
            l,
            LayerKind::Global | LayerKind::SharedKv { sliding: false, .. }
        )
    };
    let shared = |l: &LayerKind| matches!(l, LayerKind::SharedKv { .. });
    let first = &s.layers[..k as usize];
    if let Some(u) = (0..nu).find(|&u| {
        let (a, b) = ((u * k) as usize, ((u + 1) * k) as usize);
        let unit = &s.layers[a..b];
        unit.iter().map(global).ne(first.iter().map(global))
            || unit.iter().any(|l| shared(l) != shared(&unit[0]))
            || (a..b).any(|l| s.mlp.width(l as u32) != s.mlp.width(a as u32))
    }) {
        return Err(format!(
            "stacked: unit {u}'s {k} layers differ in kind from unit 0's or among themselves"
        ));
    }
    if !s.hidden.is_multiple_of(py) || !py.is_multiple_of(2) {
        return Err(format!("stacked: {} hidden over {py} rows", s.hidden));
    }
    // what the cut lays ([`crate::compile::tiles::stacked`]): a per-layer input a column, one or
    // two KV heads, every layer's MLP even over the columns, a store in each unit column
    let ple = s.ple.ok_or("stacked: no per-layer inputs")?.width;
    if ple != px || !(1..=2).contains(&s.attn.kv_heads) || nu < 2 {
        return Err(format!(
            "stacked: PLE {ple} on {px} columns, {} KV heads, {nu} units",
            s.attn.kv_heads
        ));
    }
    if let Some(l) = (0..n).find(|&l| !s.mlp.width(l).is_multiple_of(px)) {
        return Err(format!(
            "stacked: layer {l}'s MLP {} over {px} columns",
            s.mlp.width(l)
        ));
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
    // a ring of seats shares the global KV pool, NEG PYW: a sequence holds what the others'
    // least windows (a launch's laps, whole rows) leave
    let seats = pr.int("SEATS").unwrap_or(1).max(1);
    let cap = pr.int("CAP").unwrap_or(256);
    let least = (cap / seats).max(1).div_ceil(pyw) * pyw;
    if seats > neg || (seats - 1) * least >= neg * pyw {
        return Err(format!(
            "stacked: {seats} seats over {neg} KV entries a row"
        ));
    }
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
        arrangement: Some(Arrangement::derive(s, (px, py, nu))),
    };
    let contract = Contract {
        positions: neg * pyw - (seats - 1) * least,
        rope_levels: 4,
        rope_bits: 3,
        cap,
        loop_len: pr.need("LOOP")?,
        token_loop: true,
        logit_scale: 1.0,
    };
    Ok(Plan {
        shape: s.clone(),
        partition,
        contract,
        levers: pr.levers.clone(),
        seats,
    })
}

/// The words a storage PE holds of its column's side input (every layer's row of `tokens`
/// tokens); covers the pinned plans' columns.
pub const STORE_WORDS: u32 = 19_000;

/// The colours the storage columns' buses prefer (the pinned plans').
pub const STORE_BUS: [u8; 2] = [17, 10];

impl Arrangement {
    /// The arrangement of `nu` units of `px` x `py` weight PEs: column A holds the shared caches'
    /// source unit on (the KV line runs north) and at least half the units, column B the rest,
    /// levelled so the last levels hand off across. Each column's storage holds its layers' side
    /// input, [`STORE_WORDS`] words a storage PE.
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
    /// A Gemma 4 model plans by the search, on the derived arrangement: the 24-layer E4B mini;
    /// the one-layer one, whose second unit column would store no per-layer inputs, finds none.
    #[test]
    fn a_shape_with_a_derived_arrangement_plans() {
        let p = crate::compile::target::searched("gemma4-e4b-mini-l24-bf16-kv-bf16").plan;
        crate::compile::program::Program::of(p).unwrap();
        let t = poem_compiler::catalog::repository()
            .parse("gemma4-e4b-mini-l1-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let e = crate::compile::target::plan(&t, &crate::compile::target::Ask::default());
        assert!(e.is_err_and(|e| e.contains("the search found no partition")));
    }

    /// Units run unit 0's layer kinds: E4B's 42 layers (a global layer in six, 18 sharing their
    /// KV) on six units of seven would mix them (the chip hung on that pick); seven of six plan.
    #[test]
    fn units_of_mixed_layer_kinds_are_refused() {
        let t = poem_compiler::catalog::repository()
            .parse("gemma4-e4b-bf16-kv-bf16")
            .unwrap()
            .trace(poem_ir::Platform::Cerebras);
        let s = crate::compile::target::shape(&t).unwrap();
        let keys = |nu: u32| {
            crate::compile::plan::Preset::default()
                .set(&format!(
                    "PX=256,PY=160,NU={nu},NEG=28,HR=1130,G2=12,G2W=24,LOOP=64"
                ))
                .unwrap()
        };
        let e = super::plan(&s, &keys(6)).unwrap_err();
        assert!(e.contains("differ in kind"), "{e}");
        super::plan(&s, &keys(7)).unwrap();
    }
}
