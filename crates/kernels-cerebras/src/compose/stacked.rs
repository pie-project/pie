//! Stacked unit columns: units at explicit origins in two unit columns, a per-token side
//! input's storage above each column's top unit. The arrangement is the plan's
//! ([`crate::plan::Arrangement`]: the placement a pinned plan states for its unit shape); this
//! lays it out.

use std::collections::BTreeMap;

use crate::plan::{Arrangement, Plan, StoreSpec};

type R<T> = Result<T, String>;

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
