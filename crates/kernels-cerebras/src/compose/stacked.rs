//! Stacked unit columns: units at explicit origins in two unit columns, a per-token side
//! input's storage above each column's top unit (the arrangement a unit shape names; Gemma 4
//! E2B / E4B's floorplans and their tiny simulation's). A hand-made placement until the search
//! places units.

use std::collections::BTreeMap;

use crate::plan::Plan;

type R<T> = Result<T, String>;

/// A PLE storage block's size: rows above its column's top unit, tokens a storage PE, bus colour.
pub struct StoreSpec {
    pub rows: u32,
    pub tokens: u32,
    pub bus: u8,
}

/// A hand-made Gemma floorplan: unit column A at
/// x 0 holds `a_units` top to bottom, stacked from row `a_top`; unit column B, east of it, holds
/// each of `b_units` level with a column-A unit; above each column's top unit its PLE storage
/// block. Unit index order is the handoff order (unit 0 takes the embedding). The head sits east
/// of column B.
pub struct Arrangement {
    a_top: u32,
    a_units: &'static [u32],
    /// (unit, the column-A unit it sits level with), top to bottom.
    b_units: &'static [(u32, u32)],
    /// Column A's storage, column B's.
    stores: [StoreSpec; 2],
}

/// The arrangements, by unit shape (weight columns, rows, units). A new size adds one.
const ARRANGEMENTS: &[((u32, u32, u32), Arrangement)] = &[
    // the tiny simulation: E2B's arrangement in small (unit 0 shares the last unit's rows, as the
    // head's turns need)
    (
        (8, 8, 5),
        Arrangement {
            a_top: 4,
            a_units: &[4, 3, 2],
            b_units: &[(0, 4), (1, 2)],
            stores: [
                StoreSpec {
                    rows: 4,
                    tokens: 128,
                    bus: 17,
                },
                StoreSpec {
                    rows: 2,
                    tokens: 256,
                    bus: 10,
                },
            ],
        },
    ),
    // E2B: column A units 6 .. 2 down to the wafer's last row (1153), column B units 0 (level
    // with 6) and 1 (level with 2)
    (
        (256, 128, 7),
        Arrangement {
            a_top: 1153 - 129 * 5,
            a_units: &[6, 5, 4, 3, 2],
            b_units: &[(0, 6), (1, 2)],
            stores: [
                StoreSpec {
                    rows: 345,
                    tokens: 760,
                    bus: 17,
                },
                StoreSpec {
                    rows: 138,
                    tokens: 1900,
                    bus: 10,
                },
            ],
        },
    ),
    // E4B: column A units 6 .. 3, column B units 0, 1, 2 level with 6, 5, 3
    (
        (256, 160, 7),
        Arrangement {
            a_top: 1153 - 161 * 4,
            a_units: &[6, 5, 4, 3],
            b_units: &[(0, 6), (1, 5), (2, 3)],
            stores: [
                StoreSpec {
                    rows: 332,
                    tokens: 791,
                    bus: 17,
                },
                StoreSpec {
                    rows: 249,
                    tokens: 1055,
                    bus: 10,
                },
            ],
        },
    ),
];

impl Arrangement {
    pub fn of(plan: &Plan) -> R<&'static Arrangement> {
        let p = &plan.partition;
        let key = (p.cols, p.rows, p.units);
        ARRANGEMENTS
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, a)| a)
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
        for &(u, level) in self.b_units {
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
            (0, self.a_units.to_vec(), &self.stores[0]),
            (
                uw,
                self.b_units.iter().map(|&(u, _)| u).collect(),
                &self.stores[1],
            ),
        ]
    }
}
