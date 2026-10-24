//! Expert-parallel load balancing: every `every` steps the group reads how
//! many rows each rank's slots received, places each MoE layer's experts
//! anew ([`engine::placement::Placement::rebalance`]) and carries out the
//! moves: each moved expert's planes copied from the slot that held it to
//! the slot that will, then every rank's table rewritten in place.
//!
//! Every read of the loads also closes a window of the group's balance
//! ([`LayerBalance`]): at a rebalance, every `report` steps (printed), and
//! on demand. The loads arming left (its warm launches dispatch real rows)
//! are dropped at the first step's boundary, so a window counts only the
//! rows the group served.
//!
//! It runs at a step boundary, every rank's work finished and none in
//! flight: a copy lands in a slot no kernel reads, and a captured body
//! reads the planes and the table where they were.

use std::collections::BTreeMap;

use engine::placement::{Move, Placement, SPARE, Window};

use super::{EpRank, Plane};
use crate::device::Buffer;
use crate::error::{Fault, Result};

/// How often the group rebalances, and over how many steps' loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rebalance {
    /// Steps between rebalances.
    pub every: u32,
    /// Steps of loads a rebalance reads (at least `every`).
    pub window: u32,
}

/// What the boundary before a step does with the group's loads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Due {
    /// Drops what arming left: the boundary before the first step.
    Discard,
    /// Reads them, closing a window of the balance; rebalances on them,
    /// and prints the window.
    Read { rebalance: bool, report: bool },
}

/// One MoE layer's balance over a window of steps: the rows each rank's
/// slots received, and each logical expert wherever its slots sit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerBalance {
    pub layer: u32,
    /// Steps the window spans.
    pub steps: u64,
    /// Rows a rank, in rank order.
    pub ranks: Vec<u64>,
    /// Rows a logical expert.
    pub experts: Vec<u64>,
}

impl LayerBalance {
    #[must_use]
    pub fn rows(&self) -> u64 {
        self.ranks.iter().sum()
    }

    /// The busiest rank's rows over the mean rank's: 1 is even.
    #[must_use]
    pub fn rank_skew(&self) -> f64 {
        skew(&self.ranks)
    }

    /// The hottest expert's rows over the mean expert's.
    #[must_use]
    pub fn expert_skew(&self) -> f64 {
        skew(&self.experts)
    }

    /// The expert that received the most rows.
    #[must_use]
    pub fn hottest(&self) -> Option<u32> {
        (0..self.experts.len())
            .max_by_key(|&e| self.experts[e])
            .map(|e| e as u32)
    }
}

/// The largest of `loads` over their mean; 1 when no row came.
fn skew(loads: &[u64]) -> f64 {
    let total: u64 = loads.iter().sum();
    if total == 0 {
        return 1.0;
    }
    let max = loads.iter().copied().max().unwrap_or(0);
    max as f64 * loads.len() as f64 / total as f64
}

impl core::fmt::Display for LayerBalance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "[expert-balance] layer={} steps={} rows={} rank-skew={:.2} \
             expert-skew={:.2} hottest={} ranks={:?}",
            self.layer,
            self.steps,
            self.rows(),
            self.rank_skew(),
            self.expert_skew(),
            self.hottest()
                .map_or_else(|| "-".to_string(), |e| e.to_string()),
            self.ranks,
        )
    }
}

/// A layer's placement as the group holds it, and its recent loads.
struct Balance {
    placement: Placement,
    window: Window,
    stride: u32,
    /// Rows a logical expert since the last rebalance: the window's next.
    since: Vec<u32>,
}

/// The group's balancer, held by the step's leader.
pub struct Rebalancer {
    knob: Option<Rebalance>,
    /// Steps between printed reads (0: never).
    report: u32,
    /// Steps the group has begun.
    steps: u64,
    /// Steps served when the loads were last read.
    read_at: u64,
    moved: u64,
    layers: BTreeMap<u32, Balance>,
    balance: Vec<LayerBalance>,
    /// One expert's planes, a cycle's parked slot ([`SPARE`]), on the
    /// device the copies run on.
    spare: Option<Buffer>,
}

impl Rebalancer {
    /// A balancer that rebalances by `knob` (none: never) and prints the
    /// balance every `report` steps (0: never).
    #[must_use]
    pub fn new(knob: Option<Rebalance>, report: u32) -> Rebalancer {
        Rebalancer {
            knob,
            report,
            steps: 0,
            read_at: 0,
            moved: 0,
            layers: BTreeMap::new(),
            balance: Vec::new(),
            spare: None,
        }
    }

    /// The experts moved since load.
    #[must_use]
    pub fn moved(&self) -> u64 {
        self.moved
    }

    /// Each layer's balance over the window the last read closed, in layer
    /// order; empty before the first.
    #[must_use]
    pub fn balance(&self) -> &[LayerBalance] {
        &self.balance
    }

    /// Counts a step; what the boundary before it does, if anything.
    pub fn due(&mut self) -> Option<Due> {
        let served = self.steps;
        self.steps += 1;
        if served == 0 {
            return Some(Due::Discard);
        }
        let every = |n: u32| n > 0 && served.is_multiple_of(u64::from(n));
        let rebalance = self.knob.is_some_and(|knob| every(knob.every));
        let report = every(self.report);
        (rebalance || report).then_some(Due::Read { rebalance, report })
    }

    /// Does what the boundary [`Rebalancer::due`] just counted is due, the
    /// copies enqueued on `stream` (a stream of the device the caller
    /// bound). Every rank's work must have finished. The moves it made.
    pub fn run(
        &mut self,
        due: Due,
        ranks: &mut [&mut EpRank],
        stream: *mut core::ffi::c_void,
    ) -> Result<usize> {
        let Due::Read { rebalance, report } = due else {
            for rank in ranks.iter_mut() {
                rank.take_loads()?;
            }
            return Ok(0);
        };
        self.read_to(ranks, self.steps - 1)?;
        if report {
            for layer in &self.balance {
                eprintln!("engine-cuda: {layer}");
            }
        }
        if rebalance {
            self.rebalance(ranks, stream)
        } else {
            Ok(0)
        }
    }

    /// Reads every rank's loads between two steps, every rank's work
    /// finished: the window since the last read, which it closes.
    pub fn read(&mut self, ranks: &mut [&mut EpRank]) -> Result<&[LayerBalance]> {
        self.read_to(ranks, self.steps)?;
        Ok(&self.balance)
    }

    /// Reads every rank's loads, `served` steps served.
    fn read_to(&mut self, ranks: &mut [&mut EpRank], served: u64) -> Result<()> {
        let world = ranks.len() as u32;
        let loads: Vec<_> = ranks
            .iter_mut()
            .map(|rank| rank.take_loads())
            .collect::<Result<_>>()?;
        let pushes = self
            .knob
            .map_or(1, |knob| (knob.window / knob.every.max(1)).max(1) as usize);
        let mut windows = Vec::new();
        for (layer, experts, slots, stride) in ranks[0].layers() {
            let balance = match self.layers.entry(layer) {
                std::collections::btree_map::Entry::Occupied(at) => at.into_mut(),
                std::collections::btree_map::Entry::Vacant(at) => at.insert(Balance {
                    placement: Placement::uniform(experts, world, slots).map_err(|why| {
                        Fault::program("ep::rebalance", format!("layer {layer}: {why}"))
                    })?,
                    window: Window::new(experts, pushes),
                    stride,
                    since: vec![0; experts as usize],
                }),
            };
            let mut window = LayerBalance {
                layer,
                steps: served - self.read_at,
                ranks: vec![0; world as usize],
                experts: vec![0; experts as usize],
            };
            for (r, rank) in loads.iter().enumerate() {
                for (s, &rows) in rank.get(&layer).into_iter().flatten().enumerate() {
                    let expert = balance.placement.held[r * slots as usize + s] as usize;
                    window.ranks[r] += u64::from(rows);
                    window.experts[expert] += u64::from(rows);
                    balance.since[expert] = balance.since[expert].saturating_add(rows);
                }
            }
            windows.push(window);
        }
        self.read_at = served;
        self.balance = windows;
        Ok(())
    }

    /// Places each layer anew on its rows since the last rebalance and
    /// moves what the new placement moves. The moves it made.
    fn rebalance(
        &mut self,
        ranks: &mut [&mut EpRank],
        stream: *mut core::ffi::c_void,
    ) -> Result<usize> {
        let mut moved = Vec::new();
        let layers: Vec<u32> = self.layers.keys().copied().collect();
        for layer in layers {
            let balance = self.layers.get_mut(&layer).expect("listed above");
            let since = std::mem::replace(
                &mut balance.since,
                vec![0; balance.placement.experts as usize],
            );
            balance.window.push(&since);
            let next = balance.placement.rebalance(balance.window.loads());
            if next == balance.placement {
                continue;
            }
            let Ok(planes) = ranks
                .iter()
                .map(|rank| rank.planes(layer).map(<[Plane]>::to_vec))
                .collect::<std::result::Result<Vec<_>, _>>()
            else {
                continue;
            };
            let moves = balance.placement.moves(&next);
            let slots = next.slots;
            self.copy(&planes, slots, &moves, stream)?;
            moved.push((layer, next, moves.len()));
        }
        if moved.is_empty() {
            return Ok(0);
        }
        crate::device::ctx::sync(stream)?;
        let mut count = 0;
        for (layer, next, moves) in moved {
            let balance = self.layers.get_mut(&layer).expect("balanced above");
            let table = next.table(balance.stride);
            for rank in ranks.iter_mut() {
                rank.write_table(layer, &table)?;
            }
            balance.placement = next;
            count += moves;
        }
        self.moved += count as u64;
        eprintln!("engine-cuda: rebalanced the experts: {count} slot copies");
        Ok(count)
    }

    /// Enqueues `moves` in order: each copies one expert's planes, every
    /// rank's planes of the layer in `planes` (rank order), slot to slot.
    fn copy(
        &mut self,
        planes: &[Vec<Plane>],
        slots: u32,
        moves: &[Move],
        stream: *mut core::ffi::c_void,
    ) -> Result<()> {
        let expert_bytes: u64 = planes[0].iter().map(|p| p.slot_bytes).sum();
        if self
            .spare
            .as_ref()
            .is_none_or(|s| (s.bytes() as u64) < expert_bytes)
        {
            self.spare = Some(Buffer::zeroed(expert_bytes as usize)?);
        }
        let spare = self.spare.as_ref().expect("sized above").ptr();
        let at = |slot: u32, plane: usize, offset: u64| -> u64 {
            if slot == SPARE {
                return spare + offset;
            }
            let held = planes[(slot / slots) as usize][plane];
            held.ptr + u64::from(slot % slots) * held.slot_bytes
        };
        for m in moves {
            let mut offset = 0;
            for (plane, held) in planes[0].iter().enumerate() {
                crate::device::alloc::copy_any(
                    stream,
                    at(m.to, plane, offset),
                    at(m.from, plane, offset),
                    held.slot_bytes as usize,
                )?;
                offset += held.slot_bytes;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_boundary_drops_what_arming_left_and_the_knobs_read_after() {
        let knob = Rebalance {
            every: 2,
            window: 2,
        };
        let mut both = Rebalancer::new(Some(knob), 3);
        let read = |rebalance, report| Some(Due::Read { rebalance, report });
        let dues: Vec<_> = (0..7).map(|_| both.due()).collect();
        assert_eq!(
            dues,
            [
                Some(Due::Discard),
                None,
                read(true, false),
                read(false, true),
                read(true, false),
                None,
                read(true, true),
            ]
        );
        let mut neither = Rebalancer::new(None, 0);
        assert_eq!(neither.due(), Some(Due::Discard));
        assert!((0..64).all(|_| neither.due().is_none()));
    }

    #[test]
    fn a_layer_balance_is_its_peak_over_its_mean() {
        let balance = LayerBalance {
            layer: 3,
            steps: 2,
            ranks: vec![30, 10],
            experts: vec![0, 24, 8, 8],
        };
        assert_eq!(balance.rows(), 40);
        assert!((balance.rank_skew() - 1.5).abs() < 1e-9);
        assert!((balance.expert_skew() - 2.4).abs() < 1e-9);
        assert_eq!(balance.hottest(), Some(1));
        let none = LayerBalance {
            ranks: vec![0, 0],
            experts: vec![0; 4],
            ..balance
        };
        assert_eq!((none.rank_skew(), none.expert_skew()), (1.0, 1.0));
    }
}
