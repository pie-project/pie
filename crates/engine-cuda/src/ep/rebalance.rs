//! Expert-parallel load balancing: every `every` steps the group reads how
//! many rows each rank's slots received, places each MoE layer's experts
//! anew ([`engine::placement::Placement::rebalance`]) and carries out the
//! moves: each moved expert's planes copied from the slot that held it to
//! the slot that will, then every rank's table rewritten in place. The
//! loads arming left (its warm launches dispatch real rows) are dropped at
//! the first step's boundary, so the first window counts only served rows.
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
    Drop,
    /// Rebalances on them.
    Rebalance,
}

/// A layer's placement as the group holds it, and its recent loads.
struct Balance {
    placement: Placement,
    window: Window,
    stride: u32,
}

/// The group's balancer, held by the step's leader.
pub struct Rebalancer {
    knob: Rebalance,
    steps: u64,
    moved: u64,
    layers: BTreeMap<u32, Balance>,
    /// One expert's planes, a cycle's parked slot ([`SPARE`]), on the
    /// device the copies run on.
    spare: Option<Buffer>,
}

impl Rebalancer {
    #[must_use]
    pub fn new(knob: Rebalance) -> Rebalancer {
        Rebalancer {
            knob,
            steps: 0,
            moved: 0,
            layers: BTreeMap::new(),
            spare: None,
        }
    }

    /// The experts moved since load.
    #[must_use]
    pub fn moved(&self) -> u64 {
        self.moved
    }

    /// Counts a step; what the boundary before it does, if anything.
    pub fn due(&mut self) -> Option<Due> {
        let served = self.steps;
        self.steps += 1;
        if served == 0 {
            return Some(Due::Drop);
        }
        let every = u64::from(self.knob.every);
        (every > 0 && served.is_multiple_of(every)).then_some(Due::Rebalance)
    }

    /// Does what the boundary [`Rebalancer::due`] just counted is due: drops
    /// every rank's loads, or reads them and moves what the new placement
    /// moves, the copies enqueued on `stream` (a stream of the device the
    /// caller bound). Every rank's work must have finished. The moves it
    /// made.
    pub fn run(
        &mut self,
        due: Due,
        ranks: &mut [&mut EpRank],
        stream: *mut core::ffi::c_void,
    ) -> Result<usize> {
        if due == Due::Drop {
            for rank in ranks.iter_mut() {
                rank.take_loads()?;
            }
            return Ok(0);
        }
        let world = ranks.len() as u32;
        let loads: Vec<_> = ranks
            .iter_mut()
            .map(|rank| rank.take_loads())
            .collect::<Result<_>>()?;
        let shape: Vec<(u32, u32, u32, u32)> = ranks[0].layers().collect();
        let window = (self.knob.window / self.knob.every.max(1)).max(1) as usize;
        let mut moved = Vec::new();
        for (layer, experts, slots, stride) in shape {
            let balance = match self.layers.entry(layer) {
                std::collections::btree_map::Entry::Occupied(at) => at.into_mut(),
                std::collections::btree_map::Entry::Vacant(at) => at.insert(Balance {
                    placement: Placement::uniform(experts, world, slots).map_err(|why| {
                        Fault::program("ep::rebalance", format!("layer {layer}: {why}"))
                    })?,
                    window: Window::new(experts, window),
                    stride,
                }),
            };
            let mut logical = vec![0u32; experts as usize];
            for (r, rank) in loads.iter().enumerate() {
                for (s, &rows) in rank.get(&layer).into_iter().flatten().enumerate() {
                    let expert = balance.placement.held[r * slots as usize + s];
                    logical[expert as usize] += rows;
                }
            }
            balance.window.push(&logical);
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
    fn the_first_boundary_drops_what_arming_left_and_every_period_rebalances_after() {
        let mut rebalancer = Rebalancer::new(Rebalance {
            every: 2,
            window: 2,
        });
        let dues: Vec<_> = (0..6).map(|_| rebalancer.due()).collect();
        let rebalance = Some(Due::Rebalance);
        assert_eq!(
            dues,
            [Some(Due::Drop), None, rebalance, None, rebalance, None]
        );
    }
}
