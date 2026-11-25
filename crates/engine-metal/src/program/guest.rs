//! A guest program's state in the pool (`engine::GuestState`): the
//! batch sets a (program, extents) pair fires its grouped stages with, and
//! the value scratch an instance fires its own stages with, held in
//! `Instance` slots the runtime takes at bind and gives back at close.
//! Nothing of it lies in the activation span.
//!
//! One rule states the bytes and lays them: a set (or an arm's scratch) is
//! laid whole, back to back, where the runtime gives one run of slots (it
//! does when two of the state share a chunk); else in pieces of `run` whole
//! chunks, `run` the chunks the largest set needs, the sets packed in their
//! order, each whole in one piece ([`state_of`], [`places`]). The runtime
//! takes the pieces the stated bytes ask (`store::roles::scratch_part`), so
//! the two agree by construction.

use engine::program::GuestState;

use crate::error::{Fault, Result};

/// How the pool places a guest's state: an `Instance` slot's bytes, the
/// slots a chunk holds, and a chunk's bytes.
#[derive(Debug, Clone, Copy)]
pub struct Room {
    pub slot_bytes: u64,
    pub chunk_slots: u32,
    pub chunk_bytes: u64,
}

impl Room {
    /// Whether `bytes` lie in one run of slots: two of them share a chunk
    /// (the runtime's rule, `store::roles::scratch_part`).
    fn in_slots(&self, bytes: u64) -> bool {
        bytes.div_ceil(self.slot_bytes.max(1)).saturating_mul(2) <= u64::from(self.chunk_slots)
    }

    /// The chunks a piece takes: what the largest of `sets` needs, one at
    /// least.
    fn run_of(&self, sets: &[u64]) -> u32 {
        let largest = sets.iter().copied().max().unwrap_or(0);
        u32::try_from(largest.div_ceil(self.chunk_bytes.max(1)).max(1)).unwrap_or(u32::MAX)
    }
}

/// Where each of `sets` lies: the piece and the byte in it, packed in order,
/// each whole in one piece of `piece` bytes.
fn pack(sets: &[u64], piece: u64) -> Vec<(usize, u64)> {
    let mut out = Vec::with_capacity(sets.len());
    let (mut at, mut used) = (0usize, 0u64);
    for &bytes in sets {
        if used > 0 && used + bytes > piece {
            at += 1;
            used = 0;
        }
        out.push((at, used));
        used += bytes;
    }
    out
}

/// The bytes `sets` take as the pool places them, and the chunks a piece
/// takes where they lie in whole chunks (0 where they share a run of slots).
#[must_use]
pub fn state_of(sets: &[u64], room: Room) -> (u64, u32) {
    let total: u64 = sets.iter().sum();
    if total == 0 {
        return (0, 0);
    }
    if room.in_slots(total) {
        return (total, 0);
    }
    let run = room.run_of(sets);
    let piece = u64::from(run) * room.chunk_bytes;
    let pieces = pack(sets, piece).last().map_or(0, |&(at, _)| at + 1);
    (pieces as u64 * piece, run)
}

/// A pair's and an instance's state, each by [`state_of`]: what the engine
/// states for a (program, extents) pair (`engine::Engine::guest_state`).
#[must_use]
pub fn guest_state(
    pair_sets: &[u64],
    instance_sets: &[u64],
    resident: u64,
    room: Room,
) -> GuestState {
    let (per_pair, pair_run) = state_of(pair_sets, room);
    let (per_instance, instance_run) = state_of(&with_resident(resident, instance_sets), room);
    GuestState {
        // What a later fire reads as a fire left it (a beam group's state): laid first in the instance's place, copied
        // down a tier at a park and back byte for byte. A group's state
        // grows with its beams squared, past an Instance slot's chunk, so it
        // lies in the place's run of whole chunks rather than the tables.
        resident,
        // The instance's value scratch after it, dead between fires: one
        // arm (today's arms folded in), the place's packing slack with it.
        scratch: per_instance - resident,
        scratch_arms: 1,
        per_pair,
        run: pair_run.max(instance_run),
    }
}

/// An instance's sets as its place lays them: its resident state first
/// (where it holds one), then its scratch sets.
#[must_use]
pub fn with_resident(resident: u64, scratch_sets: &[u64]) -> Vec<u64> {
    (resident > 0).then_some(resident).into_iter().chain(scratch_sets.iter().copied()).collect()
}

/// Where each of `sets` lies in pieces of `pieces` bytes (the place the
/// runtime took for them, [`crate::store::Pools::guest_pieces`]): its
/// piece and the byte in it, by the rule that stated them. Refused by name
/// where they do not fit: a place the runtime took for other bytes than
/// these state.
pub fn places(sets: &[u64], pieces: &[u64]) -> Result<Vec<(usize, u64)>> {
    if sets.is_empty() {
        return Ok(Vec::new());
    }
    let short = |what: String| Fault::Residency(what);
    match pieces {
        [] => Err(short(format!(
            "{} guest set(s) of {} bytes and no place the runtime took for them",
            sets.len(),
            sets.iter().sum::<u64>()
        ))),
        [one] if *one >= sets.iter().sum::<u64>() => Ok(pack(sets, *one)),
        _ => {
            let piece = pieces.iter().copied().min().unwrap_or(0);
            let laid = pack(sets, piece);
            let fits = sets.iter().all(|&bytes| bytes <= piece)
                && laid.last().is_some_and(|&(at, _)| at < pieces.len());
            if fits {
                Ok(laid)
            } else {
                Err(short(format!(
                    "guest sets of {:?} bytes do not lie whole in {} piece(s) of {piece} bytes",
                    sets,
                    pieces.len()
                )))
            }
        }
    }
}
