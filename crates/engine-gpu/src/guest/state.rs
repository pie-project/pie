//! A guest program's state on the device (`Engine::guest_state`, REDESIGN
//! §6): what an instance of a (program, extents) pair keeps in the pool's
//! `Instance` role while bound, and what the pair keeps once while any
//! instance of it is, which the runtime takes at the pair's first bind and
//! gives back at its last close. Nothing of it lies in the activation span,
//! so a fire of guest lanes takes the rung's span as a fire of none does.
//!
//! Two things, each aligned to a storage binding's offset
//! ([`crate::arena::ALIGN`]):
//!
//! - each stage's heap, the values its launches lay over their lifetimes
//!   ([`eta_exec::tables::StageTables::scratch`]): the instance's own
//!   (`resident`), in its own `Instance` slots or run of chunks, so the
//!   pool holds a heap for every bound instance and none more. Its stages
//!   share one region as wide as the widest: a stage runs whole (staged
//!   in, dispatched, read back, its kept puts copied to their channels'
//!   cells) before the next one stages anything, so no value of one
//!   outlives it into another's. A pass is
//!   one instance's dispatches over its own heap (these backends launch no
//!   grouped kernel across instances), bound at the bind and rebound at an
//!   unpark or an evacuation, never per fire; no value lives there between
//!   fires (a kept put lies in its channel's cell).
//! - where the program reads captured attention scores, the pair's score
//!   room (`per_pair`, lane-indexed, sized by the lane budget) ([`crate::arena::CaptureRows`]): a key row and a page row for each
//!   lane a fire carries at most (`Budget::max_lanes`), sized once at the
//!   pair's first bind. A fire lays all its capturing lanes' rows, compacted
//!   from the room's first byte, in the room of one pair it carries that
//!   reads them; their contents are dead once the fire has read them out.
//!
//! The contents are dead between fires, so a park copies nothing and an
//! unpark takes fresh room, the engine told where (`Rebind::Instance`).
//! Room that changes owner is zeroed before the guest's first fire there
//! ([`Placed::clear`]): an instance's at its bind and its unpark, a pair's
//! at each new taking of its slots ([`Takings`]); never per fire.
//! Each side lies whole in one place: a run of adjacent `Instance` slots in
//! one chunk where two of its size share a chunk (the runtime's rule,
//! `store::roles`), else one run of adjacent chunks inside one segment. A
//! stage's heap binds as one span of one segment buffer, as its WGSL
//! addresses it, and the score rows as one row block, as the capture
//! kernels address them.

use engine::program::{GuestState, ScratchPlace};

use crate::arena::ALIGN;
use crate::backend::{Backend, Buffer as _};
use crate::error::{Fault, Result};
use crate::store::Pools;
use crate::store::layout::INSTANCE;

fn refused(why: String) -> Fault {
    Fault::Program {
        at: "guest::state",
        why,
    }
}

/// Which place a part lies in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Instance,
    Pair,
}

/// Where an instance's and its pair's state lay their parts, each side
/// from its own first byte.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GuestLayout {
    /// Each stage's heap: its side, its first byte there and its bytes.
    pub stages: Vec<(Side, u64, u64)>,
    /// The pair's score room: its first byte and bytes; none where the
    /// program reads no scores.
    pub scores: Option<(u64, u64)>,
    /// The bytes each instance's side takes.
    pub instance: u64,
    /// The bytes the pair's side takes.
    pub pair: u64,
}

impl GuestLayout {
    /// Stage heaps of `heaps` bytes sharing one region on the instance's
    /// side, as wide as the widest, and score rows of `scores` bytes (0:
    /// none) on the pair's.
    #[must_use]
    pub fn of(heaps: &[u64], scores: u64) -> GuestLayout {
        let mut ends = [0u64; 2];
        let mut take = |side: Side, bytes: u64| {
            let end = &mut ends[usize::from(side == Side::Pair)];
            let at = end.next_multiple_of(ALIGN);
            *end = at + bytes;
            at
        };
        let widest = heaps.iter().copied().max().unwrap_or(0);
        let region = (widest > 0).then(|| take(Side::Instance, widest));
        let stages = heaps
            .iter()
            .map(|&bytes| (Side::Instance, region.unwrap_or(0), bytes))
            .collect();
        let scores = (scores > 0).then(|| (take(Side::Pair, scores), scores));
        GuestLayout {
            stages,
            scores,
            instance: ends[0],
            pair: ends[1],
        }
    }

    /// The layout of `plans`' stages bound at `extents`, beside `scores`
    /// bytes of score rows.
    pub fn of_plans(
        plans: &[eta_compiler::codegen::launch::LaunchStagePlan],
        extents: &eta_exec::Extents,
        scores: u64,
    ) -> std::result::Result<GuestLayout, String> {
        let heaps = plans
            .iter()
            .enumerate()
            .map(|(at, plan)| {
                super::session::stage_tables(plan, extents)
                    .map(|tables| tables.scratch)
                    .map_err(|why| format!("stage {at}'s scratch: {why}"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(GuestLayout::of(&heaps, scores))
    }

    /// The bytes `side` takes.
    #[must_use]
    pub fn bytes(&self, side: Side) -> u64 {
        match side {
            Side::Instance => self.instance,
            Side::Pair => self.pair,
        }
    }

    /// The state the runtime takes for it in a pool of `facts`: each side
    /// whole, a piece of `run` chunks the most either takes in whole chunks
    /// (a side two of which share a chunk lies in slots and asks none).
    /// Refused by name where that run is past a segment.
    pub fn state(&self, facts: &Facts) -> Result<GuestState> {
        let chunks = |bytes: u64| {
            let slots = bytes.div_ceil(facts.slot.max(1));
            if slots.saturating_mul(2) <= u64::from(facts.chunk_slots) {
                0
            } else {
                bytes.div_ceil(facts.chunk.max(1))
            }
        };
        let run = chunks(self.instance).max(chunks(self.pair));
        if run > u64::from(facts.segment_chunks) {
            return Err(refused(format!(
                "a guest's state of {} bytes an instance and {} its pair takes a run of {run} \
                 chunks of {}, past a segment of {}, and each lies whole in one run of a \
                 segment's chunks",
                self.instance, self.pair, facts.chunk, facts.segment_chunks
            )));
        }
        Ok(GuestState {
            // Today's arms are folded into the instance's bytes: one arm, no
            // scratch apart.
            resident: self.instance,
            scratch: 0,
            scratch_arms: 1,
            per_pair: self.pair,
            run: u32::try_from(run).unwrap_or(u32::MAX),
        })
    }
}

/// The extents a stage's tables are laid at, as a binding states them.
#[must_use]
pub fn extents_of(extents: &engine::program::BindExtents) -> eta_exec::Extents {
    eta_exec::Extents {
        row_count: extents.row_count,
        token_count: extents.token_count,
        sampled_rows: extents.sampled_rows,
        query_len: extents.query_len,
        key_len: extents.key_len,
    }
}

/// What a pool lays a guest's state by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Facts {
    pub chunk: u64,
    pub segment_chunks: u32,
    /// An `Instance` slot's bytes.
    pub slot: u64,
    /// The `Instance` slots a chunk holds.
    pub chunk_slots: u32,
}

impl Facts {
    pub fn of<B: Backend>(pools: &Pools<B>) -> Result<Facts> {
        Ok(Facts {
            chunk: pools.layout().chunk_bytes,
            segment_chunks: pools.layout().segment_chunks,
            slot: pools.slot_bytes(INSTANCE)?,
            chunk_slots: pools.slots_per_chunk(INSTANCE)?,
        })
    }
}

/// Where a pair's state lies now: the place the runtime named and the
/// segment buffer span it resolves to, which [`Placed::rebind`] follows
/// after an evacuation and [`Placed::retarget`] after an unpark.
#[derive(Debug, Clone)]
pub struct Placed<B: Backend> {
    place: ScratchPlace,
    bytes: u64,
    span: Option<(B::Buffer, u64)>,
}

impl<B: Backend> Placed<B> {
    /// `bytes` of state at `place`; refused by name where the place does
    /// not hold them whole in one span of one segment.
    pub fn at(pools: &Pools<B>, place: &ScratchPlace, bytes: u64) -> Result<Placed<B>> {
        let mut placed = Placed {
            place: place.clone(),
            bytes,
            span: None,
        };
        placed.rebind(pools)?;
        Ok(placed)
    }

    /// Points the span at where its place's slots or chunks sit now.
    pub fn rebind(&mut self, pools: &Pools<B>) -> Result<()> {
        self.span = match &self.place {
            ScratchPlace::None if self.bytes == 0 => None,
            ScratchPlace::None => {
                return Err(refused(format!(
                    "the binding names no place for a guest's {} bytes of state",
                    self.bytes
                )));
            }
            &ScratchPlace::Run { first, slots } => {
                let slot = pools.slot_bytes(INSTANCE)?;
                let per_chunk = pools.slots_per_chunk(INSTANCE)?.max(1);
                let last = first.checked_add(slots.max(1) - 1);
                if u64::from(slots) * slot < self.bytes
                    || last.is_none_or(|last| last / per_chunk != first / per_chunk)
                {
                    return Err(refused(format!(
                        "a guest's {} bytes of state at {slots} Instance slot(s) from {first}: \
                         a run of slots holds it whole inside one chunk of {per_chunk}",
                        self.bytes
                    )));
                }
                let (buffer, at) = pools.slot_place(INSTANCE, first)?;
                Some((buffer.clone(), at))
            }
            ScratchPlace::Chunks { run, pieces } => {
                let layout = pools.layout();
                let [first] = pieces[..] else {
                    return Err(refused(format!(
                        "a guest's {} bytes of state in {} pieces: it lies in one run of chunks",
                        self.bytes,
                        pieces.len()
                    )));
                };
                let (buffer, at) = pools.run_place(first, *run)?;
                let within = at / layout.chunk_bytes.max(1);
                if u64::from(*run) * layout.chunk_bytes < self.bytes
                    || within + u64::from(*run) > u64::from(layout.segment_chunks)
                {
                    return Err(refused(format!(
                        "a guest's {} bytes of state in a run of {run} chunk(s) from chunk \
                         {first}: the run holds it whole inside one segment of {}",
                        self.bytes, layout.segment_chunks
                    )));
                }
                Some((buffer.clone(), at))
            }
        };
        Ok(())
    }

    /// Moves the state to `place`, fresh room the runtime took after an
    /// unpark: nothing is copied, its contents being dead between fires.
    pub fn retarget(&mut self, pools: &Pools<B>, place: &ScratchPlace) -> Result<()> {
        let was = std::mem::replace(&mut self.place, place.clone());
        if let Err(fault) = self.rebind(pools) {
            self.place = was;
            return Err(fault);
        }
        Ok(())
    }

    /// Zeroes the state's bytes where it lies now. The room is another
    /// owner's until then: the pool hands back slots and chunks as their
    /// last owner left them, and a guest's stages may read heap bytes no
    /// launch of its own wrote. Called when the room changes owner (a bind,
    /// an unpark's fresh room, a pair's new taking), never per fire; ordered
    /// as the tables the bind writes are, after the work of the room's last
    /// owner, which the runtime retired before it handed the room on.
    pub fn clear(&self) -> Result<()> {
        let Some((buffer, at)) = &self.span else {
            return Ok(());
        };
        let mut buffer = buffer.clone();
        buffer.zero_span(*at, self.bytes)
    }

    /// The segment buffer the state lies in and its first byte there;
    /// `None` where it takes no bytes.
    #[must_use]
    pub fn span(&self) -> Option<(&B::Buffer, u64)> {
        self.span.as_ref().map(|(buffer, at)| (buffer, *at))
    }

    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    #[must_use]
    pub fn place(&self) -> &ScratchPlace {
        &self.place
    }
}

/// The pairs whose state lies laid now: each place, the taking of its
/// slots it was laid in (`engine::InstanceBinding::pair_generation`), and
/// the instances bound on it. A pair's state is zeroed once, where an
/// instance binds on a place and taking no other instance holds: room the
/// pool handed back and took again is another owner's bytes until then.
/// The same live slots bound again are not cleared: a sibling's fire may
/// be reading its score rows.
#[derive(Debug, Default)]
pub struct Takings {
    live: Vec<(ScratchPlace, u64, u32)>,
}

impl Takings {
    /// An instance binds on `place` at `generation`: `true` where none
    /// holds that taking, so its bytes are to be cleared.
    pub fn take(&mut self, place: &ScratchPlace, generation: u64) -> bool {
        if *place == ScratchPlace::None {
            return false;
        }
        match self
            .live
            .iter_mut()
            .find(|(held, taken, _)| held == place && *taken == generation)
        {
            Some((_, _, bound)) => {
                *bound += 1;
                false
            }
            None => {
                self.live.push((place.clone(), generation, 1));
                true
            }
        }
    }

    /// An instance bound on `place` at `generation` lets it go: closed, or
    /// moved to other room.
    pub fn give(&mut self, place: &ScratchPlace, generation: u64) {
        if let Some(at) = self
            .live
            .iter()
            .position(|(held, taken, _)| held == place && *taken == generation)
        {
            self.live[at].2 -= 1;
            if self.live[at].2 == 0 {
                self.live.swap_remove(at);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Facts, GuestLayout, Side, Takings};
    use engine::program::ScratchPlace;

    /// A pair's state is cleared at the first bind on a place and taking,
    /// not while another instance holds it, and again once the place is
    /// let go and taken anew or taken in another generation.
    #[test]
    fn a_pair_is_cleared_at_each_new_taking_and_only_then() {
        let place = ScratchPlace::Run { first: 4, slots: 2 };
        let mut takings = Takings::default();
        assert!(takings.take(&place, 0), "the first bind clears");
        assert!(!takings.take(&place, 0), "a sibling on the live taking does not");
        assert!(takings.take(&place, 1), "another taking of the same refs clears");
        takings.give(&place, 0);
        assert!(!takings.take(&place, 0), "one instance still holds taking 0");
        takings.give(&place, 0);
        takings.give(&place, 0);
        assert!(takings.take(&place, 0), "taken again once every holder let go");
        assert!(!takings.take(&ScratchPlace::None, 0), "no room, nothing to clear");
    }
    use engine::program::GuestState;

    /// Qwen3-0.6B's pool on the GPU: 4 MiB chunks of 4 KiB `Instance` slots.
    const FACTS: Facts = Facts {
        chunk: 4 << 20,
        segment_chunks: 64,
        slot: 4 << 10,
        chunk_slots: 1024,
    };

    /// The heaps are the instance's, its stages sharing one region as wide
    /// as the widest; the score room is the pair's.
    #[test]
    fn the_stages_share_one_region_and_the_score_room_is_the_pairs() {
        let layout = GuestLayout::of(&[1000, 3000, 300], 5000);
        assert_eq!(
            layout.stages,
            vec![
                (Side::Instance, 0, 1000),
                (Side::Instance, 0, 3000),
                (Side::Instance, 0, 300)
            ]
        );
        assert_eq!(layout.scores, Some((0, 5000)));
        assert_eq!((layout.instance, layout.pair), (3000, 5000));
        assert_eq!(GuestLayout::of(&[], 0), GuestLayout::default());
    }

    /// A side two of which share a chunk asks no run; past it, the run is
    /// the most either side takes in whole chunks; past a segment, refused
    /// by name.
    #[test]
    fn the_run_is_the_most_a_side_takes_in_whole_chunks() {
        let small = GuestLayout::of(&[100_000], 0);
        assert_eq!(
            small.state(&FACTS).expect("fits"),
            GuestState {
                resident: 100_000,
                scratch: 0,
                scratch_arms: 1,
                per_pair: 0,
                run: 0
            }
        );
        let top_p = GuestLayout::of(&[5_056_768], 0);
        assert_eq!(
            top_p.state(&FACTS).expect("fits"),
            GuestState {
                resident: 5_056_768,
                scratch: 0,
                scratch_arms: 1,
                per_pair: 0,
                run: 2
            }
        );
        // A 9 MiB score room beside a small heap: the room's run.
        let scored = GuestLayout::of(&[100_000], 9 << 20);
        assert_eq!(scored.state(&FACTS).expect("fits").run, 3);
        let past = GuestLayout::of(&[65 * (4 << 20)], 0);
        assert!(past.state(&FACTS).is_err());
        assert_eq!(
            GuestLayout::default().state(&FACTS).expect("none"),
            GuestState::default()
        );
    }
}
