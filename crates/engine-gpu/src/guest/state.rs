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
//!   (`scratch`, dead between fires), in its own `Instance` slots or run of chunks, so the
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
//! - where the program states each beam's parent, its beam group's state
//!   (`resident`, [`BeamLayout`]), laid first in the instance's place, its
//!   heaps past it: the one part a later fire reads.
//!
//! The heaps' and score rows' contents are dead between fires, so a park
//! copies nothing and an unpark takes fresh room, the engine told where
//! (`Rebind::Instance`). The beam state is live: where an instance states
//! it (`GuestState::resident`), the runtime copies its whole place wherever
//! a park moves it and back into the fresh room it names at the unpark, so
//! the state is cleared only at the bind.
//! Room that changes owner is zeroed before the guest's first fire there
//! ([`Placed::clear`]): an instance's at its bind and, past its resident
//! state, at its unpark ([`Placed::clear_dead`]), a pair's at each new
//! taking of its slots ([`Takings`]); never per fire.
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
    /// The bytes each instance's side takes, its resident state's
    /// included.
    pub instance: u64,
    /// The bytes at the head of the instance's side a later fire reads (a
    /// beam group's state, aligned): copied with the place at a park, never
    /// cleared past the bind. Its heaps lie past them.
    pub resident: u64,
    /// The bytes the pair's side takes.
    pub pair: u64,
    /// A beam group's state (`BEAM-PAGE-LISTS.md` §4), where the program
    /// states each beam's parent: from the first byte of the instance's
    /// place ([`Self::resident`]), live from the bind to the close, laid for
    /// the bind's depth.
    pub beam: Option<BeamLayout>,
}

/// A beam group's resident state as `kernels_slang::runtime::beam` lays it:
/// its beams, the pool pages and columns it holds at most, and the frames
/// in flight a freed page waits out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeamLayout {
    pub lanes: u32,
    pub cap: u32,
    pub frames: u32,
    /// On a hybrid, the set's rs entries its resident state holds past the
    /// rest (`BEAM-PAGE-LISTS.md` §11.1): the pool's rs slots holding its
    /// buffer's rows from its base, an entry a rs page, as many as the
    /// search's depth takes; appended as the set grows. 0: none.
    pub bases: u32,
}

impl BeamLayout {
    /// A group of `lanes` beams searching to `end` (the bind's `key_len`,
    /// the slot past its last row) over `page`-slot kv pages, `frames`
    /// frames in flight: each beam's pages to `end`, the pages a fire hands
    /// out, those pending over the frames and the prompt's own page
    /// (`BEAM-PAGE-LISTS.md` §3, §4). Refused by name with no depth.
    pub fn of(
        lanes: u32,
        end: u32,
        page: u32,
        frames: u32,
    ) -> std::result::Result<BeamLayout, String> {
        if end <= 1 {
            return Err(
                "a beam group (parents) binds with no depth: its pass states the slot \
                        past the search's last row (state-geometry parents-end)"
                    .to_string(),
            );
        }
        kernels_slang::runtime::beam::admit_lanes(lanes).map_err(|why| why.to_string())?;
        let frames = frames.max(1);
        let pages = end.div_ceil(page.max(1));
        let cap = lanes
            .checked_mul(pages)
            .and_then(|held| held.checked_add(lanes.checked_mul(frames + 1)?))
            .and_then(|held| held.checked_add(1))
            .ok_or_else(|| {
                format!("a beam group of {lanes} beams to slot {end} passes u32 pages")
            })?;
        Ok(BeamLayout {
            lanes,
            cap,
            frames,
            bases: 0,
        })
    }

    /// This layout on a hybrid buffering `rs_tokens` rows an rs page over
    /// kv pages of `page` slots: its set's rs entries held in the state
    /// (§11.1), one for each rs page of the pool's pages (a beam's slot `s`
    /// lies at `(f0 + q) page + s mod page`, `q` below the pool's `cap`).
    #[must_use]
    pub fn buffering(mut self, page: u32, rs_tokens: u32) -> BeamLayout {
        self.bases = self
            .cap
            .saturating_mul(page)
            .div_ceil(rs_tokens.max(1))
            .saturating_add(1);
        self
    }

    /// Where the set's rs entries start in the state, in words.
    #[must_use]
    pub fn bases_word(&self) -> u64 {
        kernels_slang::runtime::beam::state_words(self.lanes, self.cap, self.frames)
    }

    /// The state's bytes.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        (self.bases_word() + u64::from(self.bases)) * 4
    }
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
            resident: 0,
            pair: ends[1],
            beam: None,
        }
    }

    /// This layout with a beam group's state first in the instance's
    /// place, its heaps moved past it.
    #[must_use]
    pub fn with_beam(mut self, beam: Option<BeamLayout>) -> GuestLayout {
        let resident = beam.map_or(0, |beam| beam.bytes().next_multiple_of(ALIGN));
        let shift = resident - self.resident;
        for (side, at, _) in &mut self.stages {
            if *side == Side::Instance {
                *at += shift;
            }
        }
        self.instance += shift;
        self.resident = resident;
        self.beam = beam;
        self
    }

    /// The resident bytes at the head of the instance's place: a beam
    /// group's state, or none.
    #[must_use]
    pub fn resident(&self) -> u64 {
        self.resident
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
            // One place an instance: a beam group's state first, what a later
            // fire reads, copied with the place at a park; the stage heaps
            // past it are dead between fires (one arm: a pass runs whole
            // before the next stages anything).
            resident: self.resident,
            scratch: self.instance - self.resident,
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
    /// Its first bytes a later fire reads ([`GuestLayout::resident`]),
    /// moved with the place: [`Self::clear_dead`] leaves them.
    kept: u64,
    span: Option<(B::Buffer, u64)>,
}

impl<B: Backend> Placed<B> {
    /// `bytes` of state at `place`; refused by name where the place does
    /// not hold them whole in one span of one segment.
    pub fn at(pools: &Pools<B>, place: &ScratchPlace, bytes: u64) -> Result<Placed<B>> {
        let mut placed = Placed {
            place: place.clone(),
            bytes,
            kept: 0,
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
                let (buffer, at) = pools.chunk_place(first)?;
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

    /// This state with its first `kept` bytes live across a move
    /// ([`GuestLayout::resident`]).
    #[must_use]
    pub fn keeping(mut self, kept: u64) -> Placed<B> {
        self.kept = kept.min(self.bytes);
        self
    }

    /// Moves the state to `place`, the room the runtime named after an
    /// unpark: its first [`Self::keeping`] bytes copied there by the
    /// runtime, the rest dead between fires.
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

    /// Zeroes its bytes past the ones a later fire reads
    /// ([`Self::keeping`]): an unpark's room, whose head the runtime copied
    /// in and whose rest is another owner's until then.
    pub fn clear_dead(&self) -> Result<()> {
        let Some((buffer, at)) = &self.span else {
            return Ok(());
        };
        if self.bytes == self.kept {
            return Ok(());
        }
        let mut buffer = buffer.clone();
        buffer.zero_span(*at + self.kept, self.bytes - self.kept)
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
    /// Each live taking: its place, generation, instances bound and the
    /// name its group claims its state by ([`engine::guest_claims`]).
    live: Vec<(ScratchPlace, u64, u32, u64)>,
    next: u64,
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
            .find(|(held, taken, _, _)| held == place && *taken == generation)
        {
            Some((_, _, bound, _)) => {
                *bound += 1;
                false
            }
            None => {
                self.next += 1;
                self.live.push((place.clone(), generation, 1, self.next));
                true
            }
        }
    }

    /// The name the instances bound on `place` at `generation` claim its
    /// state by, as one group; `None` where none is.
    #[must_use]
    pub fn group(&self, place: &ScratchPlace, generation: u64) -> Option<u64> {
        self.live
            .iter()
            .find(|(held, taken, _, _)| held == place && *taken == generation)
            .map(|&(_, _, _, group)| group)
    }

    /// An instance bound on `place` at `generation` lets it go: closed, or
    /// moved to other room.
    pub fn give(&mut self, place: &ScratchPlace, generation: u64) {
        if let Some(at) = self
            .live
            .iter()
            .position(|(held, taken, _, _)| held == place && *taken == generation)
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
    use super::Takings;
    use engine::program::ScratchPlace;

    /// A pair's state is cleared at the first bind on a place and taking,
    /// not while another instance holds it, and again once the place is
    /// let go and taken anew or taken in another generation.
    #[test]
    fn a_pair_is_cleared_at_each_new_taking_and_only_then() {
        let place = ScratchPlace::Run { first: 4, slots: 2 };
        let mut takings = Takings::default();
        assert!(takings.take(&place, 0), "the first bind clears");
        assert!(
            !takings.take(&place, 0),
            "a sibling on the live taking does not"
        );
        assert!(
            takings.take(&place, 1),
            "another taking of the same refs clears"
        );
        takings.give(&place, 0);
        assert!(
            !takings.take(&place, 0),
            "one instance still holds taking 0"
        );
        takings.give(&place, 0);
        takings.give(&place, 0);
        assert!(
            takings.take(&place, 0),
            "taken again once every holder let go"
        );
        assert_ne!(
            takings.group(&place, 0),
            takings.group(&place, 1),
            "each live taking is a group of its own"
        );
        assert!(
            !takings.take(&ScratchPlace::None, 0),
            "no room, nothing to clear"
        );
    }
}
