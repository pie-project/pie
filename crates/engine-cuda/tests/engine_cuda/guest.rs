//! What the runtime's role store does for a guest, done by hand: `Instance`
//! and `Channel` slots taken where the engine's facts say, so a test binds
//! instances and registers channels as a served one is.

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::{ChunkFacts, SlotRole};
use engine::program::{
    BindExtents, BoundInstance, Footprint, InstanceBinding, ProgramFacts, RingSlots,
};
use engine::{GuestState, ProgramId, ScratchPlace};

/// Slots of one role's shape, handed out from one chunk upward.
#[derive(Debug, Clone, Copy)]
struct Shape {
    per_chunk: u32,
    slot_bytes: u64,
    next: u32,
}

impl Shape {
    fn of(facts: &ChunkFacts, role: SlotRole, chunk: u32) -> Shape {
        let shape = facts
            .shapes
            .iter()
            .find(|shape| shape.role == role)
            .expect("the load states the guest shapes");
        Shape {
            per_chunk: shape.slots_per_chunk,
            slot_bytes: shape.slot_bytes,
            next: chunk * shape.slots_per_chunk,
        }
    }

    /// Room for `bytes` of guest scratch, as the runtime's role store
    /// takes it: one run of slots in a chunk where they fit a chunk's,
    /// else whole runs of `run` adjacent chunks (`engine::ScratchPlace`),
    /// each below `end`.
    fn scratch(&mut self, bytes: u64, chunk_bytes: u64, run: u32, end: u32) -> ScratchPlace {
        let per_chunk = self.per_chunk;
        let place = if bytes == 0 {
            return ScratchPlace::None;
        } else if bytes <= u64::from(per_chunk) * self.slot_bytes {
            let slots = u32::try_from(bytes.div_ceil(self.slot_bytes)).expect("a run's slots");
            if self.next % per_chunk + slots > per_chunk {
                self.next = self.next.next_multiple_of(per_chunk);
            }
            let first = self.next;
            self.next += slots;
            ScratchPlace::Run { first, slots }
        } else {
            let run = run.max(1);
            let pieces =
                u32::try_from(bytes.div_ceil(chunk_bytes * u64::from(run))).expect("whole runs");
            let first = self.next.div_ceil(per_chunk);
            self.next = (first + pieces * run) * per_chunk;
            ScratchPlace::Chunks {
                run,
                pieces: (0..pieces).map(|piece| first + piece * run).collect(),
            }
        };
        assert!(
            self.next.div_ceil(per_chunk) <= end,
            "a test's guest scratch lies in the chunks backed"
        );
        place
    }

    fn take(&mut self, footprint: &Footprint) -> Vec<u32> {
        let first = self.next;
        self.next += footprint.slots;
        assert_eq!(
            first / self.per_chunk,
            (self.next - 1) / self.per_chunk,
            "a test's items fit one chunk"
        );
        (first..self.next).collect()
    }
}

/// The two guest shapes, each in a chunk of its own, and the guests'
/// scratch in the chunks past them.
#[derive(Debug, Clone)]
pub struct Guest {
    instance: Shape,
    channel: Shape,
    scratch: Shape,
    chunk_bytes: u64,
    committed: u32,
    /// Each (program, extents) pair's state, taken at its first bind.
    pairs: Vec<(ProgramId, BindExtents, ScratchPlace)>,
}

impl Guest {
    /// The guest chunks of a pool whose first `committed` chunks are backed.
    pub fn of(facts: &ChunkFacts, committed: u32) -> Guest {
        // Chunk 0 is the null chunk; the state's chunks follow it.
        let committed = committed.min(facts.state_chunks() + 1);
        assert!(
            committed >= 3,
            "two committed chunks past the null chunk, of {} committed and {} of state",
            committed,
            facts.state_chunks()
        );
        Guest {
            // The middle of what is backed: a test's own state sits in the
            // first chunks, and a fire lays its programs in the last.
            instance: Shape::of(facts, SlotRole::Instance, committed / 2 + 1),
            channel: Shape::of(facts, SlotRole::Channel, committed / 2),
            scratch: Shape::of(facts, SlotRole::Instance, committed / 2 + 2),
            chunk_bytes: facts.chunk_bytes,
            committed,
            pairs: Vec::new(),
        }
    }

    /// Where an instance of `program` at `extents` with `state` keeps its
    /// scratch, and its pair's state: taken at the pair's first bind.
    pub fn scratch(
        &mut self,
        program: ProgramId,
        extents: &BindExtents,
        state: GuestState,
    ) -> (ScratchPlace, ScratchPlace) {
        let (chunk, end) = (self.chunk_bytes, self.committed);
        let scratch = self
            .scratch
            .scratch(state.per_instance, chunk, state.run, end);
        let pair = match self
            .pairs
            .iter()
            .find(|(held, at, _)| *held == program && at == extents)
        {
            Some((_, _, place)) => place.clone(),
            None => {
                let place = self.scratch.scratch(state.per_pair, chunk, state.run, end);
                self.pairs.push((program, extents.clone(), place.clone()));
                place
            }
        };
        (scratch, pair)
    }

    /// Fresh room for an instance's scratch and its pair's state, as an
    /// unpark takes it: elsewhere than any taken before.
    pub fn moved(
        &mut self,
        program: ProgramId,
        extents: eta_exec::Extents,
        state: GuestState,
    ) -> (ScratchPlace, ScratchPlace) {
        let extents = bind_extents(extents);
        self.pairs
            .retain(|(held, at, _)| *held != program || *at != extents);
        self.scratch(program, &extents, state)
    }

    /// The slots an instance of a program with `facts` and `state` at
    /// `extents` takes on a shell: its tables', its rings', its scratch's
    /// and its pair's.
    pub fn shell_slots(
        &mut self,
        program: ProgramId,
        facts: &ProgramFacts,
        extents: eta_exec::Extents,
        state: GuestState,
    ) -> engine_cuda::program::Slots {
        let (instance, rings) = self.slots(facts, |_| false);
        let (scratch, pair) = self.scratch(program, &bind_extents(extents), state);
        engine_cuda::program::Slots {
            instance,
            rings,
            scratch,
            pair,
        }
    }

    /// The guest chunks of `engine`'s pool, its first segment backed.
    pub fn on(engine: &mut dyn Engine, facts: &ChunkFacts) -> Guest {
        let segments = facts.segments.committed.max(1);
        engine
            .commit_segments(segments)
            .expect("the first segment commits");
        Guest::of(facts, segments * facts.segment_chunks)
    }

    /// Registers `registration` over the channel slots its ring facts state.
    pub fn register(&mut self, engine: &mut dyn Engine, registration: ChannelRegistration) {
        let slots = engine
            .ring_facts(&registration)
            .expect("the channel's ring facts")
            .map_or_else(Vec::new, |facts| self.channel.take(&facts.footprint));
        engine
            .register_channel(&ChannelRegistration {
                slots,
                ..registration
            })
            .expect("the channel registers");
    }

    /// The slots an instance of a program with `facts` takes: its tables',
    /// and each ring's but those `registered` says a channel covers.
    pub fn slots(
        &mut self,
        facts: &ProgramFacts,
        registered: impl Fn(u32) -> bool,
    ) -> (Vec<u32>, Vec<RingSlots>) {
        let instance = self.instance.take(&facts.instance);
        let rings = facts
            .rings
            .iter()
            .filter(|ring| !registered(ring.channel))
            .map(|ring| RingSlots {
                channel: ring.channel,
                slots: self.channel.take(&ring.footprint),
            })
            .collect();
        (instance, rings)
    }

    /// Binds `binding` over the slots its program's facts state; every
    /// channel it names is a registered one.
    pub fn bind(&mut self, engine: &mut dyn Engine, binding: InstanceBinding) -> BoundInstance {
        let facts = engine
            .program_facts(binding.program)
            .expect("the program's facts");
        let named: Vec<ChannelId> = binding.channels.clone();
        let (instance_slots, ring_slots) =
            self.slots(&facts, |channel| named.get(channel as usize).is_some());
        let state = engine
            .guest_state(binding.program, &binding.extents)
            .expect("the program's guest state");
        let (scratch, pair) = self.scratch(binding.program, &binding.extents, state);
        engine
            .bind_instance(&InstanceBinding {
                instance_slots,
                ring_slots,
                scratch,
                pair,
                ..binding
            })
            .expect("the instance binds")
    }
}

/// `extents` as a binding states them.
fn bind_extents(extents: eta_exec::Extents) -> BindExtents {
    BindExtents {
        row_count: extents.row_count,
        token_count: extents.token_count,
        sampled_rows: extents.sampled_rows,
        query_len: extents.query_len,
        key_len: extents.key_len,
    }
}

/// The slots of an instance on a plane with no pool: none, a ring for each
/// channel that is an item of its own, and a run each for its scratch and
/// its pair's state. Its items lie where [`alone_span`] puts them.
pub fn alone_slots(
    plane: &engine_cuda::ProgramPlane,
    program: engine::ProgramId,
) -> engine_cuda::program::Slots {
    let held = plane.held(program).expect("the program's items");
    engine_cuda::program::Slots {
        instance: Vec::new(),
        rings: held
            .ring_bytes
            .iter()
            .enumerate()
            .filter(|(_, bytes)| bytes.is_some())
            .map(|(channel, _)| RingSlots {
                channel: channel as u32,
                slots: Vec::new(),
            })
            .collect(),
        scratch: ScratchPlace::Run { first: 0, slots: 1 },
        pair: ScratchPlace::Run { first: 0, slots: 1 },
    }
}

/// A guest item of a plane with no pool: a buffer of its own, the test
/// process's until it ends.
pub fn alone_span(_: SlotRole, _: &[u32], bytes: u64) -> engine_cuda::Result<u64> {
    let buffer = engine_cuda::device::Buffer::zeroed(bytes.max(256) as usize)?;
    Ok(Box::leak(Box::new(buffer)).ptr())
}

/// A plane with no pool: its programs' tables and scratch lie in buffers of
/// their own ([`alone_span`]), its fires of up to 256 lanes, its samplers'
/// lane arenas a buffer of their own sized for every block a 256-lane
/// launch may run (the load lays them in the activation's span).
pub fn alone_plane() -> engine_cuda::ProgramPlane {
    use engine_cuda::program::launch::{Arenas, LANE_SAMPLER_BLOCKS};
    let mut plane = engine_cuda::ProgramPlane::new(engine_cuda::program::compile::Disk::disabled());
    plane.set_flight(256);
    plane.set_fire_lanes(256);
    let bytes = Arenas::bytes_for(256, 256 * LANE_SAMPLER_BLOCKS);
    if let Ok(buffer) = engine_cuda::device::Buffer::zeroed(bytes as usize) {
        let at = Box::leak(Box::new(buffer)).ptr();
        plane.set_arenas(Arenas { at, bytes });
    }
    plane
}
