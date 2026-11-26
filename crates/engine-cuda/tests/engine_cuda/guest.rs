//! What the runtime's role store does for a guest, done by hand: `Instance`
//! and `Channel` slots taken where the engine's facts say, so a test binds
//! instances and registers channels as a served one is.

#[cfg(feature = "device")]
use engine::Engine;
#[cfg(feature = "device")]
use engine::GuestState;
use engine::ScratchPlace;
#[cfg(feature = "device")]
use engine::channel::{ChannelId, ChannelRegistration};
#[cfg(feature = "device")]
use engine::chunk::{ChunkFacts, SlotRole};
use engine::program::RingSlots;
#[cfg(feature = "device")]
use engine::program::{BoundInstance, Footprint, InstanceBinding, ProgramFacts};

/// Slots of one role's shape, handed out from one chunk upward.
#[derive(Debug, Clone, Copy)]
#[cfg(feature = "device")]
struct Shape {
    per_chunk: u32,
    slot_bytes: u64,
    next: u32,
}

#[cfg(feature = "device")]
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

/// The two guest shapes, each in a chunk of its own, and the chunks past
/// them, whole, for the guest state that lies in runs of chunks.
#[derive(Debug, Clone)]
#[cfg(feature = "device")]
pub struct Guest {
    instance: Shape,
    channel: Shape,
    /// The next whole chunk, and the chunks backed, of `chunk_bytes`.
    chunk: u32,
    chunk_bytes: u64,
    committed: u32,
    /// Each pair's state, taken at its first bind ([`Guest::pair`]).
    /// Keyed by the pair's program and extents, as written.
    pairs: Vec<(String, ScratchPlace)>,
}

#[cfg(feature = "device")]
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
            // first chunks, and the pairs' runs of chunks in the last.
            instance: Shape::of(facts, SlotRole::Instance, committed / 2 + 1),
            channel: Shape::of(facts, SlotRole::Channel, committed / 2),
            chunk: committed / 2 + 2,
            chunk_bytes: facts.chunk_bytes,
            committed,
            pairs: Vec::new(),
        }
    }

    /// The guest chunks of `engine`'s pool, its first segment backed.
    pub fn on(engine: &mut dyn Engine, facts: &ChunkFacts) -> Guest {
        let segments = facts.segments.committed.max(1);
        engine
            .pool()
            .expect("the engine keeps a pool")
            .commit(segments)
            .expect("the first segment commits");
        Guest::of(facts, segments * facts.segment_chunks)
    }

    /// Fresh `Channel` slots for a ring of `registration`, where the runtime
    /// would copy it back to on a restore.
    pub fn ring(
        &mut self,
        engine: &mut dyn Engine,
        registration: &ChannelRegistration,
    ) -> Vec<u32> {
        engine
            .channels()
            .expect("the engine holds channels")
            .ring_facts(registration)
            .expect("the channel's ring facts")
            .map_or_else(Vec::new, |facts| self.channel.take(&facts.footprint))
    }

    /// Fresh `Channel` slots for an item of `footprint`: a ring a test
    /// opens itself, as the runtime registers one.
    pub fn channel(&mut self, footprint: &Footprint) -> Vec<u32> {
        self.channel.take(footprint)
    }

    /// Registers `registration` over the channel slots its ring facts state.
    pub fn register(&mut self, engine: &mut dyn Engine, registration: ChannelRegistration) {
        let channels = engine.channels().expect("the engine holds channels");
        let slots = channels
            .ring_facts(&registration)
            .expect("the channel's ring facts")
            .map_or_else(Vec::new, |facts| self.channel.take(&facts.footprint));
        channels
            .open(&ChannelRegistration {
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

    /// Where a bind of `pair` with guest state `state` finds its own
    /// scratch and its pair's, as the runtime takes them (`store::roles`):
    /// each a run of `Instance` slots where two of its size share a chunk,
    /// else runs of `state.run` adjacent chunks; the instance's fresh, the
    /// pair's taken at its first bind and named the same after.
    pub fn places(
        &mut self,
        pair: (engine::ProgramId, impl std::fmt::Debug),
        state: &GuestState,
    ) -> (ScratchPlace, ScratchPlace) {
        let scratch = self.take_state(state.instance_bytes(), state);
        let pair = format!("{pair:?}");
        if let Some((_, place)) = self.pairs.iter().find(|(held, _)| *held == pair) {
            return (scratch, place.clone());
        }
        let place = self.take_state(state.per_pair, state);
        self.pairs.push((pair, place.clone()));
        (scratch, place)
    }

    fn take_state(&mut self, bytes: u64, state: &GuestState) -> ScratchPlace {
        if bytes == 0 {
            return ScratchPlace::None;
        }
        let shape = &self.instance;
        let slots = u32::try_from(bytes.div_ceil(shape.slot_bytes)).expect("slots count");
        if slots * 2 <= shape.per_chunk {
            let first = self.instance.next;
            self.instance.take(&Footprint {
                slots,
                ..Footprint::default()
            });
            return ScratchPlace::Run { first, slots };
        }
        let run = state.chunks_a_piece();
        let piece = u64::from(run) * self.chunk_bytes;
        let pieces = (0..bytes.div_ceil(piece))
            .map(|_| {
                let first = self.chunk;
                self.chunk += run;
                assert!(
                    self.chunk <= self.committed,
                    "a test's guest state of {bytes} bytes lies in runs of {run} chunks \
                     past the {} backed",
                    self.committed
                );
                first
            })
            .collect();
        ScratchPlace::Chunks { run, pieces }
    }

    /// Binds `binding` over the slots its program's facts state, its pair's
    /// state where its guest state says; every channel it names is a
    /// registered one.
    pub fn bind(&mut self, engine: &mut dyn Engine, binding: InstanceBinding) -> BoundInstance {
        let engine = engine.programs().expect("the engine runs programs");
        let facts = engine.facts(binding.program).expect("the program's facts");
        let named: Vec<ChannelId> = binding.channels.clone();
        let (instance_slots, ring_slots) =
            self.slots(&facts, |channel| named.get(channel as usize).is_some());
        let state = engine
            .guest_state(binding.program, &binding.extents, binding.geometry)
            .expect("the program's guest state");
        let (scratch, pair) = self.places((binding.program, binding.extents), &state);
        engine
            .bind(&InstanceBinding {
                instance_slots,
                ring_slots,
                scratch,
                pair,
                ..binding
            })
            .expect("the instance binds")
    }
}

/// The slots of an instance on a plane with no pool: none, and a ring for
/// each channel that is an item of its own. Its items lie where
/// [`alone_span`] puts them.
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
        scratch: ScratchPlace::None,
        pair: ScratchPlace::None,
        pair_generation: 0,
    }
}

/// A guest item of a plane with no pool: a buffer of its own, the test
/// process's until it ends.
pub fn alone_span(_: engine_cuda::program::Item<'_>, bytes: u64) -> engine_cuda::Result<u64> {
    let buffer = engine_cuda::device::Buffer::zeroed(bytes as usize)?;
    Ok(Box::leak(Box::new(buffer)).ptr())
}

/// A plane with no pool: each pair's tables and each instance's scratch lie
/// in a buffer of their own as long as the plane states them
/// ([`alone_span`], `ProgramPlane::pair_room`), the test process's until it
/// ends.
pub fn alone_plane() -> engine_cuda::ProgramPlane {
    alone_plane_fenced().0
}

/// [`alone_plane`], and the fence past its lane arenas: a zeroed run as long
/// as they are, right after them, that no launch of the plane's writes.
pub fn alone_plane_fenced() -> (engine_cuda::ProgramPlane, Fence) {
    let mut plane = engine_cuda::ProgramPlane::new(engine_cuda::program::compile::Disk::disabled());
    // The flight and the lane arenas a load lays: the widest flight a
    // power of two of the device's units, its samplers' arenas beside.
    use engine_cuda::program::launch::{Arenas, device_units};
    let units = device_units().max(1);
    plane.set_flight(1 << units.ilog2());
    // The most lanes a test's fire carries: one a unit, the widest a lane
    // sampler's test flies.
    plane.set_fire_lanes(units);
    let bytes = Arenas::bytes_for(plane.flight(), units);
    let arenas = engine_cuda::device::Buffer::zeroed(2 * bytes as usize).expect("the lane arenas");
    let arenas: &'static engine_cuda::device::Buffer = Box::leak(Box::new(arenas));
    plane.set_arenas(Arenas {
        at: arenas.ptr(),
        bytes,
    });
    (
        plane,
        Fence {
            buffer: arenas,
            at: bytes,
        },
    )
}

/// The run past a plane's lane arenas ([`alone_plane_fenced`]).
pub struct Fence {
    buffer: &'static engine_cuda::device::Buffer,
    at: u64,
}

impl Fence {
    /// Whether every byte of the run is still zero.
    pub fn intact(&self) -> bool {
        let mut run = vec![0u8; self.at as usize];
        self.buffer
            .read(self.at, &mut run)
            .expect("the fence reads");
        run.iter().all(|&byte| byte == 0)
    }
}
