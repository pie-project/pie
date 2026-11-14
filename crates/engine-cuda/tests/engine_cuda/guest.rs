//! What the runtime's role store does for a guest, done by hand: `Instance`
//! and `Channel` slots taken where the engine's facts say, so a test binds
//! instances and registers channels as a served one is.

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::{ChunkFacts, SlotRole};
use engine::program::{BoundInstance, Footprint, InstanceBinding, ProgramFacts, RingSlots};

/// Slots of one role's shape, handed out from one chunk upward.
#[derive(Debug, Clone, Copy)]
struct Shape {
    per_chunk: u32,
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

/// The two guest shapes, each in a chunk of its own.
#[derive(Debug, Clone, Copy)]
pub struct Guest {
    instance: Shape,
    channel: Shape,
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
        engine
            .bind_instance(&InstanceBinding {
                instance_slots,
                ring_slots,
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
    }
}

/// A guest item of a plane with no pool: a buffer of its own, the test
/// process's until it ends.
pub fn alone_span(_: SlotRole, _: &[u32], bytes: u64) -> engine_cuda::Result<u64> {
    let buffer = engine_cuda::device::Buffer::zeroed(bytes as usize)?;
    Ok(Box::leak(Box::new(buffer)).ptr())
}

/// A plane with no pool: its fires lay their programs' tables and scratch
/// in a buffer of its own, the test process's until it ends.
pub fn alone_plane() -> engine_cuda::ProgramPlane {
    const SPAN: usize = 256 << 20;
    let buffer = engine_cuda::device::Buffer::zeroed(SPAN).expect("the span's buffer");
    let end = Box::leak(Box::new(buffer)).ptr() + SPAN as u64;
    let mut plane = engine_cuda::ProgramPlane::new(engine_cuda::program::compile::Disk::disabled());
    plane.set_span((end, SPAN as u64));
    plane
}
