//! What the runtime's role store does for a guest, done by hand: `Instance`
//! and `Channel` slots taken where the engine's facts say, so a test binds
//! instances and registers channels as a served one is.

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::{ChunkFacts, ShapeId, SlotRole};
use engine::program::{Footprint, InstanceBinding, RingSlots, ScratchPlace};

/// Slots of one role's shape, handed out from one chunk upward.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub id: ShapeId,
    pub per_chunk: u32,
    pub slot_bytes: u64,
    next: u32,
}

impl Shape {
    fn of(facts: &ChunkFacts, role: SlotRole, chunk: u32) -> Option<Shape> {
        let shape = facts.shapes.iter().find(|shape| shape.role == role)?;
        Some(Shape {
            id: shape.id,
            per_chunk: shape.slots_per_chunk,
            slot_bytes: shape.slot_bytes,
            next: chunk * shape.slots_per_chunk,
        })
    }

    /// The slots `footprint` takes: the next ones, adjacent, from the next
    /// chunk where the rest of this one is short of them.
    pub fn take(&mut self, footprint: &Footprint) -> Vec<u32> {
        let left = self.per_chunk - self.next % self.per_chunk;
        if footprint.slots > left && footprint.slots <= self.per_chunk {
            self.next += left;
        }
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

/// The two guest shapes the engine states, each in a chunk of its own, and
/// whole chunks a scratch past a chunk's `Instance` slots takes, handed out
/// from the last down.
#[derive(Debug, Clone)]
pub struct Slots {
    pub instance: Option<Shape>,
    pub channel: Option<Shape>,
    pub chunks: Vec<u32>,
    pub chunk_bytes: u64,
}

impl Slots {
    /// Instance slots in chunk `instance`, channel slots in `channel`.
    pub fn of(facts: &ChunkFacts, instance: u32, channel: u32) -> Slots {
        Slots {
            instance: Shape::of(facts, SlotRole::Instance, instance),
            channel: Shape::of(facts, SlotRole::Channel, channel),
            chunks: Vec::new(),
            chunk_bytes: facts.chunk_bytes,
        }
    }

    /// Where `state`'s instance bytes lie, as the runtime's role store lays
    /// them: one run of `Instance` slots where they fit a chunk's, else
    /// whole runs of [`Slots::chunks`] (adjacent ids, the last first).
    pub fn scratch(&mut self, state: &engine::GuestState) -> ScratchPlace {
        let bytes = state.per_instance;
        let run = state.chunks_a_piece();
        let chunk_bytes = self.chunk_bytes;
        if bytes == 0 {
            return ScratchPlace::None;
        }
        let shape = self.instance.as_mut().expect("scratch takes the Instance shape");
        if bytes <= u64::from(shape.per_chunk) * shape.slot_bytes {
            let footprint = Footprint {
                bytes,
                slots: u32::try_from(bytes.div_ceil(shape.slot_bytes)).expect("a chunk's slots"),
                contiguous: true,
            };
            let refs = shape.take(&footprint);
            return ScratchPlace::Run {
                first: refs[0],
                slots: footprint.slots,
            };
        }
        let need = usize::try_from(bytes.div_ceil(chunk_bytes)).expect("chunks");
        assert!(
            self.chunks.len() >= need,
            "a test's scratch of {bytes} bytes takes {need} chunks, and {} are handed",
            self.chunks.len()
        );
        let taken = self.chunks.split_off(self.chunks.len() - need);
        ScratchPlace::Chunks {
            run,
            pieces: taken.iter().step_by(run as usize).copied().collect(),
        }
    }

    /// Registers `registration` over the channel slots its ring facts state.
    pub fn register(&mut self, engine: &mut dyn Engine, registration: ChannelRegistration) {
        let slots = match engine
            .ring_facts(&registration)
            .expect("the channel's ring facts")
        {
            Some(facts) => self
                .channel
                .as_mut()
                .expect("ring facts need a channel shape")
                .take(&facts.footprint),
            None => Vec::new(),
        };
        engine
            .register_channel(&ChannelRegistration {
                slots,
                ..registration
            })
            .expect("the channel registers");
    }

    /// `binding` over the slots its program's facts state: its tables', and
    /// each ring's no registered channel (`registered`) covers.
    pub fn binding(
        &mut self,
        engine: &dyn Engine,
        binding: InstanceBinding,
        registered: &[ChannelId],
    ) -> InstanceBinding {
        let facts = engine
            .program_facts(binding.program)
            .expect("the program's facts");
        let instance_slots = match self.instance.as_mut() {
            Some(shape) if facts.instance.slots > 0 => shape.take(&facts.instance),
            _ => Vec::new(),
        };
        let ring_slots = facts
            .rings
            .iter()
            .filter(|ring| !registered.contains(&binding.channels[ring.channel as usize]))
            .map(|ring| RingSlots {
                channel: ring.channel,
                slots: self
                    .channel
                    .as_mut()
                    .expect("ring facts need a channel shape")
                    .take(&ring.footprint),
            })
            .collect();
        let state = engine
            .guest_state(binding.program, &binding.extents)
            .expect("the program's guest state");
        let scratch = self.scratch(&state);
        InstanceBinding {
            instance_slots,
            ring_slots,
            scratch,
            ..binding
        }
    }
}
