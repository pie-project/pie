//! What the runtime's role store does for a guest, done by hand: `Instance`
//! and `Channel` slots taken where the engine's facts say, so a test binds
//! instances and registers channels as a served one is.

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::chunk::{ChunkFacts, ShapeId, SlotRole};
use engine::program::{Footprint, InstanceBinding, RingSlots};

/// Slots of one role's shape, handed out from one chunk upward.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub id: ShapeId,
    pub per_chunk: u32,
    next: u32,
}

impl Shape {
    fn of(facts: &ChunkFacts, role: SlotRole, chunk: u32) -> Option<Shape> {
        let shape = facts.shapes.iter().find(|shape| shape.role == role)?;
        Some(Shape {
            id: shape.id,
            per_chunk: shape.slots_per_chunk,
            next: chunk * shape.slots_per_chunk,
        })
    }

    /// The slots `footprint` takes: the next ones, adjacent.
    pub fn take(&mut self, footprint: &Footprint) -> Vec<u32> {
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

/// The two guest shapes the engine states, each in a chunk of its own.
#[derive(Debug, Clone, Copy)]
pub struct Slots {
    pub instance: Option<Shape>,
    pub channel: Option<Shape>,
}

impl Slots {
    /// Instance slots in chunk `instance`, channel slots in `channel`.
    pub fn of(facts: &ChunkFacts, instance: u32, channel: u32) -> Slots {
        Slots {
            instance: Shape::of(facts, SlotRole::Instance, instance),
            channel: Shape::of(facts, SlotRole::Channel, channel),
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
        InstanceBinding {
            instance_slots,
            ring_slots,
            ..binding
        }
    }
}
