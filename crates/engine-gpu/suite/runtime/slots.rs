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
/// whole chunks for a guest's state past half a chunk, taken downward from
/// the chunk under both.
#[derive(Debug, Clone)]
pub struct Slots {
    pub instance: Option<Shape>,
    pub channel: Option<Shape>,
    /// The chunk the next run of whole chunks ends below, and a chunk's
    /// bytes.
    chunks: (u32, u64),
    /// Each (program, extents) pair's state, taken at its first bind.
    pairs: Vec<((engine::program::ProgramId, engine::program::BindExtents), ScratchPlace)>,
}

impl Slots {
    /// Instance slots in chunk `instance`, channel slots in `channel`;
    /// runs of whole chunks below the lower of them.
    pub fn of(facts: &ChunkFacts, instance: u32, channel: u32) -> Slots {
        Slots {
            instance: Shape::of(facts, SlotRole::Instance, instance),
            channel: Shape::of(facts, SlotRole::Channel, channel),
            chunks: (instance.min(channel), facts.chunk_bytes),
            pairs: Vec::new(),
        }
    }

    /// Room for `bytes` of a guest's state in pieces of `run` chunks, as the
    /// runtime takes it (`store::roles`): a run of `Instance` slots where two
    /// of its size share a chunk, else one run of whole chunks.
    pub fn state(&mut self, bytes: u64, run: u32) -> ScratchPlace {
        if bytes == 0 {
            return ScratchPlace::None;
        }
        let shape = self.instance.as_mut().expect("guest state needs an instance shape");
        let slots = bytes.div_ceil(shape.slot_bytes);
        if slots * 2 <= u64::from(shape.per_chunk) {
            let slots = u32::try_from(slots).expect("a run inside a chunk");
            let taken = shape.take(&Footprint {
                bytes,
                slots,
                contiguous: true,
            });
            return ScratchPlace::Run {
                first: taken[0],
                slots,
            };
        }
        let run = run.max(1);
        assert!(
            u64::from(run) * self.chunks.1 >= bytes,
            "a state of {bytes} bytes in one run of {run} chunks"
        );
        assert!(self.chunks.0 > run, "the test's chunks hold a run of {run}");
        self.chunks.0 -= run;
        ScratchPlace::Chunks {
            run,
            pieces: vec![self.chunks.0],
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
        // Its state and its pair's, the pair's taken once.
        let state = engine
            .guest_state(binding.program, &binding.extents, binding.geometry)
            .expect("the program's guest state");
        let scratch = self.state(state.instance_bytes(), state.run);
        let key = (binding.program, binding.extents);
        let pair = match self.pairs.iter().find(|(held, _)| *held == key) {
            Some((_, place)) => place.clone(),
            None => {
                let place = self.state(state.per_pair, state.run);
                self.pairs.push((key, place.clone()));
                place
            }
        };
        InstanceBinding {
            instance_slots,
            ring_slots,
            scratch,
            pair,
            ..binding
        }
    }
}
