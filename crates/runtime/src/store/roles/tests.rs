//! Instance and channel slots against a fake engine that states both
//! shapes: binds and closes give their slots back, a shared ring outlives
//! its channel until the last instance on it closes, a park and unpark move
//! an item through the host tier and tell the engine its new refs, and the
//! record math is the same with the shapes as without them.

use std::sync::{Arc, Mutex};

use ::engine::chunk::{
    ChunkFacts, Growth, Reserved, ReservedRun, SegmentFacts, ShapeId as EngineShape, SlotKind,
    SlotRole, SlotShape,
};
use ::engine::{
    BoundInstance, ChannelId, ChannelRegistration, Engine, Footprint, FrameSubmission, FrameTicket,
    InstanceBinding, InstanceId, LoadRequest, Loaded, ProgramFacts, ProgramId, Rebind,
    RegisteredChannel, RingFacts,
};

use super::{Owner, Roles, RolesHandle, bind_instance, learn_program, register_channel};
use crate::store::shared::{Shared, SharedPool};

const KIB: u64 = 1 << 10;
const MIB: u64 = 1 << 20;
const CHUNK: u64 = MIB;
const LIST: EngineShape = EngineShape(0);
const RECORD: EngineShape = EngineShape(1);
const INSTANCE: EngineShape = EngineShape(2);
const CHANNEL: EngineShape = EngineShape(3);
const PROGRAM: ProgramId = ProgramId(7);

fn shape(id: EngineShape, kind: SlotKind, slot_bytes: u64, role: SlotRole) -> SlotShape {
    SlotShape {
        id,
        kind,
        slot_bytes,
        slots_per_chunk: u32::try_from(CHUNK / slot_bytes).unwrap(),
        sets_per_working_set: 1,
        rows: Vec::new(),
        pinned: None,
        role,
    }
}

/// 32 chunks of 1 MiB, a kv list and a one-piece record, and (`roles`) a
/// 4 KiB `Instance` and a 1 KiB `Channel` shape; 8 host chunks.
fn facts(roles: bool) -> ChunkFacts {
    let mut shapes = vec![
        shape(
            LIST,
            SlotKind::List {
                tokens: 16,
                rate: 1,
                window: None,
            },
            64 * KIB,
            SlotRole::State,
        ),
        shape(
            RECORD,
            SlotKind::Fixed { pieces: 1 },
            256 * KIB,
            SlotRole::State,
        ),
    ];
    if roles {
        shapes.push(shape(
            INSTANCE,
            SlotKind::Fixed { pieces: 1 },
            4 * KIB,
            SlotRole::Instance,
        ));
        shapes.push(shape(
            CHANNEL,
            SlotKind::Fixed { pieces: 1 },
            KIB,
            SlotRole::Channel,
        ));
    }
    let facts = ChunkFacts {
        chunk_bytes: CHUNK,
        total_chunks: 32,
        segment_chunks: 32,
        segments: SegmentFacts {
            committed: 1,
            max: 1,
            growth: Growth::Fixed,
        },
        reserved: vec![ReservedRun {
            what: Reserved::Null,
            first: 0,
            chunks: 1,
        }],
        shapes,
        act_chunks: vec![(16, 1), (256, 4)],
        host_chunks: 8,
        ..ChunkFacts::default()
    };
    facts.validate().unwrap();
    facts
}

struct Pools {
    device: Shared,
    host: Shared,
    roles: RolesHandle,
}

fn pools() -> Pools {
    let facts = facts(true);
    let device = Arc::new(Mutex::new(SharedPool::new(&facts, 256).unwrap()));
    let host = Arc::new(Mutex::new(
        SharedPool::tier(&facts, facts.host_chunks)
            .unwrap()
            .unwrap(),
    ));
    let roles = Roles::new(&device, Some(&host)).expect("the pool states both shapes");
    Pools {
        device,
        host,
        roles: Arc::new(Mutex::new(roles)),
    }
}

/// Slots held in a pool, in bytes.
fn held(pool: &Shared) -> u64 {
    pool.lock().unwrap().pool().demand().live
}

/// The engine half: states an instance of 5000 bytes in three slots (its
/// cells packed whole, so more than the bytes alone take), one ring of its
/// own on channel index 1, and a channel ring of two slots; records what it
/// is bound with and rebound to.
#[derive(Default)]
struct FakeEngine {
    next: u64,
    bindings: Vec<InstanceBinding>,
    registrations: Vec<ChannelRegistration>,
    rebinds: Vec<Rebind>,
}

impl Engine for FakeEngine {
    fn kind(&self) -> &'static str {
        "fake-roles"
    }

    fn load(&mut self, _request: LoadRequest) -> ::engine::Result<Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(&mut self, _frame: &FrameSubmission) -> ::engine::Result<FrameTicket> {
        unreachable!("nothing fires here")
    }

    fn program_facts(&self, program: ProgramId) -> ::engine::Result<ProgramFacts> {
        assert_eq!(program, PROGRAM);
        Ok(ProgramFacts {
            instance: Footprint {
                bytes: 5000,
                slots: 3,
                contiguous: false,
            },
            rings: vec![
                RingFacts {
                    channel: 0,
                    cell_bytes: 64,
                    capacity: 4,
                    footprint: Footprint {
                        bytes: 352,
                        slots: 1,
                        contiguous: false,
                    },
                },
                RingFacts {
                    channel: 1,
                    cell_bytes: 600,
                    capacity: 2,
                    footprint: Footprint {
                        bytes: 1832,
                        slots: 3,
                        contiguous: false,
                    },
                },
            ],
        })
    }

    fn ring_facts(
        &self,
        _registration: &ChannelRegistration,
    ) -> ::engine::Result<Option<RingFacts>> {
        Ok(Some(RingFacts {
            channel: 0,
            cell_bytes: 64,
            capacity: 20,
            footprint: Footprint {
                bytes: 1376,
                slots: 2,
                contiguous: false,
            },
        }))
    }

    fn register_channel(
        &mut self,
        registration: &ChannelRegistration,
    ) -> ::engine::Result<RegisteredChannel> {
        self.registrations.push(registration.clone());
        Ok(RegisteredChannel {
            id: registration.id,
            reader_wait_id: 0,
            writer_wait_id: 0,
            mirror: None,
        })
    }

    fn bind_instance(&mut self, binding: &InstanceBinding) -> ::engine::Result<BoundInstance> {
        self.next += 1;
        self.bindings.push(binding.clone());
        Ok(BoundInstance {
            id: InstanceId(self.next),
            program: binding.program,
            geometry: binding.geometry,
        })
    }

    fn rebind_slots(&mut self, rebind: Rebind) -> ::engine::Result<()> {
        self.rebinds.push(rebind);
        Ok(())
    }
}

fn holder(n: u128) -> super::Holder {
    uuid::Uuid::from_u128(n)
}

fn registered(roles: &RolesHandle, engine: &mut FakeEngine, id: u64, by: u128) -> ChannelId {
    let plan = ChannelRegistration {
        id: ChannelId(id),
        capacity: 20,
        ..ChannelRegistration::default()
    };
    register_channel(
        Some(roles),
        engine,
        &plan,
        Some(holder(by)),
        |engine, plan| Ok(engine.register_channel(plan)?),
    )
    .unwrap();
    plan.id
}

/// Binds an instance of [`PROGRAM`] on `channels` (index 0 a registered
/// channel where given, index 1 the instance's own ring).
fn bind(
    roles: &RolesHandle,
    engine: &mut FakeEngine,
    channels: Vec<ChannelId>,
    by: u128,
) -> InstanceId {
    learn_program(roles, engine, PROGRAM).unwrap();
    let registered: Vec<ChannelId> = engine.registrations.iter().map(|plan| plan.id).collect();
    let binding = InstanceBinding {
        program: PROGRAM,
        channels,
        ..InstanceBinding::default()
    };
    bind_instance(Some(roles), engine, &binding, Some(holder(by)), |id| {
        registered.contains(&id)
    })
    .unwrap()
    .id
}

#[test]
fn a_bind_takes_the_slots_its_facts_count_and_a_close_gives_them_back() {
    let Pools { device, roles, .. } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);
    let channel = registered(&roles, &mut engine, 1, 1);
    let ring = engine.registrations[0].slots.clone();
    assert_eq!(ring.len(), 2, "the ring takes the slots ring_facts states");
    let id = bind(&roles, &mut engine, vec![channel, ChannelId(99)], 1);
    let binding = &engine.bindings[0];
    // Three slots, as the engine counts: not ceil(5000 / 4 KiB) = 2.
    assert_eq!(binding.instance_slots.len(), 3);
    // Ring 0 is the registered channel's; ring 1 the instance's own.
    assert_eq!(binding.ring_slots.len(), 1);
    assert_eq!(binding.ring_slots[0].channel, 1);
    assert_eq!(binding.ring_slots[0].slots.len(), 3);
    assert_eq!(held(&device) - before, 3 * 4 * KIB + (3 + 2) * KIB);

    roles.lock().unwrap().close_instance(id);
    assert_eq!(held(&device) - before, 2 * KIB, "the channel's ring stays");
    roles.lock().unwrap().close_channel(channel);
    assert_eq!(held(&device), before, "every slot is back");
}

#[test]
fn a_shared_ring_is_freed_when_the_last_instance_on_it_closes() {
    let Pools { device, roles, .. } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);
    let channel = registered(&roles, &mut engine, 1, 1);
    let first = bind(&roles, &mut engine, vec![channel], 1);
    let second = bind(&roles, &mut engine, vec![channel], 2);
    let ring = roles.lock().unwrap().slots(Owner::Channel(channel));
    assert_eq!(ring.len(), 2);

    roles.lock().unwrap().close_channel(channel);
    assert_eq!(
        roles.lock().unwrap().slots(Owner::Channel(channel)),
        ring,
        "two instances still read it"
    );
    roles.lock().unwrap().close_instance(first);
    assert_eq!(roles.lock().unwrap().slots(Owner::Channel(channel)), ring);
    roles.lock().unwrap().close_instance(second);
    assert!(
        roles
            .lock()
            .unwrap()
            .slots(Owner::Channel(channel))
            .is_empty()
    );
    assert_eq!(held(&device), before);
}

#[test]
fn a_park_and_unpark_moves_through_the_host_and_rebinds_the_new_refs() {
    let Pools {
        device,
        host,
        roles,
    } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);
    let channel = registered(&roles, &mut engine, 1, 1);
    let instance = bind(&roles, &mut engine, vec![channel], 1);
    // Another process's instance on a channel of its own: it stays.
    let other_channel = registered(&roles, &mut engine, 2, 2);
    let other = bind(&roles, &mut engine, vec![other_channel], 2);
    let others = roles.lock().unwrap().slots(Owner::Instance(other));
    let resident = held(&device);
    let was = roles.lock().unwrap().slots(Owner::Instance(instance));

    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    let copies = park.copies().to_vec();
    assert_eq!(copies.len(), 2, "an instance and a channel shape");
    let instance_copy = copies.iter().find(|copy| copy.shape == INSTANCE).unwrap();
    assert_eq!(&instance_copy.device, &was[..3]);
    roles.lock().unwrap().commit(park);
    let parked = (3 * 4 + 3 + 2) * KIB;
    assert_eq!(
        resident - held(&device),
        parked,
        "its device slots are free"
    );
    assert_eq!(held(&host), parked);
    assert_eq!(roles.lock().unwrap().slots(Owner::Instance(other)), others);
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty());

    // Something else takes the freed slots meanwhile, so the item comes
    // back elsewhere.
    let instance_pool = device.lock().unwrap().roles().instance.unwrap().pool;
    let squatter = device.lock().unwrap().take_refs(instance_pool, 3).unwrap();
    let unpark = roles
        .lock()
        .unwrap()
        .prepare_unpark(holder(1))
        .unwrap()
        .unwrap();
    for copy in unpark.copies() {
        assert_eq!(copy.device.len(), copy.host.len());
    }
    roles.lock().unwrap().commit(unpark);
    assert_eq!(held(&host), 0, "the host slots are back");
    let now = roles.lock().unwrap().slots(Owner::Instance(instance));
    assert_ne!(now, was);

    // The restore's rebind: every moved item, its new refs.
    let stale = roles.lock().unwrap().stale(holder(1));
    assert_eq!(stale.len(), 2);
    for rebind in stale {
        let owner = Owner::of(&rebind);
        engine.rebind_slots(rebind).unwrap();
        roles.lock().unwrap().rebound(owner);
    }
    assert!(roles.lock().unwrap().stale(holder(1)).is_empty());
    // Each shape apart: the tables, then its own ring (channel index 1;
    // index 0 is the registered channel's).
    assert!(engine.rebinds.contains(&Rebind::Instance {
        id: instance,
        instance: now[..3].to_vec(),
        rings: vec![::engine::RingSlots {
            channel: 1,
            slots: now[3..].to_vec(),
        }],
    }));
    assert!(engine.rebinds.contains(&Rebind::Channel {
        id: channel,
        slots: roles.lock().unwrap().slots(Owner::Channel(channel)),
    }));
    assert!(
        !engine
            .rebinds
            .iter()
            .any(|rebind| Owner::of(rebind) == Owner::Instance(other))
    );

    device.lock().unwrap().give_refs(instance_pool, &squatter);
    for id in [instance, other] {
        roles.lock().unwrap().close_instance(id);
    }
    for id in [channel, other_channel] {
        roles.lock().unwrap().close_channel(id);
    }
    assert_eq!(held(&device), before);
}

#[test]
fn an_item_closed_while_parked_gives_back_its_host_slots() {
    let Pools {
        device,
        host,
        roles,
    } = pools();
    let mut engine = FakeEngine::default();
    let before = held(&device);
    let instance = bind(&roles, &mut engine, Vec::new(), 1);
    let park = roles.lock().unwrap().prepare_park(holder(1)).unwrap();
    roles.lock().unwrap().commit(park);
    assert!(held(&host) > 0);
    roles.lock().unwrap().close_instance(instance);
    assert_eq!(held(&host), 0);
    assert_eq!(held(&device), before);
}

#[test]
fn the_record_math_is_the_same_with_the_role_shapes_as_without() {
    let (with, without) = (facts(true), facts(false));
    let a = SharedPool::new(&with, 4096).unwrap();
    let b = SharedPool::new(&without, 4096).unwrap();
    assert_eq!(a.fixed(), b.fixed(), "a record is the state shapes alone");
    assert_eq!(a.records_most(), b.records_most());
    assert_eq!(a.records_capacity(), b.records_capacity());
    assert_eq!(a.records_available(), b.records_available());
    assert_eq!(a.floor(), b.floor(), "one working set's chunks");
    let units = crate::store::rs::RsUnits {
        slots: 3,
        ..Default::default()
    };
    assert_eq!(a.list_slots_for(units), b.list_slots_for(units));
    assert_eq!(SharedPool::scale_of(&with), SharedPool::scale_of(&without));
    assert_eq!(
        crate::store::chunks::StorePools::of(&with).unwrap(),
        crate::store::chunks::StorePools::of(&without).unwrap()
    );
    assert!(b.roles().is_empty());
    assert_eq!(a.roles().instance.map(|shape| shape.id), Some(INSTANCE));
    assert_eq!(a.roles().channel.map(|shape| shape.id), Some(CHANNEL));
    // A pool that states neither shape has no roles: nothing changes.
    let b = Arc::new(Mutex::new(b));
    assert!(Roles::new(&b, None).is_none());
}
